# AIFD 在 vLLM 侧的改动说明（per-sample 选层）

> 目标读者：要接手/修改这段代码的人。
> 所有代码片段都来自仓库当前状态，行号对得上。
> 硬件：Ascend 910B　　vLLM：`/vllm-workspace/vllm`（0.23.1rc1）　　vllm-ascend：`/vllm-workspace/vllm-ascend`

---

## 1. 一句话概括

训练端需要一份「每个样本自己挑一个目标层的隐状态」。vLLM 原本只会按固定层号吐 hidden states，
我们在**不碰 attention 后端**的前提下，加了一个额外的通道：

```
隐状态通道 = [aux_0, aux_1, ..., aux_{n-1}, AIFD]
                                          ↑ 新增，由注意力熵自动选层
```

AIFD 那一通道，是**在候选层里挑一个"注意力最稳定"的层**，把该层整条序列的 hidden states 原样输出。
训练端拿它做中间层特征蒸馏（SmoothL1）。

---

## 2. 判据（per-sample 版）

```
He[l, t] = mean_h ( lse[l,t,h] − scale · ⟨q[t,h,:], O^K[t,h,:]⟩ )   ← 层 l 在 token t 的注意力熵
C_l      = mean_{t ∈ request} He[l, t]                             ← 对**整个 request** 求平均 ★per-sample 的关键
score_l  = C_l + |C_l − C_{l−1}|                                   ← 熵本身 + 熵的层间变化
best     = argmin_{l ∈ 候选层} score_l                              ← 每个 request 一个层号
```

`lse` 是 softmax 的 log-sum-exp，`O^K` 是「把 value 换成 key」的注意力输出。
两者由**同一次**算子调用给出，且**不落地 S×S 分数矩阵**：

```python
out_k, lse = npu_fused_infer_attention_score(q, k, v=key, ..., softmax_lse_flag=True)
H = lse − scale·⟨q, O^K⟩
```

推导：`H = −Σ p·log p = −Σ p·(s − lse) = lse − Σ p·s`，
而 `Σ p·s = scale·⟨q, Σ p·k⟩ = scale·⟨q, O^K⟩`。
（实测验证见 `RESULT_lse_entropy.md`，误差 ~1e-3）

---

## 3. 改动清单

| 文件 | 改动 | 作用 |
|---|---|---|
| `vllm/model_executor/models/aifd.py` | **新增 502 行** | 全部逻辑 |
| `vllm/model_executor/models/interfaces.py` | +`_aifd_hook` 注入点；`_maybe_add_hidden_state` 里加一次调用 | 把 AIFD 接进层循环 |
| `vllm/model_executor/models/extract_hidden_states.py` | `num_hidden_states` **+1** | 缓冲区分大一个通道 |
| `vllm/v1/spec_decode/extract_hidden_states.py` | `num_hidden_states` **+1** | proposer 侧的缓冲区（**容易漏**） |
| `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` | 新增 `maybe_enable_aifd()` | 共享的开关 |
| `vllm/v1/worker/gpu_model_runner.py` | 调用 `maybe_enable_aifd` | 上游 v1 runner（防御性） |
| **`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`** | 调用 `maybe_enable_aifd` | **Ascend 上真正生效的路径** |
| `speculators/scripts/launch_vllm.py` | `--aifd-candidate-layers` | 写进 `hf_config.aifd_config` |

> ⚠️ 模型 runner 有三份：`vllm/v1/worker/gpu_model_runner.py`、
> `vllm/v1/worker/gpu/model_runner.py`，而 **Ascend 实际用的是
> `vllm_ascend/worker/model_runner_v1.py`**。定位方法是看日志里的**文件名**
> （`[model_runner_v1.py:3825]`）。

---

## 4. 逐函数详解

### 4.1 层号约定（最容易搞错的地方）

```python
# aifd.py: 模块 docstring
# 与 `eagle_aux_hidden_state_layer_ids` 一致：id `L` 取的是**第 L-1 层（0-based）的输出**
```

对应到模型代码（`qwen2.py:420`，Qwen3 继承自 Qwen2）：

```python
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    hidden_states, residual = layer(positions, hidden_states, residual)
    self._maybe_add_hidden_state(
        aux_hidden_states, idx + 1, hidden_states, residual   # ← 传的是 idx+1，即"第 idx 层的输出"
    )
```

所以：**aux id `L` ⟺ 模型第 `L-1` 层（0-based）的输出 ⟺ HF `hidden_states[L]`**。
`enable_aifd` 里挂 hook 时用的 `model_idx = aux_id - 1` 就是这个换算。

### 4.2 配置读取

```python
def aifd_config_from(hf_config: Any) -> dict | None:
    """从 hf_config 读 aifd 配置；没配或候选层为空则返回 None。"""
    cfg = getattr(hf_config, "aifd_config", None)   # 用 getattr 带默认值，没配也不炸
    if not isinstance(cfg, dict):                    # 类型不对就当没配
        return None
    candidates = cfg.get("candidate_layers") or []   # `or []` 同时挡掉 None 和空 list
    if not candidates:
        return None
    return cfg


def aifd_extra_channels(hf_config: Any) -> int:
    """AIFD 会多出几个 hidden-states 通道（关闭时是 0）。"""
    return 1 if aifd_config_from(hf_config) is not None else 0
```

**为什么要有 `aifd_extra_channels`**：按通道数分配缓冲区的地方有**三处**，
必须用同一个函数算，否则会漏。漏掉任何一处的表现都是
`shape mismatch: [T, 12, H] vs [T, 13, H]`，而且报错点离原因很远。

```python
# 1) 模型侧
# vllm/model_executor/models/extract_hidden_states.py __init__
num_aux = len(getattr(self.hf_config, "eagle_aux_hidden_state_layer_ids", []))
self.num_hidden_states = num_aux + aifd_extra_channels(self.hf_config)

# 2) proposer 侧 —— 这个最容易漏
# vllm/v1/spec_decode/extract_hidden_states.py __init__
self.num_hidden_states = len(layer_ids) + aifd_extra_channels(self.hf_config)

# 3) 缓冲区本体（隐含用上面那个值）
self.hidden_states = torch.zeros(
    (self.max_num_tokens, self.num_hidden_states, self.hidden_size), ...)
```

### 4.3 挂 hook：`enable_aifd`

```python
def _find_layers(model: nn.Module) -> nn.ModuleList | None:
    """Qwen2/Qwen3/Llama 系在 ``model.model.layers``；有些模型在 ``model.layers``。"""
    for path in ("model.layers", "layers", "language_model.model.layers",
                 "language_model.layers"):
        obj: Any = model
        try:
            for part in path.split("."):     # 逐级 getattr，走不通就换下一条路径
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    return None


def enable_aifd(model: nn.Module, candidates: list[int]) -> bool:
    if not candidates:
        return False

    layers = _find_layers(model)
    if layers is None:
        logger.warning("AIFD: 找不到 decoder layers，已禁用")
        return False

    # 把回调挂到 interfaces 的注入点上（而不是让热路径每次去 import 本模块）
    from vllm.model_executor.models.interfaces import set_aifd_hook
    set_aifd_hook(note_layer_and_maybe_emit)

    num_layers = len(layers)
    _STATE.candidates = tuple(sorted(candidates))
    _STATE.num_layers = num_layers          # ★ 后面判断"是不是最后一次调用"要用
    _STATE.enabled = True
    for aux_id in _STATE.candidates:
        model_idx = aux_id - 1              # aux id L ↔ 第 L-1 层
        if not 0 <= model_idx < num_layers:
            logger.warning("AIFD: 候选层 %d 越界（共 %d 层），已禁用", aux_id, num_layers)
            _STATE.enabled = False
            return False
        attn = layers[model_idx].self_attn.attn   # ← vLLM 的 Attention 层
        attn.register_forward_hook(functools.partial(_attn_hook, aux_id))
        _STATE.attn_modules[aux_id] = attn
    return True
```

**关键设计**：`register_forward_hook` 挂在 `layer.self_attn.attn`（vLLM 的 `Attention` 模块）上，
而不是改 attention 后端。`Attention.forward(query, key, value, ...)` 收到的就是**稠密的 2D q/k/v**，
所以 hook 里能直接拿到了。

**为什么坚持不改 attention 后端**：`vllm_ascend/attention/attention_v1.py` 里有 4 条分支 +
`DeviceOperator` 间接层 + ACL graph 捕获路径（那里面还有个可疑的
`softmax_lse = torch.empty(1, ...)` 占位符）。挂 hook 一次性绕开全部。

### 4.4 算熵：`_attn_hook` + `_fia` + `_as_tn`

```python
def _attn_hook(aux_id, module, args, output):
    if not _STATE.enabled:
        return

    # Attention.forward(query, key, value, ...) 收的是 2D [T, heads*head_dim]
    if len(args) < 2:
        _DBG["skip_args"] += 1
        return
    query, key = args[0], args[1]

    num_tokens = query.shape[0]
    if num_tokens <= 1:              # 单 token 算"每 token 熵"没意义（decode 步）
        _DBG["skip_t1"] += 1
        return

    # ── 取 attn_metadata：这是 vLLM 提供 metadata 的官方入口 ──
    from vllm.model_executor.layers.attention.attention import get_attention_context
    try:
        attn_metadata, _, _, _ = get_attention_context(module.layer_name)
    except Exception:
        _DBG["skip_ctx"] += 1
        return

    # ── 我们需要两样东西：mask（TND 布局下是固定 (2048,2048)）和变长分段 ──
    attn_mask = getattr(attn_metadata, "attn_mask", None)
    seq_lengths = getattr(attn_metadata, "actual_seq_lengths_q", None)
    if attn_mask is None or not seq_lengths:
        _DBG["skip_meta"] += 1        # warmup/dummy run 走这里
        return
    if getattr(attn_metadata, "num_decode_tokens", 0) > 0:
        _DBG["skip_decode"] += 1      # 只在纯 prefill 上选层
        return

    # ── 从 Attention 模块上取形状/scale（不需要额外配置） ──
    num_heads = module.num_heads          # 32
    num_kv_heads = module.num_kv_heads    # 8（GQA）
    head_dim = module.head_size           # 128
    scale = float(getattr(module.impl, "scale", head_dim**-0.5))

    # [T, heads*head_dim] -> [T, heads, head_dim]
    q = query.view(num_tokens, num_heads, head_dim)
    k = key.view(-1, num_kv_heads, head_dim)

    # ── 一次调用同时拿到 lse 和 O^K ──
    out_k, lse = _fia(q, k, k, num_heads, num_kv_heads, head_dim, scale,
                      attn_mask, seq_lengths)
    dot = (q.float() * out_k.float()).sum(-1) * scale          # (T, heads)
    entropy = _as_tn(lse, num_tokens, num_heads).float() - dot  # (T, heads)

    _DBG["ok"] += 1
    _STATE.entropy[aux_id] = entropy.mean(dim=1)   # (T,) 对 head 平均
    if _STATE.seq_lengths is None:
        _STATE.seq_lengths = _cumulative_to_lengths(seq_lengths)
```

```python
def _fia(q, k, v, num_heads, num_kv_heads, head_dim, scale, attn_mask, seq_lengths):
    """TND 布局、稀疏因果、开 softmax_lse 的注意力。参数组合照搬
    vllm_ascend/attention/attention_v1.py 的生产路径（见 RESULT_lse_entropy.md）。"""
    import torch_npu
    return torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        atten_mask=attn_mask,                      # ⚠️ True/1 = **掩掉**，不是保留
        actual_seq_lengths=list(seq_lengths),      # ⚠️ 累加值，不是每段长度
        actual_seq_lengths_kv=list(seq_lengths),
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        scale=scale,
        input_layout="TND",                        # 变长拼接
        sparse_mode=3,                             # 因果；mask 固定 (2048,2048)
        pre_tokens=SWA_INT_MAX,
        next_tokens=SWA_INT_MAX,
        softmax_lse_flag=True,                     # ★ 开这个才有第二个返回值
    )
```

```python
def _as_tn(lse, num_tokens, num_heads):
    """把算子返回的 lse 统一成 (T, heads)。

    ⚠️ TND 下它的布局**不固定**：实测既见过 (T, N, 1) 也见过 (N, T, 1)
    （5 token 与 8 token 的结果就不同，似乎取决于 T 与 N 谁更大）。
    """
    x = lse
    if x.dim() == 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.shape == (num_heads, num_tokens):
        x = x.transpose(0, 1)
    if x.shape != (num_tokens, num_heads):
        raise RuntimeError(...)
    return x


def _cumulative_to_lengths(cu) -> list[int]:
    """[5, 10, 12] -> [5, 5, 2]

    ⚠️ `actual_seq_lengths_q` 是**累加值**，不是每段长度。直接 sum() 会得到错误的 T。
    """
    cu = [int(x) for x in cu]
    return [cu[0], *(cu[i] - cu[i - 1] for i in range(1, len(cu)))]
```

### 4.5 暂存与 emit：`note_layer_and_maybe_emit`

这是**唯一被插进 vLLM 层循环**的函数，由 `interfaces.py` 每层调一次。

```python
# vllm/model_executor/models/interfaces.py
# ── 注入点：默认 None，没开 AIFD 的进程在这行只有一次判断，零 import ──
_aifd_hook: "Callable[[list, int, torch.Tensor], None] | None" = None

def set_aifd_hook(fn) -> None:
    """由 aifd.enable_aifd() 调用。"""
    global _aifd_hook
    _aifd_hook = fn

class EagleModelMixin:
    def _maybe_add_hidden_state(self, aux_hidden_states, layer_idx,
                                hidden_states, residual):
        value = hidden_states + residual if residual is not None else hidden_states
        if layer_idx in self.aux_hidden_state_layers:   # 原有逻辑，没动
            aux_hidden_states.append(value)
        # AIFD：候选层暂存、全部过完之后追加一个"熵选层"通道。
        if _aifd_hook is not None:
            _aifd_hook(aux_hidden_states, layer_idx, value)
        return aux_hidden_states
```

```python
# aifd.py
def note_layer_and_maybe_emit(aux_list, layer_idx, value) -> None:
    if not _STATE.enabled or not _STATE.candidates:
        return

    # ① 候选层：暂存，**不 append**（append 由最后统一做）
    if layer_idx in _STATE.candidates:
        _STATE.hidden[layer_idx] = value
        return

    # ② 必须等**所有层都过完**再 append
    #    曾经用 "layer_idx > max(candidates) 就 append"，结果是候选层 [3..14]
    #    在 layer_idx=15 就插入，而 aux 层 18/33/36 排在它后面 —— 通道顺序变成
    #    [aux2, AIFD, aux18, aux33, aux36]，训练侧按「末尾是 AIFD」切分就错位。
    if _STATE.emitted or _STATE.num_layers <= 0 or layer_idx < _STATE.num_layers:
        return
    _STATE.emitted = True

    # ③ 各种异常情况的兜底（详见 §4.8）
    if value.shape[0] == 1:                     # decode 步
        _append_fallback(aux_list, value); reset_aifd_state(); return
    if not _STATE.entropy or len(_STATE.hidden) != len(_STATE.candidates):
        ...
        _append_fallback(aux_list, value); reset_aifd_state(); return

    # ④ 正常路径：选层 → gather → append
    chosen = _select_per_request(keep_bos=_STATE.keep_bos)
    if chosen is None:
        _append_fallback(aux_list, value); reset_aifd_state(); return

    aux_list.append(_gather_per_request(chosen))
    reset_aifd_state()
```

**"emit" 就是第 ④ 步**：算出每样本选中的层，把该层的 hidden gather 出来 append 进
`aux_list`。模型 forward 返回这个 list，vLLM 再 `torch.stack(..., dim=1)` 成通道维。

### 4.6 ★ per-sample 聚合：`_select_per_request`（**改动 per-token 就动这里**）

```python
def _select_per_request(keep_bos: bool) -> torch.Tensor | None:
    """返回 (T,) 的 long 张量：每个 token 所属 request 选中的候选层号。

    注意返回值仍然是"逐 token"的 —— 这是为了后面 gather 方便（每个 token 都要取
    自己那一层）。**per-sample 体现在"同一 request 内所有 token 拿到同一个层号"**。
    """
    seqlens = _STATE.seq_lengths          # 每个 request 的 token 数，如 [305, 512]
    if not seqlens:
        return None
    total = sum(seqlens)
    cands = _STATE.candidates

    # 把 L 个候选层的熵拼成 (L, T)
    ent = torch.stack([_STATE.entropy[l] for l in cands], dim=0)      # (L, T)
    if ent.shape[1] != total:
        return None
    scores = _score_matrix(ent)                                       # (L, T)

    cand_ids = torch.tensor(cands, device=ent.device)
    per_token_layer = torch.empty(total, dtype=torch.long, device=ent.device)
    offset = 0
    for n in seqlens:                     # ★ 遍历每个 request
        sl = slice(offset, offset + n)
        seg = scores[:, sl]               # (L, n) 这个 request 的所有 token
        if not keep_bos and n > 1:
            seg = seg[:, 1:]              # 可选：把位置 0（BOS）排除出平均
        # ★★★ per-sample 的关键在这里 ★★★
        # 先对**这个 request 的所有 token**求平均 (L,n)->(L,)，再 argmin -> 一个标量层号
        # 于是同一 request 内所有 token 都被赋成同一个层号
        per_token_layer[sl] = cand_ids[torch.argmin(seg.mean(dim=1))]
        offset += n
    return per_token_layer
```

```python
def _score_matrix(ent: torch.Tensor) -> torch.Tensor:
    """score_l = C_l + |C_l − C_{l−1}|（逐 token 版；l=0 行填 +inf 排除）。"""
    L = ent.shape[0]
    out = torch.full_like(ent, float("inf"))
    if L > 1:
        out[1:] = ent[1:] + (ent[1:] - ent[:-1]).abs()
    return out
```

> **注意**：`_score_matrix` 本来就是对**逐 token**的熵算的，返回 (L, T)。
> per-sample 只是在这个矩阵上**沿 token 维先聚合再 argmin**。

### 4.7 gather：`_gather_per_request`

```python
def _gather_per_request(per_token_layer: torch.Tensor) -> torch.Tensor:
    """按每个 token 各自选中的层号，把 hidden 取出来。返回 (T, H)。

    ⚠️ 全程**不能**有 device→host 同步。曾经的写法是
    `if not bool(mask.any()): continue` —— `bool()` 一个 NPU 张量就是一次同步，
    12 个候选层 = 每步 12 次，和日志那两处加起来足以把 stream 同步搞崩
    （AICPU 507018）。现在改成纯 torch.where。
    """
    cands = _STATE.candidates
    out = _STATE.hidden[cands[0]]          # 先拿第一个候选层打底
    for aux_id in cands[1:]:
        # 每个 token 选中的层必然是候选层之一（_select_per_request 保证），
        # 所以打底那个不会被漏掉
        out = torch.where((per_token_layer == aux_id).unsqueeze(-1),
                          _STATE.hidden[aux_id], out)
    return out
```

**这个函数同时支持 per-sample 和 per-token** —— 它只要求输入是 `(T,)` 的层号张量。
换成 per-token 时这里**一行都不用改**。

### 4.8 兜底：`_append_fallback`（硬性契约）

```python
def _append_fallback(aux_list, value) -> None:
    """拿不到选层结果时的兜底：仍然 append 一个形状正确的 [T, H] 张量。

    **这是硬性契约**：ExtractHiddenStatesModel 的隐状态缓冲区是按通道数
    **固定分配**的，proposer 那边是
        self.hidden_states[:num_tokens] = stacked_hidden_states
    append 少一个通道不会报"少了一个"，而是直接让 engine 崩在尺寸不匹配上
    —— 2026-09-21 那次跑了 2 小时后挂掉就是这么来的
    （27119 次正常选层，1 次不完整就够崩了）。
    """
    if _STATE.hidden:
        aux_list.append(_STATE.hidden[max(_STATE.hidden)])   # 那一层的真实隐状态
    else:
        aux_list.append(value)
```

### 4.9 调用链：谁在什么时候打开 AIFD

```
scripts/launch_vllm.py --aifd-candidate-layers 3 4 ... 14
    └─ 写进 speculative_config.draft_model_config.hf_config["aifd_config"]
        └─ vllm_ascend/worker/model_runner_v1.py:3798（模型装载时）
            └─ maybe_enable_aifd(self.model, self.speculative_config)
                └─ enable_aifd(model, candidate_layers)
                    ├─ set_aifd_hook(note_layer_and_maybe_emit)   # 接进层循环
                    └─ 给候选层的 self_attn.attn 挂 forward hook  # 算熵
```

```python
# vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py
def maybe_enable_aifd(model, spec_config) -> bool:
    """没配 aifd_config 时是纯粹的 no-op。两条 model runner 路径都要调。"""
    from vllm.model_executor.models.aifd import aifd_config_from, enable_aifd
    aifd_cfg = aifd_config_from(spec_config.draft_model_config.hf_config)
    if not aifd_cfg:
        return False
    return enable_aifd(model, list(aifd_cfg["candidate_layers"]))
```

### 4.10 ★ 完整流程串讲（一次 forward 的调用顺序）

先说两个最容易搞混的点：

**① `attn` 指什么？**

```python
attn = layers[model_idx].self_attn.attn
```

逐级拆开：

| 表达式 | 是什么 | 在哪 |
|---|---|---|
| `layers[model_idx]` | `Qwen3DecoderLayer`（vLLM 版） | `vllm/model_executor/models/qwen3.py:171` |
| `.self_attn` | `Qwen3Attention`（做 qkv_proj / qk_norm / rope 的那个） | `qwen3.py:65` |
| **`.attn`** | **vLLM 的通用 `Attention` 层**（真正调 attention 后端的那层） | `vllm/model_executor/layers/attention/attention.py` |

所以 `attn` **不是** HF 的 attention，也**不是** `Qwen3Attention`，而是
`layers/attention/attention.py` 里的 `Attention` 类。它身上有我们需要的
`register_forward_hook`、`num_heads`、`num_kv_heads`、`head_size`、`layer_name`、`impl`。

**② `_attn_hook` 在哪被调用？**

**不是我的代码调的**，是 PyTorch 调的。`register_forward_hook` 注册的是回调，
PyTorch 在**该模块 forward 返回之后**自动触发它，并把 `(module, args, output)` 传进去。

调用点在这里（`qwen3.py:166`）：

```python
class Qwen3Attention(nn.Module):
    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)      # ★ 这里进来
        #                                  ★ Attention.forward 一返回，_attn_hook 就被触发
        #                                     args = (q, k, v)，output = attn_output
        output, _ = self.o_proj(attn_output)
        return output
```

所以我们能拿到**稠密的 q/k**——这是整个方案成立的前提。

---

**③ 一次 forward 的完整时间线**

模型侧（`qwen2.py:415-422`）：

```python
aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)   # ← A
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    hidden_states, residual = layer(positions, hidden_states, residual)           # ← B（内部有 C）
    self._maybe_add_hidden_state(aux_hidden_states, idx + 1, hidden_states, residual)  # ← D
```

以本项目的配置为例：**候选层 = [3..14]，aux 层 = [2, 18, 33, 36]，模型 36 层**。

| 时刻 | 触发 | 我的代码 | 发生什么 |
|---|---|---|---|
| **A** | 循环前 | `note_layer_and_maybe_emit(aux, 0, ...)` | layer_idx=0 不是候选、且 `0 < num_layers` → **立即 return** |
| **B(idx=0)** | layer0 跑到 attention | `_attn_hook(1, ...)` | aux_id=1 不是候选（候选从 3 起）→ 立即 return |
| **D(0)** | layer0 返回后 | `note_layer_and_maybe_emit(aux, 1, ...)` | 不是候选 → return |
| … | | | （idx=1,2 同样无事） |
| **B(idx=2)** | layer2 跑到 attention | `_attn_hook(3, ...)` | ★ **第一次真正算熵**：调 `_fia(q,k,k,...)` 拿 lse + O^K → 存 `_STATE.entropy[3]` |
| **D(2)** | layer2 返回后 | `note_layer_and_maybe_emit(aux, 3, ...)` | ★ 3 是候选 → 暂存 `_STATE.hidden[3]`，**不 append** |
| … | | | （idx=3..13 逐层重复 C/D：每层算熵 + 暂存） |
| **B(idx=13)** | layer13 | `_attn_hook(14, ...)` | 最后一个候选层算熵 |
| **D(13)** | layer13 返回后 | `note_layer_and_maybe_emit(aux, 14, ...)` | 最后一个候选层暂存 —— 此刻 `_STATE.hidden` 有 12 项、`_STATE.entropy` 有 12 项 |
| … | | | （idx=14..34：`_attn_hook` 不再触发；`note_layer_and_maybe_emit` 每层 return） |
| **B(idx=17)** | layer17 跑到 attention | `_attn_hook(18, ...)` | 18 **不是**候选 → return（但 18 是 aux 层，由原有逻辑收集） |
| **D(35)** | layer35 返回后 | `note_layer_and_maybe_emit(aux, 36, ...)` | ★★ `36 == num_layers` → **触发选层 + emit** |

**关键点**：

- `_attn_hook`（算熵）在**层内部**触发，`note_layer_and_maybe_emit`（暂存/emit）在**层返回后**触发。
  每层都是「先 C 后 D」。**所以不是 `note_layer_and_maybe_emit` 最先——它是每个层周期里的第二个。**
- 但**整个 forward 里第一个被调到的我的函数确实是 `note_layer_and_maybe_emit`**（时刻 A，循环之前），
  只是那次 `layer_idx=0`，什么也不做。
- **emit 发生在最后一层（layer_idx=36）**，而不是选完候选层之后立刻。
  这是为了把 AIFD 通道放在 aux 列表的**末尾**（aux 层 18/33/36 比候选层晚收集）。

**emit 时刻具体做什么**（`note_layer_and_maybe_emit` 的后半段）：

```
1. _select_per_request()   → (T,) 层号张量，同一 request 内所有 token 同一个层号
2. _gather_per_request()   → 按层号把 hidden gather 成 (T, H)
3. aux_list.append(该张量)  → aux_hidden_states 变成 5 项（4 aux + 1 AIFD）
4. reset_aifd_state()      → 清空 entropy / hidden / seq_lengths / emitted
```

之后 vLLM 侧：

```
aux_hidden_states (list，5 个 [T,H])
    │
    ├─ gpu_model_runner: torch.cat([h[:num_tokens] for h in aux_hidden_states], dim=-1)
    │                      → 不要，实际是 stack 成 [T, 5, H]
    │
    └─ ExtractHiddenStatesModel.forward(hidden_states)      # 假 draft 模型
           └─ CacheOnlyAttentionLayer → 塞进"假 KV cache"
                  └─ ExampleHiddenStatesConnector 落盘
                         └─ 训练端读 hs_{idx}.safetensors
```

---

## 5. 改成 per-token 选层要动什么

**结论：核心只有一行。** 因为架构本来就是"逐 token 的层号张量"驱动的。

### 5.1 差异只有 `_select_per_request` 的聚合步骤

```python
# ── 当前 per-sample ──
for n in seqlens:
    seg = scores[:, offset:offset+n]
    per_token_layer[offset:offset+n] = cand_ids[torch.argmin(seg.mean(dim=1))]   # 先对 token 求平均
    offset += n

# ── per-token ──
per_token_layer = cand_ids[torch.argmin(scores, dim=0)]     # 直接对层维 argmin，每 token 一个层号
```

`argmin(scores, dim=0)` → 形状 `(T,)`，第 t 个元素是 token t 选中的候选层**在 cands 里的下标**；
再用 `cand_ids[...]` 映射回真实层号。

### 5.2 其他都不用改

| 组件 | 要改吗 | 原因 |
|---|---|---|
| `_STATE.entropy[l]` | ❌ | 本来就是 `(T,)` 逐 token 的 |
| `_score_matrix` | ❌ | 本来就是逐 token 的 `(L, T)` |
| `_gather_per_request` | ❌ | 只要求输入是 `(T,)` 层号张量 |
| `note_layer_and_maybe_emit` | ❌ | 只要 `_select_per_request` 返回 `(T,)` 就行 |
| 通道数 / 缓冲区 | ❌ | 输出仍是 `(T, H)` 一个通道 |
| **训练侧** | ❌ | 它收到的就是 `(T, H)`，不关心里面怎么选的 |
| `keep_bos` 语义 | ⚠️ | per-sample 下是"求平均时要不要含位置 0"；per-token 下没有平均，这个开关失去意义 |

### 5.3 建议的实现方式：加配置项而不是硬切

```python
# hf_config.aifd_config 里加一个：
aifd_config = {
    "candidate_layers": [3, 4, ..., 14],
    "granularity": "sample",     # 或 "token"
}

# _select_per_request 里分支：
if _STATE.granularity == "token":
    return cand_ids[torch.argmin(scores, dim=0)]
# ... 否则走原来的 per-request 循环
```

### 5.4 需要留意的

1. **DREAM-S 原版就是 per-token**，所以这个改动是"回到原版"，
   而不是"发明一个新东西"。改回去后可以和论文直接对齐。
2. **per-token 下选层会逐位置跳变**：同一个 block 内相邻 token 的监督目标可能来自不同层，
   引入空间上的不连续。DREAM-S 就是这么干的，但值不值得要在实验里看。
3. **存储不变**：仍然只多一个 `(T, H)` 通道，不是每层都存。
4. **离线实验中观察到的现象**（40 条样本，候选窗口 2..13）：
   per-sample 退化成常数（40/40 全选同一层）；per-token 则 10 个层都用到了，
   但 90% 的 token 集中在相邻的 3 层里。换 per-token 后**区分度会变好**，
   但提升幅度需要自己评估。

---

## 6. 踩过的坑（别人接手前务必看）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | engine 崩在 `self.hidden_states[:n] = ...` 尺寸不匹配 | 拿不到选层结果时**没 append**，通道少一个 | `_append_fallback` 永远 append |
| 2 | engine 崩在 `LocalScalarDenseNpu` / AICPU 507018 | 热路径上每步 **36 次 device→host 同步**（日志参数立即求值 + `counts[l]>0` + `.item()`） | 全部改成纯张量运算 / 一次性 `.cpu()` |
| 3 | AIFD 通道跑到 index 1 而不是末尾 | 在 `layer_idx=15` 就 append，aux 层 18/33/36 排到它后面 | 等 `layer_idx == num_layers` |
| 4 | 日志里 `softmax_lse` 形状时而 `(T,N,1)` 时而 `(N,T,1)` | TND 下算子布局不固定 | `_as_tn` 两种都接 |
| 5 | 分段长度算错 | `actual_seq_lengths_q` 是**累加值** | `_cumulative_to_lengths` |
| 6 | 改了 runner 但不生效 | 有三份 model runner，Ascend 用的是 `vllm_ascend` 那份 | 看日志里的文件名定位 |
| 7 | 通道数改了但还崩 | 三处缓冲区要一起改，漏了 proposer 那份 | 统一走 `aifd_extra_channels` |
