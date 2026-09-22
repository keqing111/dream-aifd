# vLLM 侧 AIFD 改造记录（Ascend）

日期：2026-09-21　　验证脚本：`scripts/test_vllm_aifd.py`（**端到端通过**）

## 结果

4 条不同 prompt，各 5~8 token：

```
--- prompt 0 ---   token 数 T=5  通道数 C=13   hidden=2560
  ✅ AIFD 通道 == 候选层 L10 的通道（逐位相同）
--- prompt 1 ---   T=8  C=13  ✅ L10
--- prompt 2 ---   T=8  C=13  ✅ L10
--- prompt 3 ---   T=8  C=13  ✅ L10
```

通道数 = `len(aux) + 1`，且 AIFD 通道与候选通道**逐位相同** —— 说明 hook / 熵计算 /
选层 / gather / 通道数（模型 + proposer 两处）全部正确。batch>1 时按 request 分段
（`L10=5` 与 `L10=10` 两轮）也验证过。

## 设计：不改 attention 后端

关键决定是**不碰** `vllm_ascend/attention/attention_v1.py`。在
`layer.self_attn.attn`（vLLM 的 `Attention` 层）上挂一个普通 forward hook 就能拿到
稠密的 q/k，于是整个 AIFD 自包含在一个新文件里，绕开了：

- `attention_v1.py` 的 4 条分支 + `DeviceOperator` 间接层
- ACL graph 捕获路径（`softmax_lse = torch.empty(1, ...)` 那个可疑的占位符）

代价：每个候选层多一次注意力调用（`value := key` + `softmax_lse_flag=True`，
一次同时拿到 `lse` 和 `O^K`）。Qwen3-4B 上约 +6% 总前向。

## 改动清单

| 文件 | 改什么 |
|---|---|
| `vllm/model_executor/models/aifd.py` | **新增**。状态、hook、熵计算、选层、gather |
| `vllm/model_executor/models/interfaces.py` | `EagleModelMixin._maybe_add_hidden_state` 里多调一次 `note_layer_and_maybe_emit`，与原有 aux 收集**互不干扰**（层号重叠时两条都留） |
| `vllm/model_executor/models/extract_hidden_states.py` | `num_hidden_states += aifd_extra_channels` |
| `vllm/v1/spec_decode/extract_hidden_states.py` | 同上（**proposer，容易漏**） |
| `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` | 新增共享的 `maybe_enable_aifd()` |
| `vllm/v1/worker/gpu_model_runner.py` | v1 上游 runner 里调用（防御性） |
| **`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`** | **Ascend 上真正生效的那条**，`~:3798` |
| `speculators-ty5537/scripts/launch_vllm.py` | `--aifd-candidate-layers`，写进 `hf_config.aifd_config` |

## ⚠️ 通道顺序 bug（真实数据才暴露，已修）

**症状**：AIFD 通道出现在 index **1**，而不是末尾。

**原因**：append 的触发条件原本是 `layer_idx > max(candidates)`。候选层是 `[3..14]`
时它在 `layer_idx=15` 就插入，而 aux 层 18/33/36 排在它**后面**，于是通道变成
`[aux2, AIFD, aux18, aux33, aux36]`。训练侧按「末尾是 AIFD」切分 → **错位**。

**怎么发现的**：用真实采集的数据逐通道去比对 HF 的逐层 `hidden_states`，
发现 `[:, -1]` 对不上任何一层（err 3.7，而别的通道是 0.006 量级）。
纯合成数据测不出来——我的合成测试里 aux 和候选层恰好是同一组，
AIFD 自然落在最后，掩盖了 bug。

**修法**：`enable_aifd` 时记下 `num_layers`，等到 `layer_idx == num_layers`
（所有层都过完）再 append。

**修后验证**（真实数据，逐通道 vs HF 逐层）：

```
通道0 → hf[2]   err=0.0007   = aux id 2
通道1 → hf[18]  err=0.0102   = aux id 18
通道2 → hf[33]  err=0.0920   = aux id 33
通道3 → hf[35]  err=3.73     = aux id 36（末层，误差大是 bf16 深层累积）
通道4 → hf[10]  err=0.0062   = AIFD 选中的 L10  ← 与服务器日志一致
```

这同时是对**整条链路**的独立验证：熵判据、选层、gather、通道位置全对。

## 踩过的坑（按浪费时间的顺序）

1. **模型 runner 有三个**。`vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/worker/gpu/model_runner.py`，
   而 **Ascend 实际用的是 `vllm_ascend/worker/model_runner_v1.py`**。我前两次都改在了不生效的地方。
   **定位方法**：日志里 `[model_runner_v1.py:3825]` 的**文件名**就是答案。

2. **`softmax_lse` 在 TND 下的形状不固定**。实测 `(T, N, 1)` 和 `(N, T, 1)` 都出现过
   （5 token vs 8 token 的结果就不同），**似乎取决于 T 与 N 谁更大**。必须两个都接住。

3. **`actual_seq_lengths_q` 是累加值**（`[5, 10]` 表示两段各 5），不是每段长度。
   直接 `sum()` 会得到 `T` 错的值。

4. **通道数有三处要一起改**：`ExtractHiddenStatesModel`（buffer）、
   `ExtractHiddenStatesProposer`（buffer）、`CacheOnlyAttentionLayer` 的 `num_heads`。
   漏一处就是 `shape mismatch: [T, 12, H] vs [T, 13, H]`。现在统一走 `aifd_extra_channels()`。

5. **`attn_metadata` 在 profile/dummy run 下是 `None`**（T=8192 那次）。属正常，
   用 `debug_once` 而不是 warning，否则日志噪音很大。

6. **`enforce_eager=True`**。别让 ACL graph 把 hook 里的算子吞掉。

7. **落盘是异步的**。`llm.generate()` 返回时 `hs_i.safetensors` 可能还没落，
   测试里要等一下。

## 还没验的

- ~~ACL graph 捕获~~ —— 现在用 `enforce_eager=True` 绕开了。要开图的话需要另做。
- **多 request 混合 prefill/decode** 的批次（现在只在纯 prefill 上选层，
  `num_decode_tokens > 0` 时直接跳过）。
- **chunked prefill**：当前配置没开。若开启，`He[l]` 会按 chunk 分段，选层语义要重新定义。
- 候选层与 aux 层**重叠**的情形（`interfaces.py` 里两条路都保留了，但没实测过）。

## 一个观察

4 条不同 prompt（长度 5~8）**全部选中 L10**，40 条样本的 HF 离线实验也是 40/40 全选同一层。
两套完全独立的实现给出同样的结论，**"per-sample 选择退化"这件事的置信度提高了**。
等采到真实规模的数据（几百条以上、长度上千）再下最终结论。
