# 验证结论：用 `softmax_lse` + 一次 `V:=K` 的注意力算注意力熵

日期：2026-09-21　　脚本：`scripts/verify_lse_entropy.py`　　硬件：Ascend 910B（卡 12）

## 结论

**恒等式成立，可以用于在 vLLM 里生产 AIFD 的选层信号。**

$$\underbrace{-\textstyle\sum_j p_{ij}\log p_{ij}}_{H_i} \;=\; \text{lse}_i \;-\; \text{scale}\cdot\langle q_i,\, O^K_i\rangle$$

其中 $O^K_i$ 是**把 value 换成 key** 后的注意力输出，$\text{lse}_i$ 由算子的
`softmax_lse_flag=True` 直接给出（**不需要额外计算，现在是被丢弃的**）。

推导：$H_i = -\sum_j p_{ij}(s_{ij}-\text{lse}_i) = \text{lse}_i - \sum_j p_{ij}s_{ij}$，
而 $\sum_j p_{ij}s_{ij} = \text{scale}\cdot\langle q_i,\sum_j p_{ij}k_j\rangle = \text{scale}\cdot\langle q_i, O^K_i\rangle$。

关键点：**全程不落地 $S\times S$ 分数矩阵**，显存与现在完全一样。

## 实测结果

参考实现用朴素 fp32 softmax 手算熵（CPU），逐元素比对。

| 配置 | `lse` vs 因果 logsumexp | `H` vs 参考 |
|---|---|---|
| MHA (KV=32), S=256, 无 mask | 9.5e-07 | **5.0e-04** |
| MHA (KV=32), S=256, 因果 | 9.5e-07 | **2.9e-03** |
| GQA (KV=8), S=512, 无 mask | 9.5e-07 | **4.4e-04** |
| GQA (KV=8), S=512, 因果 | 9.5e-07 | **1.8e-03** |
| GQA (KV=4), S=1024, 无 mask | 9.5e-07 | **3.6e-04** |
| GQA (KV=8), S=2048, 无 mask | 9.5e-07 | **3.2e-04** |
| GQA (KV=8), S=2048, 因果 | 9.5e-07 | **1.6e-03** |

误差量级 ~1e-3，来自 bf16 输入（`lse` 本身是 fp32 输出）。熵的实测范围与参考逐位吻合
（例：S=2048 因果，got/want 都是 `[-0.000, 7.539]`）。

## 算子接口细节（已实测确认）

### 1. 开启 lse

```python
out, lse = torch_npu.npu_fused_infer_attention_score(
    query, key, value,          # BNSD 布局实测通过
    num_heads=..., num_key_value_heads=...,
    scale=..., input_layout="BNSD",
    sparse_mode=..., atten_mask=...,
    softmax_lse_flag=True,      # ← 默认 False，即当前被丢弃的那一路
)
```

- `softmax_lse` 形状 **`(B, N, S, 1)`**，dtype **float32**（比 bf16 输入精度高）
- 它的定义就是**缩放并掩码后**分数的 logsumexp —— 与 `torch.logsumexp(scores)` 逐位一致，
  所以公式里不需要再乘/除 `scale`

### 2. `atten_mask` 语义（容易踩）

- **`True`(bool) / `1`(int8) = 掩掉**。因果 mask 是**上三角为 True**：
  `torch.triu(torch.ones(S,S), diagonal=1).bool()`
  —— 与 `torch.tril(...)` 直觉相反，注意。
- **形状有硬性要求**：
  - `sparse_mode` 0/1 → `(1,1,Q_S,KV_S)`（或 `(1,Q_S,KV_S)` / `(B,1,Q_S,KV_S)`）
  - `sparse_mode` 2/3/4 → **固定 `(2048,2048)`**（这就是 vLLM-Ascend 里
    `_generate_attn_mask` 硬编码 2048 的原因）。用 `(S,S)` 会报 ACL 错。

### 3. GQA

`num_key_value_heads < num_heads` 时算子内部按 `repeat_interleave` 展开 K/V，
`V:=K` 的调用同样成立（KV=4/8 都已实测）。

## 对 vLLM 侧实现的含义

- **`lse` 是免费的**：`vllm_ascend/attention/attention_v1.py:558,744` 已经在把
  `softmax_lse` 作为输出槽传给算子，只是算完就扔。把它留下来即可。
- **`O^K` 要多一次注意力调用**。只需要对**候选层**做：Qwen3-4B 在 S=2048 时
  单层 attention 约占 18% FLOPs，对 1/3 的层多跑一次 → 总前向 **+6% 左右**。
- 挂载点在 `vllm/model_executor/models/interfaces.py:1348`
  `EagleModelMixin._maybe_add_hidden_state`（所有 EAGLE3 家族模型共用的唯一咽喉），
  但 `lse`/`O^K` 需要从 attention 层往上通（可按 layer_name 走
  `get_forward_context().no_compile_layers` 那个现成的全局注册表）。

## TND / varlen / v2 算子已验证（脚本 `scripts/verify_lse_tnd.py`）

照搬 `attention_v1.py:780-784` 的**生产组合**重验：

```python
input_layout="TND"; sparse_mode=3
atten_mask = torch.triu(torch.ones(2048,2048), 1).to(torch.int8)   # 上三角=1=掩掉
pre_tokens = next_tokens = SWA_INT_MAX
actual_seq_lengths = <各序列长度累加和>
```

| 配置 | v1 `softmax_lse_flag` | v2 `return_softmax_lse` |
|---|---|---|
| varlen=[512,300,784], GQA KV=8 | lse 9.5e-07 / H **1.7e-03** ✅ | lse 9.5e-07 / H **1.7e-03** ✅ |
| lens=[2048] | H 1.7e-03 ✅ | — |
| lens=[3000] | H 1.4e-03 ✅ | — |
| lens=[1200,2400] | H 1.9e-03 ✅ | — |
| lens=[3072]（数据集最长） | H 1.8e-03 ✅ | — |
| lens=[4096] / [8192] / [4096,4096] | H 1.8e-03 / 2.1e-03 / 2.0e-03 ✅ | — |

- `lse` 形状在 TND 下是 **`(T, N, 1)`**、fp32。
- **v2 算子的参数名不同**：`return_softmax_lse`（非 `softmax_lse_flag`）、
  `actual_seq_qlen`/`actual_seq_kvlen`、`softmax_scale`、`num_query_heads`。
- **固定 2048×2048 的 mask 对 >2048 的序列照样正确**（一直测到 8192），
  不需要 chunked prefill 配合。但 **`atten_mask` 不能传 None**（`sparse_mode=3` 下会报错）。

## 还没验的

- **ACL graph 捕获下怎么把 lse 读出来。** 注意 `attention_v1.py:780` 里
  `softmax_lse = torch.empty(1, ...)` 是个**尺寸为 1 的占位符**——很可能是 graph 模式下
  仅供签名占位、并不真写。**这是实现时第一个要确认的点**，可能需要让 lse 成为
  graph 的输出缓冲。

## 踩过的坑（给未来的自己）

1. `vllm_ascend` 里有两套 mask 构造，约定**不一致**：
   `_generate_attn_mask`（bool, `tril_`）与 `get_splitfuse_attn_mask`（int8, `triu_(...,1)`）。
   不要从代码猜语义，**用 attention 输出本身去定性**最快。
2. 验证时不要只比熵 —— 熵越界（负值、超过 `log(S)`）只能说明"哪里不对"，
   先比 `attn_out` 定位 mask 语义、再比 `lse`、再比 $O^K$，逐项拆开 5 分钟就能定位。
3. 随机输入要放大（`--input-mult 8`）。不放大时 softmax 接近均匀，熵恒等于 `log(S)`，
   恒等式会"平凡地"通过而掩盖问题。
