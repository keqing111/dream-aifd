# 训练侧 AIFD 接入记录（DFlash）

日期：2026-09-21　　目标仓库：`~/dreams-rl/speculators-ty5537`

## ⚠️ 首先：`import speculators` 默认不指向这个仓库

`speculators` 是 **editable 安装指向 `~/dspark_project/speculators/src`**。跑训练必须
显式覆盖：

```bash
cd ~/dreams-rl/speculators-ty5537
PYTHONPATH=$PWD/src python scripts/train.py ...
```

（已验证 `PYTHONPATH` 能压过 editable 安装。同理 `~/.cache/.../vllm` 那边没问题：
`vllm` / `vllm_ascend` 的 editable 路径就是 `/vllm-workspace/...`。）

## 端到端验证结果

合成 hidden states（**随机噪声**，只为打通链路），48 条样本，Qwen3-4B 作 verifier：

| 配置 | 退出码 | `train/loss` | `train/aifd_loss` |
|---|---|---|---|
| `--aifd-weight 0.5`，数据 5 通道（含 AIFD） | 0 | 3.149 → 2.989 | **0.720** ✅ |
| `--aifd-weight 0`，数据 4 通道（无 AIFD） | 0 | ~2.9 | **不存在** ✅ |

`aifd_loss ≈ 0.72` 与单元测试里「两个随机向量做 SmoothL1」的理论值 0.724 吻合。
`full_acc`/`eal` 全是 0 是**预期**的——目标是随机噪声，draft 学不出东西。

## 通道排布（容易搞错，务必记住）

```
[ aux_0, aux_1, ..., aux_{n-2}, aux_{n-1}=verifier末层, AIFD ]
```

- **训练侧的 `--target-layer-ids` 不要带末层**：`launch_vllm.py` 的
  `--include-last-layer` 默认 True，会自动把 `num_hidden_layers` 追加进去。
  所以 `--target-layer-ids 2 18 33` 在 vLLM 侧产出 4 个通道，在训练侧
  `len(draft_model.target_layer_ids)` 仍是 3。
- AIFD 永远追加在**最末尾**，所以开了 AIFD 之后 verifier 末层从 `[:, -1]` 变成 `[:, -2]`。
  实现里就是用 `n_channels - 1 - aifd_channels` 算的，`aifd_channels=0` 时与旧行为**逐位一致**。

## 改动清单

| 文件 | 改什么 |
|---|---|
| `models/dflash/metrics.py` | 新增 `compute_aifd_loss` + `_rms_norm` |
| `models/dflash/config.py` | `aifd_weight`（默认 0）、`aifd_norm`（默认 True） |
| `models/dflash/core.py` | `_build_base_config_kwargs` / `get_trainer_kwargs` 带 aifd 字段；`forward` **显式**接住 `aifd_hidden_states`（否则会经 `**kwargs` 透传给 decoder layer 而报错） |
| `train/data.py` | `ArrowDataset.aifd_channels`；`_get_raw_data` 按通道数切分；`create_empty_sample` / `create_collate_fn` 带 `aifd` |
| `train/dataloader.py` | 透传 `aifd_channels` |
| `scripts/train.py` | `--aifd-weight` / `--aifd-norm`；`aifd_channels = 1 if weight>0`；非 dflash/dspark 时报错 |
| `scripts/launch_vllm.py` | `--aifd-candidate-layers` |

## 归一化：默认**关闭**（`aifd_norm=False`）

一开始我默认开了，理由是「候选层间 norm 差 4.8x + 层 6 以上有 BOS token 是中位数
400 倍」。**核实后这两个理由都不成立**，已改回默认关闭：

1. **BOS 那个 400x 离群永远进不了 loss**。`select_anchors`（`dflash/utils.py:22-68`）
   只从 `loss_mask` 里挑锚点，而 `loss_mask[0]` 恒为 False（BOS 在 prompt 里）。
   实测：58 条样本、14624 个锚点块位置，**落在位置 0 的 0 个**，最小锚点位置 20。
2. **剩下只有候选层间 4.83x，确实在一个数量级内**，SmoothL1 完全扛得住。

`--aifd-norm` 保留着（默认 False），将来若把候选窗口放宽、或选择开始逐样本变化，
再打开即可。

## Loss 形式

照搬 DREAM-S 的 `compute_mid_loss`（`train/main_deepspeed.py:244-247`）：
`SmoothL1Loss(reduction="none")` 的 masked mean，作用在
`anchored_block_indices` 选出的位置上。

**与 DREAM-S 的一个偏离**：DREAM-S 除以 `B*S`（整条序列长），不是 mask 里 1 的个数，
导致该项被系统性压小两三个数量级。这里改为「sum / mask 计数」，
所以 `--aifd-weight` 的标定量级和 DREAM-S 不可直接类比。

metric 走 `aifd_loss_sum` / `aifd_loss_total`，由 `normalize_counted_metrics` 折算成
`aifd_loss`（= 每个有效位置的平均 loss）。

## DSpark + 真实数据：已跑通

用带 AIFD 的 vLLM server 采集了 **19 条真实 hidden states**（5 通道），
跑 `--speculator-type dspark --aifd-weight 0.5`：

```
train/loss=2.469  train/full_acc=0.361  train/aifd_loss=0.535   ← 真实在学
```

（日志里的 0.00 是空 batch——48 行里只有 19 行有 hidden states，
`--on-missing skip` 把其余滤掉了。）

**DSpark 已接入**：`models/dspark/core.py` 的 `get_trainer_kwargs` 与 `forward`
都加了 aifd 参数和 loss 项。

## 还没验的（诚实清单）

- **没有和 baseline 比过效果**（`--aifd-weight 0` vs 非 0 的 `eal` / `position_acc` 曲线）。
  19 条样本不够，且训练步数太少。
- **全量真实数据没跑过**：只用 19 条验证了链路。**但不需要预先完整采集**——
  见下"在线生成"一节。
- `sliding_window` / chunked prefill 与 AIFD 的交互没测。

## 在线生成：不需要预先完整采集

`--on-missing generate` **本来就是默认值**（`scripts/train.py:863-872`）：
训练时按需调 vLLM endpoint 生成，`--on-generate cache|delete` 控制用完后留不留。
所以 170 GiB 那个数字只适用于"离线预采集"这条可选路径，**不是必须的**。

```bash
# 1) server 必须带 --aifd-candidate-layers，否则没有 AIFD 通道
python scripts/launch_vllm.py <model> --aifd-candidate-layers 3 4 ... -- --port 8199 ...

# 2) 训练：默认就是在线生成
PYTHONPATH=$PWD/src python scripts/train.py ... \
  --vllm-endpoint http://localhost:8199/v1 \
  --on-missing generate --on-generate delete     # delete = 用完即删，零落盘
```

- `cache`：生成一次后留在 `hidden-states-path`，后续 epoch 命中缓存（快，但会累积到 ~170 GiB）
- `delete`：每次用完删掉（零落盘，但每个 epoch 都要重新生成）
- 哪个划算取决于 epoch 数和磁盘，先用 `delete` 跑通、要提速再换 `cache`

## "空 batch" 是什么

用 `--on-missing skip` 跑一个**缓存不完整**的数据集时会出现：

1. 某些行没有 `hs_{idx}.safetensors` → `_get_raw_data` 返回 `None`
2. `create_collate_fn` 把 `None` 过滤掉（`data.py:461`）
3. **如果整个 packed batch 全是 None**，就回退成 `create_empty_sample()`
   —— 0 长度的占位样本，`loss_mask` 全 0、`document_ids` 全 -1（`data.py:463-470`）
4. 这一步前向拿到全 0 的 mask → `loss`/`aifd_loss`/`full_acc` **全报 0**

实测确认（`/tmp/sub50k` 48 行，只缓存了 19 行）：

```
有缓存的行 → 正常样本；没缓存的行 → None
全 None 的 batch 经 collate: loss_mask 非零数 = 0, document_ids 有效位置数 = 0
```

**这是 `skip` 的产物，不是 bug**。默认的 `generate` 模式下每行都能拿到 hidden states，
不会出现空 batch。所以正式跑训练时**不要用 `skip`**（除非确实想丢弃缺数据的样本）。

## 环境上的两个坑

1. **Ascend 上不能用默认的 `--draft-attn-impl simple_flex_attention`**：
   Triton/BiShengHIR 编译 mask kernel 时报 `ub overflow`。用 `--draft-attn-impl eager`。
2. `--draft-vocab-size` 必须和已有的 vocab mapping 文件一致（默认 32000），否则报
   `dim 0 should match provided value`。
