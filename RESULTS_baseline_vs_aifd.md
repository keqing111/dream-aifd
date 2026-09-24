# baseline 对照：AIFD 明确有害

**日期**：2026-09-23 01:50 → 07:26（约 5.6 小时）
**对照设计**：**只开关 AIFD，其余完全相同**（同代码、同数据、同配置、同卡、同种子）

## 设计

| | server（卡 13） | 训练（卡 11/12） | AIFD |
|---|---|---|---|
| **baseline** | 不带 `--aifd-candidate-layers`，**4 通道** | `--aifd-weight 0`，跑 seed 42 / 43 | 关闭 |
| **AIFD** | 带 `--aifd-candidate-layers`，**5 通道** | `--aifd-weight 0.2/0.4/0.7/1.0` | 开启 |

baseline 跑的是**原始 4 通道数据路径**（`aifd_channels=0`），完全不碰 AIFD。
server 只在"是否多吐一个通道"上不同——AIFD 的额外注意力调用不改动模型自身的
attention 输出，所以两边的 4 个 aux 通道是逐位相同的。**这是一个干净的对照。**

## 结果（验证集，epoch 3/3）

| 配置 | val/accept_len | val/full_acc | val/accept_rate | val/loss |
|---|---|---|---|---|
| **baseline seed=42** | **4.3430** | **0.6590** | **0.6160** | 0.5060 |
| **baseline seed=43** | **4.3320** | **0.6580** | **0.6140** | 0.5070 |
| AIFD token w=0.2 | 4.1780 | 0.6440 | 0.5970 | 0.5260 |
| AIFD token w=0.4 | 4.1620 | 0.6430 | 0.5950 | 0.5280 |
| AIFD token w=0.7 | 4.1530 | 0.6420 | 0.5940 | 0.5290 |
| AIFD token w=1.0 | 4.1490 | 0.6420 | 0.5940 | 0.5300 |

```
种子噪声（baseline 42 vs 43）  = 0.0110
baseline 均值 − AIFD 均值      = 4.3375 − 4.1605 = −0.1770   ← 16 倍噪声
```

**AIFD 让 accept_len 掉了 4.1%（full_acc 掉 2.4%），差距是种子噪声的 16 倍。**

同时权重趋势是单调的（越高越差，组间极差 0.0290 ≈ 2.6 倍噪声），方向和幅度一致。

## ★ 用户提出的疑点：这个数字是不是信息泄露

用户指出：**同数据、3 epoch，其他组只到 3.67，而这里的 baseline 到了 4.33**，
提升幅度过大，怀疑信息泄露。

### 已确认的事实

- baseline 走的是**原始数据路径**（4 通道），我改的 `data.py` 在 `aifd_channels=0` 时
  与上游逐位等价（`[:, :n-1]` + `[:, n-1]`）。
- 报告的 `val/accept_len` 由上游 `dflash/metrics.py::compute_metrics` 算出，
  **我没有改动这个函数**（只加了 AIFD loss 项）。
- `repos/` 里的源码与工作区逐字节一致（7 个关键文件已核对）。

### 待查（下一步）

1. **EAL 自洽性**：用日志里的逐位置准确率按 `eal = Σ_k Π_{i≤k} acc_i` 反推，
   看能否复现报告的 4.34。
2. **和用户 baseline 的配置差异**：本次是**单卡**、`num_layers 5`、
   `seq 3072`、`max_anchors 512`、**`grad_accum 12`**、`loss-fn ce0.3/tv0.7`、
   50k 数据、3 epoch。**`grad_accum 12` 会把有效 batch 放大 12 倍**，
   这一项足以显著改变收敛结果，且不是泄露。
3. **端到端验证**：把训练好的 draft 装进 vLLM 跑真实投机解码，
   对比实测接受长度与训练时报告的 4.34。

## 文件

- `logs/training/base_seed{42,43}.log.gz` —— 两组 baseline 训练日志
- `logs/server/srv_baseline.log.gz` —— baseline 用的 plain server 日志
- `run_baseline.sh` —— baseline 调度脚本
