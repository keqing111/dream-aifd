# per-token 粒度下扫 4 个 aifd-weight

**日期**：2026-09-22 11:22 → 22:51（约 11.5 小时，两批各 ~5h45m）
**分支**：`per-token-4weights`

## 配置

- server（卡 13）：`--aifd-candidate-layers 3..14 --aifd-granularity token`
- 训练（卡 11/12）：`--speculator-type dspark --num-layers 5 --lr 3e-4
  --total-seq-len 3072 --max-anchors 512 --loss-fn '{"ce":0.3,"tv":0.7}'
  --enable-confidence-head --confidence-head-with-markov
  --aifd-draft-layer 2 --grad-accum-steps 12 --epochs 3`
- 数据：`open_perfectblend_qwen3_4b_50k`
- 生成：在线生成、用完即删（`--on-missing generate --on-generate delete`）
- 与上一批（per-sample @ 0.1）**只差粒度**，其余配置完全相同

## 调度

两组并行（卡 11/12），跑完自动接下一批（见 `run_4weights.sh`）：

```
第一批  11:22 → 17:06   w=0.2 (卡11) / w=0.4 (卡12)
第二批  17:06 → 22:51   w=0.7 (卡11) / w=1.0 (卡12)
```

## 结果（验证集，epoch 3/3）

| weight | AIFD 占 loss | val/loss | val/accept_len | val/full_acc | val/accept_rate | val/position_0_acc |
|---|---|---|---|---|---|---|
| 0.2 | 5.4%  | 0.5260 | **4.1780** | **0.6440** | 0.5970 | 0.8150 |
| 0.4 | 10.6% | 0.5280 | 4.1620 | 0.6430 | 0.5950 | 0.8140 |
| 0.7 | 18.4% | 0.5290 | 4.1530 | 0.6420 | 0.5940 | 0.8120 |
| 1.0 | 26.0% | 0.5300 | 4.1490 | 0.6420 | 0.5940 | 0.8120 |

**AIFD 占比跨了 5 倍（5.4% → 26.0%），accept_len 只从 4.178 动到 4.149（0.7%）。**

逐 epoch 看趋势一致（单调、极小）：

```
epoch 1  accept_len:  3.306 / 3.280 / 3.269 / 3.268
epoch 2  accept_len:  3.882 / 3.857 / 3.857 / 3.849
epoch 3  accept_len:  4.178 / 4.162 / 4.153 / 4.149
```

## 观察

### 1. 权重越高，指标**略差**（但幅度在噪声内）

单调下降，但差异只有 0.7%。单种子、无误差棒，**不认为有统计意义**。

### 2. ★ `aifd_loss` 收敛到 ~0.14，和权重几乎无关

| weight | 0.2 | 0.4 | 0.7 | 1.0 |
|---|---|---|---|---|
| 收敛后 `val/aifd_loss_epoch` | 0.141 | 0.140 | 0.139 | **0.138** |

**权重放大 5 倍，AIFD loss 只降 2%。** 如果瓶颈是"梯度不够大"，加大权重应显著压低它。
这更像撞到了**表达能力的地板** —— 与"draft 中间层接不住 target 中间层"的猜测一致。

### 3. per-token vs per-sample 也看不出差别

| | accept_len | full_acc |
|---|---|---|
| per-sample @ 0.1（上一批） | 4.163 | 0.643 |
| per-token @ 0.2（本批） | 4.178 | 0.644 |

per-token 确实带来逐位置变化（server 每批用到 6~8 个层，vs per-sample 的 100% 全选 L10），
**但指标上分辨不出**。选层分布仍集中在 L10/L11。

```
L9=50,  L10=590,  L11=540,  L12=15,  L13=4,  L14=18
L4=1, L5=2, L9=36, L10=597, L11=604, L12=23, L13=7, L14=11
```

## ⚠️ 结论的边界

这 4 组**没有 `--aifd-weight 0` 的对照**，所以只能得出：

> **AIFD 的权重在 0.2~1.0 区间内对结果无影响。**

**不能**得出"AIFD 有没有用"。那需要 weight=0（或外部 baseline）来对照。

## 文件

- `logs/training/tok_w{0.2,0.4,0.7,1.0}.log.gz` —— 4 组训练日志
- `logs/server/srv_token.log.gz` —— 本批的 server 日志（52 MB → gz）
- `run_4weights.sh` —— 调度脚本

检查点（每组 9.1 GB，共 36 GB）在本地 `~/aifd-work/checkpoints/`，未入库。
