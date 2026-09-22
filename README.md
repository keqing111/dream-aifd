# dream-aifd

把 **DREAM-S 的 AIFD**（per-sample 选层 + 中间层特征蒸馏）迁移到
**speculators 的 DSpark 训练**，作为可选的 loss 插件。

配套一个在 **vLLM / vLLM-Ascend** 侧实现的在线选层通道：
vLLM 数据生成 server 按注意力熵自动挑一层，把该层的 hidden states 作为
**额外的第 5 个通道**随 aux 通道一起落盘，训练侧直接消费。

---

## 核心思路

### 判据（per-sample）

```
He[l, t] = mean_h ( lse[l,t,h] − scale · ⟨q[t,h,:], O^K[t,h,:]⟩ )   # 层 l 在 token t 的注意力熵
C_l      = mean_{t ∈ request} He[l, t]                             # ★ per-sample：对整条 request 求平均
score_l  = C_l + |C_l − C_{l−1}|
best     = argmin_{l ∈ 候选层} score_l
```

`lse` 是 softmax 的 log-sum-exp，`O^K` 是「把 value 换成 key」的注意力输出。
**关键技巧**：两者由**同一次**算子调用同时给出，且不落地 S×S 分数矩阵。

```python
out_k, lse = npu_fused_infer_attention_score(q, k, v=key, ..., softmax_lse_flag=True)
H = lse − scale·⟨q, O^K⟩        # 推导：H = −Σ p·log p = lse − Σ p·s
```

代价：每个候选层多跑一次注意力（12/36 层）→ 总前向约 **+6%**，显存不变。

### 数据流

```
target 模型跑一次 forward
   ├─ aux 层（2/18/33/36）      → 原有通道
   └─ 候选层（3..14）的 attention hook 算熵
          ↓ 全部层走完后
      选层 → gather → 追加**一个**通道
          ↓
   隐状态通道 = [aux_2, aux_18, aux_33, aux_36, AIFD]
                                              ↑ 末尾，训练侧按此切分
```

---

## 仓库结构

```
dream-aifd/
├── README.md                  本文件
├── RESULTS.md                 最终训练结果 + loss 构成 + 规模
├── docs/                      四份设计/结果文档
│   ├── AIFD_vllm_改动说明.md    ★ vLLM 侧逐函数详解（含调用时序）
│   ├── RESULT_lse_entropy.md   lse + V:=K 恒等式的推导与实测验证
│   ├── RESULT_vllm_integration.md  vLLM 集成记录 + 踩坑
│   └── RESULT_training_side.md 训练侧接入 + 归一化决策
├── repos/                     **三个仓库的完整源码副本**（改动已在其中）
│   ├── vllm/                  基线 822865845a
│   ├── vllm-ascend/           基线 6e784075d
│   └── speculators-ty5537/    基线 28ad834（含 vllm train 侧的改动）
├── patches/                   可直接 git apply 的补丁（用于在干净仓库上复现改动）
│   ├── vllm-aifd.patch
│   ├── vllm-ascend-aifd.patch
│   └── speculators-aifd.patch
├── scripts/                   探针 / 验证 / 分析脚本
└── logs/                      训练与服务日志（>1MB 的已 gzip）
```

> `repos/` 是**改动后**的完整源码（直接读、直接用）。
> `patches/` 是同一批改动的 diff（用于在你自己干净的仓库上 `git apply`）。
> 两者内容一致，选一个用即可。
>
> `repos/` 里**排除了** `.git/`、`__pycache__/`、编译产物，以及两个 `csrc/` 目录
> （vllm 的 4.9M + vllm-ascend 的 100M C++/CANN 内核源码，与 AIFD 无关，为控制体积剔除）。

### 本地完整工作区

本仓库只放**代码 + 文档 + 日志**。训练产物（检查点 25 GB、采集的隐状态 1.8 GB、
原始日志 37 MB）在本地目录：

```
~/aifd-work/
├── logs/         全部原始日志（未压缩）
├── checkpoints/  aifd_run_w01（3 epoch 完整跑）+ 6 个测试检查点
├── data/         采集的 hidden states（5 通道 / 4 通道 / 合成）
└── repos/        三个仓库副本（与本仓库 repos/ 同源，含 csrc）
```

---

## 改了哪些文件

### vLLM（`/vllm-workspace/vllm`，基线 `822865845a`）

| 文件 | 改动 |
|---|---|
| `vllm/model_executor/models/aifd.py` | **新增**，全部选层逻辑 |
| `vllm/model_executor/models/interfaces.py` | `_aifd_hook` 注入点；`_maybe_add_hidden_state` 里加一次调用 |
| `vllm/model_executor/models/extract_hidden_states.py` | `num_hidden_states` +1 |
| `vllm/v1/spec_decode/extract_hidden_states.py` | `num_hidden_states` +1（proposer，**容易漏**） |
| `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` | `maybe_enable_aifd()` |
| `vllm/v1/worker/gpu_model_runner.py` | 调用（上游 v1 runner，防御性） |

### vLLM-Ascend（`/vllm-workspace/vllm-ascend`，基线 `6e784075d`）

| 文件 | 改动 |
|---|---|
| `vllm_ascend/worker/model_runner_v1.py` | **Ascend 上真正生效的调用点**（约 3798 行） |

> ⚠️ 模型 runner 有三份，Ascend 实际用的是 vllm-ascend 那份。
> 定位方法是看日志里的**文件名**（`[model_runner_v1.py:3825]`）。

### speculators（`/home/y50063564/dreams-rl/speculators-ty5537`，基线 `28ad834`）

| 文件 | 改动 |
|---|---|
| `models/dflash/metrics.py` | 新增 `compute_aifd_loss` |
| `models/dflash/config.py` | `aifd_weight` / `aifd_norm` / `aifd_draft_layer` |
| `models/dflash/core.py` | `_backbone_forward` 返回监督点；`forward` 加 loss 项 |
| `models/dspark/core.py` | 同上（DSpark 是实际训练用的家族） |
| `train/data.py` | `ArrowDataset.aifd_channels`；按通道数切分 |
| `train/dataloader.py` | 透传 `aifd_channels` |
| `train/trainer.py` | **梯度累积**（从 `sy5537` 那份移植） |
| `scripts/train.py` | `--aifd-weight` / `--aifd-norm` / `--aifd-draft-layer` / `--grad-accum-steps` |
| `scripts/launch_vllm.py` | `--aifd-candidate-layers` / `--aifd-granularity` |

---

## 怎么用

```bash
# ── 1) 打补丁 ──
cd /path/to/vllm          && git apply /path/to/patches/vllm-aifd.patch
cd /path/to/vllm-ascend   && git apply /path/to/patches/vllm-ascend-aifd.patch
cd /path/to/speculators   && git apply /path/to/patches/speculators-aifd.patch

# ── 2) 起带 AIFD 的 vLLM server（卡 11）──
cd speculators-ty5537
PYTHONPATH=$PWD/src:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=11 \
python scripts/launch_vllm.py /path/to/Qwen3-4B \
  --hidden-states-path /tmp/aifd_hs \
  --target-layer-ids 2 18 33 \
  --aifd-candidate-layers 3 4 5 6 7 8 9 10 11 12 13 14 \
  -- --port 8200 --max-model-len 4096 --enforce-eager --gpu-memory-utilization 0.85

# ── 3) 训练（卡 12，在线生成用完即删）──
PYTHONPATH=$PWD/src:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=12 \
python scripts/train.py \
  --verifier-name-or-path /path/to/Qwen3-4B \
  --speculator-type dspark --num-layers 5 --draft-vocab-size 32000 \
  --draft-attn-impl eager --max-anchors 512 \
  --data-path /path/to/prepared_dataset \
  --vllm-endpoint http://localhost:8200/v1 \
  --on-missing generate --on-generate delete \
  --target-layer-ids 2 18 33 \
  --total-seq-len 3072 --epochs 3 --lr 3e-4 \
  --loss-fn '{"ce": 0.3, "tv": 0.7}' \
  --enable-confidence-head --confidence-head-with-markov \
  --aifd-weight 0.1 --aifd-draft-layer 2 --grad-accum-steps 12
```

**注意**：训练必须用 `PYTHONPATH=$PWD/src` 覆盖——`speculators` 是 editable 安装，
默认指向 `~/dspark_project/speculators`，不加 `PYTHONPATH` 改动不生效。

---

## 结果

见 `RESULTS.md`。一句话：**3 epoch 跑完，稳定收敛**。

| 指标 | epoch 2/3 | epoch 3/3 |
|---|---|---|
| val/loss | 0.567 | **0.527** |
| val/accept_len | 3.861 | **4.163** |
| val/full_acc | 0.613 | **0.643** |

---

## ⚠️ 重要：这个结果**不能**用来判断 AIFD 有没有用

**没有对照组。** 这一跑只跑了 `--aifd-weight 0.1`，没跑 `--aifd-weight 0`。
而且和已有的 baseline 相比，配置差异不止 AIFD 一项：
`num_layers`、`lr`、`seq_len`、`max_anchors`、**单卡 vs 多卡**、数据规模（50k vs 700k）。
**任何一项都足以解释差距。**

另外 AIFD 项在收敛后的占比只有 **2.8%**（`aifd_loss` 从初始的 ~1.3 降到 0.145），
权重 0.1 是按照**未收敛**时的量级选的——这个标定本身就是错的。
要拿到 ~10% 占比，权重应该在 **0.4** 左右。

---

## 已知问题 / 待办

1. **没有对照组**（见上）。
2. **`aifd-weight` 标定用的是未收敛量级**，建议改 0.4 附近重跑。
3. **候选窗口 2..13 下 per-sample 会退化**：离线实验里 40/40 全选同一层；
   vLLM 侧也是 3 条 prompt 全选 L10。per-token 粒度能带来逐位置变化
   （验证到用到的层 `[9,10,14]` / `[9,10,11,13]`），已经做成开关：
   `--aifd-granularity {sample,token}`。
4. **归一化默认关闭**。依据：候选层间 norm 只差 4.8x（一个数量级内），
   而 BOS 的 400x 离群 token 永远进不了 loss（`select_anchors` 只从 `loss_mask`
   里挑，实测 14624 个锚点位置 0 次命中位置 0）。
5. **每样本按温度取一个层，但监督点是 draft 的第 2 层**（`--aifd-draft-layer 2`）。
   中间层没有 RMSNorm，两边逐元素尺度差 ~3x（draft 侧 ~2.5，target 侧 ~0.78）。
   这个组合是否有害**没验证过**。

---

## 修复过的 bug（供参考）

| 现象 | 根因 | 修法 |
|---|---|---|
| engine 崩在 `hidden_states[:n] = ...` 尺寸不匹配 | 拿不到选层结果时**没 append**，通道少一个 | `_append_fallback` 永远 append 一个形状正确的张量 |
| engine 崩在 `LocalScalarDenseNpu` / AICPU 507018 | 热路径每步 **36 次 device→host 同步**（日志参数立即求值 + NPU 标量真值判断 + `.item()`） | 全部改成纯张量运算 / 一次性 `.cpu()` |
| AIFD 通道跑到 index 1 而不是末尾 | 在 `layer_idx=15` 就 append，aux 层 18/33/36 排到它后面 | 等 `layer_idx == num_layers` 再 append |
| `softmax_lse` 形状时而 `(T,N,1)` 时而 `(N,T,1)` | TND 布局下算子行为不固定 | `_as_tn` 两种都接 |
| 分段长度算错 | `actual_seq_lengths_q` 是**累加值** | `_cumulative_to_lengths` |
| 通道数改了还崩 | 三处缓冲区要一起改，漏了 proposer 那份 | 统一走 `aifd_extra_channels` |

---

## 环境

Ascend 910B（16 卡）· torch 2.10.0+cpu / torch_npu 2.10.0.post2 · CANN 9.0.0
transformers 5.13.0 · vLLM 0.23.1rc1 · vllm-ascend 0.19.1rc2 · speculators 0.6.0.dev0
target 模型 Qwen3-4B（36 层 / hidden 2560 / head_dim 128 / 32 Q 头 / 8 KV 头 / 无滑窗）

**Ascend 上的两个坑**：
- 不能用默认的 `--draft-attn-impl simple_flex_attention`，Triton/BiShengHIR 编译器报
  `ub overflow`，要用 `eager`。
- `--draft-vocab-size` 必须与已有 vocab mapping 一致（32000）。
