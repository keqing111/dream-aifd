#!/usr/bin/env python3
"""AIFD 事后分析：训练完的 draft 在监督点上到底多大程度模仿上了 target。

要回答的问题
------------
1. draft 第 2 层的输出（监督点）和 AIFD 选的 target 层，逐位置有多像？
2. 这个"像"是**均匀**的，还是**集中在部分维度上**？
   （如果只有部分维度能对上，说明 loss 里有相当一部分梯度是在逼 draft
     去够它够不着的东西 —— 这会挤压真正有用的表征。）
3. 残差是**结构化的**（说明 draft 表达力不够，有系统性偏差）
   还是**近似各向同性的噪声**（说明只是没学够）？

做法
----
在 draft 的 `layers[1]`（= 监督点，1-based 的"第 2 层"）上挂 forward hook，
跑一次前向拿到**全序列**的输出 (1, T, H)；数据里同一位置的 AIFD target 也是 (T, H)。
两者**逐位置直接对齐**，不需要重建 anchor 逻辑。

用法
----
    ASCEND_RT_VISIBLE_DEVICES=12 python analyze_aifd_match.py \
        --ckpt /tmp/aifd_run_w01/checkpoint_best
"""

from __future__ import annotations

import argparse
import sys

import torch

REPO = "/home/y50063564/dreams-rl/speculators-ty5537/src"
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/tmp/aifd_run_w01/checkpoint_best",
                    help="训练好的 draft checkpoint 目录")
    ap.add_argument("--data-path",
                    default="/home/y50063564/data/open_perfectblend_qwen3_4b_50k")
    ap.add_argument("--hidden-states-path", default="/tmp/aifd_real_hs")
    ap.add_argument("--batches", type=int, default=4, help="跑几个 batch")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--aifd-draft-layer", type=int, default=2, help="1-based")
    ap.add_argument("--device", default="npu:0")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    import torch_npu  # noqa: F401, PLC0415
    from pathlib import Path

    from hs_connectors import FileTransfer
    from speculators.model import SpeculatorModel
    from speculators.train.data import ArrowDataset, create_collate_fn

    # ── 1) 加载训练好的 draft ──
    print(f"加载 checkpoint: {args.ckpt}")
    model = SpeculatorModel.from_pretrained(args.ckpt)
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = len(model.layers)
    print(f"  draft 层数 = {n_layers}，监督点 = 第 {args.aifd_draft_layer} 层"
          f"（0-based index {args.aifd_draft_layer - 1}）")

    # ⚠️ 必须强制成 eager：checkpoint 是用 --draft-attn-impl eager 训的，
    # 但重载后 config 里的 _attn_implementation 未必留住，会走 simple_flex_attention
    # 的 torch.compile 路径 —— Ascend 上那个 BiShengHIR 编译器会 ub overflow。
    from speculators.models.attention import create_float_mask  # noqa: PLC0415

    model._attn_impl = "eager"
    model._create_mask_fn = create_float_mask
    # 层是在 forward 时读 `self.config._attn_implementation` 才决定用哪个 attn_fn，
    # 所以每个层的 config 也要一起改（只改 model 上的是不够的）。
    # attention 模块自己存了 config（`Qwen3DFlashAttention.config`），
    # forward 时读它决定用哪个 attn_fn。
    for _layer in model.layers:
        _layer.self_attn.config._attn_implementation = "eager"  # noqa: SLF001
    print("  已强制 attn impl = eager（model + 每一层）")

    # ── 2) 在监督点上挂 hook，抓全序列输出 ──
    captured: dict[str, torch.Tensor] = {}

    def hook(module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        captured["h"] = t.detach()          # (1, T, H)

    model.layers[args.aifd_draft_layer - 1].register_forward_hook(hook)

    # ── 3) 组一个真 batch（和训练同一套 collate）──
    ds = ArrowDataset(
        datapath=args.data_path,
        transfer=FileTransfer(Path(args.hidden_states_path)),
        max_len=args.max_len, on_missing="skip", split_ratio=1.0,
        aifd_channels=1,
    )
    collate = create_collate_fn(args.max_len, 2560, num_target_layers=4, aifd=True)

    samples = [ds[i] for i in range(len(ds.data)) if ds[i] is not None]
    print(f"  可用样本 {len(samples)} 条")

    all_dim_corr, all_dim_res, all_dim_tgt = [], [], []
    n_used = 0
    for bi in range(min(args.batches, max(1, len(samples) // 2))):
        chunk = samples[bi * 2: bi * 2 + 2] or samples[:2]
        if not chunk:
            continue
        batch = collate([c for c in chunk if c is not None])
        captured.clear()
        gpu_batch = {
            k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.no_grad():
            # 只为跑前向拿监督点的输出，aifd_weight=0 就不会算 AIFD loss
            model(**gpu_batch, aifd_weight=0.0)
        if "h" not in captured:
            print("  ⚠️ hook 没触发，跳过")
            continue

        pred = captured["h"].float().cpu()                       # (1, T, H)
        tgt = batch["aifd_hidden_states"].float().cpu()           # (1, T, H)
        mask = batch["loss_mask"].bool().cpu()                    # (1, T)
        if pred.shape != tgt.shape:
            print(f"  ⚠️ 形状不符 pred={tuple(pred.shape)} tgt={tuple(tgt.shape)}")
            continue

        p = pred[0][mask[0]]      # (n, H) 只在 loss_mask 位置上比
        t = tgt[0][mask[0]]
        if p.shape[0] < 16:
            continue
        all_dim_corr.append(p)
        all_dim_res.append(p)
        all_dim_tgt.append(t)
        n_used += 1

    if not all_dim_corr:
        print("❌ 没拿到有效对比数据")
        return 1

    P = torch.cat(all_dim_corr, 0)      # (N, H) draft 输出
    T = torch.cat(all_dim_tgt, 0)       # (N, H) target
    N, H = P.shape
    print(f"\n对比了 {n_used} 个 batch，共 {N} 个位置 × {H} 维")

    # ── 4) 整体相似度 ──
    cos = torch.nn.functional.cosine_similarity(P, T, dim=-1)
    print(f"\n【整体】逐位置余弦相似度: 均值 {cos.mean():.4f}  中位 {cos.median():.4f}"
          f"  p5 {cos.quantile(0.05):.4f}  p95 {cos.quantile(0.95):.4f}")

    # 参考：把 target 打乱（跨位置）后应该接近 0
    perm = torch.randperm(N)
    cos_rand = torch.nn.functional.cosine_similarity(P, T[perm], dim=-1)
    print(f"        对照（打乱位置）: 均值 {cos_rand.mean():.4f}")

    # ── 5) 逐维分析：哪些维度对上了、哪些没有 ──
    # 逐维相关（跨位置）
    Pc = P - P.mean(0, keepdim=True)
    Tc = T - T.mean(0, keepdim=True)
    denom = (Pc.norm(dim=0) * Tc.norm(dim=0)).clamp_min(1e-8)
    dim_corr = (Pc * Tc).sum(0) / denom                          # (H,)

    # 逐维残差 / 逐维 target 尺度
    resid = (P - T)
    dim_resid_std = resid.std(0)                                 # (H,)
    dim_tgt_std = T.std(0)                                       # (H,)
    ratio = dim_resid_std / dim_tgt_std.clamp_min(1e-8)          # 残差 / 目标尺度

    print(f"\n【逐维相关】均值 {dim_corr.mean():.4f}  中位 {dim_corr.median():.4f}")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"  p{int(q*100):02d} = {dim_corr.quantile(q):.4f}")
    print(f"  相关 > 0.5 的维度: {int((dim_corr > 0.5).sum())} / {H}")
    print(f"  相关 < 0.1 的维度: {int((dim_corr < 0.1).sum())} / {H}")

    print(f"\n【逐维残差/目标尺度比】中位 {ratio.median():.3f}")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"  p{int(q*100):02d} = {ratio.quantile(q):.3f}")

    # ── 6) 残差是结构化的还是噪声？（SVD）──
    R = resid - resid.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(R.float())
    sv = sv / sv.sum()
    for p in (0.001, 0.01, 0.05, 0.1):
        k = max(1, int(N * p))
        print(f"  残差 SVD: 前 {p*100:4.1f}% 奇异值（{k} 个）占 {sv[:k].sum()*100:5.1f}%")

    # 逐维系统性偏置
    bias = resid.mean(0).abs()
    print(f"\n【系统性偏置】逐维 |mean(resid)|: 中位 {bias.median():.4f}  "
          f"最大 {bias.max():.4f}（相对逐维目标尺度中位 "
          f"{(bias / dim_tgt_std.clamp_min(1e-8)).median():.3f}）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
