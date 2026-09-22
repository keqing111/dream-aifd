#!/usr/bin/env python3
"""AIFD 探针：Qwen3-4B 每层 hidden 的尺度 + 每层 attention 熵 + 候选层打分。

这个脚本先回答一个问题：**不同层 hidden_states 的尺度差多少**，顺带把 AIFD 的
打分公式实算一遍，看候选窗口（前 40%、跳过前两层）合不合理。

口径（与 DREAM-S 对齐）
----------------------
记 `h_l` = **第 l 层的输出**（= HF `hidden_states[l+1]`；`hidden_states[0]` 是 embedding）。

1. 每层每 token 的注意力熵，照抄 `dream_s/ge_data/ge_data_suppliments.py:52-59`：

       He[l,t] = - sum_k  A[l,t,k] * log(A[l,t,k] + eps)      # 对 key 维求和
       He[l,t] = He[l,t].mean(over heads)                      # 再对 head 平均

2. per-sample 聚合（本仓库的选择；DREAM-S 原版是 per-token）：

       C_l = mean_t He[l,t]                                    # 每样本每层一个标量

3. 打分公式（上层 = l-1）：

       score_l = C_l + |C_l - C_{l-1}|
       best    = argmin_l score_l

4. 候选层：前 40% 的层、跳过前两层。36 层 ⇒ 层 2..13。

为什么要 hook 而不是 output_attentions=True
------------------------------------------
`attn_implementation="eager"` 下 `eager_attention_forward` **总是**返回
`attn_weights` `(B, 32, S, S)`；模型级 `output_attentions=False` 时这些权重
不会被 `all_self_attentions` 收集，每层算完即释放。所以在 `self_attn` 上挂一个
forward hook、在 hook 里**当场**把熵算出来只留 `(S,)`，驻留显存 ≈ 1 层的
`S x S`，而不是 36 层。

用法
----
    ASCEND_RT_VISIBLE_DEVICES=11 python probe_layer_stats.py --n-samples 4
    ASCEND_RT_VISIBLE_DEVICES=11 python probe_layer_stats.py --n-samples 8 --out-json stats.json
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import statistics as st
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

log = logging.getLogger("probe")


# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
N_LAYERS = 36
HIDDEN_SIZE = 2560
CANDIDATE_FRACTION = 0.4  # 前 40% 的层
SKIP_FIRST = 2            # 跳过前两层
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def candidate_layers(num_layers: int = N_LAYERS) -> list[int]:
    """前 40% 的层里跳过前两层。36 层 ⇒ 前 40% = 层 0..13 ⇒ 层 2..13。"""
    return list(range(SKIP_FIRST, int(CANDIDATE_FRACTION * num_layers)))


# --------------------------------------------------------------------------
# hook
# --------------------------------------------------------------------------
@dataclass
class ProbeBuffers:
    """每层只留 `(S,)` 的两个派生量；候选层额外留完整 `(S, H)` fp32。"""

    seq_len: int
    keep_full: set[int]
    norm: list[torch.Tensor | None] = field(default_factory=lambda: [None] * N_LAYERS)
    ce: list[torch.Tensor | None] = field(default_factory=lambda: [None] * N_LAYERS)
    full: dict[int, torch.Tensor] = field(default_factory=dict)
    embed_norm: torch.Tensor | None = None


def attach_probe_hooks(model, seq_len: int, keep_full: set[int]):
    """挂 `2 x 层数 + 1` 个 hook。返回 (Bufers, handles)，用完必须 remove_hooks。"""
    layers = model.model.layers
    if len(layers) != N_LAYERS:
        raise ValueError(f"expected {N_LAYERS} layers, got {len(layers)}")

    buf = ProbeBuffers(seq_len=seq_len, keep_full=keep_full)
    handles: list = []

    def _layer_hook(i, module, args, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        if h.shape[1] != buf.seq_len:
            raise RuntimeError(f"layer {i}: seq {h.shape[1]} != {buf.seq_len}")
        # token 级 L2 norm —— 在 device 上算完只留 (S,)
        buf.norm[i] = h.detach().float().norm(dim=-1).squeeze(0).to("cpu")
        if i in buf.keep_full:
            buf.full[i] = h.detach().float().squeeze(0).to("cpu")

    def _attn_hook(i, module, args, output):
        if not isinstance(output, (tuple, list)) or output[1] is None:
            raise RuntimeError(
                f"layer {i}: self_attn 没返回 attn_weights —— "
                "确认 attn_implementation='eager'"
            )
        w = output[1]  # (B, H, S, S) bf16
        if w.shape[-1] != buf.seq_len:
            raise RuntimeError(f"layer {i}: attn 末维 {w.shape[-1]} != {buf.seq_len}")
        # bf16 的 log 精度不够，升到 fp32 算；算完立刻只剩 (S,)
        w = w.float()
        ce = -(w * w.clamp_min(1e-8).log()).sum(dim=-1)  # (B,H,S) 对 key 维求和
        buf.ce[i] = ce.mean(dim=1).squeeze(0).to("cpu")   # (B,S) 再对 head 平均
        del w, ce

    def _embed_hook(module, args, output):
        buf.embed_norm = output.detach().float().norm(dim=-1).squeeze(0).to("cpu")

    handles.append(model.model.embed_tokens.register_forward_hook(_embed_hook))
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(functools.partial(_layer_hook, i)))
        handles.append(
            layer.self_attn.register_forward_hook(functools.partial(_attn_hook, i))
        )
    return buf, handles


def remove_hooks(handles) -> None:
    for h in handles:
        h.remove()
    handles.clear()


def assert_complete(buf: ProbeBuffers) -> None:
    """hook 漏触发会静默产出 None —— 在用之前拦一道。"""
    missing = [i for i in range(N_LAYERS) if buf.norm[i] is None or buf.ce[i] is None]
    if missing:
        raise RuntimeError(f"hook 漏触发的层: {missing}")
    missing = sorted(buf.keep_full - buf.full.keys())
    if missing:
        raise RuntimeError(f"候选层完整张量缺失: {missing}")
    if buf.embed_norm is None:
        raise RuntimeError("embedding hook 未触发")


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def q_summary(x: torch.Tensor) -> dict:
    x = x.float()
    vals = torch.quantile(x, torch.tensor(QUANTILES)).tolist()
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        **{f"p{int(q * 100)}": v for q, v in zip(QUANTILES, vals)},
        "max": float(x.max()),
        "min": float(x.min()),
    }


def score_layers(ce_mean: list[float]) -> list[float | None]:
    """score_l = C_l + |C_l - C_{l-1}|；l=0 没有上层，记 None。"""
    return [None] + [ce_mean[l] + abs(ce_mean[l] - ce_mean[l - 1])
                     for l in range(1, len(ce_mean))]


def score_matrix(ce_mat: torch.Tensor) -> torch.Tensor:
    """逐 token 版：`(L, S)` 的熵 -> `(L, S)` 的 score，l=0 行填 inf 排除。

    score[l,t] = He[l,t] + |He[l,t] - He[l-1,t]|
    """
    out = torch.full_like(ce_mat, float("inf"))
    out[1:] = ce_mat[1:] + (ce_mat[1:] - ce_mat[:-1]).abs()
    return out


def pick_per_token(ce_mat: torch.Tensor, cand: list[int]) -> torch.Tensor:
    """逐 token 在候选层内 argmin，返回 (S,) 的层号。"""
    sc = score_matrix(ce_mat)[cand]           # (len(cand), S)
    return torch.tensor(cand)[sc.argmin(dim=0)]  # (S,)


def pick_per_token_plain_entropy(ce_mat: torch.Tensor, cand: list[int]) -> torch.Tensor:
    """对照：DREAM-S 原版口径，直接对熵 argmin（没有 |ΔC| 那一项）。"""
    sc = ce_mat[cand]                         # (len(cand), S)
    return torch.tensor(cand)[sc.argmin(dim=0)]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset",
                    default="/home/y50063564/data/open_perfectblend_qwen3_4b_700k")
    ap.add_argument("--verifier", default="/home/y50063564/Qwen3-4B")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-len", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=2048,
                    help="超过就截断（eager 注意力显存 O(S^2)）")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None,
                    help="逐 token 熵矩阵 (36, S) 落盘，key = ce_<row_idx>")
    ap.add_argument("--no-full", action="store_true",
                    help="不留候选层完整张量（只看分布时用，省显存）")
    return ap.parse_args()


def _as_int_list(col) -> list[int]:
    """torch 格式的数据集取列可能返回 tensor，也可能返回 0-dim tensor 的列表。"""
    if isinstance(col, torch.Tensor):
        return [int(x) for x in col.reshape(-1).tolist()]
    return [int(x) for x in col]


def pick_rows(ds, n: int, seed: int, min_len: int) -> list[int]:
    seq_lens = _as_int_list(ds["seq_len"])
    cand = [i for i, s in enumerate(seq_lens) if s >= min_len]
    if len(cand) < n:
        raise SystemExit(f"长度 >= {min_len} 的样本只有 {len(cand)} 条 < {n}")
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(cand), generator=g).tolist()
    rows = [cand[k] for k in order[:n]]
    log.info("抽中 %d 行: %s（原长 %s）", len(rows), rows, [seq_lens[r] for r in rows])
    return rows


def run_one(model, ds, row_idx: int, cand: list[int], max_len: int, device: str,
            keep_full_tensors: bool = True) -> dict:
    item = ds[row_idx]
    ids = item["input_ids"]
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.int64)
    ids = ids.to(torch.int64)
    t_full = int(ids.shape[0])
    t = min(t_full, max_len)
    ids = ids[:t].unsqueeze(0).to(device)

    buf, handles = attach_probe_hooks(
        model, t, keep_full=set(cand) if keep_full_tensors else set()
    )
    try:
        with torch.no_grad():
            model.model(ids, use_cache=False)
        assert_complete(buf)
    finally:
        remove_hooks(handles)

    ce_mean = [float(buf.ce[l].mean()) for l in range(N_LAYERS)]
    scores = score_layers(ce_mean)
    best = min(cand, key=lambda l: scores[l])

    def _outlier(l):
        """离群 token 诊断：norm 超过本层中位数 10 倍的位置与个数。"""
        n = buf.norm[l]
        med = float(n.median())
        mask = n > 10 * med
        return {
            "argmax_pos": int(n.argmax()),
            "max_over_median": float(n.max()) / (med + 1e-9),
            "n_gt_10x_median": int(mask.sum()),
        }

    layers = {
        l: {
            "norm": q_summary(buf.norm[l]),
            "outlier": _outlier(l),
            "ce_mean": ce_mean[l],
            "ce_token_std": float(buf.ce[l].std()),
            "score": scores[l],
            "in_candidates": l in cand,
        }
        for l in range(N_LAYERS)
    }
    # 逐 token 分析用的原始熵矩阵 (L, S) fp32 —— 留一份，省得为了换口径重跑模型
    ce_mat = torch.stack([buf.ce[l] for l in range(N_LAYERS)]).float()

    out = {
        "row_idx": row_idx,
        "seq_len_full": t_full,
        "seq_len_used": t,
        "embed_norm_mean": float(buf.embed_norm.mean()),
        "best_layer": best,
        "best_score": scores[best],
        "layers": layers,
    }
    del buf
    return out, ce_mat


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    import torch_npu  # noqa: F401, PLC0415  —— 必须在建 NPU tensor 之前 import
    from datasets import load_from_disk  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    log.info("加载 %s (eager, bf16) 到 %s", args.verifier, args.device)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.verifier, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(args.device)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)

    ds = load_from_disk(args.dataset)
    rows = pick_rows(ds, args.n_samples, args.seed, args.min_len)
    cand = candidate_layers()

    results = []
    ce_mats: dict[int, torch.Tensor] = {}
    for row_idx in rows:
        r, ce_mat = run_one(model, ds, row_idx, cand, args.max_len, args.device,
                            keep_full_tensors=not args.no_full)
        results.append(r)
        ce_mats[row_idx] = ce_mat
        log.info("row %d  T=%d  best_layer=%d  score=%.4f",
                 row_idx, r["seq_len_used"], r["best_layer"], r["best_score"])
        try:
            torch.npu.empty_cache()
        except Exception:  # noqa: BLE001 —— 非 NPU 环境下静默跳过
            pass

    report(results, cand)
    report_per_token(results, ce_mats, cand)
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(results, indent=2, ensure_ascii=False)
        )
        log.info("原始结果写入 %s", args.out_json)
    if args.out_npz:
        _np = __import__("numpy")
        _np.savez_compressed(
            args.out_npz,
            **{f"ce_{k}": v.numpy() for k, v in ce_mats.items()},
            rows=_np.array(rows),
        )
        log.info("逐 token 熵矩阵写入 %s（key = ce_<row_idx>，形状 (36, S)）", args.out_npz)
    return 0


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def _mean_over(results: list[dict], fn) -> list[float]:
    return [st.mean(fn(r, l) for r in results) for l in range(N_LAYERS)]


def report(results: list[dict], cand: list[int]) -> None:
    n = len(results)
    print()
    print("=" * 104)
    print(f"AIFD 探针报告 —— {n} 条样本，候选层 = {cand[0]}..{cand[-1]}（{len(cand)} 层）")
    print("=" * 104)

    # ---- 表 1：每层 hidden 的尺度 ----
    print("\n【表 1】每层 h_l 的 token 级 L2 norm（h_l = 第 l 层输出；相对值以层 0 为 1.0）")
    print(f"{'层':>3} {'候选':>4} {'mean':>9} {'std':>8} {'p5':>8} {'中位':>8} {'p95':>8}"
          f" {'max':>9} {'相对层0':>9} {'CV':>7}")
    base = st.mean(r["layers"][0]["norm"]["mean"] for r in results)
    for l in range(N_LAYERS):
        m = st.mean(r["layers"][l]["norm"]["mean"] for r in results)
        s = st.mean(r["layers"][l]["norm"]["std"] for r in results)
        p5 = st.mean(r["layers"][l]["norm"]["p5"] for r in results)
        med = st.mean(r["layers"][l]["norm"]["p50"] for r in results)
        p95 = st.mean(r["layers"][l]["norm"]["p95"] for r in results)
        mx = st.mean(r["layers"][l]["norm"]["max"] for r in results)
        mark = "*" if l in cand else ""
        print(f"{l:>3} {mark:>4} {m:>9.1f} {s:>8.1f} {p5:>8.1f} {med:>8.1f} {p95:>8.1f}"
              f" {mx:>9.1f} {m / base:>9.2f} {s / m:>7.3f}")

    # ---- 表 2：候选层之间的尺度离散度（决定要不要归一化）----
    print("\n【表 2】候选层之间的尺度离散度  ← 决定要不要归一化的关键")
    cand_means = [st.mean(r["layers"][l]["norm"]["mean"] for r in results) for l in cand]
    lo_l = min(cand, key=lambda l: st.mean(r["layers"][l]["norm"]["mean"] for r in results))
    hi_l = max(cand, key=lambda l: st.mean(r["layers"][l]["norm"]["mean"] for r in results))
    for l in cand:
        m = st.mean(r["layers"][l]["norm"]["mean"] for r in results)
        print(f"  层 {l:>2}: mean norm = {m:>9.1f}   （相对最小候选层 {m / min(cand_means):>5.2f}x）")
    print(f"\n  候选层内：最小 {min(cand_means):.1f}（层 {lo_l}）"
          f" / 最大 {max(cand_means):.1f}（层 {hi_l}）"
          f"  ⇒ 极差 {max(cand_means) / min(cand_means):.2f}x")
    print(f"  候选层内 CV = {st.pstdev(cand_means) / st.mean(cand_means):.3f}")
    all_means = _mean_over(results, lambda r, l: r["layers"][l]["norm"]["mean"])
    print(f"  （对照：全部 36 层的极差 {max(all_means) / min(all_means):.2f}x，"
          f"CV = {st.pstdev(all_means) / st.mean(all_means):.3f}）")

    # 中位数口径（对离群 token 稳健）
    cand_meds = [
        st.mean(r["layers"][l]["norm"]["p50"] for r in results) for l in cand
    ]
    print(f"\n  换用**中位数**口径：候选层 {min(cand_meds):.1f} .. {max(cand_meds):.1f}"
          f"  ⇒ 极差 {max(cand_meds) / min(cand_meds):.2f}x"
          f"，CV = {st.pstdev(cand_meds) / st.mean(cand_meds):.3f}")

    print("\n【表 2b】离群 token 诊断（max 远超中位数 ⇒ 逐 token 归一化几乎是必须的）")
    print(f"{'层':>3} {'候选':>4} {'max/中位':>10} {'>10x中位的token数':>18} {'argmax位置':>11}")
    for l in cand:
        r10 = st.mean(r["layers"][l]["outlier"]["max_over_median"] for r in results)
        n10 = st.mean(r["layers"][l]["outlier"]["n_gt_10x_median"] for r in results)
        ap = [r["layers"][l]["outlier"]["argmax_pos"] for r in results]
        print(f"{l:>3} {'*':>4} {r10:>10.1f} {n10:>18.1f} {str(ap):>11}")

    # ---- 表 3：每层注意力熵与打分 ----
    print("\n【表 3】每层 attention 熵 C_l 与 score_l = C_l + |C_l - C_{l-1}|")
    print(f"{'层':>3} {'候选':>4} {'C_l':>10} {'|C_l - C_{l-1}|':>16} {'score_l':>10}")
    for l in range(N_LAYERS):
        c = st.mean(r["layers"][l]["ce_mean"] for r in results)
        sc = [r["layers"][l]["score"] for r in results if r["layers"][l]["score"] is not None]
        # 层 0 没有上层，score 恒为 None
        s_txt = f"{st.mean(sc):>10.4f}" if sc else f"{'—':>10}"
        d_txt = f"{st.mean(sc) - c:>16.4f}" if sc else f"{'—':>16}"
        mark = "*" if l in cand else ""
        print(f"{l:>3} {mark:>4} {c:>10.4f} {d_txt} {s_txt}")

    # ---- 表 4：每样本选中的层 ----
    print("\n【表 4】每样本选中的层")
    cnt: Counter = Counter()
    for r in results:
        cnt[r["best_layer"]] += 1
        bn = r["layers"][r["best_layer"]]["norm"]["mean"]
        print(f"  row {r['row_idx']:>6}  T={r['seq_len_used']:>5}  →  层 {r['best_layer']:>2}"
              f"   score={r['best_score']:.4f}   该层 norm={bn:.1f}")
    print(f"\n  选中层分布: {dict(sorted(cnt.items()))}")

    # ---- 结论 ----
    picked_means = [
        st.mean(r["layers"][l]["norm"]["mean"] for r in results if r["best_layer"] == l)
        for l in sorted(cnt)
    ]
    print("\n【结论】")
    print(f"  候选层内 norm 极差 = {max(cand_means) / min(cand_means):.2f}x"
          f"（层 {lo_l} vs 层 {hi_l}）")
    if len(picked_means) > 1:
        print(f"  本批样本实际选中层的 norm 极差 = "
              f"{max(picked_means) / min(picked_means):.2f}x"
              f"（{min(picked_means):.1f} .. {max(picked_means):.1f}）")
    else:
        print("  本批样本全落在同一层，实际尺度差需要更多样本才能看出来")
    print()


def _hist(counts: torch.Tensor, cand: list[int]) -> str:
    total = counts.sum().item()
    return "  ".join(f"L{l}:{int(counts[l])}" for l in cand if counts[l] > 0) + \
           f"   (共 {int(total)} token)"


def report_per_token(results: list[dict], ce_mats: dict[int, torch.Tensor],
                     cand: list[int]) -> None:
    """逐 token 选择：每个 token 各自在候选层内 argmin。"""
    print()
    print("=" * 104)
    print("【逐 token 选择】对比 per-sample（后者在本批样本里退化成常数）")
    print("=" * 104)

    agg_ours = torch.zeros(N_LAYERS)
    agg_plain = torch.zeros(N_LAYERS)
    frac_mode_ours: list[float] = []
    n_mode_ours: list[float] = []
    per_sample_best: list[int] = []

    print("\n每样本：逐 token 选中层的分布")
    print(f"{'row':>8} {'T':>6} {'众数层':>7} {'众数占比':>9} {'不同层数':>9} "
          f"{'分布熵':>8} {'per-sample层':>12} {'位置0选中':>10}")
    for r in results:
        row = r["row_idx"]
        ce = ce_mats[row]                       # (36, S)
        pick = pick_per_token(ce, cand)          # (S,)
        plain = pick_per_token_plain_entropy(ce, cand)
        S = pick.shape[0]

        counts = torch.bincount(pick, minlength=N_LAYERS).float()
        agg_ours += counts
        agg_plain += torch.bincount(plain, minlength=N_LAYERS).float()
        p = counts / counts.sum()
        mode_layer = int(p.argmax())
        mode_frac = float(p[mode_layer])
        frac_mode_ours.append(mode_frac)
        n_mode_ours.append(float((counts > 0).sum()))
        nz = p[p > 0]
        dist_entropy = float(-(nz * nz.log()).sum())
        per_sample_best.append(r["best_layer"])

        print(f"{row:>8} {S:>6} {mode_layer:>7} {mode_frac:>8.1%} "
              f"{int((counts > 0).sum()):>9} {dist_entropy:>8.3f} "
              f"{r['best_layer']:>12} {int(pick[0]):>10}")

    total_tok = agg_ours.sum().item()
    print(f"\n全局逐 token 分布（{int(total_tok)} 个 token，"
          "score_l = He_l + |He_l - He_{l-1}|）：")
    for l in cand:
        c = int(agg_ours[l])
        print(f"  L{l:>2}: {c:>7}  {c / total_tok:>7.2%}  {'#' * int(60 * c / total_tok)}")
    top1 = agg_ours.argmax().item()
    print(f"  ⇒ 众数层 L{top1} 占 {agg_ours[top1] / total_tok:.1%}"
          f"；per-token 用到的不同层数 = {int((agg_ours > 0).sum())}")

    print(f"\n对照：DREAM-S 原版口径（直接对熵 argmin，无 |ΔC| 项）：")
    for l in cand:
        c = int(agg_plain[l])
        if c:
            print(f"  L{l:>2}: {c:>7}  {c / agg_plain.sum():>7.2%}  "
                  f"{'#' * int(60 * c / agg_plain.sum())}")
    print(f"  ⇒ 众数层 L{int(agg_plain.argmax())} 占 {agg_plain.max() / agg_plain.sum():.1%}"
          f"；用到的不同层数 = {int((agg_plain > 0).sum())}")

    print("\n对比：")
    print(f"  per-sample : {len(set(per_sample_best))} 个不同层 → {sorted(set(per_sample_best))}")
    print(f"  per-token  : 用到的层数 = {int((agg_ours > 0).sum())}"
          f"，每样本众数占比均值 {st.mean(frac_mode_ours):.1%}"
          f"（{min(frac_mode_ours):.1%} .. {max(frac_mode_ours):.1%}）")
    print(f"  ⇒ per-token 确实给出了逐位置不同的层号；"
          f"但每样本内的集中度仍有 {st.mean(frac_mode_ours):.1%}")

    # BOS / 位置 0 是否特殊
    poss0 = [int(pick_per_token(ce_mats[r["row_idx"]], cand)[0]) for r in results]
    print(f"\n位置 0（BOS）选中的层: {poss0}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
