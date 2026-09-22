#!/usr/bin/env python3
"""检验「少数维度决定输出，但 AIFD 在逼 draft 模仿全部维度」这个假设。

假设的推理链
------------
1. draft 最终只需要让 top-k 概率和 target 一致；
2. 对这少数 token 起作用的只是**个别维度**；
3. 而 AIFD 拿中间层的 hidden 做 SmoothL1，**2560 维全都要对** ——
   大量维度对 top-k 没影响，梯度却照样在推，挤压了真正有用的表征。
4. 补充：这个"少数维度"要在**靠输出那一端**看；中间层的特征空间已经
   "混乱"，逐维方差看不出结构。

两个量
------
* **任务相关度**：扰动第 L 层第 d 维，最终 logits 会变多少。
  用 Hutchinson 估 `‖∂logits/∂h[L][d]‖²`：对 vocab 维撒随机向量 g，
  反传 `logits·g`，累积梯度的逐维平方。
* **逐维方差**：同一批数据上 h[L] 的逐维 std。

判定
----
若 (相关度高度集中) 且 (与方差几乎不相关) → 假设成立；
若 (相关度也均匀) → 假设不成立（loss 没有在浪费梯度）。

⚠️ 逐 token 的 hidden 里位置 0 是 BOS 的 massive activation（norm 是中位数的
几百倍）。不剔除它，方差会被它一个人吃光（实测 CV 从 0.4 飙到 46）。

用法
----
    ASCEND_RT_VISIBLE_DEVICES=12 python analyze_dim_relevance.py --layers 10 20 30 35
"""

from __future__ import annotations

import argparse
import sys

import torch

DS = "/home/y50063564/data/open_perfectblend_qwen3_4b_50k"
MODEL = "/home/y50063564/Qwen3-4B"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 20, 30, 35],
                    help="要分析的层（HF hidden_states 索引；k = 第 k-1 层的输出）")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--n-probe", type=int, default=8,
                    help="Hutchinson 随机投影数，越多越准")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--device", default="npu:0")
    return ap.parse_args()


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """秩相关。逐维方差是重尾的，Pearson 会被离群点带偏，所以用 Spearman。"""
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (ra.std() * rb.std()))


def concentration(v: torch.Tensor, name: str) -> dict:
    o = torch.sort(v, descending=True).values
    cum = o.cumsum(0) / o.sum()
    out = {"CV": float(v.std() / v.mean())}
    for p in (0.05, 0.10, 0.50):
        k = max(1, int(len(o) * p))
        out[f"top{int(p*100)}%"] = float(cum[k - 1])
    return out


def main() -> int:
    args = parse_args()
    import torch_npu  # noqa: F401, PLC0415
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM

    ds = load_from_disk(DS)
    rows = [i for i in range(len(ds)) if int(ds[i]["seq_len"]) >= 128][: args.n_samples]

    print(f"加载 {MODEL}（eager）")
    model = (AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager")
        .to(args.device).eval())
    for p in model.parameters():
        p.requires_grad_(False)

    report = {}
    for L in args.layers:
        rel_acc, hid_list = None, []
        for r in rows:
            ids = ds[r]["input_ids"].to(torch.long)[: args.max_len].unsqueeze(0)
            ids = ids.to(args.device)
            store: dict = {}

            def hook(m, i, o, _s=store):
                h = o if isinstance(o, torch.Tensor) else o[0]
                h.retain_grad()
                _s["h"] = h

            # HF: hidden_states[k]，k=0 是 embedding，k=j+1 是第 j 层的输出。
            # 要"第 L 层的输出"就挂 model.model.layers[L-1]。
            hd = model.model.layers[L - 1].register_forward_hook(hook)
            try:
                # 参数全冻结 → 计算图是断的，retain_grad() 会报
                # "can't retain_grad on Tensor that has requires_grad=False"。
                # 让 embedding 输出 require grad 即可把整条链接回图里，
                # 参数依旧冻结（backward 不为它们分配梯度内存）。
                emb = model.model.embed_tokens(ids).detach().requires_grad_(True)
                logits = model(inputs_embeds=emb).logits          # (1, T, V)
                Hs = store["h"]                                   # (1, T, H)
                H = Hs.shape[-1]
                acc = torch.zeros(H, dtype=torch.float64, device=args.device)
                for _ in range(args.n_probe):
                    g = torch.randn_like(logits)
                    # retain_graph=True: 要反传多次；每次反传后清 Hs.grad，
                    # 否则 PyTorch 会把梯度**累加**。
                    (logits.float() * g).sum().backward(retain_graph=True)
                    acc += (Hs.grad.float() ** 2).mean(dim=(0, 1))
                    Hs.grad = None
                rel = (acc / args.n_probe).cpu()
                rel_acc = rel if rel_acc is None else rel_acc + rel
                hid_list.append(Hs.detach().float().cpu().reshape(-1, H))
            finally:
                hd.remove()

        R = rel_acc / len(rows)                                   # (H,) 任务相关度
        X = torch.cat(hid_list, 0)                                # (N, H)

        # 剔除 BOS 那种离群 token（norm 远超中位数）
        nrm = X.norm(dim=-1)
        keep = nrm < nrm.median() * 10
        Xk = X[keep]
        var = Xk.var(0)

        cR = concentration(R, "relevance")
        cV = concentration(var, "variance")
        rho = spearman(R, var)
        k5 = max(1, int(len(R) * 0.05))
        share = float(var[R.topk(k5).indices].sum() / var.sum())

        report[L] = (cR, cV, rho, share, int((~keep).sum()))
        print(f"  L{L:>2} 完成（剔除离群 {int((~keep).sum())}/{X.shape[0]} token）")

    # ── 汇总表 ──
    print("\n" + "=" * 96)
    print("【任务相关度】‖∂logits/∂h[L][d]‖²  （扰动该维对最终输出影响多大）")
    print(f"{'层':>4} {'CV':>7} {'前5%维占比':>11} {'前10%':>8} {'前50%':>8}   "
          f"{'逐维方差CV':>10} {'ρ(相关度,方差)':>15} {'相关度top5%维占方差':>20}")
    for L, (cR, cV, rho, share, _) in report.items():
        print(f"{L:>4} {cR['CV']:>7.3f} {cR['top5%']*100:>10.1f}% "
              f"{cR['top10%']*100:>7.1f}% {cR['top50%']*100:>7.1f}%   "
              f"{cV['CV']:>10.3f} {rho:>15.4f} {share*100:>19.1f}%")

    print("\n读法：")
    print("  * 相关度若集中在少数维 → 前5%/前10% 应该远大于 5%/10%，CV 应该很大")
    print("  * ρ(相关度,方差) 接近 0 → 方差完全不能指示哪些维重要（支持假设）")
    print("  * 最后两列：相关度 top5% 的维占总方差的比例；无关时期望 ≈5%")

    # ── lm_head 列范数（静态对照）──
    W = model.lm_head.weight.detach().float().cpu()
    colnorm = W.norm(dim=0)
    cW = concentration(colnorm, "lm_head")
    print(f"\n【对照】lm_head 权重逐维列范数: CV={cW['CV']:.3f}  "
          f"前5%维占 {cW['top5%']*100:.1f}%")
    print(f"  （若 lm_head 均匀读所有维，CV 会很小）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
