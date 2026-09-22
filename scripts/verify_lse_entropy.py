#!/usr/bin/env python3
"""验证：能否用 softmax_lse + 一次 V:=K 的注意力，免落地 S×S 地算出注意力熵。

数学
----
记 s_ij = scale·<q_i, k_j>，p_ij = softmax_j(s_ij)，lse_i = log Σ_j exp(s_ij)。

    H_i = -Σ_j p_ij·log p_ij
        = -Σ_j p_ij·(s_ij - lse_i)
        = lse_i - Σ_j p_ij·s_ij                       (因为 Σ_j p_ij = 1)

而第二项不需要落地分数矩阵：

    Σ_j p_ij·s_ij = scale·<q_i, Σ_j p_ij·k_j>
                  = scale·<q_i, attn(q, k, V:=k)_i>

所以：

    H_i = lse_i - scale·<q_i, O^K_i>

`lse` 由 npu_fused_infer_attention_score(softmax_lse_flag=True) 直接给出，
`O^K` 由同一次调用把 value 换成 key 得到。

本脚本用朴素 fp32 softmax 手算的熵做对照，逐项比对。

用法
----
    ASCEND_RT_VISIBLE_DEVICES=12 python verify_lse_entropy.py
"""

from __future__ import annotations

import argparse
import sys

import torch


def ref_entropy(q, k, scale, mask_keep=None, num_heads=None):
    """朴素 fp32 参考：返回 (H, lse)，形状 (B, N, S)。mask_keep: (S, S) bool，True=保留。

    一律搬到 CPU 算，避免与 NPU 上的 mask 设备不匹配。
    GQA 下 k 的头数少于 q，按 repeat_interleave 展开（与 npu 算子的语义一致）。
    """
    q = q.detach().float().cpu()
    k = k.detach().float().cpu()
    if num_heads is not None and k.shape[1] != num_heads:
        k = k.repeat_interleave(num_heads // k.shape[1], dim=1)
    if mask_keep is not None:
        mask_keep = mask_keep.detach().cpu()
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B,N,S,S_kv)
    if mask_keep is not None:
        scores = scores.masked_fill(~mask_keep, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)                   # (B,N,S)
    p = torch.softmax(scores, dim=-1)
    H = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
    return H, lse


def call_fia(q, k, v, scale, num_kv_heads, sparse_mode=0, atten_mask=None,
             softmax_lse_flag=True, next_tokens=2147483647, pre_tokens=2147483647):
    """调 npu_fused_infer_attention_score（BNSD 布局）。返回 (attn_out, lse)。

    因果（无 atten_mask）用 sparse_mode=3 + next_tokens=0 —— 与
    vllm_ascend/attention/attention_v1.py 的用法一致。
    """
    out, lse = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        atten_mask=atten_mask,
        num_heads=q.shape[1],
        num_key_value_heads=num_kv_heads,
        scale=scale,
        input_layout="BNSD",
        sparse_mode=sparse_mode,
        next_tokens=next_tokens,
        pre_tokens=pre_tokens,
        softmax_lse_flag=softmax_lse_flag,
    )
    return out, lse


def report(name, got, want, atol=2e-2):
    got = got.detach().float().cpu().reshape(-1)
    want = want.detach().float().cpu().reshape(-1)
    diff = (got - want).abs()
    ok = bool(diff.max().item() < atol)
    print(f"  [{name}] max|Δ| = {diff.max().item():.3e}  "
          f"mean|Δ| = {diff.mean().item():.3e}  "
          f"范围 got[{got.min():.3f},{got.max():.3f}] "
          f"want[{want.min():.3f},{want.max():.3f}]   {'✅' if ok else '❌'}")
    return ok


def entropy_from_lse(lse, q, o_k, scale):
    """H = lse - scale·<q, O^K>，逐 (B,N,S) 求和 head_dim。"""
    return lse.float() - (q.float() * o_k.float()).sum(-1) * scale


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=32,
                    help="< heads 时为 GQA")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--input-mult", type=float, default=8.0,
                    help="放大输入让分数有动态范围；=1 时注意力接近均匀，测不出问题")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    global torch_npu
    import torch_npu  # noqa: PLC0415

    dt = getattr(torch, args.dtype)
    B, N, KV, S, D = args.batch, args.heads, args.kv_heads, args.seq, args.dim
    scale = D ** -0.5
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    def rnd(*shape):
        # 乘 input_mult 是为了让分数有动态范围：input_mult=1 时 softmax 几乎是均匀的，
        # 熵恒等于 log(S)，恒等式会被"平凡地"验证通过而看不出问题。
        return (torch.randn(*shape, generator=g, dtype=torch.float32)
                / D ** 0.5 * args.input_mult).to(dt).npu()

    q, k, v = rnd(B, N, S, D), rnd(B, KV, S, D), rnd(B, KV, S, D)
    print(f"配置: B={B} N={N} KV={KV} S={S} D={D} dtype={args.dtype} "
          f"scale={scale:.6f} input_mult={args.input_mult} "
          f"{'(GQA)' if KV < N else '(MHA)'}")

    # ---------- 测试 1：无 mask ----------
    print("\n【测试 1】无 mask —— 隔离出 lse / 熵恒等式本身")
    H_ref, lse_ref = ref_entropy(q, k, scale, num_heads=N)
    out, lse = call_fia(q, k, v, scale, KV)
    out_k, _ = call_fia(q, k, k, scale, KV)      # V := K
    print(f"  attn_out {tuple(out.shape)}   softmax_lse {tuple(lse.shape)} "
          f"dtype={lse.dtype}")
    report("lse vs logsumexp(scores)", lse.reshape(lse_ref.shape), lse_ref)
    H_got = entropy_from_lse(lse.reshape(lse_ref.shape), q, out_k, scale)
    ok1 = report("H = lse - scale·<q, O^K>", H_got, H_ref)

    # 顺便交叉验：attention_out 本身对不对
    p_ref = torch.softmax(
        torch.matmul(q.float(), k.float().repeat_interleave(N // KV, dim=1).transpose(-1, -2))
        * scale, dim=-1)
    out_ref = torch.matmul(p_ref, v.float().repeat_interleave(N // KV, dim=1))
    report("attn_out vs 朴素 softmax·V", out, out_ref, atol=5e-2)

    # ---------- 测试 2：因果 mask ----------
    print("\n【测试 2】因果 mask")
    mask_keep = torch.tril(torch.ones(S, S, dtype=torch.bool)).npu()
    H_ref_c, lse_ref_c = ref_entropy(
        q, k, scale, mask_keep=mask_keep.cpu(), num_heads=N
    )

    # 形状是有讲究的：sparse_mode 0/1 要 (1,1,Q_S,KV_S)；2/3/4 要固定 (2048,2048)。
    # 先用合法形状把极性测出来，不要靠猜。
    m41 = mask_keep.reshape(1, 1, S, S).contiguous()      # tril=True
    m41_inv = (~mask_keep).reshape(1, 1, S, S).contiguous()
    variants = [
        ("sm=1 (1,1,S,S) tril=True ", dict(sparse_mode=1, atten_mask=m41)),
        ("sm=1 (1,1,S,S) tril=False", dict(sparse_mode=1, atten_mask=m41_inv)),
        ("sm=1 (1,1,S,S) tril=True  int8", dict(sparse_mode=1,
                                               atten_mask=m41.to(torch.int8))),
        ("sm=1 (1,1,S,S) 上三角=True int8",
         dict(sparse_mode=1, atten_mask=m41_inv.to(torch.int8))),
    ]
    # 先用 attention 输出本身定性 mask 语义 —— 比拿熵去猜可靠得多
    v_rep = v.float().repeat_interleave(N // KV, dim=1)
    p_c = torch.softmax(
        torch.matmul(q.float(), k.float().repeat_interleave(N // KV, dim=1)
                     .transpose(-1, -2)).masked_fill(
                         ~mask_keep, float("-inf")) * scale, dim=-1)
    out_ref_c = torch.matmul(p_c, v_rep)

    print("  —— 先看 attn_out 对不对（这直接说明 mask 语义）——")
    good = None
    for name, kw in variants:
        try:
            out_c, lse_c = call_fia(q, k, v, scale, KV, **kw)
            d_out = (out_c.detach().float().cpu() - out_ref_c.cpu()).abs().max().item()
            hit = d_out < 5e-2
            if hit and good is None:
                good = (name, kw)
            print(f"  {name:34s}  max|Δattn_out| = {d_out:.4e}  "
                  f"{'✅ ← mask 语义就是这个' if hit else '❌'}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:34s}  失败: {type(e).__name__}: {str(e)[:160]}")

    if good is None:
        print("  ⇒ 没有一种 mask 组合复现出因果输出，需要再查")
    else:
        name, kw = good
        out_c, lse_c = call_fia(q, k, v, scale, KV, **kw)
        out_k_c, _ = call_fia(q, k, k, scale, KV, **kw)   # V := K
        lse_c = lse_c.reshape(lse_ref_c.shape)
        print(f"\n  —— 用「{name}」逐项拆开看 ——")
        report("lse vs 因果 logsumexp", lse_c, lse_ref_c, atol=5e-2)
        # 逐位置看：位置 0 只允许看 j=0，熵必须为 0，lse 必须等于 s_00
        k_rep_diag = k.float().repeat_interleave(N // KV, dim=1)
        s00 = (q.float()[0, :, 0, :] * k_rep_diag[0, :, 0, :]).sum(-1) * scale
        print(f"    位置0: lse_ref = {lse_ref_c[0, :, 0].mean():.4f}  "
              f"lse_op = {lse_c.float().cpu()[0, :, 0].mean():.4f}  "
              f"s_00 = {s00.cpu().mean():.4f}")
        print(f"    位置0 的 H_ref = {H_ref_c[0, :, 0].mean():.4f}（应为 0）")
        # O^K 本身对不对？参考 = Σ_j p_ij·k_j
        k_rep = k.float().repeat_interleave(N // KV, dim=1)
        report("O^K vs Σ_j p_ij·k_j", out_k_c, torch.matmul(p_c, k_rep), atol=5e-2)
        H_c = entropy_from_lse(lse_c, q, out_k_c, scale)
        # 逐位置拆开
        print(f"\n    {'位置':>5} {'lse':>10} {'Σps':>10} {'H_got':>10} {'H_ref':>10} {'Σp':>8}")
        for i in (0, 1, 2, 8, 64, 200, 255):
            lse_i = lse_c.float().cpu()[0, :, i].mean()
            sps = (q.float().cpu()[0, :, i, :] * out_k_c.float().cpu()[0, :, i, :]
                   ).sum(-1).mean() * scale
            hg = H_c.detach().float().cpu()[0, :, i].mean()
            hr = H_ref_c.float()[0, :, i].mean()
            psum = p_c.cpu()[0, :, i, :].sum(-1).mean()
            print(f"    {i:>5} {lse_i:>10.4f} {sps:>10.4f} {hg:>10.4f} "
                  f"{hr:>10.4f} {psum:>8.4f}")
        report("H (因果) = lse - scale·<q, O^K>", H_c, H_ref_c, atol=5e-2)

    # ---------- 测试 3：把 head 平均，看 DREAM-S 口径 ----------
    if ok1:
        print("\n【测试 3】head 平均后的量级（DREAM-S 口径是 -(p·logp) 再 mean(head)）")
        H_head_mean_got = H_got.mean(dim=1)
        H_head_mean_ref = H_ref.mean(dim=1)
        report("mean over heads", H_head_mean_got, H_head_mean_ref)

    print("\n" + ("=" * 70))
    if ok1:
        print("结论：恒等式成立 —— H = lse - scale·<q, O^K> 可以用")
        print(f"      代价 = attention 里多一次 V:=K 的调用（lse 是免费的）")
    else:
        print("结论：恒等式在此硬件/版本上不成立，需要看上面哪一项对不上")
    print("=" * 70)
    return 0 if ok1 else 1


if __name__ == "__main__":
    sys.exit(main())
