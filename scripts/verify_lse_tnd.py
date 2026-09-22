#!/usr/bin/env python3
"""TND/varlen 布局 + v2 算子上重验熵恒等式。

`verify_lse_entropy.py` 是在 BNSD、单序列、dense mask 上验的。但 vLLM-Ascend 的
生产路径是（`attention_v1.py:780-784`）：

    input_layout = "TND"
    sparse_mode  = 3 if causal else 0
    atten_mask   = get_splitfuse_attn_mask()   # triu(ones(2048,2048), 1).int8
    pre_tokens = next_tokens = SWA_INT_MAX
    actual_seq_lengths_q = <各序列长度的累加和>

本脚本照搬这套组合，在**多条不等长序列拼接**的 TND 上重验

    H = lse - scale·<q, O^K>

v1 / v2 两个算子都测（v2 的参数名不同：return_softmax_lse / actual_seq_qlen /
softmax_scale / num_query_heads）。

用法
----
    ASCEND_RT_VISIBLE_DEVICES=12 python verify_lse_tnd.py
    ASCEND_RT_VISIBLE_DEVICES=12 python verify_lse_tnd.py --lengths 1024 300 700
"""

from __future__ import annotations

import argparse
import sys

import torch

SWA_INT_MAX = 2147483647
MASK_SIZE = 2048


def ref_per_sequence(q, k, lengths, scale, causal=True):
    """逐序列算参考 (H, lse)。q/k 是 TND 的 (T, heads, D)，按 lengths 切开。

    返回 (H, lse)，形状 (T, heads)。
    """
    H_all, lse_all = [], []
    off = 0
    for n in lengths:
        qs = q[off:off + n].detach().float().cpu().transpose(0, 1)  # (heads, n, D)
        ks = k[off:off + n].detach().float().cpu().transpose(0, 1)
        num_heads, kv_heads = qs.shape[0], ks.shape[0]
        if kv_heads != num_heads:
            ks = ks.repeat_interleave(num_heads // kv_heads, dim=0)
        scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale     # (heads,n,n)
        if causal:
            allow = torch.tril(torch.ones(n, n, dtype=torch.bool))
            scores = scores.masked_fill(~allow, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        p = torch.softmax(scores, dim=-1)
        H = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
        H_all.append(H.transpose(0, 1))          # (n, heads)
        lse_all.append(lse.transpose(0, 1))
        off += n
    return torch.cat(H_all, 0), torch.cat(lse_all, 0)


def call_v1(q, k, v, scale, n_heads, n_kv, cu, mask, sparse_mode):
    return torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        atten_mask=mask,
        actual_seq_lengths=cu,
        actual_seq_lengths_kv=cu,
        num_heads=n_heads,
        num_key_value_heads=n_kv,
        scale=scale,
        input_layout="TND",
        sparse_mode=sparse_mode,
        pre_tokens=SWA_INT_MAX,
        next_tokens=SWA_INT_MAX,
        softmax_lse_flag=True,
    )


def call_v2(q, k, v, scale, n_heads, n_kv, cu, mask, sparse_mode):
    return torch_npu.npu_fused_infer_attention_score_v2(
        q, k, v,
        atten_mask=mask,
        actual_seq_qlen=cu,
        actual_seq_kvlen=cu,
        num_query_heads=n_heads,
        num_key_value_heads=n_kv,
        softmax_scale=scale,
        input_layout="TND",
        sparse_mode=sparse_mode,
        pre_tokens=SWA_INT_MAX,
        next_tokens=SWA_INT_MAX,
        return_softmax_lse=True,
    )


def as_TH(lse, T):
    """把 lse 统一成 (T, heads)。实测形状可能是 (N,T,1) / (T,N,1) / (T,N)。"""
    x = lse.detach().float().cpu()
    x = x.squeeze(-1) if x.dim() == 3 and x.shape[-1] == 1 else x
    if x.shape[0] != T and x.shape[1] == T:
        x = x.transpose(0, 1)
    return x


def entropy_from_lse(lse_TH, q, o_k, scale):
    """H = lse - scale·<q, O^K>，TND 下 q/o_k 是 (T, heads, D)。"""
    return lse_TH - (q.detach().float().cpu() * o_k.detach().float().cpu()
                     ).sum(-1) * scale


def cmp(name, got, want, atol=5e-2):
    got, want = got.reshape(-1), want.reshape(-1)
    d = (got - want).abs()
    ok = bool(d.max().item() < atol)
    print(f"  {name:34s} max|Δ| = {d.max().item():.3e}  "
          f"范围 got[{got.min():.3f},{got.max():.3f}] "
          f"want[{want.min():.3f},{want.max():.3f}]  {'✅' if ok else '❌'}")
    return ok


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 300, 784],
                    help="各序列长度（varlen 拼接）")
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--input-mult", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    global torch_npu
    import torch_npu  # noqa: PLC0415

    N, KV, D = args.heads, args.kv_heads, args.dim
    lens = args.lengths
    T = sum(lens)
    cu = torch.tensor(lens).cumsum(0).tolist()
    scale = D ** -0.5
    g = torch.Generator().manual_seed(args.seed)

    def rnd(shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32)
                / D ** 0.5 * args.input_mult).to(torch.bfloat16).npu()

    q, k, v = rnd((T, N, D)), rnd((T, KV, D)), rnd((T, KV, D))
    # splitfuse 风格的因果 mask：上三角 = 1 = 掩掉
    mask = torch.triu(torch.ones(MASK_SIZE, MASK_SIZE), diagonal=1).to(torch.int8).npu()

    print(f"配置: TND  varlen={lens}  T={T}  N={N}  KV={KV}  D={D}  "
          f"cu={cu}  mask={tuple(mask.shape)} int8(上三角=1)  sparse_mode=3")

    H_ref, lse_ref = ref_per_sequence(q, k, lens, scale, causal=True)
    print(f"  参考: H 范围 [{H_ref.min():.3f}, {H_ref.max():.3f}]  "
          f"lse 范围 [{lse_ref.min():.3f}, {lse_ref.max():.3f}]")

    all_ok = True
    for opname, fn in (("v1 (softmax_lse_flag)", call_v1),
                       ("v2 (return_softmax_lse)", call_v2)):
        print(f"\n########## {opname} ##########")
        try:
            out, lse = fn(q, k, v, scale, N, KV, cu, mask, 3)
            out_k, _ = fn(q, k, k, scale, N, KV, cu, mask, 3)   # V := K
        except Exception as e:  # noqa: BLE001
            print(f"  调用失败: {type(e).__name__}: {str(e)[:200]}")
            all_ok = False
            continue
        print(f"  attn_out {tuple(out.shape)}   softmax_lse {tuple(lse.shape)} "
              f"dtype={lse.dtype}")
        lse_TH = as_TH(lse, T)
        ok = cmp("lse vs 逐序列因果 logsumexp", lse_TH, lse_ref)
        H_got = entropy_from_lse(lse_TH, q, out_k, scale)
        ok &= cmp("H = lse - scale·<q, O^K>", H_got, H_ref)
        all_ok &= ok

    print("\n" + "=" * 78)
    print("结论：TND/varlen 下恒等式" + ("成立 —— 生产路径的布局可以直接用"
          if all_ok else "仍有问题，见上"))
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
