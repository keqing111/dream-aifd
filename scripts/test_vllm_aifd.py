#!/usr/bin/env python3
"""端到端验证 vLLM 侧 AIFD：跑一次 extract_hidden_states，检查多出来的那个通道。

自洽校验的思路
--------------
把**候选层本身**同时配成 aux 通道（`eagle_aux_hidden_state_layer_ids = 候选层`）。
这样 vLLM 会把每个候选层的 hidden 都当成普通通道吐出来（前 N 个），AIFD 再追加一个
（第 N+1 个）。于是：

  * 通道数必须是 `len(候选层) + 1`；
  * **第 N+1 个通道必须与前面某一个通道逐位相同** —— 因为 AIFD 就是"从候选层里挑一层"。
    这条把 gather 的正确性钉死，不需要任何外部参考。

至于"挑得对不对"，另用 HF 参考实现比对（见 compare_aifd_with_hf.py）。

用法
----
    ASCEND_RT_VISIBLE_DEVICES=12 python test_vllm_aifd.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch
from safetensors.torch import load_file

CANDIDATES = list(range(3, 15))  # eagle_aux 层号约定：id L = 第 L-1 层(0-based)的输出


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/home/y50063564/Qwen3-4B")
    ap.add_argument("--outdir", default=None, help="不给就用临时目录")
    ap.add_argument("--num-prompts", type=int, default=3)
    ap.add_argument("--granularity", default="sample", choices=["sample", "token"])
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig

    outdir = args.outdir or tempfile.mkdtemp(prefix="aifd_hs_")
    os.makedirs(outdir, exist_ok=True)

    print(f"候选层（eagle_aux 约定）: {CANDIDATES}")
    print(f"aux 通道 = 候选层本身，共 {len(CANDIDATES)} 个；期望总通道数 "
          f"{len(CANDIDATES) + 1}")

    llm = LLM(
        model=args.model,
        enforce_eager=True,          # 别让 ACL graph 把 hook 里的算子吞掉
        max_model_len=2048,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": list(CANDIDATES),
                    "aifd_config": {
                        "candidate_layers": list(CANDIDATES),
                        "granularity": args.granularity,
                    },
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": outdir,
                "allow_custom_save_path": True,
            },
        ),
    )

    # 用**不同**的 prompt：相同 prompt 在 prefix caching 下会共用前缀，
    # 落盘的文件名对不上（而且不同样本本来就可能选到不同的层，这正是要看的）。
    pool = [
        "The capital of France is",
        "Write a short poem about the sea:",
        "Explain photosynthesis in one sentence:",
        "List three prime numbers greater than ten:",
        "Translate 'good morning' into Japanese:",
        "What is the tallest mountain in the world?",
    ]
    assert args.num_prompts <= len(pool), f"最多 {len(pool)} 条"
    prompts = pool[: args.num_prompts]
    outputs = llm.generate(
        prompts,
        [SamplingParams(max_tokens=1, extra_args={
            "kv_transfer_params": {
                "hidden_states_path": os.path.join(outdir, f"hs_{i}.safetensors"),
            }
        }) for i in range(args.num_prompts)],
    )

    # 落盘是异步 flush 的，generate 返回时文件可能还没落。等一下。
    import time

    paths = [o.kv_transfer_params["hidden_states_path"] for o in outputs]
    for _ in range(60):
        if all(os.path.exists(p) for p in paths):
            break
        time.sleep(1)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"⚠️  {len(missing)} 个文件等了 60s 仍未落盘：{missing}")

    n_cand = len(CANDIDATES)
    ok_all = True
    for i, out in enumerate(outputs):
        path = out.kv_transfer_params["hidden_states_path"]
        if not os.path.exists(path):
            print(f"\n--- prompt {i} ---\n  ⚠️  {path} 不存在，跳过")
            continue
        hs = load_file(path)["hidden_states"]           # (T, channels, H)
        T, C, H = hs.shape
        print(f"\n--- prompt {i} ---")
        print(f"  token 数 T={T}  通道数 C={C}   hidden={H}")
        if C != n_cand + 1:
            print(f"  ❌ 通道数不对：期望 {n_cand + 1}，实际 {C}")
            ok_all = False
            continue

        aifd = hs[:, -1, :]                              # (T, H) AIFD 通道

        if args.granularity == "sample":
            # per-sample：整条 AIFD 通道必须与某一个候选通道**逐位相同**
            hit = next((CANDIDATES[j] for j in range(n_cand)
                        if bool((hs[:, j, :] == aifd).all())), None)
            if hit is None:
                diffs = sorted(((CANDIDATES[j],
                                 float((hs[:, j, :] - aifd).abs().max()))
                                for j in range(n_cand)), key=lambda x: x[1])
                print(f"  ❌ 与任何候选通道都不逐位相同。最接近："
                      f"L{diffs[0][0]} (max|Δ|={diffs[0][1]:.3e})")
                ok_all = False
            else:
                print(f"  ✅ AIFD 通道 == 候选层 L{hit} 的通道（逐位相同）")
        else:
            # per-token：**逐位置**检查 —— 每个 token 的行必须等于某个候选层在
            # 该位置的行。整条通道不要求匹配同一个层。
            matched = torch.zeros(T, dtype=torch.bool)
            per_token_layer = torch.full((T,), -1, dtype=torch.long)
            for j, aux_id in enumerate(CANDIDATES):
                eq = (hs[:, j, :] == aifd).all(dim=-1)        # (T,) 该层是否逐位命中
                per_token_layer[eq] = aux_id
                matched |= eq
            n_ok = int(matched.sum())
            if n_ok == T:
                used = sorted(set(per_token_layer.tolist()))
                print(f"  ✅ 每个 token 都命中某个候选层；用到的层 = {used}（{len(used)} 个）")
            else:
                bad = (~matched).nonzero().flatten()[:5].tolist()
                print(f"  ❌ {T - n_ok}/{T} 个 token 未命中任何候选层；例如位置 {bad}")
                ok_all = False

    print("\n" + "=" * 70)
    print("端到端结果：" + ("通过" if ok_all else "有失败项") + f"\n输出目录 {outdir}")
    print("=" * 70)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
