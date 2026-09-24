#!/bin/bash
# baseline 对照：不带 AIFD 的 server（4 通道）+ --aifd-weight 0。
#
# 目的：判断「4 组 AIFD 实验 vs 原先的 baseline 的偏移」是不是我改的代码造成的。
# 这一路**完全不碰 AIFD**：server 不吐那个通道，加载器 aifd_channels=0，
# 走的是原始数据路径 —— 所以它能直接检验基础路径有没有被我的改动影响。
#
# 跑两个种子：
#   seed=42 —— 和那 4 组 AIFD 实验同种子，做公平 A/B
#   seed=43 —— 用来估计 run-to-run 噪声（0.7% 那个差异到底算不算差异）
#
# 用法: setsid bash run_baseline.sh > logs/driver_baseline.log 2>&1 < /dev/null &

set -u
cd /home/y50063564/dreams-rl/speculators-ty5537 || exit 1
W=/home/y50063564/aifd-work
mkdir -p "$W/logs" "$W/checkpoints"

run_one() {
  local seed="$1" card="$2"
  local tag="base_seed${seed}"
  echo "=== [$(date '+%H:%M:%S')] 启动 seed=$seed 卡=$card → $W/checkpoints/$tag"
  PYTHONPATH="$PWD/src:$PYTHONPATH" ASCEND_RT_VISIBLE_DEVICES="$card" \
  python scripts/train.py \
    --verifier-name-or-path /home/y50063564/Qwen3-4B \
    --speculator-type dspark --num-layers 5 --draft-vocab-size 32000 \
    --draft-attn-impl eager --max-anchors 512 \
    --data-path /home/y50063564/data/open_perfectblend_qwen3_4b_50k \
    --vllm-endpoint http://localhost:8200/v1 \
    --on-missing generate --on-generate delete \
    --target-layer-ids 2 18 33 \
    --total-seq-len 3072 --epochs 3 --lr 3e-4 \
    --loss-fn '{"ce": 0.3, "tv": 0.7}' \
    --enable-confidence-head --confidence-head-with-markov \
    --markov-rank 256 --markov-head-type vanilla \
    --aifd-weight 0 --grad-accum-steps 12 \
    --checkpoint-freq 0.1 --num-workers 4 --log-freq 10 \
    --seed "$seed" \
    --save-path "$W/checkpoints/$tag" \
    > "$W/logs/train_$tag.log" 2>&1
  echo "=== [$(date '+%H:%M:%S')] seed=$seed 结束，退出码 $?"
}

run_one 42 11 & run_one 43 12 & wait
echo "########## baseline 全部完成 [$(date '+%H:%M:%S')] ##########"
