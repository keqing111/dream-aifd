#!/bin/bash
# per-token 粒度下扫 4 个 aifd-weight。
#
# 资源：卡 13 = vLLM server（粒度 token，需先起）
#       卡 11 / 12 = 训练，两两一批，跑完自动接下一批
# 生成：在线生成，用完即删（--on-generate delete）
#
# 用法: setsid bash run_4weights.sh > logs/driver.log 2>&1 < /dev/null &

set -u
cd /home/y50063564/dreams-rl/speculators-ty5537 || exit 1
W=/home/y50063564/aifd-work
mkdir -p "$W/logs" "$W/checkpoints"

run_one() {
  local w="$1" card="$2"
  local tag="tok_w${w}"
  echo "=== [$(date '+%H:%M:%S')] 启动 weight=$w 卡=$card → $W/checkpoints/$tag"
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
    --aifd-weight "$w" --aifd-draft-layer 2 --grad-accum-steps 12 \
    --checkpoint-freq 0.1 --num-workers 4 --log-freq 10 \
    --save-path "$W/checkpoints/$tag" \
    > "$W/logs/train_$tag.log" 2>&1
  echo "=== [$(date '+%H:%M:%S')] weight=$w 结束，退出码 $?"
}

echo "########## 第一批 ##########"
run_one 0.2 11 & run_one 0.4 12 & wait

echo "########## 第二批 ##########"
run_one 0.7 11 & run_one 1.0 12 & wait

echo "########## 全部完成 [$(date '+%H:%M:%S')] ##########"
