# 最终训练结果（3 epoch 完整跑完）

生成时间: 2026-09-22 09:36
来源日志: logs/training/aifd_run_w01d.log.gz

## 训练命令
```
# Timestamp: 2026-09-22T02:28:12.625144+00:00 # Git SHA: 28ad83452121203db2ceacfac16b5db274b7fcff # World size: 1 # speculators: 0.6.0.dev0 # vllm: 0.23.1rc1.dev1002+g822865845.empty # transformers: 5.13.0 # torch: 2.10.0+cpu # compressed-tensors: 0.17.0 scripts/train.py --verifier-name-or-path /home/y50063564/Qwen3-4B --speculator-type dspark --num-layers 5 --draft-vocab-size 32000 --draft-attn-impl eager --max-anchors 512 --data-path /home/y50063564/data/open_perfectblend_qwen3_4b_50k --vllm-endpoint http://localhost:8200/v1 --on-missing generate --on-generate delete --target-layer-ids 2 18 33 --total-seq-len 3072 --epochs 3 --lr 3e-4 --loss-fn '{"ce": 0.3, "tv": 0.7}' --enable-confidence-head --confidence-head-with-markov --markov-rank 256 --markov-head-type vanilla --aifd-weight 0.1 --aifd-draft-layer 2 --grad-accum-steps 12 --checkpoint-freq 0.1 --num-workers 4 --save-path /tmp/aifd_run_w01 --log-freq 10
```

## 验证集曲线（续跑部分：epoch 2/3 与 3/3）
```
val/confidence_loss_epoch=0.210 val/confidence_abs_error_epoch=0.191 val/confidence_pred_mean_epoch=0.573 val/loss_epoch=0.567 val/ce_loss_epoch=0.765 val/tv_loss_epoch=0.182 val/accept_rate_epoch=0.561 val/accept_len_epoch=3.861 val/full_acc_epoch=0.613 val/position_0_acc_epoch=0.793 val/position_1_acc_epoch=0.721 val/position_2_acc_epoch=0.664 val/position_3_acc_epoch=0.616 val/position_4_acc_epoch=0.576 val/position_5_acc_epoch=0.541 val/position_6_acc_epoch=0.509 val/position_7_acc_epoch=0.478 val/aifd_loss_epoch=0.162 
val/confidence_loss_epoch=0.203 val/confidence_abs_error_epoch=0.182 val/confidence_pred_mean_epoch=0.618 val/confidence_cumprod_bias_epoch=0.009 val/loss_epoch=0.527 val/ce_loss_epoch=0.692 val/tv_loss_epoch=0.167 val/accept_rate_epoch=0.595 val/accept_len_epoch=4.163 val/full_acc_epoch=0.643 val/position_0_acc_epoch=0.813 val/position_1_acc_epoch=0.746 val/position_2_acc_epoch=0.693 val/position_3_acc_epoch=0.648 val/position_4_acc_epoch=0.609 val/position_5_acc_epoch=0.575 val/position_6_acc_epoch=0.543 val/position_7_acc_epoch=0.512 val/aifd_loss_epoch=0.145 
```

## loss 构成（末次 val）
```
  CE  (w=0.3)            0.2076  占  39.4%
  TV  (w=0.7)            0.1169  占  22.2%
  confidence (w=1.0)     0.2030  占  38.5%
  AIFD (w=0.1)           0.0145  占   2.8%
```

## 训练规模
```
  数据: open_perfectblend_qwen3_4b_50k (50k 行 / 3557 万 token)
  每 epoch 约 12280 步，3 epoch 共约 36840 步
  实测 0.513 秒/步（单卡 Ascend 910B）
  总耗时约 6 小时（含一次 server 崩溃后从检查点续跑）
```
