# 日志索引

原件（未压缩）在 `~/aifd-work/logs/`。这里放压缩版，>1MB 的已 gzip。

## training/ —— 训练日志

| 文件 | 内容 |
|---|---|
| `run_final_3epoch.log.gz` | ★ **最终完整跑**：epoch 1-2，3 epoch 全部完成，`Validation epoch 3/3 completed`，`checkpoint_best` 更新。**最重要的日志**。 |
| `run_first_crashed.log.gz` | 第一次跑：epoch 0，跑到 step ~11740 时 server 崩（AICPU），之后 ~800 步是空 batch（指标全 0），最后也崩了。 |
| `run_resume_1.log.gz` / `run_resume_2.log.gz` | 两次重启续跑。篇幅小，主要是崩溃诊断。 |

## server/ —— vLLM 数据生成 server 日志

| 文件 | 内容 |
|---|---|
| `server_final.log.gz` | ★ **最终 server**：服务 17329+ 请求，成功选层 2.7 万次，**0 次通道数异常**。含 `AIFD: 已挂 12 个候选层的 attention hook... 粒度=sample`。 |
| `server_crash_channel_mismatch.log.gz` | 第一次崩溃：`RuntimeError: The expanded size of the tensor (5) must match the existing size (4)` —— 拿不到选层结果时**没 append 通道**，缓冲区尺寸不匹配打死 engine。 |
| `server_crash_aicpu.log.gz` | 第二次崩溃：`AICPU exception 507018`。**traceback 指向我自己写的一行日志**——热路径上每步 36 次 device→host 同步。 |

## analysis/ —— 验证与分析

| 文件 | 内容 |
|---|---|
| `verify_granularity_sample.log` | 粒度 `sample` 端到端验证：AIFD 通道与某个候选通道**逐位相同**。 |
| `verify_granularity_token.log` | 粒度 `token` 验证：逐位置命中不同层（用到的层 `[9,10,14]` / `[6,10,14]` / `[9,10,11,13]`）。 |
| `dim_relevance.log` | 逐维任务相关度分析（检验「少数维度决定输出」假设）。 |

（另有 `verify_e2e_offline.log` 等离线验证日志）

## 常用查询

```bash
# 选层分布
zcat logs/server/server_final.log.gz | grep "选中层分布"

# 异常诊断（attention hook 为什么跳过）
zcat logs/server/server_crash_*.log.gz | grep "hook 跳过原因计数"

# 验证集曲线
zcat logs/training/run_final_3epoch.log.gz | grep -E "val/(loss|accept_len|full_acc)_epoch"

# 续跑点
zcat logs/training/*.log.gz | grep "Resuming training"
```
