# 方法流程与指标说明

本文档说明当前项目的同步 baseline、异步多流水方案，以及结果文件中各指标的含义和计算方式。默认脚本会把同步结果写入 `results/sync/`，把异步结果写入 `results/async/`。

## 1. 当前方法流程

当前项目支持两种运行模式：

- `--mode sync`：同步 barrier baseline。
- `--mode async`：异步多验证流水线方案。

同步 baseline 是“三端异构设备 + 边缘 7B 服务器”的同步 speculative decoding 仿真/测量流程。

1. 数据选择
   - 入口：`python -m edge_spec.run`。
   - 默认数据：`data/spec_bench/question.jsonl`。
   - 正式脚本通过 `DATASET_MODE` 切换数据集使用方式。
   - `DATASET_MODE=one-per-category`：每个类别取 1 条样本。
   - `DATASET_MODE=all`：运行所有选中样本，不丢弃任何样本；最后一个 microbatch 可以少于 3 条。

2. 模型分工
   - `device-0` 使用 `Qwen/Qwen2.5-0.5B-Instruct` 作为 draft 模型。
   - `device-1` 使用 `Qwen/Qwen2.5-1.5B-Instruct` 作为 draft 模型。
   - `device-2` 使用 `Qwen/Qwen2.5-3B-Instruct` 作为 draft 模型。
   - 边缘服务器使用 `Qwen/Qwen2.5-7B-Instruct` 作为 target/verifier 模型。

3. 每轮同步推测解码
   - 每个未完成的端侧请求从当前 prefix 开始生成最多 `gamma` 个 draft token。
   - 端侧 draft 耗时直接按运行记录计算：`draft_start_s = client.available_at_s`，`draft_end_s = draft_start_s + measured_draft_time_s`，`draft_time_s = draft_end_s - draft_start_s`。
   - 每个端侧把 prefix、draft token、draft 分布上传到边缘服务器，上行延迟由 payload、带宽、RTT、jitter、拥塞概率共同仿真。
   - draft 包到达时间为 `arrival_s = draft_end_s + uplink_s`。
   - 边缘服务器等待本轮所有活跃设备的 draft 包到达，形成同步 barrier：`barrier_time_s = max(arrival_s)`。
   - 边缘服务器对本轮 batch 做 target forward，拿到每个 draft 位置以及 bonus token 位置的 target 分布。
   - 对每条请求执行 Leviathan-style exact speculative sampling：
     - draft token 以 `min(1, p_target(token) / p_draft(token))` 的概率被接受。
     - 一旦拒绝，用 `max(p_target - p_draft, 0)` 的残差分布采样替代 token。
     - 如果 `gamma` 个 draft token 全部接受，则额外从 target 的 bonus 分布采样 1 个 token。
   - 边缘服务器把本轮 emitted token 下发给端侧，下行延迟同样按网络 profile 仿真。
   - 请求生成到 EOS 或达到 `max_new_tokens` 后结束。

4. microbatch 与全局时间
   - 同一个 microbatch 内的 3 个请求同步推进。
   - 下一个 microbatch 会在上一个 microbatch 全部请求完成后开始。
   - `total_virtual_time_s` 是所有 microbatch 串行执行后的虚拟完成时间。

5. target-only baseline
   - 若未使用 `--skip-target-baseline`，每条请求还会跑一次公平的端到端 target-only baseline。
   - `target_only_latency_s = target_only_uplink_s + target_only_model_latency_s + target_only_downlink_s`。
   - `speedup_vs_target_only = target_only_latency_s / speculative_latency_s`。
   - 该值大于 1 表示 speculative 流程更快；小于 1 表示当前 speculative 流程比 target-only 慢。

## 1.1 异步多流水方案

异步模式由 `HeteroAsyncPipelineRunner` 实现，入口参数是 `--mode async --pipeline-count N`。

1. 多流水划分
   - 边缘服务器被抽象为 `N` 条相互独立的验证流水线。
   - 每条流水线维护自己的 `available_at_s`、累计忙碌时间和验证次数。
   - 流水线内部顺序处理草稿片段；不同流水线之间可并行推进。

2. 端侧异步起草
   - 每个设备维护自己的请求队列。
   - 同一设备上的请求串行推进：当前请求完成后，该设备才开始下一个请求。
   - 不同设备之间互不等待。
   - 每个请求每轮从当前 prefix 起草最多 `gamma` 个 token，生成 draft packet 后经上行网络到达边缘侧。

3. 流水线调度
   - draft packet 到达后不会等待其他设备形成 batch。
   - 调度器选择“能最早开始处理该 packet”的流水线。
   - 选择规则：最小化 `max(pipeline.available_at_s, packet.arrival_s)`；若并列，再按流水线当前可用时间和 `pipeline_id` 打破平局。
   - packet 在流水线上的排队时间为 `pipeline_queue_wait_s = pipeline_start_s - arrival_s`。

4. 非阻塞验证与局部回退
   - 每次异步验证的 `target_batch_size` 为 1，即单个草稿片段进入某条流水线。
   - 验证仍使用 exact speculative sampling。
   - 如果发生拒绝，只影响该请求本轮输出：用 target/draft 残差分布采样替代 token，并从新的 prefix 继续下一轮起草。
   - 其他设备和其他流水线不回退、不等待。

5. 异步时间语义
   - `total_virtual_time_s` 是所有设备请求队列全部完成的最大完成时间。
   - 异步模式没有全局 barrier，因此 `barrier_wait_s` 固定为 0，`slowest_device_rounds` 不适用。
   - 异步模式新增流水线队列等待、流水线忙碌时间和利用率指标。

## 2. 汇总指标：`summary.json`

### 全局规模与吞吐

- `request_count`
  - 总请求数。
  - 公式：`len(records)`。

- `round_count`
  - `sync` 模式下是总同步轮次数。
  - `async` 模式下是总验证事件数，也就是进入流水线的 draft packet 数。
  - 公式：`len(round_trace)`。

- `total_generated_tokens`
  - 所有请求实际输出的 token 数总和。
  - 公式：`sum(generated_token_count)`。

- `total_virtual_time_s`
  - 整次实验的虚拟总耗时，单位秒。
  - `sync` 模式：所有 microbatch 串行执行后的完成时间。
  - `async` 模式：所有设备请求队列全部完成时的最大完成时间。

- `throughput_tokens_per_s`
  - 全局吞吐，单位 token/s。
  - 公式：`total_generated_tokens / total_virtual_time_s`。
  - 注意：这是端到端虚拟吞吐，不是 GPU 单次 forward 的裸吞吐。

### speculative decoding 质量

- `mean_acceptance_rate`
  - 请求级 acceptance rate 的平均值。
  - 单请求公式：`accepted_draft_tokens / proposed_draft_tokens`。
  - 汇总公式：`mean(record.acceptance_rate)`。
  - 含义：draft token 被 target 接受的比例。越高表示 draft 模型越贴近 target 模型，通常能减少同步轮数。

### barrier 与同步开销

- `mean_barrier_wait_s`
  - 所有轮次、所有设备的 barrier 等待时间均值，单位秒。
  - 单设备单轮公式：`barrier_wait_s = barrier_time_s - arrival_s`。
  - 含义：某设备 draft 包已到达后，为等待最慢设备而空等的时间。

- `barrier_wait_fraction`
  - barrier 等待在“端侧 draft 生成 + 上行 + barrier 等待”中的占比。
  - 公式：`sum(barrier_wait_s) / (sum(draft_time_s + uplink_s) + sum(barrier_wait_s))`。
  - 注意：该比例不包含 target forward 和 downlink 时间，只衡量同步等待在端侧到达阶段的占比。

- `slowest_device_rounds`
  - 每个设备成为本轮最慢到达设备的次数。
  - 实现上取 `barrier_wait_s` 最小的设备，因为最慢设备的等待时间为 0。
  - 含义：哪个设备最常决定 barrier 时间。
  - `async` 模式没有全局 barrier，该字段为空对象。

### 异步流水线指标

以下字段只在 `--mode async` 下有明确含义。

- `mode`
  - 运行模式，异步方案为 `"async"`。

- `pipeline_count`
  - 边缘侧验证流水线数量。

- `verification_event_count`
  - 异步验证事件数。
  - 公式：`len(round_trace)`。

- `pipeline_verification_count`
  - 所有流水线处理过的 draft packet 总数。

- `mean_pipeline_queue_wait_s`
  - draft packet 到达边缘后，在流水线前排队的平均时间。
  - 公式：`mean(pipeline_queue_wait_s)`。

- `pipeline_queue_wait_fraction`
  - 流水线排队等待在“排队等待 + target forward”中的占比。
  - 公式：`sum(pipeline_queue_wait_s) / (sum(pipeline_queue_wait_s) + sum(target_forward_s))`。

- `pipeline_busy_s`
  - 每条流水线累计执行 target verification 的时间。

- `pipeline_verifications`
  - 每条流水线处理的 draft packet 数。

- `pipeline_utilization`
  - 每条流水线利用率。
  - 公式：`pipeline_busy_s[pipeline] / total_virtual_time_s`。

- `mean_pipeline_utilization`
  - 所有流水线平均利用率。
  - 公式：`sum(pipeline_busy_s) / (pipeline_count * total_virtual_time_s)`。

### 网络指标

- `mean_uplink_effective_mbps`
  - 所有上行传输采样后的有效带宽均值，单位 Mbps。
  - 有效带宽由 profile 基准带宽、带宽 jitter、拥塞共同决定。

- `mean_downlink_effective_mbps`
  - 所有下行传输采样后的有效带宽均值，单位 Mbps。

- `mean_uplink_effective_rtt_ms`
  - 所有上行传输采样后的有效 RTT 均值，单位毫秒。
  - 有效 RTT 由 profile 基准 RTT、RTT jitter、拥塞共同决定。

- `mean_downlink_effective_rtt_ms`
  - 所有下行传输采样后的有效 RTT 均值，单位毫秒。

- `network_congestion_events`
  - 上行和下行发生拥塞的总次数。
  - 每个设备每轮有两个网络样本：uplink 和 downlink。

- `network_congestion_fraction`
  - 网络样本中的拥塞比例。
  - 公式：`network_congestion_events / network_samples`，其中 `network_samples = 2 * 设备轮次样本数`。

### target-only 对比

- `mean_target_only_latency_s`
  - target-only 端到端 baseline 的请求级平均延迟，单位秒。
  - 单请求公式：`target_only_uplink_s + target_only_model_latency_s + target_only_downlink_s`。
  - 仅在启用 target baseline 时存在；否则为 `null`。

- `mean_speedup_vs_target_only`
  - 请求级 speedup 的平均值。
  - 单请求公式：`target_only_latency_s / latency_s`。
  - 解释：
    - `> 1`：当前 speculative 流程比 target-only 快。
    - `= 1`：两者相当。
    - `< 1`：当前 speculative 流程比 target-only 慢。
  - 该字段使用端到端 target-only latency 计算，不再只使用 target 模型裸生成耗时。

### 设备级延迟

- `mean_latency_by_device`
  - 每个设备上请求的端到端平均延迟，单位秒。
  - 单请求延迟从 microbatch 开始计时，到该请求收到最后一轮下行 token 为止。

### 任务级指标：`task_metrics`

每个 task/category 下都有以下字段。

- `request_count`
  - 该任务的请求数。

- `microbatch_count`
  - 该任务覆盖的 microbatch 数。
  - `all` 模式下，microbatch 按数据顺序每 3 条切分，最后一个 microbatch 可以少于 3 条。

- `generated_token_count`
  - 该任务的输出 token 总数。
  - 当前实现中与 `effective_received_token_count` 相同。

- `effective_received_token_count`
  - 端侧实际收到的 token 总数。
  - 当前没有丢包或截断建模，因此等于输出 token 总数。

- `effective_duration_s`
  - 该任务的有效完成时长，单位秒。
  - 计算方式：对该任务涉及的每个 microbatch，取该 microbatch 内该任务请求的最大 `latency_s`，再求和。
  - 在当前每任务一个 microbatch 的设置下，等于该任务 3 个设备请求中最慢请求的延迟。

- `effective_throughput_tokens_per_s`
  - 该任务有效吞吐，单位 token/s。
  - 公式：`effective_received_token_count / effective_duration_s`。

- `effective_received_throughput_tokens_per_s`
  - 与 `effective_throughput_tokens_per_s` 相同。
  - 保留这个字段是为了以后区分“生成 token”和“端侧实际收到 token”。

- `e2e_first_token_latency_s`
  - 该任务请求的平均首 token 延迟，单位秒。
  - 单请求首 token 延迟：从 microbatch 开始，到端侧第一次收到 emitted token。

- `e2e_mean_latency_s`
  - 该任务请求的平均端到端延迟，单位秒。
  - 公式：`mean(record.latency_s)`。

### profile 与数据选择

- `profiles`
  - 每个设备的仿真 profile。
  - `uplink_mbps` / `downlink_mbps`：基准上下行带宽。
  - `rtt_ms`：基准 RTT。
  - `jitter_ms`：最终传输 delay 上的一阶随机抖动。
  - `bandwidth_jitter_ratio`：带宽随机波动比例。
  - `rtt_jitter_ms`：RTT 随机波动范围。
  - `congestion_probability`：单次传输发生拥塞的概率。
  - `congestion_slowdown`：拥塞时带宽除以该值，RTT 乘以该值。
  - 当前默认配置让三台设备使用相同的网络基准参数，网络状态每次传输动态采样；端侧 draft 耗时来自真实生成记录。

- `dataset_selection`
  - 本次实验的数据选择方式。
  - `dataset_path`：数据路径。
  - `one_per_category`：是否每类取 1 条。
  - `one_per_category_per_device`：是否每类每设备取 1 条。
  - `dataset_mode`：数据集使用模式，常用值为 `one-per-category` 或 `all`。
  - `all`：是否运行所有选中样本且不丢弃尾部样本。
  - `selected_request_count`：筛选后的原始请求数。
  - `request_count`：实际进入 microbatch 运行的请求数。
  - `dropped_request_count`：未进入运行的请求数。`all` 模式下应为 0。
  - `microbatch_count`：microbatch 数。
  - `categories`：选中的任务类别。

## 3. 请求级指标：`specbench_sync_hetero.jsonl`

每一行是一条请求的记录。

- `microbatch_id`
  - 请求所属 microbatch。

- `device_id`
  - 请求分配到的端侧设备。

- `draft_model`
  - 端侧 draft 模型。

- `target_model`
  - 边缘服务器 target/verifier 模型。

- `task`
  - SpecBench 类别。

- `prompt_id`
  - 数据集里的请求 ID。

- `prompt`
  - 输入 prompt。

- `generated_text`
  - speculative 流程生成的文本。

- `generated_token_count`
  - speculative 流程输出 token 数。

- `effective_received_token_count`
  - 端侧收到 token 数；当前等于 `generated_token_count`。

- `acceptance_rate`
  - draft token 接受率。
  - 公式：`accepted_draft_tokens / proposed_draft_tokens`。

- `accepted_draft_tokens`
  - 被 target 接受的 draft token 数。

- `proposed_draft_tokens`
  - draft 模型提出的 token 数。

- `sync_rounds`
  - `sync` 模式下是该请求参与的同步轮数。
  - `async` 模式下沿用该字段表示该请求发起并完成的异步验证轮数。

- `execution_mode`
  - 请求来自同步 baseline 还是异步方案，取值为 `sync` 或 `async`。

- `async_rounds`
  - 异步模式下该请求的验证轮数。
  - 与异步模式里的 `sync_rounds` 相同，用于避免读结果时误解。

- `pipeline_ids`
  - 异步模式下该请求每轮使用的流水线 ID 序列。

- `pipeline_switches`
  - 异步模式下该请求相邻两轮切换流水线的次数。

- `first_token_latency_s`
  - 首 token 端到端延迟，单位秒。

- `latency_s`
  - 请求端到端延迟，单位秒。

- `tokens_per_s`
  - 请求级吞吐。
  - 公式：`generated_token_count / latency_s`。

- `effective_received_tokens_per_s`
  - 请求级端侧接收吞吐；当前等于 `tokens_per_s`。

- `target_only_latency_s`
  - target-only 端到端 baseline 延迟，单位秒。
  - 公式：`target_only_uplink_s + target_only_model_latency_s + target_only_downlink_s`。

- `target_only_model_latency_s`
  - target 模型单独生成的耗时，单位秒。
  - 该字段用于拆解 baseline，不直接作为 speedup 分子。

- `target_only_uplink_s`
  - target-only baseline 中客户端上传 prompt/prefix 到边缘侧的网络延迟，单位秒。

- `target_only_downlink_s`
  - target-only baseline 中边缘侧把完整生成结果下发给端侧的网络延迟，单位秒。

- `target_only_uplink_payload_bytes`
  - target-only baseline 上行 payload 估算字节数。

- `target_only_downlink_payload_bytes`
  - target-only baseline 下行 payload 估算字节数。

- `target_only_text`
  - target-only baseline 生成文本。

- `speedup_vs_target_only`
  - 相对 target-only 的加速比。
  - 公式：`target_only_latency_s / latency_s`。

## 4. 轮次级指标：`round_trace.jsonl`

每一行是一轮同步 trace。

### 轮次字段

- `microbatch_id`
  - 所属 microbatch。

- `round_index`
  - microbatch 内的轮次编号，从 0 开始。

- `target_batch_size`
  - `sync` 模式：本轮参与 target verification 的活跃请求数。
  - `async` 模式：固定为 1，表示每个 draft packet 独立注入流水线。
  - 请求提前 EOS 后，后续轮次可能小于 3。

- `barrier_time_s`
  - 本轮所有 draft 包到齐的虚拟时间点，单位秒。
  - 仅同步模式使用。

- `target_forward_s`
  - target 模型本轮 batch forward 的真实测量耗时，单位秒。

- `pipeline_id`
  - 异步模式下处理该 draft packet 的流水线 ID。

- `pipeline_start_s`
  - 异步模式下该 packet 开始验证的虚拟时间点。

- `pipeline_finish_s`
  - 异步模式下该 packet 完成验证的虚拟时间点。

- `pipeline_queue_wait_s`
  - 异步模式下 packet 到达后等待流水线可用的时间。
  - 公式：`pipeline_start_s - arrival_s`。

### `devices` 内单设备字段

- `device_id`
  - 设备 ID。

- `request_id`
  - 请求 ID。

- `draft_model`
  - 端侧 draft 模型。

- `draft_time_s`
  - 端侧 draft 模型生成本轮 draft token 的真实测量耗时。

- `draft_start_s`
  - 本轮端侧开始生成 draft 的虚拟时间点。

- `draft_end_s`
  - 本轮端侧完成 draft 生成的虚拟时间点。

- 关系
  - `draft_time_s = draft_end_s - draft_start_s`。

- `uplink_s`
  - 上行传输延迟，单位秒。
  - 公式近似：`effective_rtt_ms / 2000 + payload_bytes * 8 / (effective_mbps * 1e6) + jitter_s`，并截断为非负。

- `uplink_effective_mbps`
  - 本次上行采样后的有效带宽。

- `uplink_effective_rtt_ms`
  - 本次上行采样后的有效 RTT。

- `uplink_jitter_s`
  - 本次上行 delay 抖动项。

- `uplink_congested`
  - 本次上行是否发生拥塞。

- `arrival_s`
  - draft 包到达边缘服务器的虚拟时间点。
  - 公式：`draft_end_s + uplink_s`。

- `barrier_wait_s`
  - 到达后等待本轮最慢设备的时间。
  - 公式：`barrier_time_s - arrival_s`。
  - 异步模式下固定为 0。

- `downlink_s`
  - 下行传输延迟，单位秒。

- `downlink_effective_mbps`
  - 本次下行采样后的有效带宽。

- `downlink_effective_rtt_ms`
  - 本次下行采样后的有效 RTT。

- `downlink_jitter_s`
  - 本次下行 delay 抖动项。

- `downlink_congested`
  - 本次下行是否发生拥塞。

- `uplink_payload_bytes`
  - 上行 payload 估算字节数。
  - 公式：`len(prefix_ids) * 4 + len(draft_ids) * 4 + sum(draft_dist.payload_bytes()) + 128`。

- `downlink_payload_bytes`
  - 下行 payload 估算字节数。
  - 公式：`len(emitted_ids) * 4 + 64`。

- `accepted_count`
  - 本轮被接受的 draft token 数。

- `proposed_count`
  - 本轮提出的 draft token 数。

- `emitted_count`
  - 本轮最终发给端侧的 token 数。
  - 如果发生拒绝，通常是“已接受 token + 替代 token”。
  - 如果本轮所有 draft token 都接受，则通常是 `proposed_count + 1`，额外的 1 是 bonus token。

## 5. 当前 `summary.json` 的主要结论

- 总请求数取决于当前 `dataset_selection.request_count`；正式脚本默认 `DATASET_MODE=all`，会运行全部选中样本。
- 总输出 token 数是 6986，总虚拟时间约 212.16 秒，全局吞吐约 32.93 token/s。
- 平均 draft 接受率约 0.631，说明 draft 与 target 有一定一致性，但并不稳定到可以抵消全部同步和网络开销。
- 平均 barrier 等待约 0.026 秒，barrier 等待占端侧到达阶段约 18.45%。
- `device-0` 成为最慢设备的次数通常由真实 draft 生成耗时以及当轮动态采样到的网络状态共同决定；具体次数需要结合当前重新生成的 `summary.json` 查看。
- `mean_speedup_vs_target_only` 需要结合当前重新生成的 `summary.json` 查看；该值现在使用端到端 target-only baseline。
- task 维度里，`translation`、`extraction` 的延迟和总 token 都低；`rag`、`summarization`、`writing` 等任务的端到端延迟更高，主要和输出长度、接受率、同步轮数以及设备等待有关。
