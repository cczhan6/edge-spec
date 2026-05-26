# 指标字段说明


本文档只包含结果文件中的指标字段解释与计算方式。

## 1. 汇总指标: summary.json

### 全局规模与吞吐

- request_count
  - 总请求数。
  - 公式: len(records)。

- round_count
  - sync 模式: 总同步轮次数。
  - async 模式: 总验证事件数 (进入流水线的 draft packet 数)。
  - 公式: len(round_trace)。

- total_generated_tokens
  - 所有请求实际输出的 token 数总和。
  - 公式: sum(generated_token_count)。

- total_virtual_time_s
  - 整次实验的虚拟总耗时, 单位秒。
  - sync 模式: 所有 microbatch 串行执行后的完成时间。
  - async 模式: 所有设备请求队列全部完成时的最大完成时间。

- throughput_tokens_per_s
  - 全局吞吐, 单位 token/s。
  - 公式: total_generated_tokens / total_virtual_time_s。
  - 注意: 这是端到端虚拟吞吐, 不是 GPU 单次 forward 的裸吞吐。

### speculative decoding 质量

- mean_acceptance_rate
  - 请求级 acceptance rate 的平均值。
  - 单请求公式: accepted_draft_tokens / proposed_draft_tokens。
  - 汇总公式: mean(record.acceptance_rate)。
  - 含义: draft token 被 target 接受的比例。

### barrier 与同步开销

- mean_barrier_wait_s
  - 所有轮次、所有设备的 barrier 等待时间均值, 单位秒。
  - 单设备单轮公式: barrier_wait_s = barrier_time_s - arrival_s。

- barrier_wait_fraction
  - barrier 等待在"端侧 draft 生成 + 上行 + barrier 等待"中的占比。
  - 公式: sum(barrier_wait_s) / (sum(draft_time_s + uplink_s) + sum(barrier_wait_s))。
  - 注意: 该比例不包含 target forward 和 downlink 时间。

- slowest_device_rounds
  - 每个设备成为本轮最慢到达设备的次数。
  - 实现上取 barrier_wait_s 最小的设备, 因为最慢设备的等待时间为 0。
  - async 模式没有全局 barrier, 该字段为空对象。

### 异步流水线指标

以下字段只在 --mode async 下有明确含义。

- mode
  - 运行模式, 异步方案为 "async"。

- pipeline_count
  - 边缘侧验证流水线数量。

- verification_event_count
  - 异步验证事件数。
  - 公式: len(round_trace)。

- pipeline_verification_count
  - 所有流水线处理过的 draft packet 总数。

- mean_pipeline_queue_wait_s
  - draft packet 到达边缘后, 在流水线前排队的平均时间。
  - 公式: mean(pipeline_queue_wait_s)。

- pipeline_queue_wait_fraction
  - 流水线排队等待在"排队等待 + target forward"中的占比。
  - 公式: sum(pipeline_queue_wait_s) / (sum(pipeline_queue_wait_s) + sum(target_forward_s))。

- pipeline_busy_s
  - 每条流水线累计执行 target verification 的时间。

- pipeline_verifications
  - 每条流水线处理的 draft packet 数。

- pipeline_utilization
  - 每条流水线利用率。
  - 公式: pipeline_busy_s[pipeline] / total_virtual_time_s。

- mean_pipeline_utilization
  - 所有流水线平均利用率。
  - 公式: sum(pipeline_busy_s) / (pipeline_count * total_virtual_time_s)。

### 网络指标

- mean_uplink_effective_mbps
  - 所有上行传输采样后的有效带宽均值, 单位 Mbps。

- mean_downlink_effective_mbps
  - 所有下行传输采样后的有效带宽均值, 单位 Mbps。

- mean_uplink_effective_rtt_ms
  - 所有上行传输采样后的有效 RTT 均值, 单位毫秒。

- mean_downlink_effective_rtt_ms
  - 所有下行传输采样后的有效 RTT 均值, 单位毫秒。

- network_congestion_events
  - 上行和下行发生拥塞的总次数。

- network_congestion_fraction
  - 网络样本中的拥塞比例。
  - 公式: network_congestion_events / network_samples, 其中 network_samples = 2 * 设备轮次样本数。

### target-only 对比

- mean_target_only_latency_s
  - target-only 端到端 baseline 的请求级平均延迟, 单位秒。
  - 单请求公式: target_only_uplink_s + target_only_model_latency_s + target_only_downlink_s。
  - 仅在启用 target baseline 时存在; 否则为 null。

- mean_speedup_vs_target_only
  - 请求级 speedup 的平均值。
  - 单请求公式: target_only_latency_s / latency_s。
  - 解释: > 1 表示 speculative 更快, < 1 表示更慢。

### 设备级延迟

- mean_latency_by_device
  - 每个设备上请求的端到端平均延迟, 单位秒。

### 任务级指标: task_metrics

每个 SpecBench 六类任务下都有以下字段。任务标签为 Sum、Math、MT、QA、RAG、Trans。

- request_count
  - 该任务的请求数。

- microbatch_count
  - 该任务覆盖的 microbatch 数。

- generated_token_count
  - 该任务的输出 token 总数。

- effective_received_token_count
  - 端侧实际收到的 token 总数。

- effective_duration_s
  - 该任务的有效完成时长, 单位秒。

- effective_throughput_tokens_per_s
  - 该任务有效吞吐, 单位 token/s。
  - 公式: effective_received_token_count / effective_duration_s。

- effective_received_throughput_tokens_per_s
  - 与 effective_throughput_tokens_per_s 相同。

- e2e_first_token_latency_s
  - 该任务请求的平均首 token 延迟, 单位秒。

- e2e_mean_latency_s
  - 该任务请求的平均端到端延迟, 单位秒。

### profile 与数据选择

- profiles
  - 每个设备的仿真 profile。
  - uplink_mbps / downlink_mbps: 基准上下行带宽。
  - rtt_ms: 基准 RTT。
  - jitter_ms: 最终传输 delay 上的一阶随机抖动。
  - bandwidth_jitter_ratio: 带宽随机波动比例。
  - rtt_jitter_ms: RTT 随机波动范围。
  - congestion_probability: 单次传输发生拥塞的概率。
  - congestion_slowdown: 拥塞时带宽除以该值, RTT 乘以该值。

- dataset_selection
  - 本次实验的数据选择方式。
  - dataset_path: 数据路径。
  - category: 本次筛选的 SpecBench 六类任务标签; 未筛选时为 null。
  - one_per_category: 是否每个六类任务取 1 条。
  - one_per_category_per_device: 是否每个六类任务每设备取 1 条。
  - dataset_mode: 数据集使用模式。
  - all: 是否运行所有选中样本且不丢弃尾部样本。
  - selected_request_count: 筛选后的原始请求数。
  - request_count: 实际进入 microbatch 运行的请求数。
  - dropped_request_count: 未进入运行的请求数。
  - microbatch_count: microbatch 数。
  - categories: 选中的六类任务标签。

## 2. 请求级指标: specbench_sync_hetero.jsonl

每一行是一条请求的记录。

- microbatch_id
  - 请求所属 microbatch。

- device_id
  - 请求分配到的端侧设备。

- draft_model
  - 端侧 draft 模型名称。

- target_model
  - 边缘侧 target/verifier 模型名称。

- task
  - SpecBench 六类任务标签: Sum、Math、MT、QA、RAG、Trans。

- prompt_id
  - 数据集里的请求 ID。

- prompt
  - 输入 prompt。

- generated_text
  - speculative 流程生成的文本。

- generated_token_count
  - speculative 流程输出 token 数。

- effective_received_token_count
  - 端侧实际收到的 token 数。

- acceptance_rate
  - draft token 接受率。
  - 公式: accepted_draft_tokens / proposed_draft_tokens。

- accepted_draft_tokens
  - 被 target 接受的 draft token 数。

- proposed_draft_tokens
  - draft 模型提出的 token 数。

- sync_rounds
  - sync 模式: 该请求参与的同步轮数。
  - async 模式: 该请求完成的验证轮数。

- execution_mode
  - 请求来自同步 baseline 还是异步方案, 取值为 sync 或 async。

- async_rounds
  - 异步模式下该请求的验证轮数。

- pipeline_ids
  - 异步模式下该请求每轮使用的流水线 ID 序列。

- pipeline_switches
  - 异步模式下该请求相邻两轮切换流水线的次数。

- first_token_latency_s
  - 首 token 端到端延迟, 单位秒。

- latency_s
  - 请求端到端延迟, 单位秒。

- tokens_per_s
  - 请求级吞吐。
  - 公式: generated_token_count / latency_s。

- effective_received_tokens_per_s
  - 请求级端侧接收吞吐。

- target_only_latency_s
  - target-only 端到端 baseline 延迟, 单位秒。

- target_only_model_latency_s
  - target-only baseline 中 target 模型生成耗时, 单位秒。

- target_only_uplink_s
  - target-only baseline 中客户端上传的网络延迟, 单位秒。

- target_only_downlink_s
  - target-only baseline 中边缘侧下发的网络延迟, 单位秒。

- target_only_uplink_payload_bytes
  - target-only baseline 上行 payload 估算字节数。

- target_only_downlink_payload_bytes
  - target-only baseline 下行 payload 估算字节数。

- target_only_text
  - target-only baseline 生成文本。

- speedup_vs_target_only
  - 相对 target-only 的加速比。
  - 公式: target_only_latency_s / latency_s。

## 3. 轮次级指标: round_trace.jsonl

每一行是一轮同步 trace。

### 轮次字段

- microbatch_id
  - 所属 microbatch。

- round_index
  - microbatch 内的轮次编号, 从 0 开始。

- target_batch_size
  - sync 模式: 本轮参与 target verification 的活跃请求数。
  - async 模式: 固定为 1。

- barrier_time_s
  - 本轮所有 draft 包到齐的虚拟时间点, 单位秒。

- target_forward_s
  - target 模型本轮 batch forward 的真实测量耗时, 单位秒。

- pipeline_id
  - 异步模式下处理该 draft packet 的流水线 ID。

- pipeline_start_s
  - 异步模式下该 packet 开始验证的虚拟时间点。

- pipeline_finish_s
  - 异步模式下该 packet 完成验证的虚拟时间点。

- pipeline_queue_wait_s
  - 异步模式下 packet 到达后等待流水线可用的时间。
  - 公式: pipeline_start_s - arrival_s。

### devices 内单设备字段

- device_id
  - 设备 ID。

- request_id
  - 请求 ID。

- draft_model
  - 端侧 draft 模型。

- draft_time_s
  - 端侧 draft 模型生成本轮 draft token 的真实测量耗时。

- draft_start_s
  - 本轮端侧开始生成 draft 的虚拟时间点。

- draft_end_s
  - 本轮端侧完成 draft 生成的虚拟时间点。

- uplink_s
  - 上行传输延迟, 单位秒。

- uplink_effective_mbps
  - 本次上行采样后的有效带宽。

- uplink_effective_rtt_ms
  - 本次上行采样后的有效 RTT。

- uplink_jitter_s
  - 本次上行 delay 抖动项。

- uplink_congested
  - 本次上行是否发生拥塞。

- arrival_s
  - draft 包到达边缘服务器的虚拟时间点。
  - 公式: draft_end_s + uplink_s。

- barrier_wait_s
  - 到达后等待本轮最慢设备的时间。
  - 公式: barrier_time_s - arrival_s。

- downlink_s
  - 下行传输延迟, 单位秒。

- downlink_effective_mbps
  - 本次下行采样后的有效带宽。

- downlink_effective_rtt_ms
  - 本次下行采样后的有效 RTT。

- downlink_jitter_s
  - 本次下行 delay 抖动项。

- downlink_congested
  - 本次下行是否发生拥塞。

- uplink_payload_bytes
  - 上行 payload 估算字节数。

- downlink_payload_bytes
  - 下行 payload 估算字节数。

- accepted_count
  - 本轮被接受的 draft token 数。

- proposed_count
  - 本轮提出的 draft token 数。

- emitted_count
  - 本轮最终发给端侧的 token 数。
