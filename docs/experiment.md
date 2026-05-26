# 实验设置与方法流程
数据按照 SpecBench 六类任务口径划分: Summarization (Sum), Math Reasoning (Math), Multi-turn Dialogue (MT), QA, Retrieval-Augmented Generation (RAG), Machine Translation (Trans)。

本文档集中描述实验设置、方法流程、主要结论和摘要, 不包含结果文件的字段逐项解释。

## 1. 当前方法流程

当前项目支持两种运行模式:

- --mode sync: 同步 barrier baseline。
- --mode async: 异步多验证流水线方案。

### 实验仿真设置

实验模拟"三台端侧设备 + 一台边缘服务器"的边缘推理场景。三台设备分别绑定 0.5B、1.5B、3B draft 模型, 边缘侧使用 7B target/verifier 模型, 因此异构性主要来自端侧模型规模、真实 draft 生成耗时和 draft/target 接受率差异。动态网络不是固定延迟, 而是每次上行、下行都基于 profile 独立采样带宽抖动、RTT 抖动、传输 jitter 和拥塞事件; 默认三台设备使用相同网络基准参数, 用随机动态采样刻画边缘网络波动。同步模式用 barrier 暴露慢设备和网络抖动造成的等待, 异步模式则让 draft packet 到达后直接进入边缘侧多条验证流水线, 用队列等待和流水线利用率刻画非阻塞调度效果。

同步 baseline 是"三端异构设备 + 边缘 7B 服务器"的同步 speculative decoding 仿真/测量流程。

1. 数据选择
   - 入口: python -m edge_spec.run。
   - 默认数据: data/spec_bench/question.jsonl。
   - 原始 question.jsonl 中的 13 个 category 会在加载时归并为 6 个 SpecBench 任务: Sum、Math、MT、QA、RAG、Trans。
   - MT 由原始 MT-Bench 子类 writing、roleplay、reasoning、math、coding、extraction、stem、humanities 归并得到。
   - 正式脚本默认命令为 bash scripts/run_qwen25_specbench.sh 和 bash scripts/run_qwen25_specbench_async.sh。
   - 选择类别时在命令前加 CATEGORY=Sum/Math/MT/QA/RAG/Trans; 不指定 CATEGORY 时默认跑 Sum。
   - 正式脚本通过 DATASET_MODE 切换数据集使用方式。
   - DATASET_MODE=one-per-category: 当前类别取 1 条样本; 直接调用 CLI 且不传 --category 时为每个六类任务取 1 条样本, 共 6 条。
   - DATASET_MODE=all: 运行当前类别下所有选中样本, 不丢弃任何样本; 最后一个 microbatch 可以少于 3 条。

2. 模型分工
   - device-0 使用 Qwen/Qwen2.5-0.5B-Instruct 作为 draft 模型。
   - device-1 使用 Qwen/Qwen2.5-1.5B-Instruct 作为 draft 模型。
   - device-2 使用 Qwen/Qwen2.5-3B-Instruct 作为 draft 模型。
   - 边缘服务器使用 Qwen/Qwen2.5-7B-Instruct 作为 target/verifier 模型。

3. 每轮同步推测解码
   - 每个未完成的端侧请求从当前 prefix 开始生成最多 gamma 个 draft token。
   - 端侧 draft 耗时直接按运行记录计算: draft_start_s = client.available_at_s, draft_end_s = draft_start_s + measured_draft_time_s, draft_time_s = draft_end_s - draft_start_s。
   - 每个端侧把 prefix、draft token、draft 分布上传到边缘服务器, 上行延迟由 payload、带宽、RTT、jitter、拥塞概率共同仿真。
   - draft 包到达时间为 arrival_s = draft_end_s + uplink_s。
   - 边缘服务器等待本轮所有活跃设备的 draft 包到达, 形成同步 barrier: barrier_time_s = max(arrival_s)。
   - 边缘服务器对本轮 batch 做 target forward, 拿到每个 draft 位置以及 bonus token 位置的 target 分布。
   - 对每条请求执行 Leviathan-style exact speculative sampling:
     - draft token 以 min(1, p_target(token) / p_draft(token)) 的概率被接受。
     - 一旦拒绝, 用 max(p_target - p_draft, 0) 的残差分布采样替代 token。
     - 如果 gamma 个 draft token 全部接受, 则额外从 target 的 bonus 分布采样 1 个 token。
   - 边缘服务器把本轮 emitted token 下发给端侧, 下行延迟同样按网络 profile 仿真。
   - 请求生成到 EOS 或达到 max_new_tokens 后结束。

4. microbatch 与全局时间
   - 同一个 microbatch 内的 3 个请求同步推进。
   - 下一个 microbatch 会在上一个 microbatch 全部请求完成后开始。
   - total_virtual_time_s 是所有 microbatch 串行执行后的虚拟完成时间。

5. target-only baseline
   - 若未使用 --skip-target-baseline, 每条请求还会跑一次公平的端到端 target-only baseline。
   - target_only_latency_s = target_only_uplink_s + target_only_model_latency_s + target_only_downlink_s。
   - speedup_vs_target_only = target_only_latency_s / speculative_latency_s。
   - 该值大于 1 表示 speculative 流程更快; 小于 1 表示当前 speculative 流程比 target-only 慢。

## 1.1 异步多流水方案

异步模式由 HeteroAsyncPipelineRunner 实现, 入口参数是 --mode async --pipeline-count N。

1. 多流水划分
   - 边缘服务器被抽象为 N 条相互独立的验证流水线。
   - 每条流水线维护自己的 available_at_s、累计忙碌时间和验证次数。
   - 流水线内部顺序处理草稿片段; 不同流水线之间可并行推进。

2. 端侧异步起草
   - 每个设备维护自己的请求队列。
   - 同一设备上的请求串行推进: 当前请求完成后, 该设备才开始下一个请求。
   - 不同设备之间互不等待。
   - 每个请求每轮从当前 prefix 起草最多 gamma 个 token, 生成 draft packet 后经上行网络到达边缘侧。

3. 流水线调度
   - draft packet 到达后不会等待其他设备形成 batch。
   - 调度器选择"能最早开始处理该 packet"的流水线。
   - 选择规则: 最小化 max(pipeline.available_at_s, packet.arrival_s); 若并列, 再按流水线当前可用时间和 pipeline_id 打破平局。
   - packet 在流水线上的排队时间为 pipeline_queue_wait_s = pipeline_start_s - arrival_s。

4. 非阻塞验证与局部回退
   - 每次异步验证的 target_batch_size 为 1, 即单个草稿片段进入某条流水线。
   - 验证仍使用 exact speculative sampling。
   - 如果发生拒绝, 只影响该请求本轮输出: 用 target/draft 残差分布采样替代 token, 并从新的 prefix 继续下一轮起草。
   - 其他设备和其他流水线不回退、不等待。

5. 异步时间语义
   - total_virtual_time_s 是所有设备请求队列全部完成的最大完成时间。
   - 异步模式没有全局 barrier, 因此 barrier_wait_s 固定为 0, slowest_device_rounds 不适用。
   - 异步模式新增流水线队列等待、流水线忙碌时间和利用率指标。

## 2. 主要结论与摘要

- 总请求数取决于当前 dataset_selection.request_count; 正式脚本默认 CATEGORY=Sum DATASET_MODE=all, 会运行单个选中类别下的全部样本。
- 总输出 token 数、总虚拟时间与全局吞吐应以当前 summary.json 为准。
- 平均 draft 接受率决定同步轮数与 speculative 效果, 需要结合当前 summary.json 查看。
- 同步模式的 barrier 等待会放大慢设备与网络抖动的影响; 异步模式则以流水线排队等待和利用率刻画非阻塞效果。
- 任务维度中, 输出长度更长的任务通常具有更高端到端延迟, 需要结合具体任务的 token 数和接受率综合判断。
