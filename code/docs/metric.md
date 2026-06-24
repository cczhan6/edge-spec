# 指标定义

本文档定义仿真输出 CSV 的字段含义和统计口径。实验流程、方法说明和推荐图表见 [experiment.md](experiment.md)；单次 run 的数值解读应写入 `experiment_analysis_<RUN_ID>.md`。

## 输出层级

每次运行默认写入 `outputs/runs/<RUN_ID>/`：

| 层级 | 路径 | 用途 |
|---|---|---|
| 汇总结果 | `summary/all_results.csv` | 汇总本次 run 中所有 scenario/method 的主结果。 |
| 类别汇总 | `summary/category_results.csv` | 汇总本次 run 中所有 scenario/method/category 的类别结果。 |
| 场景主表 | `raw/main_results_<scenario>.csv` | 单个场景下各方法的总体性能。 |
| 场景类别表 | `raw/category_results_<scenario>.csv` | 单个场景下按 SpecBench 顶层类别聚合的性能。 |
| 系统指标 | `raw/system_metrics_<scenario>.csv` | 资源利用率、排队、浪费 token、通信和等待时间。 |
| 设备指标 | `raw/device_metrics_<scenario>_<method>.csv` | 每台虚拟 client 的利用率、队列和 token 统计。 |
| 请求明细 | `raw/request_details_<scenario>_<method>.csv` | 每个请求的完成时延、类别、设备和 token 统计。 |
| Segment 明细 | `raw/segment_details_<scenario>_<method>.csv` | 每个 segment 的 gamma、acceptance、lane、状态和时延。 |
| 逐轮 trace | `raw/round_trace_<scenario>_<method>.csv` | 与 segment 明细相近，用于按调度轮次排查行为。 |
| 事件明细 | `raw/event_details_<scenario>_<method>.csv` | draft、verify、batch、target-only 和完成事件。 |

## 主结果

`main_results_<scenario>.csv` 和 `category_results_<scenario>.csv` 使用同一组核心字段。区别是前者按 scenario/method 聚合，后者额外按 SpecBench 顶层类别聚合。

| 字段 | 含义 |
|---|---|
| `avg_latency_ms` | 请求完成时延平均值。 |
| `p50_latency_ms` | 请求完成时延 P50。 |
| `p95_latency_ms` | 请求完成时延 P95。 |
| `p99_latency_ms` | 请求完成时延 P99。 |
| `avg_tpot_ms` | decode-only TPOT；当前按每个请求完成时延除以最终提交 token 数后取平均。 |
| `avg_tbt_ms` | decode-only TBT；steady-state decode 口径下与 `avg_tpot_ms` 使用同一统计。 |
| `goodput_tok_s` | 最终提交给用户的输出 token 数除以 makespan。 |
| `avg_acceptance_rate` | 真实 accepted draft tokens 除以总 proposed draft positions；线性段等于主路径 draft tokens，树形段等于候选树可验证路径深度。 |
| `avg_selected_gamma` | 调度器选择的平均 gamma。 |
| `latency_speedup_vs_autoregressive` | 相对 `target_only` 的时延加速比。 |
| `latency_ratio_vs_*` | 当前方法与指定基线的时延比值。 |
| `relative_latency_reduction_vs_*` | 相对指定基线的时延降低比例。 |
| `goodput_gain_vs_*` | 相对指定基线的 goodput 提升比例。 |

`goodput_tok_s` 只统计最终提交给用户的 token。被 rejected、stale、discarded 或 proactive miss 消耗的 draft token 不计入 goodput。

SpecBench 原始类别会归并为 6 个顶层类别：`MT`、`QA`、`Math`、`RAG`、`Sum`、`Trans`。请求明细中 `category` 为顶层类别，`raw_category` 保留原始类别。

## 系统指标

`system_metrics_<scenario>.csv` 用于解释主结果背后的资源瓶颈。

| 指标组 | 含义 |
|---|---|
| 利用率 | drafter 设备、verifier lane 和 target 服务资源的忙碌比例。 |
| 排队与等待 | drafter 本地队列、lane 队列、同步 batch barrier、SpecEdge proactive 等等待时间。 |
| 一致性状态 | `stale`、`discarded`、`absorbed` segment 数。 |
| 浪费与复用 | rollback 次数、wasted draft tokens、bonus reused tokens。 |
| SpecEdge 专有项 | proactive segment、proactive hit/waste、pipeline idle bubble 和 pipeline timing error。 |
| 时延分解 | draft、verify、target-only、上行通信和下行通信的解析时延汇总。 |
| 通信载荷 | 上下行 token-ID payload 字节数。 |

解释系统指标时应结合主表。单独的高利用率不一定表示性能好；它也可能表示资源已经成为瓶颈。

## 明细表

| 文件 | 主要用途 |
|---|---|
| `request_details_<scenario>_<method>.csv` | 排查单个请求的时延、arrival/decode-ready time、origin device、类别、committed token、rollback、max outstanding、max unconfirmed draft tokens 和 target-only 分项。 |
| `segment_details_<scenario>_<method>.csv` | 排查 segment 级 scheduled/verify gamma、真实 acceptance、bonus reuse、payload、lane、draft 预算、SpecEdge pipeline 字段和最终状态。 |
| `round_trace_<scenario>_<method>.csv` | 按调度轮次复现 segment 行为，适合定位动态 gamma、回滚和 stale segment。 |
| `event_details_<scenario>_<method>.csv` | 查看 draft、verify、batch、target-only、server_only direct verify 和 request completion 事件。 |
| `device_metrics_<scenario>_<method>.csv` | 比较虚拟 client 的固定 drafter、利用率、空闲时间、draft busy time、队列等待、请求数、生成 token、接受 token 和平均 gamma。 |

`SpecEdge` 的对比字段名为 `latency_ratio_vs_specedge` 和 `relative_latency_reduction_vs_specedge`。其 pipeline-aware scheduling 事件会写入 `event_details` 的 `pipeline_schedule`、`global_batch_verify.batch_type` 和 `pipeline_idle_bubble_ms` 字段。draft 预算相关字段包括 `tree_strategy`、`tree_budget_nodes`、`draft_compute_nodes`、`processed_candidate_count`、`retained_tree_nodes`、`target_verify_tree_nodes`、`proposed_count` 和 `tree_path_switched`；默认 `specexec_approx` 时，`processed_candidate_count` 表示 draft forward 处理的 candidate 数，`retained_tree_nodes` 表示 logprob/budget pruning 后保留的候选树节点数，`target_verify_tree_nodes` 表示 target verify latency 使用的计费节点数，`proposed_count` 表示 acceptance 统计使用的可验证路径深度；显式切到 `linear` 时树预算等于主路径 gamma，target verify 按单段固定 forward 计费。其中 `tree_path_switched` 表示 target greedy 命中了候选树中的非主路径分支。

`event_details` 不包含 prefill 分项。`server_only` 不产生 `global_batch_verify` 事件。
request 明细中的 `arrival_time_ms` 与 `decode_ready_time_ms` 相等，表示请求进入仿真器时
已经具备解码条件；server-only/target-only 的下行字段记录最终输出下载。
