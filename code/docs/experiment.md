# 实验设计与运行说明

本文档是稳定实验协议，说明如何用当前仿真器组织实验、比较方法、读取指标，并给出推荐的性能分析、消融实验和敏感性分析设置。单次 run 的数值解读不放在本文档中，应写入 `experiment_analysis_<RUN_ID>.md` 或 [latest_experiment_analysis.md](latest_experiment_analysis.md)。

## 阅读方式

| 目标 | 阅读位置 |
|---|---|
| 确认实验是否符合论文口径 | 第 1-3 节 |
| 查询主表、系统表和明细表含义 | 第 4 节，或直接阅读 [metric.md](metric.md) |
| 运行主性能对比 | 第 5 节 |
| 运行组件消融 | 第 6 节 |
| 运行 lane、窗口和手动 sweep | 第 7 节 |
| 检查正式实验是否遗漏步骤 | 第 9 节 |

## 1. 实验目标

实验目标是评估异构端侧 drafter + 边缘 target 的推测解码系统，在真实 greedy
语义和解析时延模型下的端到端收益。

核心口径：

- 真实 drafter 和 target 只负责生成 token、greedy acceptance、correction 和 bonus；
- 模型 forward 的宿主机 wall time 不进入虚拟时间；
- draft、verify、target-only 和通信均使用解析公式推进事件时间；
- 请求固定绑定到 origin device，不迁移；
- 每台设备固定部署一个 drafter，并以 segment 级 FIFO 串行 draft；
- 所有方法使用同一数据集、seed、输出长度分布和虚拟网络配置。

`smoke` 和单元测试可使用 fake 模型推理器。正式实验应关闭 `USE_FAKE_MODEL_RUNNER`，
加载真实 Hugging Face 模型。

## 2. 实验设置

### 数据集

默认数据集为 `data/spec_bench/question.jsonl`，共 480 条。仿真读取每条样本的第一轮
prompt，并按 `simulation.seed` 无放回抽样。

`simulation.num_requests` 表示所有设备合计请求数。默认值为 200，默认输出长度从
`[64, 128, 256]` 中随机选择。

SpecBench 原始 13 个 `category` 会归并为 6 个顶层类别：

| 输出标签 | 原始类别 |
|---|---|
| `MT` | `writing`、`coding`、`math`、`reasoning`、`roleplay`、`extraction`、`stem`、`humanities` |
| `QA` | `qa` |
| `Math` | `math_reasoning` |
| `RAG` | `rag` |
| `Sum` | `summarization` |
| `Trans` | `translation` |

`request_details` 中 `category` 为顶层标签，`raw_category` 保留原始类别。

初步验证建议使用每类 10 条、总共 60 条：

```bash
SAMPLES_PER_CATEGORY=10 bash scripts/run.sh all
```

`scripts/run.sh` 会为每次运行创建独立目录 `outputs/runs/<RUN_ID>/`，`RUN_ID`
默认为实验开始时间 `YYYYMMDD-HHMMSS`。每个 run 都会保存 `manifest.yaml`，记录命令、
配置、数据集、场景、方法、输出路径和 git 状态。若显式指定的 `RUN_ID` 或 `RUN_DIR`
已存在，脚本会退出，避免覆盖旧实验。

`SAMPLES_PER_CATEGORY` 启用后按 6 个顶层类别均衡抽样，并忽略
`simulation.num_requests` 的全局随机抽样。抽样完成后会打乱请求顺序，避免同类请求
连续分配到设备轮询队列。若只检查流程，可再加 `USE_FAKE_MODEL_RUNNER=1`；若要验证真实语义
趋势，应加载真实模型运行。

### 模型与模型推理器

默认真实模型配置在 `configs/default.yaml`：

| 角色 | 模型 |
|---|---|
| small drafter | `Qwen/Qwen2.5-0.5B-Instruct` |
| medium drafter | `Qwen/Qwen2.5-1.5B-Instruct` |
| large drafter | `Qwen/Qwen2.5-3B-Instruct` |
| target | `Qwen/Qwen2.5-7B-Instruct` |

真实 drafter 和 target 必须共享 tokenizer/vocab；不兼容时启动失败。drafter 和
target-only 自回归 forward 使用 KV cache，verify 使用一次 target forward，虚拟时延按
固定 verification forward 近似计费。

### 虚拟设备与边缘

默认 8 台设备，异构池为 small/mid/large 数量 `3/3/2`。每个设备模板固定：
`drafter_profile`、`draft_token_rate_tok_s`、`draft_startup_ms`、上下行带宽、RTT 和 jitter。
`low_end`、`mid_end` 和 `high_end` 均表示请求的 origin client 设备，只负责本地
drafter；它们不表示边缘服务器。边缘服务器由 `edge` 单独建模，负责 target verify
和 target-only 生成。

三档 client 设备按设备形态和部署 drafter 划分：

| 设备层级 | 典型设备 | 部署 drafter | 强异构场景解析参数 |
|---|---|---|---|
| `low_end` | 普通手机、旧款 Android、Raspberry Pi + NPU、低功耗 IoT/小板卡 | small drafter，如 0.5B 量化模型 | `25 tok/s`，上/下行 `5/30 Mbps`，RTT `90 ms`，jitter `25 ms` |
| `mid_end` | 旗舰手机、平板、Apple Silicon 入门笔记本、Jetson Orin Nano、轻薄本 GPU | medium drafter，如 1.5B 量化模型 | `60 tok/s`，上/下行 `25/100 Mbps`，RTT `40 ms`，jitter `10 ms` |
| `high_end` | 用户侧 RTX 工作站、高性能笔记本、Jetson AGX、企业内网工作站 | large drafter，如 3B 量化模型 | `100 tok/s`，上/下行 `100/300 Mbps`，RTT `10 ms`，jitter `2 ms` |

这些 token rate 是虚拟时间中的 sustained decode 解析速率，不是 Hugging Face forward
的宿主机 wall time。取值依据如下：

- Tummalapalli et al. 对 Qwen2.5-1.5B 4-bit 的端侧测量显示，RTX 4050 laptop
  约 `131.7 tok/s`，iPhone 16 Pro sustained hot 状态约 `22.6 tok/s`，
  Samsung S24 Ultra 约 `9.9 tok/s`，RPi 5 + Hailo-10H 约 `6.9 tok/s`。
- Transformer-Lite 在 Snapdragon 8 Gen 3 上报告 Gemma 2B decode 约 `30 tok/s`、
  ChatGLM2 6B decode 约 `14 tok/s`，并显示移动端 decode 速度随模型变大下降。
- 移动网络测量显示 100 Mb/s 吞吐并不罕见，但约 60% 移动用户低于 50 Mb/s，
  只有 top 5% 用户能经常获得低于 20 ms 的最小延迟；商业 5G/4G 的吞吐和 RTT
  还会受频段、位置、遮挡、设备发热和边缘服务位置影响。

参考文献：

- Pranay Tummalapalli et al. 2026. [LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load](https://arxiv.org/abs/2603.23640).
- Luchang Li et al. 2024. [Transformer-Lite: High-efficiency Deployment of Large Language Models on Mobile Phone GPUs](https://arxiv.org/abs/2403.20041).
- ASM Rizvi et al. 2025. [Third-Party Assessment of Mobile Performance in the 5G Era](https://arxiv.org/abs/2507.18834).
- Arvind Narayanan et al. 2020. [A First Look at Commercial 5G Performance on Smartphones](https://arxiv.org/abs/1909.07532).
- Peixuan Song et al. 2023. [A case study on latency, bandwidth and energy efficiency of mobile 5G and YouTube Edge service in London](https://arxiv.org/abs/2310.14090).

边缘默认 4 条 verifier lane，供 `full` 和异步消融方法进行并行 verify；
`target_only` 不使用这些 lane，而是使用单个边缘 target 服务资源：

```text
verify_startup_ms = 8
target_only_token_rate_tok_s = 80
```

边缘服务器按 A100/RTX 4090 级单卡、7B 级 target 模型的低延迟 serving 口径建模。
`target_only_token_rate_tok_s = 80` 表示逐 token 自回归 decode 的解析速率，也用于
验证阶段的固定 forward 近似。一个 segment 的 target verification 视为一次目标模型
自回归 decode step 的等效时间，忽略 `gamma` 对该 forward 时延的二阶影响。并行
verify 不是免费操作，但一次 target forward 可以同时覆盖多个候选 token，因此其时间
应明显短于同等 token 数的逐 token 自回归 decode。

按当前公式，在单请求单段验证时：

```text
gamma = 4: verify_ms = 8 + 1000 / 80 = 20.5 ms
           target-only 生成 5 tokens = 1000 * 5 / 80 = 62.5 ms
           verify / target-only ~= 33%

gamma = 8: verify_ms = 8 + 1000 / 80 = 20.5 ms
           target-only 生成 9 tokens = 1000 * 9 / 80 = 112.5 ms
           verify / target-only ~= 18%
```

因此，对常用 `gamma=4-8`，并行验证大约占同等 token 数 target-only 自回归解码时间的
`18%-33%`。这个比例的分母是“同等 token 数的 target-only 解码时间”，且来自固定
verification forward 约等于一次 target 自回归 decode step 的近似。若换成
speculative decoding 自身的运行时间分解，verification 仍可能占主要部分；
SpecDecode-Bench 对 vLLM 生产级实现的 profiling 报告显示，target verification
在若干 speculative decoding 方法中约占 `42%-95%` 的执行时间。

边缘参数参考：

- BALI benchmark 报告 A100 上 7B 级模型 batch-1 serving 的 token rate 约为几十到
  百 token/s 量级；相关实验中 Mistral-7B 在 A100 上约为 `70 tok/s` 量级。
- Skeleton-of-Thought 的 profiling 显示，Vicuna-7B 在 A100 上逐 token decode 的
  单步开销约为数十毫秒量级，并指出 decode 阶段受内存带宽和顺序 token 依赖限制。
- Leviathan et al. 的 speculative decoding 公式化了“一次 target 并行验证多个 draft
  token”的机制；这正是把 verify 近似为固定 target verification forward，而不是
  同等 token 数自回归 decode 的依据。
- SpecDecode-Bench 指出，尽管并行验证相对逐 token 自回归更短，verification 在整体
  speculative decoding runtime 中仍经常是 dominant cost。

参考文献：

- Yaniv Leviathan, Matan Kalman, Yossi Matias. 2023. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192).
- Ning Zhang et al. 2023. [Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding](https://openreview.net/forum?id=mqVgBbNCm9).
- Vikas Kumar et al. 2024. [BALI - A Benchmark for Accelerated Language Model Inference](https://doaj.org/article/1cae2238947b431cafffd9ed72168330).
- Maozheng Wang et al. 2026. [SpecDecode-Bench: The Benchmark for Real-World Speculative Decoding](https://specdecode-bench.github.io/).

解析时延：

```text
draft_ms       = draft_startup_ms + 1000 * gamma / draft_token_rate_tok_s
verify_ms      = verify_startup_ms + 1000 * B / target_only_token_rate_tok_s
target_only_ms = target_only_startup_ms + 1000 * output_tokens / target_only_token_rate_tok_s
```

通信使用 token-ID payload：

```text
payload_bytes = packet_header_bytes + token_count * packet_token_bytes
delay_ms = RTT_ms / 2 + payload_bytes * 8 / (bandwidth_mbps * 1000) + deterministic_jitter_ms
```

### 场景

`scripts/run.sh all` 默认运行以下场景：

| 场景 | 目的 |
|---|---|
| `homogeneous` | 8 台同构虚拟 client，全部使用 `medium` drafter，并沿用 `combined_strong_heterogeneous` 中 `mid_end` 的 draft 速率、网络参数和 Poisson 到达；用于观察强负载口径下没有草稿设备异构时 `full` 与 SpecEdge-style baseline 的差异 |
| `balanced_drafter` | 8 台设备，网络相同，主要观察 drafter 质量/速度异构 |
| `network_heterogeneous` | 8 台设备，强化网络 RTT/带宽差异，观察通信瓶颈 |
| `combined_strong_heterogeneous` | 8 台设备，Poisson 到达，设备与网络都强异构，观察动态负载稳定性 |

场景覆盖文件位于 `configs/<scenario>.yaml`，通过 `load_config(default, scenario)` 与
默认配置深度合并。

## 3. 方法流程

### Target-only

`target_only` 将请求上传到边缘，单个 target 服务资源自回归生成全部输出 token，再返回设备。
语义由 target 真实 greedy decoding 决定，时延由 `target_only_ms` 和通信公式决定。

### 推测解码公共流程

1. 请求按 `request_id % num_devices` 映射到 origin device；
2. 设备从本地 FIFO 队列取一个 segment，基于真实前缀调用 drafter 生成 `gamma` 个 token；
3. segment 上传边缘，边缘 target 验证 draft token；
4. 若 target argmax 匹配 draft token，则 token 被接受；
5. 首个不匹配位置产生 correction；
6. 若 draft 全接受，则 target 产生一个 bonus token；
7. 设备收到结果后提交 emitted token，必要时重起草后续 segment。

在 `full` 及其异步消融方法中，同一请求可以维护多段乐观前缀。`full` 不使用固定
`W_default` 截断在飞 segment 数，而是按 DSI 原仓库逻辑持续向后起草，直到覆盖剩余
输出、请求完成或前缀被 rejection/bonus 重定位打断。后续位置的 segment 到达边缘后，
不必等待当前 `edge_frontier_pos` 验证完成；它可以提前进入空闲 verifier lane。
边缘会缓存这些 out-of-order verification result，并只在所有祖先 segment 都确认后，
按 frontier 顺序解析、下发和提交。因此 target 侧可以并行验证同一请求的不同
speculative position，但用户可见输出仍严格等价于 target greedy 前缀推进。

除 SpecEdge/server-only 树形基线外，动态 gamma 从
`speculation.gamma_candidates = [1, 2, 4, 6, 8]` 枚举，过滤超过剩余长度的候选值，并最大化：

```text
expected emitted tokens / (draft + predicted target queue + verify + communication)
```

Acceptance 估计使用每请求最近 `acceptance_window_rounds = 8` 轮真实观测；无历史时回退
到 drafter 的 `acceptance_prior`。Prior 只用于冷启动调度，不决定真实 acceptance。
SpecEdge 和 server-only 树形基线不使用这套 gamma 枚举；它们按 `kaist-ina/specedge`
原仓库思路使用配置的 `max_beam_len` 作为树生长上限，仅在剩余输出不足时截断。

### 方法分类

`full` 是主方法。它使用 heterogeneous 设备池、动态 gamma、least-finish verifier lane、
DSI-style 持续乐观起草、同请求多位置并行验证和 bonus 匹配重定位。

对比基线只包含 target-only 和已有/增强 SD baseline：

| 方法 | 基线含义 | 设备池与行为 |
|---|---|---|
| `target_only` | 边缘自回归基线 | 请求上传边缘后由单个 target 服务资源完整生成 |
| `sync_batch_sd` | 异构 drafter + 动态 token + 同步 batch SD 基线 | heterogeneous，动态 gamma，global batch |
| `SpecEdge` | SpecExec-style 树形近似基线 | 树形 draft、server batch validation、proactive continuation、pipeline-aware scheduling；可显式切到线性近似 |
| `server_only` | SpecEdge-style server-only 近似基线 | 请求级上传 prompt/下载最终输出；服务器侧树形 draft 后直接 target verify，不做 segment 级端边往返或 global batch；可显式切到线性近似 |

组件消融方法不是对比基线，只用于分析 `full` 中不同组件的贡献：

| 方法 | 消融含义 | 行为 |
|---|---|---|
| `wo_async` | 去掉持续乐观起草 | heterogeneous，`W=1`，least-finish lane |
| `wo_scheduling` | 去掉 heterogeneity-aware lane scheduling | heterogeneous，动态 gamma，round-robin lane，DSI-style 持续起草 |
| `conservative_rollback` | 去掉精细 bonus 重定位/局部保留 | DSI-style 持续起草，但前缀变化后完整丢弃后续链 |

`full` 拒绝时，后续乐观链作废；全接受产生 bonus 时，如果 bonus 匹配下一
segment 首 token，则裁掉该 token 并重定位后续 segment；若下一 segment 已经提前
验证，则同步转换缓存的 verification result。未验证且被裁空的 segment 标记为
`absorbed`；若 bonus 不匹配，则后续链标记 `stale` 并重新起草。

SD baseline 的具体解释：

- `sync_batch_sd`：使用 heterogeneous drafter、动态 gamma，并采用全局同步 batch verify。
  每轮要等待 batch 条件或 timeout，再统一 verify。它用于衡量“异构 drafter + 动态 token”
  在同步 barrier 下的表现。
- `SpecEdge`：SpecExec-style 树形近似基线。主实验默认使用 `specexec_approx` draft 预算：
  本地 tree builder 按 cumulative logprob 选择 top `max_n_beams` candidate nodes，
  对每个 candidate 取 top `max_branch_width` children，加入 `log(0.9)` 父分数衰减，
  并按 `max_budget` 对 retained tree nodes 做 logprob pruning。proactive draft
  从当前树叶子中选 best bonus token candidate 后继续扩展。它仍是解析仿真中的
  SpecExec-style approximation，不等同于逐行复现上游运行时和 CUDA/KV-cache 行为。
  若需要延续旧固定段级近似，可显式切换到 `linear`；此时
  `tree_budget_nodes = draft_compute_nodes = gamma`，target verify 按单段 forward 计费。
  SpecEdge 同时实现 server batch validation、
  发送 validate 后立即启动的 proactive continuation，以及 pipeline-aware scheduling。树生长深度按
  `max_beam_len` 配置固定上限执行；运行时动态变化的是 beam candidate、budget pruning、
  proactive hit 后的树重排/续接，以及多请求 validation 的 interleave。server 侧按
  `kaist-ina/specedge` 的 `static`/`dynamic` batch loop 语义 interleave 多请求 verification：
  `static` 等满 `server_batch_size`，`dynamic` 收集当前已到达的 validate 请求后立即启动 batch。
- `server_only`：对应 SpecEdge-style server-only baseline 的近似版本。每个请求按
  `target_only` 相同口径上传 prompt、下载最终输出；服务器侧 draft/verify 共址，
  不产生 segment 级端边往返。它按原仓库默认 `max_batch_size=1` 语义只保留一个
  active 请求。该请求在服务器侧使用固定 1.5B 级 `medium` drafter 交替执行 draft
  和 target verify，服务器侧生成结束后
  才启动下一个排队请求；不进入 global batch，最终输出仍由 target greedy 语义决定。
  `server_only.draft_token_rate_tok_s` 单独控制该服务器侧 drafter 的解析速度，默认
  504 tok/s。该值固定自 ComputingForGeeks 的 RTX 4090 + Ollama 实测：`qwen2.5:1.5b`
  在五个 prompt 上的生成速率为 496.4、506.5、509.6、516.6、490.4 tok/s，均值约
  503.9 tok/s。`server_only` 默认使用 `specexec_approx` draft 预算，并使用上游 server-only 配置中的
  独立 `server_only.max_budget = 64`，避免隐式继承本地 `specedge.max_budget = 32`。
  若需要旧固定段级近似，可显式切换到 `server_only.tree_draft_strategy: linear`。
  语义路径深度默认为 `max_beam_len = 4`，仅在剩余输出不足时截断。

## 4. 主要指标

主表：

- `avg_latency_ms`、`p50_latency_ms`、`p95_latency_ms`、`p99_latency_ms`：请求完成时延；
- `avg_ttft_ms`：首 token 返回时延；
- `goodput_tok_s`：最终提交给用户的输出 token 数 / makespan，不包含 rejected、stale、
  discarded 或 proactive miss 产生的浪费 draft token；
- `avg_acceptance_rate`：真实接受 token / proposed draft positions；线性段等于主路径 draft token，树形段等于候选树可验证路径深度；
- `avg_selected_gamma`：平均调度 gamma；
- `latency_speedup_vs_autoregressive`：相对 `target_only` 的时延加速；
- `latency_ratio_vs_*` 和 `relative_latency_reduction_vs_*`：相对同步和 vanilla baseline；
- `goodput_gain_vs_*`：相对基线 goodput 提升。

系统表：

- 设备、lane 和 target 利用率；
- 设备队列、lane 队列、batch barrier、SpecEdge proactive 和等待时间；
- `stale`、`discarded`、`absorbed` segment 数；
- rollback 次数、wasted draft tokens、bonus reused tokens；
- draft、verify、target-only 和上下行通信的时延与 payload 汇总。

明细表：

- `request_details_<scenario>_<method>.csv`：请求级时延、类别、设备、token 和 target-only 分项；
- `segment_details_<scenario>_<method>.csv`：每个 segment 的 gamma、acceptance、lane、状态和通信；
- `device_metrics_<scenario>_<method>.csv`：每台设备的固定 drafter、请求数、利用率和 token 统计；
- `event_details_<scenario>_<method>.csv`：事件级 draft、verify、batch 和完成 trace。

类别结果：

- `category_results_<scenario>.csv`：每个 scenario/method/category 的主指标；
- `outputs/runs/<RUN_ID>/summary/category_results.csv`：本次 run 中所有 scenario 的类别结果汇总。

## 5. 性能分析设置

### 完整对比

主性能对比只报告 `full` 与对比基线，不把组件消融变体作为基线。运行默认场景和
主方法/基线：

```bash
bash scripts/run.sh all
```

使用真实模型时保持 `USE_FAKE_MODEL_RUNNER=0`。快速检查可运行：

```bash
bash scripts/run.sh smoke
```

建议主表展示：

1. 每个场景下主方法和基线的 `avg_latency_ms`、`p95_latency_ms`、`avg_ttft_ms`；
2. `full` 相对 `target_only`、`sync_batch_sd`、
   `SpecEdge` 和 `server_only` 的加速比或时延降低；
3. `goodput_tok_s` 与 `target_utilization`，判断是否由 target 端瓶颈主导；
4. `category_results` 中 6 类任务的分项表现，观察长上下文或低 acceptance 类别的影响。

推荐图表：

- 场景 × 方法的平均时延柱状图；
- p95/p99 tail latency 对比；
- goodput 与 target utilization 对比；
- 6 类任务的 `full` speedup 热力图；
- stale waste、bonus reuse 和 rollback 的堆叠柱状图。

## 6. 消融实验设置

消融实验使用同一场景、同一 seed、同一请求集合，只比较 `full` 和组件移除版本。
`wo_async`、`wo_scheduling`、`conservative_rollback` 是组件消融，不作为外部基线报告。

推荐使用 `combined_strong_heterogeneous` 作为主消融场景，因为它同时包含设备、网络和
到达过程异构：

```bash
SCENARIOS=combined_strong_heterogeneous \
METHODS="full wo_async wo_scheduling conservative_rollback" \
bash scripts/run.sh all
```

核心消融对照：

| 消融目标 | 对比方法 | 观察指标 |
|---|---|---|
| 持续乐观起草收益 | `full` vs `wo_async` | 时延、TTFT、lane utilization、device queue wait |
| 调度收益 | `full` vs `wo_scheduling` | p95 时延、lane queue wait、target utilization |
| 精细回滚与 bonus 重定位 | `full` vs `conservative_rollback` | wasted draft tokens、stale ratio、rollback count |

解读时避免只看平均时延。若 `full` 平均时延提升不大，但 p95、device queue wait 或
stale waste 明显下降，说明异步和调度主要改善尾部与资源利用。

## 7. 敏感性分析设置

### 固定窗口 W

当前 `full` 已按 DSI-style 持续向后起草，不再读取 `W_default` 作为固定窗口限制。
因此旧的 `sensitivity-w` 不再适合作为 `full` 的核心敏感性实验；保留该脚本只用于
兼容旧运行流程。若需要分析窗口限制，应新增一个显式 fixed-window 方法，或直接比较
`full` 与 `wo_async`。

```bash
SCENARIO=combined_strong_heterogeneous W_VALUES="1 2 3 4" bash scripts/run.sh sensitivity-w
```

输出：

```text
outputs/runs/<RUN_ID>/raw/sensitivity_w.csv
```

旧脚本输出主指标，但对当前 `full` 的解释力有限。更推荐观察：

- `full` 相对 `wo_async` 的时延、TTFT 和 stale waste；
- `edge.num_lanes` 对同请求多位置验证重叠率和 lane queue wait 的影响；
- `avg_acceptance_rate`、`avg_selected_gamma` 与 wasted draft tokens 是否表明策略过度激进。

如果需要同时分析 `wasted_draft_tokens` 和 `stale_segment_ratio`，应读取对应
`system_metrics` 和 `segment_details`。

### Verifier lane 数

lane 数越多，边缘排队下降，但超过负载需求后收益会饱和。

```bash
SCENARIO=combined_strong_heterogeneous LANE_VALUES="1 2 4 8" bash scripts/run.sh sensitivity-lanes
```

输出：

```text
outputs/runs/<RUN_ID>/raw/sensitivity_lanes.csv
```

内置脚本输出主指标，重点观察：

- `avg_latency_ms`、`p95_latency_ms` 和 `avg_ttft_ms` 是否下降；
- `goodput_tok_s` 是否随 lane 数增长；
- 时延收益是否在 4 或 8 lanes 后趋于平坦。

如果需要分析 `lane_queue_wait_ms_p95`、`target_utilization` 和 lane utilization，应为每个
lane 数生成独立 config 后运行 `full`，再读取对应 `system_metrics`。

### 可选手动 sweep

如果需要更完整的论文级分析，可额外复制 YAML 并修改：

- `speculation.gamma_candidates`：评估 `full`、`sync_batch_sd` 和异步消融方法中更保守或更激进的 lookahead；SpecEdge/server-only 树形基线不使用该项选择树深度；
- `sync_batch.B_global` 和 `global_batch_timeout_ms`：评估同步 batch 大小和等待超时；
- `simulation.poisson_rate_per_s`：评估到达负载；
- `device_pools.*.uplink_mbps/rtt_ms`：评估网络瓶颈；
- `edge.target_only_token_rate_tok_s`：评估边缘 target 自回归 decode 和 verification forward 能力；
- `device_pools.*.draft_token_rate_tok_s`：评估端侧算力异构；
- `server_only.draft_token_rate_tok_s`：评估 server-only 固定 1.5B drafter 的服务器侧生成速度；
- `specedge.tree_draft_strategy` / `server_only.tree_draft_strategy`：默认 `specexec_approx`；可手动切换到 `linear` 运行旧固定段级线性近似；
- `specedge.max_n_beams/max_beam_len/max_branch_width/max_budget`：评估 `specexec_approx` 树形起草宽度、深度和预算；trace 中 `processed_candidate_count`、`retained_tree_nodes`、`target_verify_tree_nodes` 和 `proposed_count` 分别记录 draft forward candidate 数、预算剪枝后保留节点数、target verify 计费节点数和 acceptance 统计使用的可验证路径深度；
  默认值 504 tok/s 来自
  [ComputingForGeeks RTX 4090 Ollama benchmark](https://computingforgeeks.com/ollama-models-cheat-sheet/)。

手动 sweep 应固定 seed，并记录修改后的 config 文件名，避免把配置差异误认为方法收益。

## 8. 预期效果

主性能对比预期：

- `target_only` 语义最直接，但所有生成集中在边缘，平均时延和 TTFT 通常较高；
- `sync_batch_sd` 能利用异构 drafter、动态 gamma 和 batch verify，但 barrier waiting 会放大尾部时延，网络异构场景更明显；
- `SpecEdge` 表示 SpecExec-style 树形近似 baseline，预期通过 server batch validation、
  proactive continuation 和 pipeline-aware request interleaving 减少 server idle bubble；
  收益取决于 proactive 命中率、batch interleaving 充分性和网络 RTT；
- `server_only` 只承担请求级 prompt 上传和最终输出下载，不承担 segment 级端边往返；
  所有 draft 与单段 verify 都占用服务器侧解析资源；
- `full` 预期在强异构和高负载场景收益最大，体现为更低平均/p95 时延、更高 goodput、更低 lane idle，以及可控的 stale waste。

组件消融预期：

- `wo_async` 去掉持续乐观起草后，资源重叠不足，TTFT 和平均时延通常劣于 `full`；
- `wo_scheduling` 使用 round-robin lane，强异构场景下 p95 和 lane queue wait 应高于 `full`；
- `conservative_rollback` 会减少复杂重定位，但 stale/discarded 和 wasted draft tokens 通常更高。

类别上，acceptance 较高或输出较规则的任务更容易从大 gamma 和 bonus 复用中获益；
acceptance 较低或 correction 频繁的任务应更依赖动态 gamma 和精细回滚控制。

## 9. 复现实验检查清单

1. 确认真实模型可加载，drafter 与 target tokenizer/vocab 兼容；
2. 固定 `simulation.seed`，正式结果不要使用 fake 模型推理器；
3. 运行第 5 节主性能对比命令生成主结果；
4. 检查 `outputs/runs/<RUN_ID>/manifest.yaml`、`summary/category_results.csv`、`system_metrics` 和明细表；
5. 运行第 6 节组件消融命令；
6. 运行 `sensitivity-lanes`；`sensitivity-w` 仅作为旧 fixed-window 流程保留，不作为当前 `full` 的主敏感性实验；
7. 汇总平均、尾部、goodput、利用率和 stale waste，而不是只报告单一 speedup；
8. 修改 YAML 做手动 sweep 时，保留配置副本并使用独立 `RUN_ID`。
