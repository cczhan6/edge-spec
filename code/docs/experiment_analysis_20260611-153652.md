# 实验结果分析：20260611-153652

> 文档状态：历史单次 run 记录，用于追溯 `homogeneous` 场景结果。该文件不代表最终论文结果；最终结果应以多场景、多 seed 和完整 baseline 的汇总为准。

数据来源为 `outputs/runs/20260611-153652`。该 run 于 `2026-06-11 15:36:52 +0800`
启动，命令为 `bash scripts/run.sh all`，使用 `homogeneous` 场景、每类 10 条样本，总计
60 个请求，且 `use_fake_model_runner=false`。本次结果包含 `full`、`target_only`、
`SpecEdge` 和 `server_only` 四种方法；没有包含 `sync_batch_sd`，因此不能基于该 run
对同步 batch baseline 作数值结论。

`homogeneous` 场景中 8 台虚拟 client 均使用 `medium` drafter，解析速率为 60 tok/s，
上下行为 25/100 Mbps，RTT 为 40 ms，jitter 为 10 ms。该设置用于观察没有端侧设备强异构时，
异步端边推测解码与 SpecEdge-style baseline 的差异。

## 总体结果

| 方法 | 平均时延 (s) | P50 (s) | P95 (s) | P99 (s) | 平均 TTFT (s) | Makespan (s) | Goodput (tok/s) | Acceptance | 平均 Gamma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 13.55 | 13.66 | 22.55 | 26.74 | 0.26 | 27.59 | 329.44 | 0.690 | 2.35 |
| `target_only` | 59.65 | 59.58 | 106.41 | 110.05 | 59.65 | 113.65 | 79.96 | 0.000 | 0.00 |
| `SpecEdge` | 51.46 | 53.35 | 75.00 | 75.54 | 0.64 | 76.98 | 118.06 | 0.727 | 2.06 |
| `server_only` | 43.53 | 43.70 | 78.38 | 80.74 | 43.53 | 84.07 | 108.10 | 0.581 | 3.96 |

`full` 在同构场景中取得最优端到端表现。相对 `target_only`，`full` 的平均时延从
59.65 s 降到 13.55 s，降低 77.3%，等价于 4.40x 加速；P95 时延降低 78.8%，P99
时延降低 75.7%；goodput 从 79.96 tok/s 提升到 329.44 tok/s，提升 312.0%。

相对 `SpecEdge`，`full` 的平均时延降低 73.7%，P95 降低 69.9%，P99 降低 64.6%，goodput
提升 179.0%。值得注意的是，`SpecEdge` 的 acceptance 更高，为 0.727，而 `full` 为 0.690；
因此本次 run 中 `full` 的优势不是由 draft token 接受率更高带来的，而是来自异步持续起草、
多 lane 验证和精细前缀一致性控制对端到端流水线的改善。

相对 `server_only`，`full` 的平均时延降低 68.9%，P95 降低 71.2%，P99 降低 66.9%，goodput
提升 204.7%。`server_only` 的服务器侧 drafter 更快，平均 gamma 也更大，但它只保留一个
active request 串行执行服务器侧 draft/verify；在 60 请求 Poisson 到达负载下，该串行路径的
完成时延明显高于 `full` 的端侧并行 draft 与边缘多 lane 验证。

## 类别结果

| 类别 | `full` 平均时延 (s) | `full` P95 (s) | Goodput (tok/s) | Acceptance | 平均 Gamma | vs `target_only` | vs `SpecEdge` | vs `server_only` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MT | 12.12 | 18.69 | 64.23 | 0.693 | 2.20 | 3.57x / 72.0% | 4.07x / 75.4% | 2.60x / 61.6% |
| QA | 16.93 | 21.85 | 75.80 | 0.610 | 1.91 | 3.88x / 74.2% | 3.64x / 72.6% | 2.82x / 64.6% |
| Math | 10.95 | 16.90 | 86.33 | 0.812 | 4.20 | 5.03x / 80.1% | 4.21x / 76.2% | 3.63x / 72.4% |
| RAG | 12.62 | 21.96 | 60.12 | 0.658 | 2.25 | 4.57x / 78.1% | 3.94x / 74.7% | 3.35x / 70.1% |
| Sum | 13.93 | 21.82 | 51.89 | 0.708 | 2.35 | 4.76x / 79.0% | 3.55x / 71.8% | 3.48x / 71.3% |
| Trans | 14.79 | 26.76 | 58.34 | 0.659 | 2.25 | 4.72x / 78.8% | 3.54x / 71.8% | 3.47x / 71.2% |

所有 6 个类别中，`full` 的平均时延均低于三个基线。相对 `target_only`，Math 的加速最高，
达到 5.03x，平均时延降低 80.1%；Sum 和 Trans 也分别达到 4.76x 和 4.72x。相对 `SpecEdge`，
Math 和 MT 的收益最突出，分别达到 4.21x 和 4.07x。

类别差异与 acceptance 和动态 gamma 的行为一致。Math 的 acceptance 最高，为 0.812，平均
gamma 达到 4.20，说明调度器能在高接受率任务上选择更长的 lookahead。QA 的 acceptance 最低，
为 0.610，平均 gamma 降到 1.91，但仍相对 `SpecEdge` 达到 3.64x 加速，说明低接受率任务中
保守 gamma 与异步 lane 并行仍能显著降低完成时延。

## 系统指标解释

| 方法 | Target 利用率 | 设备利用率 | Lane 利用率 | Lane P95 排队 (ms) | Stale 比例 | Wasted Draft Tokens | Bonus Reuse | Rollback | Batch Wait (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.638 | 0.761 | 0.638 | 19.9 | 0.134 | 3523 | 338 | 1215 | 0.0 |
| `target_only` | 1.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0 | 0 | 0 | 0.0 |
| `SpecEdge` | 0.974 | 0.412 | 0.000 | 0.0 | 0.000 | 6914 | 0 | 1194 | 2742.5 |
| `server_only` | 0.999 | 0.000 | 0.000 | 0.0 | 0.000 | 4598 | 0 | 1567 | 0.0 |

`full` 的设备利用率为 0.761，lane/target 利用率为 0.638，说明同构场景下端侧 draft
和边缘验证都被有效使用。lane P95 排队仅 19.9 ms，且没有 batch waiting，表明边缘 lane
排队不是主要瓶颈。相比之下，`target_only` 和 `server_only` 的 target 利用率接近 1.0，
说明它们主要受服务器侧串行生成或服务器侧 draft/verify 资源限制。

`SpecEdge` 的 target 利用率为 0.974，但 batch waiting time 总计达到 2742.5 s，并产生
6914 个 wasted draft tokens，其中 proactive wasted tokens 为 4854。它的 TTFT 很低，
为 0.64 s，但完成时延仍高于 `full` 和 `server_only`。这说明在当前线性 SpecEdge 近似中，
proactive continuation 和 server batch validation 能改善首 token，但会在完成路径上引入大量
等待和无效 draft 开销。

`full` 产生了 13.4% stale segment 和 1215 次 rollback，说明持续乐观起草确实带来一致性维护成本。
不过它的 wasted draft tokens 为 3523，明显低于 `SpecEdge` 的 6914，也低于 `server_only` 的
4598；同时有 338 个 bonus token 被复用。因此，该 run 支持的机制性解释是：`full` 通过多 lane
异步验证和 bonus 重定位吸收了部分乐观起草风险，使额外 draft 浪费换来了更低时延和更高 goodput。

## 与强异构 Run 的对照

与 `outputs/runs/20260611-155747` 的 `combined_strong_heterogeneous` 场景相比，本 run 中
`full` 的平均时延从 20.04 s 降到 13.55 s，降低 32.4%；goodput 从 148.64 tok/s 提升到
329.44 tok/s，提升 121.6%。主要原因是同构场景消除了 low-end 设备的 25 tok/s draft 速率、
弱 uplink 和 90 ms RTT，所有请求都使用 medium drafter 与中等网络条件。结果上表现为
`full` 的设备利用率从 0.480 提高到 0.761，target/lane 利用率从 0.298 提高到 0.638。

这个对照说明：`full` 在强异构场景下仍然胜出，但同构 medium-client 场景更能发挥持续端侧 draft
和边缘多 lane 验证的重叠收益。换句话说，设备和网络异构会削弱但不会消除 `full` 的优势。

## 结论与后续实验

结论 1：在 `homogeneous` 场景中，`full` 相对 `target_only`、`SpecEdge` 和 `server_only`
的平均时延分别降低 77.3%、73.7% 和 68.9%。

结论 2：`full` 的 goodput 分别比 `target_only`、`SpecEdge` 和 `server_only` 高 312.0%、
179.0% 和 204.7%，同时保持最低的 P95 和 P99 时延。

结论 3：`full` 在 6 个 SpecBench 顶层类别中均优于三个基线。Math 的收益最突出，
相对 `target_only` 达到 5.03x，相对 `SpecEdge` 达到 4.21x。

结论 4：`full` 的 acceptance 低于 `SpecEdge`（0.690 vs. 0.727），但时延和 goodput
仍明显更优。因此，该 run 中的主要收益更应归因于异步执行、verifier lane 并行和
一致性保持的回滚机制，而不是 draft 质量更高。

限制：该 run 只覆盖一个场景、60 个请求和一个 seed，且不包含 `sync_batch_sd`。最终论文结果应补充多 seed 实验、缺失的同步 batch baseline、组件消融和 lane 敏感性分析。
