# 最新实验结果分析

> 文档状态：当前最近一次可讨论 run 的结果摘要，不等同于最终论文结果。若后续产生新的主实验 run，应更新本文件的数据来源和结论，或将旧文件改名为 `experiment_analysis_<RUN_ID>.md` 保存。

数据来源为 `outputs/runs/20260611-155747`。该 run 于 `2026-06-11 15:57:47 +0800`
启动，命令为 `bash scripts/run.sh all`，使用 `combined_strong_heterogeneous`
场景、每类 10 条样本，总计 60 个请求，且 `use_fake_model_runner=false`。本次最新结果只包含
`full`、`target_only`、`SpecEdge` 和 `server_only` 四种方法；没有包含
`sync_batch_sd`，因此不能基于该 run 对同步 batch baseline 作数值结论。

## 总体结果

| 方法 | 平均时延 (s) | P50 (s) | P95 (s) | P99 (s) | 平均 TTFT (s) | Goodput (tok/s) | Acceptance | 平均 Gamma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 20.04 | 16.88 | 49.01 | 58.43 | 0.33 | 148.64 | 0.693 | 2.22 |
| `target_only` | 59.53 | 60.41 | 106.48 | 110.09 | 59.53 | 79.92 | 0.000 | 0.00 |
| `SpecEdge` | 53.65 | 53.63 | 88.70 | 97.07 | 0.67 | 91.84 | 0.716 | 2.03 |
| `server_only` | 43.51 | 43.72 | 78.44 | 80.88 | 43.51 | 108.01 | 0.581 | 3.96 |

`full` 在最新强异构场景中是整体最优方法。相对 `target_only`，`full` 的平均时延从
59.53 s 降到 20.04 s，降幅为 66.3%，等价于 2.97x 加速；P95 时延降低 54.0%，P99
时延降低 46.9%，goodput 从 79.92 tok/s 提升到 148.64 tok/s，提升 86.0%。

相对 `SpecEdge`，`full` 的平均时延降低 62.7%，P95 降低 44.7%，P99 降低 39.8%，goodput
提升 61.8%。这说明收益不只是来自 speculative decoding 本身，而是来自异步乐观起草、
多 verifier lane 并行验证和前缀一致性控制对流水线空泡的压缩。需要注意的是，`full`
的 acceptance 为 0.693，低于 `SpecEdge` 的 0.716；因此该 run 中 `full` 的优势不是由
更高接受率解释的，而是主要来自调度和流水线结构。

相对 `server_only`，`full` 的平均时延降低 53.9%，P95 降低 37.5%，P99 降低 27.8%，goodput
提升 37.6%。`server_only` 避免了 segment 级端边往返，但它把 draft 和 verify 都放到服务器侧，
target 利用率接近饱和；`full` 将 draft 计算移到异构端侧并用边缘 lane 做并行验证，因此端到端
时延和吞吐仍明显更优。

## 类别结果

| 类别 | `full` 平均时延 (s) | `full` P95 (s) | Acceptance | 平均 Gamma | vs `target_only` | vs `SpecEdge` | vs `server_only` |
|---|---:|---:|---:|---:|---:|---:|---:|
| MT | 15.37 | 28.74 | 0.696 | 2.22 | 2.83x / 64.7% | 3.22x / 68.9% | 2.05x / 51.2% |
| QA | 21.30 | 44.01 | 0.621 | 1.89 | 3.05x / 67.3% | 2.89x / 65.4% | 2.24x / 55.4% |
| Math | 17.30 | 33.65 | 0.825 | 3.90 | 3.20x / 68.8% | 2.75x / 63.6% | 2.31x / 56.7% |
| RAG | 15.96 | 45.38 | 0.686 | 2.15 | 3.60x / 72.2% | 3.04x / 67.1% | 2.62x / 61.9% |
| Sum | 27.52 | 49.83 | 0.670 | 1.94 | 2.38x / 58.0% | 2.10x / 52.3% | 1.76x / 43.1% |
| Trans | 22.77 | 59.03 | 0.655 | 2.06 | 3.09x / 67.6% | 2.52x / 60.3% | 2.27x / 55.9% |

所有 6 个类别中，`full` 的平均时延都低于三个基线。相对 `target_only`，RAG 的收益最大，
平均时延降低 72.2%，达到 3.60x；相对 `SpecEdge`，MT 的收益最大，平均时延降低 68.9%，达到
3.22x。Sum 是最弱类别，但相对 `target_only`、`SpecEdge` 和 `server_only` 仍分别有 2.38x、
2.10x 和 1.76x 加速。

类别差异与 acceptance 和动态 gamma 基本一致。Math 的 acceptance 最高，为 0.825，平均 gamma
达到 3.90，说明调度器能在高接受率任务上选择更激进的 lookahead。QA 的 acceptance 最低，为
0.621，平均 gamma 降到 1.89，但它仍相对 `SpecEdge` 取得 2.89x 加速，说明在低接受率任务上，
降低 gamma 后的异步验证仍能保留明显的流水线收益。

## 系统指标解释

| 方法 | Target 利用率 | 设备利用率 | Lane 利用率 | Lane P95 排队 (ms) | Stale 比例 | Wasted Draft Tokens | Bonus Reuse | Rollback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.298 | 0.480 | 0.298 | 16.2 | 0.103 | 3256 | 339 | 1250 |
| `target_only` | 0.999 | 0.000 | 0.000 | 0.0 | 0.000 | 0 | 0 | 0 |
| `SpecEdge` | 0.770 | 0.478 | 0.000 | 0.0 | 0.000 | 7228 | 0 | 1275 |
| `server_only` | 0.999 | 0.000 | 0.000 | 0.0 | 0.000 | 4598 | 0 | 1567 |

`target_only` 和 `server_only` 的 target 利用率都接近 1.0，说明这两个方法主要受服务器侧
自回归生成或服务器侧 draft/verify 串行流程限制。`SpecEdge` 的 target 利用率为 0.770，但
batch waiting time 总计达到 2359.1 s，表明同步验证和 proactive pipeline 中仍存在大量等待。
`full` 没有 batch waiting，lane P95 排队只有 16.2 ms，说明多 lane 验证不是最新 run 的主要瓶颈。

`full` 的 target/lane 利用率为 0.298，明显低于其他方法。这一方面说明边缘 target 压力被释放，
另一方面也说明瓶颈已经转移到端侧 draft、设备队列或网络路径。对应地，`full` 的设备队列等待总量为
945.0 s，高于 `SpecEdge` 的 427.5 s；后续如果继续优化，优先应检查低端设备上的 draft 队列、
动态 gamma 是否过于保守，以及请求到设备的轮询分配是否放大了低端设备排队。

`full` 的 wasted draft tokens 为 3256，低于 `SpecEdge` 的 7228 和 `server_only` 的 4598。
虽然 `full` 产生了 10.3% stale segment，并进行了 1250 次 rollback，但 bonus reuse 达到
339 token，且精细重定位减少了无效 draft 的总体浪费。因此，该 run 支持一个机制性结论：
持续乐观起草会带来 stale 和 rollback，但配合前缀一致性控制后，浪费保持在可控范围内，并换来了更低时延和更高 goodput。

## 结论与后续实验

结论 1：在 `combined_strong_heterogeneous` 场景中，`full` 相对 `target_only`、`SpecEdge`
和 `server_only` 的平均时延分别降低 66.3%、62.7% 和 53.9%。

结论 2：`full` 的 goodput 分别比 `target_only`、`SpecEdge` 和 `server_only` 高 86.0%、
61.8% 和 37.6%，同时保持最低的 P95 和 P99 时延。

结论 3：`full` 在 6 个 SpecBench 顶层类别中均优于三个基线。相对 `target_only`，RAG
的收益最大，达到 3.60x；相对 `SpecEdge`，MT 的收益最大，达到 3.22x。

结论 4：`full` 的 acceptance 低于 `SpecEdge`（0.693 vs. 0.716），但时延和 goodput
仍明显更优。因此，该 run 中的主要收益更应归因于异步执行、verifier lane 并行和
一致性保持的回滚机制，而不是 draft 质量更高。

限制：该 run 只覆盖一个场景、60 个请求和一个 seed，且不包含 `sync_batch_sd`。在写入最终论文结果前，应补充多 seed 实验、缺失的同步 batch baseline、组件消融和 lane 敏感性分析。
