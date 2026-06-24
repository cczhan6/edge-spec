# 异构端边推测解码仿真器

本目录实现一个用于论文实验的端边推测解码事件仿真器。真实 drafter 和 target 模型负责生成 token、计算 greedy acceptance、correction 和 bonus；虚拟设备、网络和边缘服务器的时延由解析公式给出；事件仿真器负责推进 draft、verify、通信、排队、回滚和请求完成状态。

除特别说明外，以下命令均在仓库根目录的 `code/` 目录中执行。

真实模型 forward 的 wall time 不进入虚拟时间。因此，该仿真器可以在不部署大量真实端侧设备的情况下，评估异构端侧 drafter、边缘 target、异步验证和调度策略对 decode-only 时延与吞吐的影响。

## 文档导航

| 目标 | 文档 |
|---|---|
| 文档维护规则和阅读顺序 | [docs/README.md](docs/README.md) |
| 实验口径、场景、方法和推荐实验 | [docs/experiment.md](docs/experiment.md) |
| 输出文件和指标定义 | [docs/metric.md](docs/metric.md) |
| 最近一次实验结果分析 | [docs/latest_experiment_analysis.md](docs/latest_experiment_analysis.md) |
| 历史 run 分析 | `docs/experiment_analysis_<RUN_ID>.md` |

## 核心口径

- 实验范围是 steady-state autoregressive decoding；请求到达即视为 decode-ready，
  端侧 drafter 和边缘 target 已分别建立该请求的 prefix KV cache。
- Prompt prefill、初始 prompt 传输及 KV 建立过程不纳入仿真时间和资源调度。
- 请求固定绑定到 origin client device，不在设备之间迁移。
- 每台虚拟 client 固定部署一个 drafter，并以 segment 级 FIFO 串行服务本地请求。
- 边缘服务器部署 target model，负责 target verification 和 target-only 生成。
- `draft_ms`、`verify_ms`、`target_only_ms` 和 `network_ms` 均由配置中的解析参数计算。
- acceptance、correction 和 bonus 由真实模型在线产生，不使用预设 acceptance trace。
- `full` 支持持续乐观起草、同请求多位置并行 verify、least-finish lane 调度和 bonus 重定位。

## 方法

| 方法 | 用途 | 简要说明 |
|---|---|---|
| `full` | 主方法 | 异构端侧 drafter、动态 gamma、多 verifier lane、持续乐观起草和精细回滚。 |
| `target_only` | 自回归基线 | decode-ready 请求由单个边缘 target 服务资源完整生成；decode-only 口径下无通信。 |
| `server_only_linear` | Server-only SD-Linear | 服务器侧线性 draft + target verify，draft/target 使用独立逻辑资源，无端边通信。 |
| `specedge_linear` | SpecEdge-Linear | 端侧线性 draft、端边往返、server batch validation、proactive continuation。 |
| `dip_sd` | DiP-SD | 固定流水线加确定性在线优化器，按 epoch 有序 batch verify。 |
| `server_only_tree` | Server-only SD-Tree | 服务器侧 SpecExec-style 树形 draft + target verify，无 proactive、无端边通信。 |
| `specedge_tree` | SpecEdge-Tree | 端侧 SpecExec-style 树形 draft、server batch validation、proactive continuation。 |
| `wo_async` | 组件消融 | 去掉持续乐观起草。 |
| `wo_scheduling` | 组件消融 | 去掉 heterogeneity-aware lane scheduling。 |
| `conservative_rollback` | 组件消融 | 去掉精细 bonus 重定位和局部保留。 |

Legacy aliases `sync_batch_sd`, `SpecEdge`, and `server_only` remain accepted for
old experiments and tests, but the baseline rebuild validates the canonical names
listed above.

## 场景

`scripts/run.sh all` 默认运行以下场景：

| 场景 | 目的 |
|---|---|
| `homogeneous` | 所有虚拟 client 使用 `medium` drafter，用于观察无设备强异构时的表现。 |
| `combined_strong_heterogeneous` | 设备、网络和到达过程均强异构，主要观察动态负载稳定性。 |

场景覆盖文件位于 `configs/<scenario>.yaml`，会与 `configs/default.yaml` 深度合并。参数依据和解释见 [docs/experiment.md](docs/experiment.md)。

## 环境

```bash
pip install -r requirements.txt
```

默认真实模型：

| 角色 | 模型 | 宿主设备 |
|---|---|---|
| small drafter | `Qwen/Qwen2.5-0.5B-Instruct` | `cuda:0` |
| medium drafter | `Qwen/Qwen2.5-1.5B-Instruct` | `cuda:0` |
| large drafter | `Qwen/Qwen2.5-3B-Instruct` | `cuda:0` |
| target | `Qwen/Qwen2.5-7B-Instruct` | `cuda:1` |

宿主设备只影响真实语义计算的执行位置，不影响虚拟时间。若模型已在 Hugging Face cache 中，但网络或代理不稳定，可在配置的 `model_runner` 下设置 `local_files_only: true`，或运行时设置 `HF_HUB_OFFLINE=1`。

## 运行

```bash
# 不加载真实模型的快速检查
bash scripts/run.sh smoke

# 默认 combined_strong_heterogeneous + full
bash scripts/run.sh single

# 默认场景 + 主方法/基线
bash scripts/run.sh all

# verifier lane 数敏感性分析
bash scripts/run.sh sensitivity-lanes
```

常用覆盖：

| 环境变量 | 默认值 |
|---|---|
| `CONFIG` | `configs/default.yaml` |
| `DATASET` | `data/spec_bench/question.jsonl` |
| `RUN_ROOT` | `outputs/runs` |
| `RUN_ID` | 当前开始时间，格式 `YYYYMMDD-HHMMSS` |
| `SCENARIOS` | `homogeneous combined_strong_heterogeneous` |
| `METHODS` | `full target_only server_only_linear specedge_linear dip_sd server_only_tree specedge_tree` |
| `USE_FAKE_MODEL_RUNNER` | `0` |
| `SAMPLES_PER_CATEGORY` | 空，默认使用 `simulation.num_requests` 全局抽样 |
| `TREE_DRAFT_STRATEGY` | 空，默认使用配置文件中的 `specexec_approx`；可设 `linear` 或 `specexec_approx` |

`specedge_tree` 和 `server_only_tree` 采用 `specexec_approx` 树形口径。该模式会记录 `processed_candidate_count`、`retained_tree_nodes` 和 `target_verify_tree_nodes` 三类节点数。`specedge_linear` 和 `server_only_linear` 强制使用线性候选，不依赖树形配置。

也可以直接用命令覆盖：

```bash
# 使用 SpecEdge 树形 baseline
METHOD=specedge_tree bash scripts/run.sh single

# 使用 SpecEdge 线性 baseline
METHOD=specedge_linear bash scripts/run.sh single
```

初步验证可按 SpecBench 6 类均衡抽样，例如每类 10 条、总共 60 条：

```bash
SAMPLES_PER_CATEGORY=10 bash scripts/run.sh all
```

只运行强异构场景：

```bash
SCENARIOS=combined_strong_heterogeneous \
SAMPLES_PER_CATEGORY=10 \
bash scripts/run.sh all
```

## 输出

每次运行会创建独立目录 `outputs/runs/<RUN_ID>/`，并写入 `manifest.yaml` 记录命令、配置、数据集、场景、方法、输出路径和 git 状态。若显式指定的 `RUN_ID` 或 `RUN_DIR` 已存在，脚本会退出，避免覆盖旧实验。

主要输出包括：

- `outputs/runs/<RUN_ID>/summary/all_results.csv`
- `outputs/runs/<RUN_ID>/summary/category_results.csv`
- `outputs/runs/<RUN_ID>/raw/main_results_<scenario>.csv`
- `outputs/runs/<RUN_ID>/raw/category_results_<scenario>.csv`
- `outputs/runs/<RUN_ID>/raw/system_metrics_<scenario>.csv`
- `outputs/runs/<RUN_ID>/raw/device_metrics_<scenario>_<method>.csv`
- `outputs/runs/<RUN_ID>/raw/request_details_<scenario>_<method>.csv`
- `outputs/runs/<RUN_ID>/raw/segment_details_<scenario>_<method>.csv`
- `outputs/runs/<RUN_ID>/raw/round_trace_<scenario>_<method>.csv`
- `outputs/runs/<RUN_ID>/raw/event_details_<scenario>_<method>.csv`

指标定义见 [docs/metric.md](docs/metric.md)。实验流程和推荐图表见 [docs/experiment.md](docs/experiment.md)。

## 测试

```bash
pytest -q

bash scripts/verify_baseline_rebuild.sh
```
