# 三端设备 + 边缘 7B 推测解码实验

这个项目现在按“方法”和“基线”组织代码：

- `--method proposed`：三组件异步端边推测验证方法。
- `--method sync_batch`：同步 barrier batch speculative baseline。
- `--method target_only`：纯 target-only baseline。

Proposed 方法拆成三个互相解耦的组件：

1. `Asynchronous Edge Speculation`：端侧异步起草、上传、边缘接收和结果返回。
2. `Heterogeneity-aware Verification Scheduling`：adaptive lookahead、prefix-aware lane 调度、lane-local micro-batching。
3. `Prefix-version Consistency Control`：prefix state、stale draft、rejection 和局部回滚。

## 环境

```bash
mamba env create -f environment.yml
conda activate edge-spec
python scripts/prepare_specbench.py
```

## 快速自检

不下载真实模型，使用 deterministic fake model 跑完整协议：

```bash
python -m edge_spec.run --method proposed --use-fake-models --max-new-tokens 16 --gamma 8 --initial-lookahead 4
python -m edge_spec.run --method sync_batch --use-fake-models --max-new-tokens 16 --gamma 4
python -m edge_spec.run --method target_only --use-fake-models --max-new-tokens 16
```


## 脚本用法

三个正式脚本保持原名，默认按方法名写结果目录：

| 方法 | 脚本 | 默认结果目录 |
|---|---|---|
| proposed | `scripts/run_qwen25_specbench_async.sh` | `results/proposed/<CATEGORY>/` |
| sync_batch | `scripts/run_qwen25_specbench.sh` | `results/sync_batch/<CATEGORY>/` |
| target_only | `scripts/run_qwen25_specbench_target_only.sh` | `results/target_only/<CATEGORY>/` |

常用运行方式：

```bash
bash scripts/run_qwen25_specbench_async.sh
bash scripts/run_qwen25_specbench.sh
bash scripts/run_qwen25_specbench_target_only.sh
```

所有脚本都支持这些环境变量：

- `CATEGORY`: `Sum`, `Math`, `MT`, `QA`, `RAG`, `Trans`; 默认 `Sum`。
- `DATASET_MODE`: `all`, `limit`, `one-per-category`, `one-per-category-per-device`; 默认 `all`。
- `RESULTS_DIR`: 覆盖输出目录。
- `DATASET_PATH`, `PROFILE_CONFIG`: 覆盖数据集和网络 profile 路径。
- `TARGET_MODEL`, `SERVER_DEVICE`, `TORCH_DTYPE`: 覆盖边缘 target 模型加载配置。
- `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K`: 覆盖生成参数。
- `SEED`: 实验主 seed, 默认 `42`。
- `NETWORK_SEED`: 动态网络 trace seed, 默认等于 `SEED`。
- `NETWORK_TRACE_SLOT_S`: 网络 replay 时间片长度, 默认 `0.05`。

`proposed` 和 `sync_batch` 额外支持：

- `DRAFT_MODEL_0`, `DRAFT_MODEL_1`, `DRAFT_MODEL_2`: 覆盖三台端侧 draft 模型。
- `CLIENT_DEVICE`: 覆盖 draft 模型加载设备。
- `GAMMA`: 覆盖 draft chunk 上限；proposed adaptive 默认 `8`，sync_batch 默认 `4`。

`proposed` 额外支持：

- `LANE_COUNT`: verifier lane 数量, 默认 `3`。
- `MAX_INFLIGHT_SEGMENTS`: 每请求最多未验证 segment 数, 默认 `2`。
- `LOOKAHEAD_POLICY`: `adaptive` 或 `fixed`。
- `INITIAL_LOOKAHEAD`: adaptive lookahead 初始草稿长，默认 `4`。
- `SCHEDULER`: `prefix-aware` 或 `queue-only`。
- `LANE_BATCH_SIZE`: lane-local micro-batch 大小。
- `LANE_BATCH_TIMEOUT_S`: lane-local micro-batch 等待超时。

示例：

```bash
CATEGORY=RAG LANE_COUNT=2 bash scripts/run_qwen25_specbench_async.sh
CATEGORY=Math bash scripts/run_qwen25_specbench.sh
CATEGORY=QA bash scripts/run_qwen25_specbench_target_only.sh
RESULTS_DIR=results/proposed/debug MAX_NEW_TOKENS=32 bash scripts/run_qwen25_specbench_async.sh
```

正式对比建议成对运行同一组 seed 和网络 trace：

```bash
SEED=42 NETWORK_SEED=42 CATEGORY=MT bash scripts/run_qwen25_specbench_target_only.sh
SEED=42 NETWORK_SEED=42 CATEGORY=MT bash scripts/run_qwen25_specbench.sh
SEED=42 NETWORK_SEED=42 CATEGORY=MT bash scripts/run_qwen25_specbench_async.sh
```

## 同步 baseline

默认运行命令：

```bash
bash scripts/run_qwen25_specbench.sh
```

脚本内部使用 `--method sync_batch`。选择类别时，在命令前加 `CATEGORY=...`：

```bash
CATEGORY=Sum bash scripts/run_qwen25_specbench.sh
CATEGORY=Math bash scripts/run_qwen25_specbench.sh
```

不指定 `CATEGORY` 时默认跑 `Sum`。可选类别为 `Sum`、`Math`、`MT`、`QA`、`RAG`、`Trans`。同步结果默认写入 `results/sync_batch/<CATEGORY>/`。

## 仅边缘 baseline

仅边缘 baseline 使用 `--method target_only`，只加载边缘侧 target model，不加载三台端侧 draft model。它用于衡量“请求上传到边缘后完全由 7B target 生成”的端到端基线。

默认运行命令：

```bash
bash scripts/run_qwen25_specbench_target_only.sh
```

选择类别时同样使用 `CATEGORY=...`：

```bash
CATEGORY=Sum bash scripts/run_qwen25_specbench_target_only.sh
CATEGORY=RAG bash scripts/run_qwen25_specbench_target_only.sh
```

结果默认写入 `results/target_only/<CATEGORY>/`。该 baseline 没有 draft verification 事件，因此 `event_trace.jsonl` 为空，主要看 `request_records.jsonl` 和 `summary.json`。

## Proposed 方法

默认运行命令：

```bash
bash scripts/run_qwen25_specbench_async.sh
```

脚本内部使用 `--method proposed --lane-count ${LANE_COUNT:-3}`。选择类别或 lane 数量：

```bash
CATEGORY=RAG bash scripts/run_qwen25_specbench_async.sh
LANE_COUNT=2 bash scripts/run_qwen25_specbench_async.sh
```

常用 proposed 消融参数：

```bash
python -m edge_spec.run \
  --method proposed \
  --lane-count 3 \
  --max-inflight-segments 2 \
  --gamma 8 \
  --initial-lookahead 4 \
  --lookahead-policy adaptive \
  --scheduler prefix-aware \
  --lane-batch-size 2 \
  --lane-batch-timeout-s 0.001
```

默认 proposed adaptive 从 `--initial-lookahead 4` 起步，并在 `[1, --gamma]` 范围内调整；当前脚本默认 `--gamma 8`。可将 `--lookahead-policy fixed`、`--scheduler queue-only`、不同 `--gamma` 或不同 `--lane-batch-size` 用于消融。

## 数据选择

脚本默认使用 `DATASET_MODE=all`，每次只跑一个类别。也可以切换：

```bash
DATASET_MODE=one-per-category bash scripts/run_qwen25_specbench.sh
DATASET_MODE=all bash scripts/run_qwen25_specbench_async.sh
DATASET_MODE=all bash scripts/run_qwen25_specbench_target_only.sh
```

`question.jsonl` 应为 480 条。原始 13 个 category 会归并为 SpecBench 六类任务：`Sum`、`Math`、`MT`、`QA`、`RAG`、`Trans`。

默认模型：

- 端侧 `device-0`: `Qwen/Qwen2.5-0.5B-Instruct`
- 端侧 `device-1`: `Qwen/Qwen2.5-1.5B-Instruct`
- 端侧 `device-2`: `Qwen/Qwen2.5-3B-Instruct`
- 边缘服务器: `Qwen/Qwen2.5-7B-Instruct`

输出：

- `request_records.jsonl`: 请求级结果和指标
- `event_trace.jsonl`: 同步轮次或 proposed lane 验证事件 trace；target_only 下为空
- `summary.json`: 汇总指标、数据选择、运行配置

指标字段说明见 `docs/metric.md`。

## 网络 profile

默认 profile 中的网络基准参数相同；网络差异来自每次传输的动态采样，而不是固定绑定到某个设备。动态网络由 `NETWORK_SEED + device_id + direction + virtual_time_slot` replay, 因此 target_only、sync_batch 和 proposed 可以分开运行但共享同一条外部网络轨迹。网络 profile 支持：

- `bandwidth_jitter_ratio`
- `rtt_jitter_ms`
- `congestion_probability`
- `congestion_slowdown`

每轮实际采样到的上下行带宽、RTT、jitter 和拥塞标记会写入 `event_trace.jsonl`。

## 测试

```bash
python3 -m unittest discover -s tests
```
