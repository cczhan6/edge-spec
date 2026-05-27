# 实验设置与方法流程

数据按照 SpecBench 六类任务口径划分: Summarization (Sum), Math Reasoning (Math), Multi-turn Dialogue (MT), QA, Retrieval-Augmented Generation (RAG), Machine Translation (Trans)。

当前项目支持三种运行方法:

- `--method proposed`: 本文三组件异步端边推测验证方法。
- `--method sync_batch`: 同步 barrier batch speculative baseline。
- `--method target_only`: 纯 target-only baseline。

## 1. Proposed 三组件方法

### 1.1 Asynchronous Edge Speculation

端侧设备维护本地 speculative prefix, 在未收到验证结果时继续起草, 默认每个请求最多保留 `--max-inflight-segments 2` 个未验证 draft segment。segment 经网络异步到达边缘侧 ingress, 验证结果再异步返回端侧。该组件只负责运行时事件流, 不直接决定 lookahead、lane 选择或一致性策略。

### 1.2 Heterogeneity-aware Verification Scheduling

调度组件包含三类策略:

- `--lookahead-policy adaptive|fixed`: 根据设备网络、历史接受率和边缘队列压力选择 draft chunk length。adaptive 默认从 `--initial-lookahead 4` 起步, 并夹到 `[1, --gamma]`; proposed 脚本默认 `--gamma 8`。
- `--scheduler prefix-aware|queue-only`: prefix-aware 调度综合 lane queue delay、估计验证开销、KV locality 和 rollback risk 选择 verifier lane。
- `--lane-batch-size` 与 `--lane-batch-timeout-s`: 每条 lane 内局部 micro-batch, 避免全局 barrier。

### 1.3 Prefix-version Consistency Control

`PrefixStateManager` 为每个请求维护 committed prefix、prefix version、branch prefix hash 和 rejection history。边缘侧只验证处在 committed frontier 的 ready segment。发生 rejection 时, 只增加该请求的 prefix version, 并使同请求旧版本 pending segments 失效, 不影响其他请求和 lane。

## 2. Baselines

### 2.1 sync_batch

三个端侧设备在同一 microbatch 内同步推进。每轮端侧生成 draft, 上传到边缘侧后等待本轮所有活跃设备到达, 形成 barrier batch, 再由 target model 统一验证。

### 2.2 target_only

每条请求只由边缘侧 target model 生成, 延迟包含 prompt 上行、target 生成和结果下行。该方法用于端到端 target-only 对比, 对应脚本为 `scripts/run_qwen25_specbench_target_only.sh`。

该 baseline 是“仅边缘基线”: 运行时只加载 `--target-model`, 不加载三台端侧 draft model。三台虚拟设备仍用于分配请求和采样网络 profile, 但不进行端侧起草、draft 上传或 verifier lane 调度。

## 3. 动态网络 replay 与公平对比

实验网络不是固定延迟。每次上行或下行传输都会基于 profile 中的基准带宽、RTT、jitter 和拥塞概率生成一个动态网络状态。为了避免 target_only、sync_batch 和 proposed 分开运行时因为网络随机性不同而影响结论, 当前实现使用 seed 驱动的 network trace replay。

网络状态由以下四个量决定:

```text
NETWORK_SEED + device_id + direction + virtual_time_slot
```

其中 `direction` 为 `uplink` 或 `downlink`, `virtual_time_slot = floor(virtual_time_s / NETWORK_TRACE_SLOT_S)`。因此, 在同一个 `NETWORK_SEED` 和 `NETWORK_TRACE_SLOT_S` 下, 三种方法会暴露在同一条外部动态网络轨迹中。方法本身导致请求发生在不同虚拟时间时, 会自然落到不同 time slot; 这不是误差, 而是 replay 语义下方法与动态网络交互的结果。

正式对比应使用配对实验: 每个 category 和每个 seed 下, target_only、sync_batch、proposed 使用相同的 `SEED`, `NETWORK_SEED`, `NETWORK_TRACE_SLOT_S`, 数据选择参数和生成参数。三种方法可以分开运行, 不要求在同一个进程内完成。

示例:

```bash
SEED=42 NETWORK_SEED=42 NETWORK_TRACE_SLOT_S=0.05 CATEGORY=MT bash scripts/run_qwen25_specbench_target_only.sh
SEED=42 NETWORK_SEED=42 NETWORK_TRACE_SLOT_S=0.05 CATEGORY=MT bash scripts/run_qwen25_specbench.sh
SEED=42 NETWORK_SEED=42 NETWORK_TRACE_SLOT_S=0.05 CATEGORY=MT bash scripts/run_qwen25_specbench_async.sh
```

报告性能结论时建议使用多个 seed 做配对统计, 例如 `SEED=42,43,44,45,46`。target_only 不再内嵌在 proposed 或 sync_batch 中; 需要 AR baseline 时, 使用单独的 `--method target_only` 结果作为公共 baseline。

## 4. 脚本与结果目录

正式脚本保持原名, 但结果目录按方法名组织:

- `bash scripts/run_qwen25_specbench_async.sh`: proposed, 默认输出 `results/proposed/<CATEGORY>/`。
- `bash scripts/run_qwen25_specbench.sh`: sync_batch, 默认输出 `results/sync_batch/<CATEGORY>/`。
- `bash scripts/run_qwen25_specbench_target_only.sh`: target_only, 默认输出 `results/target_only/<CATEGORY>/`。

常用环境变量包括 `CATEGORY`, `DATASET_MODE`, `RESULTS_DIR`, `SEED`, `NETWORK_SEED`, `NETWORK_TRACE_SLOT_S`, `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `TARGET_MODEL`, `SERVER_DEVICE`, `TORCH_DTYPE`。proposed 和 sync_batch 还支持 `DRAFT_MODEL_0/1/2`, `CLIENT_DEVICE`, `GAMMA`; proposed 额外支持 `INITIAL_LOOKAHEAD`, `LANE_COUNT`, `MAX_INFLIGHT_SEGMENTS`, `LOOKAHEAD_POLICY`, `SCHEDULER`, `LANE_BATCH_SIZE`, `LANE_BATCH_TIMEOUT_S`。

`NETWORK_SEED` 默认等于 `SEED`。公平对比的网络 replay 语义见第 3 节。

输出文件固定为:

- `request_records.jsonl`
- `event_trace.jsonl`
- `summary.json`

## 5. 输出与时间语义

- `total_virtual_time_s`: 整次实验虚拟完成时间。
- `request_records.jsonl`: 请求级结果。
- `event_trace.jsonl`: sync_batch 的同步轮次或 proposed 的 lane 验证事件; target_only 下为空。
- `summary.json`: 汇总吞吐、延迟、接受率、网络和 lane 指标。

Proposed 方法没有全局 barrier, 因此 `barrier_wait_s` 固定为 0; 额外记录 `lane_id`, `lane_queue_wait_s`, `lane_start_s`, `lane_finish_s`, lane busy time 和 utilization。target_only 没有 speculative acceptance、barrier 或 lane 指标, 这些字段为 0、空对象或 null。
