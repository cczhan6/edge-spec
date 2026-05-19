# 三端设备 + 边缘 7B 推测解码实验

这个项目实现两个可对比方案：

- 同步 baseline：三个端侧异构设备各自处理不同请求，边缘服务器等待三端 draft 到达后，用 `Qwen/Qwen2.5-7B-Instruct` 做统一 batch verification。
- 异步多流水方案：端侧 draft 到达后直接注入边缘侧多条非阻塞验证流水线，按流水线负载选择最早可用的验证资源，不等待全局 batch barrier。

## 环境

```bash
mamba env create -f environment.yml
conda activate edge-spec
python scripts/prepare_specbench.py
```

## 快速自检

不下载真实模型，使用 deterministic fake model 跑完整协议：

```bash
python -m edge_spec.run --use-fake-models --max-new-tokens 16 --gamma 4 --skip-target-baseline
python -m edge_spec.run --mode async --pipeline-count 3 --use-fake-models --max-new-tokens 16 --gamma 4 --skip-target-baseline
```

## 同步 baseline

```bash
bash scripts/run_qwen25_specbench.sh
```

默认脚本使用 `DATASET_MODE=all`：运行全部选中样本，不丢弃最后不足 3 条的尾部样本。

可以随时切换数据集使用方式：

```bash
DATASET_MODE=one-per-category bash scripts/run_qwen25_specbench.sh
DATASET_MODE=all bash scripts/run_qwen25_specbench.sh
```

- `one-per-category`：每类 1 条。
- `all`：跑所有选中样本，不丢弃任何样本；最后一个 microbatch 可以少于 3 条。

同步结果默认写入 `results/sync/`。

## 异步多流水方案

```bash
bash scripts/run_qwen25_specbench_async.sh
```

异步脚本使用 `--mode async --pipeline-count 3`。也可以用环境变量调整流水线数量：

```bash
PIPELINE_COUNT=2 bash scripts/run_qwen25_specbench_async.sh
```

异步模式仍复用同一组端侧 draft 模型、target 模型、数据选择和网络 profile，因此可以和同步 baseline 直接比较 `summary.json` 里的吞吐、延迟、接受率和网络指标。异步 trace 中会额外记录 `pipeline_id`、`pipeline_queue_wait_s`、`pipeline_start_s`、`pipeline_finish_s`，summary 中会额外记录流水线利用率和队列等待。

异步结果默认写入 `results/async/`。两个脚本都支持用 `RESULTS_DIR` 覆盖输出目录：

```bash
RESULTS_DIR=results/sync_run_1 bash scripts/run_qwen25_specbench.sh
RESULTS_DIR=results/async_p2 PIPELINE_COUNT=2 bash scripts/run_qwen25_specbench_async.sh
```

运行时默认显示进度条。同步模式按 microbatch 更新，异步模式按完成请求数更新。直接调用 CLI 时可以用 `--no-progress` 关闭：

```bash
python -m edge_spec.run --mode async --no-progress
```

如果 Hugging Face 权重下载进度长时间不动，通常是代理连接中断。脚本默认设置了
`HF_HUB_DISABLE_XET=1`，中断后重跑即可从已有缓存续传。

当前 `question.jsonl` 应为 480 条、13 个类别。`DATASET_MODE=all` 会运行全部 480 条。

默认模型：

- 端侧 `device-0`: `Qwen/Qwen2.5-0.5B-Instruct`
- 端侧 `device-1`: `Qwen/Qwen2.5-1.5B-Instruct`
- 端侧 `device-2`: `Qwen/Qwen2.5-3B-Instruct`
- 边缘服务器: `Qwen/Qwen2.5-7B-Instruct`

输出：

- `results/sync/specbench_sync_hetero.jsonl`: 同步请求级结果和指标
- `results/sync/round_trace.jsonl`: 同步轮次 trace
- `results/sync/summary.json`: 同步汇总指标
- `results/async/specbench_sync_hetero.jsonl`: 异步请求级结果和指标
- `results/async/round_trace.jsonl`: 异步流水线验证事件 trace
- `results/async/summary.json`: 异步汇总指标

指标和方法流程的中文说明见 `docs/metrics.md`。

端侧 draft 耗时直接使用运行时记录的真实生成时间。每轮 trace 写入
`draft_start_s`、`draft_end_s` 和 `draft_time_s`，其中
`draft_time_s = draft_end_s - draft_start_s`；draft 包到达时间按
`arrival_s = draft_end_s + uplink_s` 计算，不再使用 FLOPs 折算或额外慢速开销指标。

默认 profile 中的网络基准参数相同；网络差异来自每次传输的动态采样，而不是固定绑定到某个设备。网络 profile 支持动态仿真：

- `bandwidth_jitter_ratio`: 每次上下行传输的带宽随机波动比例，例如 `0.35` 表示 `±35%`
- `rtt_jitter_ms`: 每次传输的 RTT 随机波动范围
- `congestion_probability`: 每次传输发生拥塞的概率
- `congestion_slowdown`: 拥塞时带宽除以该倍数，RTT 乘以该倍数

每轮实际采样到的 `uplink_effective_mbps`、`downlink_effective_mbps`、
`uplink_effective_rtt_ms`、`downlink_effective_rtt_ms` 和拥塞标记会写入
`results/sync/round_trace.jsonl` 或 `results/async/round_trace.jsonl`。

## 测试

```bash
python3 -m unittest discover -s tests
```
