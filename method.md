# Method：面向异构端边场景的异步推测验证框架

本文提出一种面向异构端边推理场景的异步推测验证框架。核心思想是将 Distributed Speculative Inference（DSI）中的 speculation parallelism 从单机多处理器环境迁移到端边协同网络：端侧设备独立、异步地生成 draft segments，边缘侧不再等待全局批次形成，而是将异步到达的草稿片段注入多个非阻塞验证通道中进行验证。该设计旨在缓解集中式批量验证中的同步等待、慢设备拖累和异构链路导致的验证资源利用不足问题。

## 1. System Overview

系统由多个端侧设备和一个边缘验证服务器组成。每个端侧设备部署轻量级 drafter model，负责本地生成 draft tokens；边缘服务器部署 target model，负责对端侧上传的 draft segments 进行 lossless verification。与传统同步式 speculative decoding 不同，端侧设备在上传草稿后无需阻塞等待验证结果，而是可以继续基于当前 speculative prefix 生成后续草稿。

边缘侧包含四个核心组件：

1. **Async Ingress Queue**：接收不同端侧设备异步上传的 draft segments。
2. **Prefix-State Manager**：维护每个请求的 committed prefix、prefix version、验证进度和 KV cache 位置信息。
3. **Verifier-Lane Pool**：由多条非阻塞验证通道组成，每条通道绑定局部验证队列、局部 micro-batch buffer 和局部 KV cache。
4. **Async Result Dispatcher**：将验证结果异步返回端侧设备，用于确认接受、触发回滚或同步最新前缀。

整体流程如下：

```text
Device_i generates draft segment
        ↓
Device_i asynchronously sends draft segment to edge
        ↓
Edge checks prefix version and prefix hash
        ↓
Scheduler assigns the segment to a verifier lane
        ↓
Lane performs local micro-batched target verification
        ↓
Edge asynchronously returns accepted length or rejection signal
        ↓
Device_i commits accepted tokens or rolls back stale drafts
```

## 2. Asynchronous Drafting on Edge Devices

每个端侧设备维护两个逻辑线程：draft thread 和 receive thread。draft thread 持续生成草稿片段并异步上传；receive thread 负责接收边缘侧返回的验证结果并更新本地前缀状态。

端侧上传的草稿片段表示为：

```text
D_i = (request_id, device_id, prefix_version, base_position,
       draft_tokens, draft_logits, timestamp)
```

其中，`prefix_version` 表示该片段基于哪个已确认前缀生成，`base_position` 表示该片段起始位置，`draft_tokens` 是端侧生成的候选 token 序列，`draft_logits` 用于 lossless verification 或接受率估计。端侧发送 `D_i` 后不等待验证完成，而是继续生成后续 speculative tokens，从而实现通信与本地起草的重叠。

该机制将传统的阻塞式流程：

```text
draft → upload → wait for verification → draft next segment
```

改为非阻塞式流程：

```text
draft → upload → continue drafting
              ↘ receive verification result asynchronously
```

## 3. Heterogeneity-Aware Adaptive Lookahead

在异构端边环境中，不同设备的计算能力、上行带宽、RTT 和 draft accuracy 可能差异很大。因此，本文不采用统一固定的 draft length，而是为每个端侧设备动态选择 draft chunk length，即自适应 lookahead。

对设备 `i`，其草稿长度定义为：

```text
gamma_i = f(C_i, B_i, R_i, A_i, Q_e)
```

其中，`C_i` 表示端侧计算能力，`B_i` 表示上行带宽，`R_i` 表示网络往返时延，`A_i` 表示历史接受率，`Q_e` 表示边缘验证队列负载。

调节原则如下：

- 当端侧计算快、链路稳定、接受率高时，减小 `gamma_i`，以降低 rejection 后的无效起草代价。
- 当端侧计算慢、链路差或 RTT 较高时，增大 `gamma_i`，以减少频繁上传带来的通信开销。
- 当边缘验证端拥塞时，增大 `gamma_i`，降低验证任务注入频率。
- 当历史接受率下降时，减小 `gamma_i`，避免在错误前缀上生成过多无效草稿。

该机制对应于将 DSI 中用于匹配 target server 并行度的 lookahead 扩展为端边异构环境下的设备级动态草稿长度控制。

## 4. Prefix-State Management

由于端侧设备会在验证结果返回前继续起草，系统必须处理 stale prefix 问题。本文为每个请求维护一个前缀状态表：

```text
RequestState[q] = {
    committed_prefix,
    committed_position,
    prefix_version,
    prefix_hash,
    accepted_length_history,
    rejected_position_history,
    kv_cache_locations
}
```

边缘服务器收到 draft segment 后，首先执行前缀一致性检查：

```text
if D_i.prefix_version < RequestState[q].prefix_version:
    discard D_i or return stale-prefix signal

if D_i.prefix_hash != RequestState[q].prefix_hash:
    reject D_i and request resynchronization

otherwise:
    inject D_i into a verifier lane
```

该设计将 DSI 中的 descendant termination 机制迁移为端边异步网络中的 prefix-version invalidation。也就是说，当某个 token 被 target model 拒绝后，系统不会回滚整条验证通道，而是更新对应请求的 prefix version，并使所有依赖旧前缀版本的 pending segments 失效。

## 5. Verifier-Lane Pool

边缘服务器被组织为一个 verifier-lane pool：

```text
VerifierPool = {Lane_1, Lane_2, ..., Lane_P}
```

每条验证通道包含：

```text
Lane_p = {
    target_worker,
    local_queue,
    micro_batch_buffer,
    local_kv_cache,
    load_statistics
}
```

验证通道不是传统 pipeline parallelism 中的模型层切分，而是 speculation parallelism 在边缘验证端的资源组织形式。每条 lane 可以独立接收草稿片段、形成局部 micro-batch 并执行 target verification。多条 lane 并行工作，使一个请求或设备的延迟不再阻塞其他请求的验证过程。

## 6. Prefix-Aware Asynchronous Injection

为了避免简单“到达即随机分配”导致 KV cache 命中率下降或前缀一致性错误，本文设计 prefix-aware asynchronous injection。调度器根据草稿片段的前缀版本、目标请求、KV cache locality、lane 负载和预计验证时间选择验证通道。

对草稿片段 `D_i` 和验证通道 `Lane_p`，定义调度代价：

```text
Cost(D_i, Lane_p) =
    alpha · QueueDelay(Lane_p)
  + beta  · VerifyLatency(D_i, Lane_p)
  + chi   · KVCacheMissCost(D_i, Lane_p)
  + delta · RollbackRisk(D_i)
```

调度器选择代价最小的通道：

```text
Lane* = argmin_p Cost(D_i, Lane_p)
```

其中，`QueueDelay` 表示当前 lane 的排队时延，`VerifyLatency` 表示 target model 验证该片段的预计开销，`KVCacheMissCost` 表示缺失对应 prefix KV cache 时的重算或迁移成本，`RollbackRisk` 表示由低接受率或过长 draft segment 带来的潜在回滚浪费。

## 7. Lane-Local Micro-Batching

本文不采用全局同步批量验证，而是在每条 verifier lane 内执行局部 micro-batching。这样既保留 GPU 批处理效率，又避免等待所有设备或请求形成统一 batch。

每条 lane 的 micro-batch 触发条件为：

```text
if micro_batch_size >= B:
    verify immediately
elif waiting_time >= tau:
    verify current micro-batch
elif request_deadline is close:
    verify immediately
```

与集中式 batch verification 相比，lane-local micro-batching 具有两个优势：

1. **消除全局同步屏障**：慢设备或迟到草稿不会阻塞其他 lane 的验证。
2. **保留计算效率**：每条 lane 仍然可以通过小批量验证提高 target model 的 GPU 利用率。

因此，该机制不是取消 batching，而是将全局同步 batch 转换为局部异步 micro-batch。

## 8. Rejection Handling and Local Rollback

当 target model 拒绝某个 draft token 时，边缘侧执行请求级局部回滚，而不是通道级回滚。设请求 `q` 的某个草稿片段在位置 `r` 被拒绝，则系统执行：

```text
1. commit target token at position r
2. RequestState[q].prefix_version += 1
3. invalidate all pending segments satisfying:
       same request_id q
       old prefix_version
       base_position >= r
4. return rollback signal to the corresponding device
5. keep other requests and verifier lanes running
```

该机制保证 rejection 只影响同一请求中依赖错误前缀的 speculative branch，而不会阻塞其他请求或其他验证通道。因此，系统在保持 target model 输出一致性的同时，实现了请求级错误隔离。

## 9. Algorithm Summary

整体算法可以概括为：

```text
For each device i:
    while request is unfinished:
        gamma_i ← adaptive_lookahead(device_state, network_state, edge_feedback)
        D_i ← generate gamma_i draft tokens
        asynchronously send D_i to edge
        continue drafting without waiting

At the edge server:
    upon receiving D_i:
        check prefix_version and prefix_hash
        if D_i is stale:
            discard or request resynchronization
        else:
            Lane* ← prefix-aware scheduler(D_i, VerifierPool)
            inject D_i into Lane*

For each verifier lane:
    collect compatible draft segments into local micro-batch
    if triggering condition is met:
        run target verification
        return accepted length or rejection position

Upon verification result:
    if accepted:
        update committed_position and acceptance history
    else:
        update prefix_version
        invalidate stale pending segments
        notify device to rollback and resynchronize
```

## 10. Difference from Centralized Batch Verification

现有端边推测解码框架通常依赖集中式批量验证，即多个端侧设备的草稿需要在边缘侧同步形成 batch 后再统一验证。这类方法虽然提高了 GPU 利用率，但容易受到慢设备、链路抖动和批次形成延迟的影响。

本文方法的关键差异在于：

| 维度 | 集中式批量验证 | 本文方法 |
|---|---|---|
| 通信模式 | 偏同步 | 异步上传与异步返回 |
| 验证触发 | 全局 batch 形成后触发 | draft 到达后注入 lane |
| 批处理方式 | 全局 batch | lane-local micro-batch |
| 慢设备影响 | 可能拖慢 batch 形成 | 主要影响自身请求 |
| 验证资源组织 | 单一集中 verifier | 多条非阻塞 verifier lanes |
| 回滚粒度 | 批次级或请求级 | request-prefix branch 级 |
| 异构适配 | 较弱 | 设备级 adaptive lookahead + lane 调度 |

## 11. Relation to DSI

本文方法借鉴 DSI 的核心思想，但不直接复用其单机线程树结构。具体迁移关系如下：

| DSI 机制 | 本文端边迁移 |
|---|---|
| speculation parallelism | 端侧异步起草与边缘非阻塞验证重叠 |
| target server pool | edge verifier-lane pool |
| lookahead | heterogeneity-aware adaptive draft chunk length |
| thread tree | 多请求、多设备 speculative branches |
| terminate descendants | prefix-version invalidation |
| current verifier | Prefix-State Manager 中的 committed frontier |
| per-server KV cache | 每条 verifier lane 的 local KV cache |
| 单机多处理器调度 | 异构端边网络下的异步注入调度 |

因此，本文并非简单将 DSI 部署到边缘侧，而是将其非阻塞 draft-verify 思想推广到异构端边网络，重点解决多设备草稿异步到达、前缀过期、验证队列不均衡、KV cache locality 和局部回滚隔离问题。

## 12. Summary

本文方法可以概括为：端侧设备异步生成并上传 draft segments，边缘侧维护多个非阻塞 verifier lanes，根据前缀版本、队列负载、KV cache locality 和回滚风险将草稿片段注入合适的验证通道；每条通道执行局部 micro-batching 以兼顾异步通信和 GPU 利用率；当发生 rejection 时，系统通过 prefix-version invalidation 仅失效对应请求的 speculative branch，从而避免全局同步等待和慢节点拖累。
