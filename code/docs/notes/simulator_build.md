# 真实语义 + 解析时延事件仿真设计

## 分层

```text
ModelRunner: 真实或 fake 模型 token 语义
Analytical latency: 虚拟设备 token-rate 与通信公式
Simulator: 本地 draft FIFO、target queue、回滚和事件推进
```

ModelRunner 不返回 wall time。Hugging Face drafter 和 target-only 路径启用
KV cache；target verify 通过一次 padded batch forward 返回 greedy correction
或 bonus。

## 设备

设备由 YAML 模板展开。每台设备固定一种 drafter、token rate、startup 和网络
属性。请求按 ID 轮询成为设备本地请求，不做迁移。设备每次只起草一个 segment，
多个请求按 segment FIFO 共享设备容量。

## 异步状态

Request 分别维护设备已收到的 committed tokens 和 edge 已验证的 tokens。
窗口内后续 segment 可基于乐观 draft 链继续生成。拒绝造成 prefix version
变化并作废后续链。Bonus 与下一 segment 首 token 匹配时裁剪并重定位 segment，
否则作废后续链。

## 方法兼容

九种 CLI 方法名保持不变。区别仍由 medium-only 或 heterogeneous 设备池、窗口、
同步 global batch、phase overlap、lane assignment 和 conservative rollback
开关表达。所有方法共享同一模型推理器和解析时延内核。
