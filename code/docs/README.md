# 文档入口

本目录用于保存代码仓库中需要长期维护的说明文档。文档按“稳定说明”和“实验记录”分层管理，避免把项目介绍、实验设置、指标定义和单次结果分析混在同一个文件中。

文档中的运行命令默认从仓库根目录的 `code/` 目录执行。论文手稿、投稿材料和最终排版材料放在 `paper/` 下。

## 阅读顺序

| 读者目标 | 建议阅读 |
|---|---|
| 快速了解项目和运行命令 | [../README.md](../README.md) |
| 确认实验口径、场景、方法和推荐实验 | [experiment.md](experiment.md) |
| 查询输出文件和指标含义 | [metric.md](metric.md) |
| 查看当前最近一次实验分析 | [latest_experiment_analysis.md](latest_experiment_analysis.md) |
| 追溯历史单次 run 分析 | `experiment_analysis_<RUN_ID>.md` |

## 文档职责

| 文档 | 状态 | 维护规则 |
|---|---|---|
| `../../README.md` | 工作区入口 | 只保留 `code/` 和 `paper/` 的导航，不展开实验论证。 |
| `../README.md` | 代码入口 | 只保留项目定位、快速运行、输出路径和文档导航；避免展开长篇实验论证。 |
| `docs/experiment.md` | 稳定实验协议 | 记录数据集、场景、方法、消融和敏感性分析。只有实验口径变化时才更新。 |
| `docs/metric.md` | 稳定指标字典 | 记录 CSV 字段、统计口径和易混淆指标。新增输出字段时同步更新。 |
| `docs/latest_experiment_analysis.md` | 当前结果摘要 | 指向最近一次值得讨论的 run。若产生新主结果，应整体替换或明确标注旧结果。 |
| `docs/experiment_analysis_<RUN_ID>.md` | 历史结果记录 | 每个 run 一份，文件名必须包含 run id，正文开头写清日期、场景、样本数、方法集合和限制。 |
| `docs/notes/` | 草稿笔记 | 保存早期设计和实验构想，不作为当前实验协议的权威来源。 |

## 写作规则

1. 先给结论，再给依据。每个文档开头应说明“本文档回答什么问题”和“不回答什么问题”。
2. 避免口语化表达。不要使用“差不多”“应该还行”“看起来”等模糊判断；改写为可验证条件、数值或明确假设。
3. 统一术语。使用 `drafter`、`target`、`verify`、`target_only`、`full`、`SpecEdge` 等代码中的名字；第一次出现时解释含义。
4. 单次实验结果不能写成最终结论。涉及 run 的文档必须说明场景、请求数、seed、是否使用 fake runner、缺失的 baseline 和可推广范围。
5. 指标解释放在 `metric.md`，实验流程放在 `experiment.md`，结果解读放在 analysis 文档。不要在多个文档中维护同一段长解释。
6. 新增代码仓库文档时优先放在 `code/docs/` 下。论文手稿和投稿材料放在 `paper/` 下，仓库根目录只保留工作区入口。

## 推荐命名

```text
code/docs/
  README.md
  experiment.md
  metric.md
  latest_experiment_analysis.md
  experiment_analysis_<RUN_ID>.md
  notes/
```

如果后续实验记录继续增多，建议再引入：

```text
code/docs/runs/
  <RUN_ID>.md
code/docs/archive/
  <deprecated-or-draft-doc>.md
```

迁移历史文件时只移动文件位置，不改写实验数值；迁移后在原引用位置补链接，避免破坏论文写作追溯链。
