# Edge SD Simulator Workspace

本工作区按用途分成两个顶层目录：

- `code/`: 仿真器代码、配置、数据、运行脚本、测试和代码仓库文档。
- `paper/`: 论文写作材料、投稿材料、论文图表和参考 PDF。

代码相关操作先进入 `code/`：

```bash
cd code
pip install -r requirements.txt
bash scripts/run.sh smoke
python3 -m unittest discover -s tests -v
```

常用入口：

- [代码说明](code/README.md)
- [代码文档维护规则](code/docs/README.md)
- [实验协议](code/docs/experiment.md)
- [指标定义](code/docs/metric.md)
- [论文材料目录](paper/README.md)

当前默认实验口径为 steady-state autoregressive decoding：`simulation.include_prefill`
默认为 `false`，不统计 prompt prefill、prompt 传输或 TTFT；主指标为请求完成时延、
TPOT/TBT、goodput 和相对基线加速比。
