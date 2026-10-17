# 17类旁路分类器：现有数据训练与本地恢复

分类器处理飞轮中已归并的问题；在线九类意图仍由对话主链处理。
现有 `dataset/` 为 v1 数据：训练2594、验证161、测试161条。保留原划分和两条同标签训练重复记录。
该数据缺少逐条来源记录；本次test结果属于现有数据回归，不能解释为新建独立盲测。

## 本地训练

在仓库根目录执行（首次需要联网下载基座和依赖）：

```bash
uv sync --frozen --group ml
uv run --group ml python -m scripts.ch10.artifacts
make ch10-train
make ch10-eval
make ch10-export
make classifier-up
curl -fsS http://127.0.0.1:8110/healthz
make ch10-threshold-scan
```

训练前冻结三个分割哈希与标签顺序，只用train更新权重、val选轮次和阈值。
种子在分类头初始化前固定为42；学习率2e-5，batch16，最多8轮，早停耐心2，序列上限128。
阈值应用与线上服务共用“过线多标签，全部不过线取最高分”规则。
设备自动选择CUDA/MPS/CPU；`training_report.json`记录实际设备、依赖、基座revision和代码哈希。
已有模型时训练命令明确失败；需要重新训练时先将旧 `data/ch10/model/` 归档到独立版本目录。

`DATA_DIR=/absolute/dataset make ch10-train` 可覆盖数据目录；评测、导出和阈值扫描支持同一参数。
当前服务恢复要求根目录 `dataset/` 与产物清单一致；其他数据版本应在独立项目副本中部署。

模型权重位于 `data/ch10/model/`，ONNX包位于 `data/ch10/onnx/`，均为本地忽略文件。
包内的 `artifact-manifest.json` 校验权重、tokenizer、阈值和标签顺序。
根目录 `data/ch10/artifact-manifest.json` 是可提交的ONNX发布清单；目前 `artifact_uri=null`，
尚未发布远程下载地址。其他机器可按以上步骤重训，但不保证不同硬件的权重逐字节相同。

`make classifier-up` 先校验再启动，失败返回非零；`make classifier-down` 停止记录的服务进程。
服务监听本地8110，健康接口返回数据/ONNX哈希与阈值。

## 报告和旁路验收

- `data/ch10/dataset-manifest.json`：冻结数据版本、各标签支持数与精确重叠。
- `data/ch10/reports/training_report.json`：训练日志、参数、设备与val选择结果。
- `data/ch10/reports/eval_report.json`：v1 test逐类指标和错误样本。
- `data/ch10/reports/export_report.json`：Torch/ONNX一致性、大小及CPU单条延迟；延迟含预热后20次运行，峰值内存另标未测。
- `data/ch10/reports/history/pre-restoration-0de8e19/`：本次重训前的历史报告。

真实HTTP和旁路幂等验收使用独立MySQL测试库；仅复制现有样本到测试库，业务问题池不增加数据：

```bash
RUN_CLASSIFIER_LIVE=1 uv run --group ml pytest -q tests/test_classifier_live.py
```

日常 `make classify-pool FORCE=1` 会处理业务库中**已归并且未分类**的问题，应在需要实际归类时运行。

## 远程分发（Task 4 Step 4 / Task 5 Step 6）

普通 Git 只提交代码、manifest 和报告；`data/ch10/model/`、`data/ch10/onnx/`（约 391 MB/份）
继续被忽略。`data/ch10/artifact-manifest.json` 的 `artifact_uri` 目前仍为 `null`。
**本仓库尚未配置已授权的对象存储/发布地址**，因此：

- `make classifier-bootstrap` 会读取发布清单；`artifact_uri` 为空时以非零状态明确拒绝，
  不会从公开 GitHub Release 或猜测地址下载，也不会把大文件放进普通 Git。
- 获取存储授权后，把 ONNX 目录打成 zip，确保根目录扁平包含
  `model.onnx`、`tokenizer.json`、`threshold.json`（可选 `tokenizer_config.json`），
  在清单中填写 `artifact_uri`（https 或 file）与 `archive_sha256`，随后运行：

```bash
make classifier-bootstrap
make classifier-up
```

下载器先落临时文件，校验归档 SHA256 与逐文件 SHA256 后原子替换
`data/ch10/onnx/`；目标目录已有完整且一致的本地产物时直接跳过，
已有不完整/不同版本产物时拒绝覆盖。下载失败不会留下半包，原目录保持不变。
空 `data/ch10/onnx/` 加远程下载的可执行验收再用独立临时目录测试；真实空环境启动
在本机完成前属于**阻塞项**，原因是没有已授权存储目标，而不是代码缺失。
