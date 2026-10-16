# BGE-M3 向量引擎

独立的 PyTorch gRPC 稠密向量服务。业务侧只安装轻量客户端，不加载 ONNX、Torch 或模型权重。模型本地加载、CLS pooling、FP32 输出及 L2 归一化与旧 `BgeM3EmbeddingRouter` 对齐。仅提供稠密向量；分类原型、证据窗口、聚类和事实修订属于业务层。

## 安装与启动

使用 Python 3.10 和 Poetry。现有环境使用 PyTorch 2.8.0 + CUDA 12.8；部署时先按设备安装对应 Torch wheel，再安装项目依赖。权重不提交 Git，不打入代码包，也不在服务启动时下载。

```bash
poetry env use python3.10
poetry install
poetry run bgem3-emb serve \
  --model-dir /workspace/model-bin/BAAI/bge-m3 \
  --device cuda:3 --dtype float16 --port 50054
```

CLI 安全默认值为 CPU、FP32，加速设备必须显式选择。支持 `cpu`、`cuda:N`、`npu:N`，不存在的设备报错，不自动回退。`--model-dir` 必填；支持 `BGE_MODEL_DIR`、`BGE_DEVICE`、`BGE_DTYPE`、`BGE_HOST`、`BGE_PORT`、`BGE_MAX_TOKENS`、`BGE_BATCH_SIZE`、`BGE_MAX_TEXTS`、`BGE_QUEUE_CAPACITY`。默认单批 8 条、每次 RPC 最多 128 条、512 token、队列 16 个请求，另有总排队字节和单请求字节限制。8192 是模型上限，长序列显存和批大小须单独验证。

默认只监听 `127.0.0.1:50054`。跨机器部署时显式配置监听地址，并通过受控网络访问。业务客户端的 `MEETNOTE_EMBEDDING_TARGET` 必须指向服务。

## 协议与调用

协议真源：[embedding.proto](src/bgem3_emb/proto/embedding.proto)，包名 `bgem3.v1`。

- `Encode`：输入有序文本数组；输出同序的 FP32、1024 维单位向量、截断前 token 数、截断标记、模型标识、实例 generation、排队和推理耗时。推理耗时是含预处理、数据传输的主机端耗时，不冒充纯 GPU kernel 时间。
- `max_tokens=0` 使用服务配置上限。默认不允许截断；超长输入返回 `INVALID_ARGUMENT`，调用方明确设 `truncate=true` 才截断。截断保留特殊 token。
- `Metadata`：就绪、模型权重与 tokenizer 的联合 SHA256 标识、设备、精度、runner、revision、容量、队列字节、累计完成/失败/取消/过期/拒绝、推理批分布及最近一次耗时。
- `RESOURCE_EXHAUSTED` 表示队列满；`DEADLINE_EXCEEDED` 表示排队或推理超时；`UNAVAILABLE` 表示未就绪、设备故障或关闭。模型故障后停止接受请求，不能静默回退 CPU。

```python
import asyncio
import grpc
from bgem3_emb.proto import embedding_pb2 as pb, embedding_pb2_grpc as rpc

async def main():
    async with grpc.aio.insecure_channel("127.0.0.1:50054") as channel:
        client = rpc.EmbeddingServiceStub(channel)
        info = await client.Metadata(pb.MetadataRequest(), timeout=5)
        assert info.ready
        result = await client.Encode(
            pb.EncodeRequest(texts=["周五交付。", "改为下周一交付。"], max_tokens=512),
            timeout=30,
        )
        print(len(result.embeddings), len(result.embeddings[0].values))

asyncio.run(main())
```

`poetry run python` 运行上面的客户端代码。生成协议：`bash scripts/run_grpcio_tools.sh`；生成工具版本与 grpcio 一致，并生成 `.py` / `.pyi`。MeetNote 中保留同一 proto 的客户端副本，不依赖本项目的重型模型包；更新协议时同步源文件及生成文件。

## 执行与生命周期

调用链为 `Async gRPC -> 有界 FIFO Engine -> 唯一 Worker 线程 -> Runtime -> Torch eager`。gRPC handler 只校验、入队、等待结果，不执行同步推理。一个线程拥有一份模型和 CUDA 上下文；一次只执行一个请求，内部按批大小拆分；不人为等待跨请求组批。

请求 deadline 使用单调时钟，覆盖排队与执行。排队取消项在出队时跳过；已执行 kernel 的请求取消后只丢弃结果，不影响同队列其他请求。这里没有模型会话、向量数据库或服务端文本缓存。请求内容、向量及运行事件不写文件，日志只输出启动及故障元数据。

加载与预热成功后才启动 gRPC。SIGINT/SIGTERM 停止接收、失败 pending 请求、等待当前 kernel 并释放模型和端口。超过 30 秒关闭宽限期时终止进程，避免同步 CUDA 工作留下无法退出的线程；部署系统应据退出码判断失败。

## 测试与性能对照

```bash
poetry run pytest -q
poetry run ruff check src tests scripts
poetry run python scripts/benchmark.py --device cuda:3 --rounds 20
```

基准用同一中文文本、相同 token 长度和批大小，对照旧 FP32 ONNX CPU 与 GPU FP32/BF16；多次预热和多轮实测，记录原始耗时、P50/P95/P99、吞吐、精度对齐及资源信息。`reports/cpu-gpu.json` 只含合成输入的性能数值，不含会议原文或生成结果。CPU 和 GPU 均在同一服务器，差值包含设备及运行时改变。

20 轮对照（Xeon Gold 6326 / A100 40GB，预热 3 次）的中位数：

| 批大小 × token | CPU ONNX FP32 | GPU FP16 | 加速比 |
| --- | ---: | ---: | ---: |
| 1 × 64 | 34.68ms | 12.58ms | 2.76× |
| 1 × 256 | 86.23ms | 12.48ms | 6.91× |
| 1 × 512 | 185.52ms | 12.49ms | 14.85× |
| 8 × 128 | 380.97ms | 14.62ms | 26.06× |
| 8 × 512 | 1626.76ms | 39.18ms | 41.52× |

CPU 数字仅计 ONNX 执行和归一化，GPU 还包含 tokenizer 及数据传输。CPU 有尾延迟波动，原始轮次全部保留，不能仅看加速比推断其他服务器的性能。验收容差在测量前固定：GPU FP32 与 ONNX 最小余弦 0.99999，低精度为 0.999，成对相似度最大变化 0.01。

最终部署精度选择 FP16。BF16 在等长输入矩阵通过，但在混合长度批次中的重复长文本得到最低余弦 0.998998，低于预设 0.999 门槛，因此不作为已验收部署配置；失败数值保留于 `reports/service-bf16-rejected.json`。同样输入 FP16 最低余弦约 0.999977，成对相似度最大变化约 0.00030。`scripts/check_service.py` 进一步验证真实 gRPC、混合 padding、显式截断、并发 1/2/4/8 及过载 32；性能及对齐结果保存为纯数值报告。

模型标识还包含运行精度、CLS pooling 和归一化方式；部署中改变这些语义时，业务缓存不能继续混用旧向量。完整性能矩阵包含 FP32、FP16、BF16，等长矩阵通过不能代替混合长度验收。

## Ascend 910B 部署

硬件依赖按平台安装，不加入所有环境共用的项目依赖。当前 ARM64 环境使用
CANN 8.3.RC1、PyTorch 2.8.0+cpu、torch-npu 2.8.0，版本对应关系见
[昇腾官方兼容矩阵](https://github.com/Ascend/pytorch/tree/v2.8.0#ascend-auxiliary-software)。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
poetry run pip install 'torch-npu==2.8.0' pyyaml setuptools
poetry run bgem3-emb serve \
  --model-dir /home/model-bin/BAAI/bge-m3 \
  --device npu:0 --dtype float16 --batch-size 8 --port 50054
```

`npu:0` 是容器内逻辑编号，即使 `npu-smi` 显示物理卡 4 也不能直接使用
`npu:4`。CPU 版 Torch 通过 torch_npu 扩展调用 NPU，无需 CUDA。
公共 FIFO 调度位于 `engines/scheduler.py`，共享分词、批处理与向量后处理位于
`engines/torch_runtime.py`，平台设备初始化和资源管理分别位于 CPU/CUDA/NPU 实现。
NPU 插件仅在选择 NPU 后端时显式导入。旧 `engine` 和 `runtime` 导入保留兼容入口。
