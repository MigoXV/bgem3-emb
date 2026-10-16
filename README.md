# BGE-M3 向量引擎

独立的 PyTorch gRPC 稠密向量服务。业务侧只安装轻量客户端，不加载 ONNX、Torch 或模型权重。模型本地加载、CLS pooling、FP32 输出及 L2 归一化与旧 `BgeM3EmbeddingRouter` 对齐。仅提供稠密向量；分类原型、证据窗口、聚类和事实修订属于业务层。

## 安装与启动

使用 Python 3.10 和 Poetry。现有环境使用 PyTorch 2.8.0 + CUDA 12.8；部署时先按设备安装对应 Torch wheel，再安装项目依赖。权重不提交 Git，不打入代码包，也不在服务启动时下载。

```bash
poetry env use python3.10
poetry install
poetry run bgem3-emb serve \
  --model-dir /workspace/model-bin/BAAI/bge-m3 \
  --device cuda:3 --dtype bfloat16 --port 50054
```

CLI 安全默认值为 CPU、FP32，GPU 必须显式选择。`--model-dir` 必填；支持 `BGE_MODEL_DIR`、`BGE_DEVICE`、`BGE_DTYPE`、`BGE_HOST`、`BGE_PORT`、`BGE_MAX_TOKENS`、`BGE_BATCH_SIZE`、`BGE_MAX_TEXTS`、`BGE_QUEUE_CAPACITY`。默认单批 8 条、每次 RPC 最多 128 条、512 token、队列 16 个请求，另有总排队字节和单请求字节限制。8192 是模型上限，长序列显存和批大小须单独验证。

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

首次十轮对照（Xeon Gold 6326 / A100 40GB）：8×512 token 为 CPU 约 1627ms、GPU FP32 约 205ms、GPU BF16 约 40ms；最终报告以重新运行的原始数据为准。验收容差在测量前固定：GPU FP32 与 ONNX 最小余弦相似度 0.99999，BF16 为 0.999，成对相似度最大变化 0.01。批量及不同精度通过容差后再用于业务。
