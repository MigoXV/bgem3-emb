"""真实 gRPC 的对齐、顺序、截断和并发冒烟；不记录输入正文或向量。"""

import asyncio
import json
from pathlib import Path
from time import perf_counter

import grpc
import numpy as np
import onnxruntime as ort
import typer
from tokenizers import Tokenizer

from bgem3_emb.proto import embedding_pb2 as pb
from bgem3_emb.proto import embedding_pb2_grpc as rpc

app = typer.Typer()


@app.command()
def main(
    target: str = typer.Option("127.0.0.1:50054", envvar="BGE_TARGET"),
    model_dir: Path = typer.Option(
        Path("/workspace/model-bin/BAAI/bge-m3"), envvar="BGE_MODEL_DIR"
    ),
    output: Path = typer.Option(Path("reports/service.json")),
):
    asyncio.run(check(target, model_dir, output))


async def check(target, model_dir, output):
    texts = [
        "周五交货。",
        "取消旧的采购安排，改为下周一交付计算设备。",
        "报价是否含税尚未确认，不能据此批准采购。",
        "回滚应用版本，不回滚客户已经成功写入的数据。",
        "顾珊去核原工单，今天别猜。",
        "小李负责回归测试，小王只负责发布。",
        "陈宁提交报告。" * 120,
        "明天开始回归测试，测试尚未执行。",
    ]
    tok = Tokenizer.from_file(str(model_dir / "onnx/tokenizer.json"))
    tok.enable_truncation(max_length=512)
    enc = tok.encode_batch(texts)
    width = max(len(e.ids) for e in enc)
    ids = np.full((len(enc), width), 1, dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, item in enumerate(enc):
        ids[i, : len(item.ids)] = item.ids
        mask[i, : len(item.ids)] = 1
    session = ort.InferenceSession(
        str(model_dir / "onnx/model.onnx"), providers=["CPUExecutionProvider"]
    )
    reference = session.run(["sentence_embedding"], {"input_ids": ids, "attention_mask": mask})[0]
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    del session
    async with grpc.aio.insecure_channel(target) as channel:
        client = rpc.EmbeddingServiceStub(channel)
        info = await client.Metadata(pb.MetadataRequest(), timeout=5)
        assert info.ready
        result = await client.Encode(pb.EncodeRequest(texts=texts, truncate=True), timeout=30)
        out = np.asarray([r.values for r in result.embeddings], dtype=np.float32)
        cosine = np.sum(out * reference, axis=1)
        delta = float(np.max(np.abs(out @ out.T - reference @ reference.T)))
        print(
            json.dumps({"alignment_cosines": cosine.tolist(), "similarity_delta": delta}),
            flush=True,
        )
        if float(cosine.min()) < 0.999 or delta > 0.01:
            output.parent.mkdir(exist_ok=True, parents=True)
            output.write_text(
                json.dumps(
                    {
                        "accepted": False,
                        "dtype": info.dtype,
                        "model_id": info.model_id,
                        "generation": info.generation,
                        "revision": info.revision,
                        "cosines": cosine.tolist(),
                        "max_similarity_delta": delta,
                        "min_cosine_required": 0.999,
                        "max_similarity_delta_allowed": 0.01,
                    },
                    indent=2,
                )
                + "\n"
            )
            raise typer.Exit(1)
        assert result.embeddings[6].truncated
        try:
            await client.Encode(pb.EncodeRequest(texts=[texts[6]]), timeout=10)
        except grpc.aio.AioRpcError as exc:
            assert exc.code() == grpc.StatusCode.INVALID_ARGUMENT
        else:
            raise AssertionError("implicit truncation accepted")
        rows = []
        request = pb.EncodeRequest(texts=texts[:4], truncate=True)
        for _ in range(3):
            await client.Encode(request, timeout=30)
        for concurrency in (1, 2, 4, 8, 32):
            latencies = []
            errors = {}

            async def one():
                started = perf_counter()
                try:
                    response = await client.Encode(request, timeout=30)
                    assert response.generation == info.generation
                    latencies.append((perf_counter() - started) * 1000)
                except grpc.aio.AioRpcError as exc:
                    errors[exc.code().name] = errors.get(exc.code().name, 0) + 1

            started = perf_counter()
            await asyncio.gather(*(one() for _ in range(concurrency)))
            seconds = perf_counter() - started
            rows.append(
                {
                    "concurrency": concurrency,
                    "runs_ms": latencies,
                    "errors": errors,
                    "p50_ms": float(np.median(latencies)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "p99_ms": float(np.percentile(latencies, 99)),
                    "throughput_texts_s": len(latencies) * 4 / seconds,
                }
            )
        final = await client.Metadata(pb.MetadataRequest(), timeout=5)
        assert final.ready and not final.active and final.queued == 0
    report = {
        "accepted": True,
        "min_cosine": float(cosine.min()),
        "max_similarity_delta": delta,
        "device": info.device,
        "dtype": info.dtype,
        "model_id": info.model_id,
        "generation": info.generation,
        "revision": info.revision,
        "concurrency": rows,
        "truncation_rejected_without_flag": True,
        "stats": {
            key: getattr(final, key)
            for key in (
                "requests",
                "completed",
                "rejected",
                "failed",
                "cancelled",
                "expired",
                "batches",
                "texts",
            )
        },
    }
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    app()
