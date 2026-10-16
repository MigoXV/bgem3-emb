"""独立实例的真实 gRPC 吞吐与生命周期验证；客户端不加载 Torch/ONNX。"""

import asyncio
import json
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import grpc
import numpy as np
import typer
from google.protobuf.json_format import MessageToDict
from tokenizers import Tokenizer

from bgem3_emb.proto import embedding_pb2 as pb
from bgem3_emb.proto import embedding_pb2_grpc as rpc

app = typer.Typer()


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def census():
    return command("npu-smi", "info")


def validate(response, count, info):
    assert response.generation == info.generation
    assert response.model_id == info.model_id
    assert len(response.embeddings) == count
    vectors = np.asarray([e.values for e in response.embeddings], dtype=np.float32)
    assert vectors.shape == (count, 1024) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    return vectors


async def expect_error(client, request, code):
    try:
        await client.Encode(request, timeout=30)
    except grpc.aio.AioRpcError as exc:
        assert exc.code() == code
    else:
        raise AssertionError(f"expected {code}")


async def smoke(client, info):
    texts = ["周五交货。", "取消旧安排，改为下周一交付。", "交付安排需要确认。" * 100]
    request = pb.EncodeRequest(texts=texts + [texts[0]], max_tokens=64, truncate=True)
    result = await client.Encode(request, timeout=30)
    vectors = validate(result, 4, info)
    assert result.embeddings[2].truncated
    assert result.embeddings[2].token_count > 64
    assert not result.embeddings[0].truncated
    assert np.allclose(vectors[0], vectors[3], atol=1e-5)
    # 验证混合长度中的响应顺序，同平台同精度单条结果作参照。
    for index, text in enumerate(texts[:2]):
        single = await client.Encode(pb.EncodeRequest(texts=[text]), timeout=30)
        row = validate(single, 1, info)[0]
        assert float(row @ vectors[index]) >= 0.999
    await expect_error(client, pb.EncodeRequest(), grpc.StatusCode.INVALID_ARGUMENT)
    await expect_error(
        client,
        pb.EncodeRequest(texts=[texts[2]], max_tokens=64),
        grpc.StatusCode.INVALID_ARGUMENT,
    )
    health = await client.Metadata(pb.MetadataRequest(), timeout=5)
    assert health.ready
    return {"mixed_lengths": True, "order": True, "explicit_truncation": True,
            "invalid_argument": True, "finite_unit_vectors": True,
            "cross_backend_alignment": "not_run"}


def make_cases(model_dir):
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    base = "本次会议讨论项目交付进度，研发团队周五前完成联调，测试团队记录问题并跟进风险。"
    cases = []
    for batch in (1, 8):
        for length in (64, 256, 512):
            texts = []
            for i in range(batch):
                text = f"事项{i}：" + base * 30
                while len(tokenizer.encode(text).ids) > length:
                    text = text[:-1]
                texts.append(text)
            cases.append((batch, length, texts))
    return cases


async def measure(client, info, cases, rounds, warmups, rows):
    for batch, length, texts in cases:
        request = pb.EncodeRequest(texts=texts, max_tokens=512)
        for concurrency in (1, 2, 4, 8):
            for _ in range(warmups):
                responses = await asyncio.gather(*(
                    client.Encode(request, timeout=30) for _ in range(concurrency)
                ))
                for response in responses:
                    validate(response, batch, info)
            row = {"batch": batch, "target_tokens": length, "concurrency": concurrency,
                   "rounds": [], "errors": {}, "successful_requests": 0, "tokens": 0}
            rows.append(row)
            for _ in range(rounds):
                async def one():
                    started = perf_counter()
                    try:
                        response = await client.Encode(request, timeout=30)
                        elapsed = perf_counter() - started
                        return response, elapsed, None
                    except grpc.aio.AioRpcError as exc:
                        return None, perf_counter() - started, exc.code().name

                started = perf_counter()
                results = await asyncio.gather(*(one() for _ in range(concurrency)))
                seconds = perf_counter() - started
                samples, tokens, counts = [], 0, []
                inference, queue = [], []
                for response, elapsed, error in results:
                    if error:
                        row["errors"][error] = row["errors"].get(error, 0) + 1
                        continue
                    validate(response, batch, info)
                    counts = [e.token_count for e in response.embeddings]
                    tokens += sum(min(e.token_count, 512) for e in response.embeddings)
                    samples.append(elapsed * 1000)
                    inference.append(response.inference_ms)
                    queue.append(response.queue_ms)
                row["rounds"].append({"wall_seconds": seconds, "latencies_ms": samples,
                                      "inference_ms": inference, "queue_ms": queue,
                                      "tokens": tokens, "actual_tokens": counts})
                row["successful_requests"] += len(samples)
                row["tokens"] += tokens
            wall = sum(r["wall_seconds"] for r in row["rounds"])
            latencies = [x for r in row["rounds"] for x in r["latencies_ms"]]
            row["wall_seconds"] = wall
            row["tokens_per_s"] = row["tokens"] / wall
            row["latency_ms"] = dict(zip(
                ("p50", "p95", "p99"),
                np.percentile(latencies, [50, 95, 99]).tolist() if latencies else [None] * 3,
            ))
            print(json.dumps({k: v for k, v in row.items() if k != "rounds"}), flush=True)
            if row["errors"]:
                raise RuntimeError(f"gRPC performance case failed: {row['errors']}")


async def run_dtype(model_dir, device, dtype, rounds, warmups, cases, record):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    target = f"127.0.0.1:{port}"
    args = [sys.executable, "-m", "bgem3_emb.commands.app", "serve", "--model-dir",
            str(model_dir), "--device", device, "--dtype", dtype, "--port", str(port),
            "--batch-size", "8"]
    record.update(dtype=dtype, target=target, command=args, before=census(), cases=[])
    with tempfile.TemporaryFile(mode="w+") as log:
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        record["pid"] = proc.pid
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                client = rpc.EmbeddingServiceStub(channel)
                info = None
                for _ in range(600):
                    if proc.poll() is not None:
                        raise RuntimeError(f"server exited during startup: {proc.returncode}")
                    try:
                        info = await client.Metadata(pb.MetadataRequest(), timeout=1)
                        if info.ready:
                            break
                    except grpc.aio.AioRpcError:
                        pass
                    await asyncio.sleep(0.5)
                else:
                    raise TimeoutError("server startup timeout")
                assert info.device == device and info.dtype == dtype and info.ready
                record["metadata"] = MessageToDict(info, preserving_proto_field_name=True)
                record["loaded"] = census()
                record["smoke"] = await smoke(client, info)
                await measure(client, info, cases, rounds, warmups, record["cases"])
                final = await client.Metadata(pb.MetadataRequest(), timeout=5)
                assert final.ready and final.active == 0 and final.queued == 0
                record["final_metadata"] = MessageToDict(final, preserving_proto_field_name=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait)
                record["forced_kill"] = True
            record["exit_code"] = proc.returncode
            log.seek(0)
            record["server_log"] = log.read()
            record["after"] = census()
            probe = socket.socket()
            try:
                record["port_released"] = probe.connect_ex(("127.0.0.1", port)) != 0
            finally:
                probe.close()
        assert record["exit_code"] == 0 and record["port_released"]
        assert not record.get("forced_kill")
        record["accepted"] = True


@app.command()
def main(
    model_dir: Path = typer.Option(..., envvar="BGE_MODEL_DIR", exists=True, file_okay=False),
    device: str = typer.Option("npu:0", envvar="BGE_DEVICE"),
    output: Path = typer.Option(Path("reports/npu-grpc.json")),
    rounds: int = typer.Option(20, min=3),
    warmups: int = typer.Option(3, min=1),
    dtypes: str = typer.Option("float32,float16,bfloat16"),
):
    """启动独立服务，经 gRPC 测试后 SIGTERM 关闭；仅测 NPU。"""
    if not device.startswith("npu:"):
        raise typer.BadParameter("此报告仅测 NPU，请选择 npu:N")
    selected = dtypes.split(",")
    if any(x not in ("float32", "float16", "bfloat16") for x in selected):
        raise typer.BadParameter("dtypes 必须为逗号分隔的 float32,float16,bfloat16")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": platform.platform(), "cpu": command("lscpu"),
        "versions": {name: version(name) for name in ("torch", "torch-npu", "transformers", "grpcio")},
        "revision": command("git", "rev-parse", "HEAD"),
        "workspace_status": command("git", "status", "--porcelain"),
        "model_dir": str(model_dir), "device": device, "rounds": rounds, "warmups": warmups,
        "metric": "successful non-padding tokens including special tokens / total gRPC wall seconds; loading, warmups and client validation excluded",
        "cross_backend_alignment": "not_run", "runs": [],
    }
    cases = make_cases(model_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    for dtype in selected:
        record = {}
        report["runs"].append(record)
        try:
            asyncio.run(run_dtype(model_dir, device, dtype, rounds, warmups, cases, record))
        except Exception as exc:
            record["accepted"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"dtype": dtype, "error": record["error"],
                              "server_log": record.get("server_log", "")}), flush=True)
        finally:
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not all(r.get("accepted") for r in report["runs"]):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
