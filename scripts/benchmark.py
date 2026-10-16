"""相同中文输入、相同 tokenizer、同一机器，对照旧 CPU ONNX 和 Torch GPU。只保存性能数值。"""

import gc
import json
import platform
import resource
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
import typer
from tokenizers import Tokenizer

from bgem3_emb.config import Config
from bgem3_emb.runtime import Runtime

app = typer.Typer()


@app.command()
def main(
    model_dir: Path = typer.Option(
        Path("/workspace/model-bin/BAAI/bge-m3"), envvar="BGE_MODEL_DIR"
    ),
    device: str = typer.Option("cuda:3", envvar="BGE_DEVICE"),
    output: Path = typer.Option(Path("reports/cpu-gpu.json")),
    rounds: int = typer.Option(20, min=3),
):
    case_shapes = [(1, 64), (1, 256), (1, 512), (8, 128), (8, 512)]
    tok = Tokenizer.from_file(str(model_dir / "onnx/tokenizer.json"))
    base = "本次会议讨论项目交付进度，张经理负责确认接口方案，研发团队周五前完成联调，测试团队记录问题并跟进风险。"
    cases = []
    for batch, length in case_shapes:
        # 不依赖不同后端的截断：选取实际编码长度接近目标的完整文本。
        values = []
        for i in range(batch):
            value = f"事项{i}：" + base * 30
            while len(tok.encode(value).ids) > length:
                value = value[:-1]
            values.append(value)
        cases.append(values)
    record = {
        "hardware": platform.platform(),
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "workspace_status": subprocess.check_output(["git", "status", "--porcelain"], text=True)
        if Path(".git").exists()
        else "uncommitted",
        "torch": torch.__version__,
        "ort": ort.__version__,
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "device": device,
        "gpu_census": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv"],
            text=True,
        ),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if Path(".git").exists()
        else "uncommitted",
        "rounds": rounds,
        "warmups": 3,
        "cases": [],
        "acceptance": {
            "fp32_min_cosine": 0.99999,
            "bf16_min_cosine": 0.999,
            "bf16_max_similarity_delta": 0.01,
        },
    }
    session = ort.InferenceSession(
        str(model_dir / "onnx/model.onnx"), providers=["CPUExecutionProvider"]
    )
    references = []
    for (batch, length), texts in zip(case_shapes, cases):
        enc = tok.encode_batch(texts)
        width = max(len(e.ids) for e in enc)
        ids = np.full((batch, width), 1, dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, e in enumerate(enc):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        inputs = {"input_ids": ids, "attention_mask": mask}
        values = []
        for i in range(rounds + 3):
            started = perf_counter()
            out = session.run(["sentence_embedding"], inputs)[0]
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
            elapsed = (perf_counter() - started) * 1000
            if i >= 3:
                values.append(elapsed)
        references.append(out)
        record["cases"].append(
            {
                "batch": batch,
                "target_tokens": length,
                "actual_tokens": [len(e.ids) for e in enc],
                "cpu_onnx": summarize(values, batch),
            }
        )
        print(json.dumps(record["cases"][-1]), flush=True)
    del session
    gc.collect()
    for dtype in ("float32", "float16", "bfloat16"):
        started = perf_counter()
        runtime = Runtime(Config(model_dir=model_dir, device=device, dtype=dtype))
        record[dtype + "_model_id"] = runtime.model_id
        record[dtype + "_load_ms"] = (perf_counter() - started) * 1000
        for index, ((batch, length), texts, reference) in enumerate(
            zip(case_shapes, cases, references)
        ):
            values = []
            for i in range(rounds + 3):
                torch.cuda.synchronize(torch.device(device))
                started = perf_counter()
                out = runtime.encode(texts, 512, False).vectors
                torch.cuda.synchronize(torch.device(device))
                elapsed = (perf_counter() - started) * 1000
                if i >= 3:
                    values.append(elapsed)
            cosine = np.sum(out * reference, axis=1)
            delta = float(np.max(np.abs(out @ out.T - reference @ reference.T)))
            row = summarize(values, batch)
            row.update(
                min_cosine=float(cosine.min()),
                max_abs=float(np.max(np.abs(out - reference))),
                max_similarity_delta=delta,
                speedup=record["cases"][index]["cpu_onnx"]["p50_ms"] / row["p50_ms"],
            )
            record["cases"][index][dtype] = row
            print(
                json.dumps({"batch": batch, "target_tokens": length, "dtype": dtype, **row}),
                flush=True,
            )
        record[dtype + "_peak_gpu_mib"] = (
            torch.cuda.max_memory_allocated(torch.device(device)) / 1024**2
        )
        runtime.close()
        del runtime
        gc.collect()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    record["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    aligned = all(
        c["float32"]["min_cosine"] >= 0.99999
        and c["float16"]["min_cosine"] >= 0.999
        and c["float16"]["max_similarity_delta"] <= 0.01
        and c["bfloat16"]["min_cosine"] >= 0.999
        and c["bfloat16"]["max_similarity_delta"] <= 0.01
        for c in record["cases"]
    )
    print(json.dumps({"aligned": aligned, "output": str(output)}), flush=True)
    if not aligned:
        raise typer.Exit(1)


def summarize(values, batch):
    return dict(
        runs_ms=values,
        p50_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        texts_per_s=batch * 1000 / float(np.median(values)),
    )


if __name__ == "__main__":
    app()
