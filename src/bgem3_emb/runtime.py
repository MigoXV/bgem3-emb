"""唯一 Worker 线程持有模型；本模块不依赖 gRPC 或会议业务。"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from bgem3_emb.config import Config


@dataclass
class Encoded:
    vectors: np.ndarray
    token_counts: list[int]
    truncated: list[bool]
    batch_sizes: list[int]
    elapsed_ms: float


def model_identity(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "tokenizer.json", "pytorch_model.bin", "model.safetensors"):
        file = path / name
        if file.is_file():
            digest.update(name.encode())
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
    return "bge-m3:" + digest.hexdigest()


class Runtime:
    dimensions = 1024

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        if self.device.type == "cuda":
            if not torch.cuda.is_available() or self.device.index >= torch.cuda.device_count():
                raise ValueError("CUDA device 不可用")
            if self.dtype == torch.bfloat16:
                with torch.cuda.device(self.device):
                    if not torch.cuda.is_bf16_supported():
                        raise ValueError("设备不支持 bfloat16")
        if not config.model_dir.is_dir():
            raise ValueError("本地模型目录不存在")
        torch.set_num_threads(4)
        # 固定 eager 作为可对齐基线；不隐式启用编译、TF32 或远程代码。
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model_id = model_identity(config.model_dir) + ":" + config.dtype + ":cls:l2"
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_dir, local_files_only=True)
        self.model = (
            AutoModel.from_pretrained(
                config.model_dir,
                local_files_only=True,
                torch_dtype=self.dtype,
                attn_implementation="eager",
            )
            .eval()
            .to(self.device)
        )
        if self.model.config.hidden_size != self.dimensions:
            raise ValueError("模型不是预期的 1024 维 BGE-M3")

    def encode(self, texts: list[str], max_tokens: int, truncate: bool) -> Encoded:
        started = perf_counter()
        raw = self.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
        counts = [len(ids) for ids in raw]
        if not truncate and any(count > max_tokens for count in counts):
            raise ValueError("input exceeds max_tokens; set truncate explicitly")
        # 保留 CLS/SEP，和旧 ONNX tokenizer 的截断语义保持一致。
        ids = [x if len(x) <= max_tokens else x[: max_tokens - 1] + [x[-1]] for x in raw]
        outputs, sizes = [], []
        with torch.inference_mode():
            for start in range(0, len(ids), self.config.batch_size):
                rows = ids[start : start + self.config.batch_size]
                batch = self.tokenizer.pad(
                    {"input_ids": rows, "attention_mask": [[1] * len(x) for x in rows]},
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state[:, 0].float()
                vectors = torch.nn.functional.normalize(hidden, p=2, dim=1)
                output = vectors.cpu().numpy().copy()
                if not np.isfinite(output).all():
                    raise RuntimeError("non-finite embeddings")
                outputs.append(output)
                sizes.append(len(rows))
        return Encoded(
            np.concatenate(outputs),
            counts,
            [x > max_tokens for x in counts],
            sizes,
            (perf_counter() - started) * 1000,
        )

    def close(self):
        del self.model
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
