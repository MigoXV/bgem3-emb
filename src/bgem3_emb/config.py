"""显式运行配置；CLI 负责环境变量，核心不读取进程环境。"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Config(BaseModel):
    model_dir: Path
    device: str = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    max_tokens: int = Field(default=512, ge=8, le=8192)
    batch_size: int = Field(default=8, ge=1, le=64)
    max_texts: int = Field(default=128, ge=1, le=512)
    max_text_chars: int = Field(default=32768, ge=1)
    queue_capacity: int = Field(default=16, ge=1)
    queue_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    request_bytes: int = Field(default=1024 * 1024, ge=256, le=4 * 1024 * 1024)
    request_timeout: float = Field(default=30, gt=0, le=300)
    shutdown_seconds: float = Field(default=30, gt=0)

    @model_validator(mode="after")
    def validate_runtime(self):
        if self.device != "cpu" and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("device 必须是 cpu 或 cuda:N")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU 仅支持 float32 基线")
        if self.queue_bytes < self.request_bytes:
            raise ValueError("queue_bytes 不能小于 request_bytes")
        return self
