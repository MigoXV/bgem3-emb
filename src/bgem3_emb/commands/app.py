import asyncio
import logging
from pathlib import Path

import typer

from bgem3_emb.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
app = typer.Typer(no_args_is_help=True, help="BGE-M3 稠密向量推理引擎")


@app.callback()
def main():
    """模型推理及服务命令。"""


@app.command()
def serve(
    model_dir: Path = typer.Option(..., envvar="BGE_MODEL_DIR", help="本地模型目录，不自动下载"),
    device: str = typer.Option("cpu", envvar="BGE_DEVICE", help="cpu、cuda:N 或 npu:N（容器内逻辑编号）"),
    dtype: str = typer.Option("float32", envvar="BGE_DTYPE"),
    host: str = typer.Option("127.0.0.1", envvar="BGE_HOST"),
    port: int = typer.Option(50054, envvar="BGE_PORT", min=1, max=65535),
    max_tokens: int = typer.Option(512, envvar="BGE_MAX_TOKENS"),
    batch_size: int = typer.Option(8, envvar="BGE_BATCH_SIZE"),
    max_texts: int = typer.Option(128, envvar="BGE_MAX_TEXTS"),
    queue_capacity: int = typer.Option(16, envvar="BGE_QUEUE_CAPACITY"),
):
    """启动单模型、有界队列的异步 gRPC 服务。"""
    from bgem3_emb.server import serve as run

    config = Config(
        model_dir=model_dir,
        device=device,
        dtype=dtype,
        max_tokens=max_tokens,
        batch_size=batch_size,
        max_texts=max_texts,
        queue_capacity=queue_capacity,
    )
    asyncio.run(run(config, host, port))


if __name__ == "__main__":
    app()
