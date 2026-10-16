"""服务组合根与信号关闭；模型只存在于 Engine 的唯一 Worker 中。"""

import asyncio
import logging
import os
import signal
import subprocess

import grpc

from bgem3_emb.engine import Engine
from bgem3_emb.proto import embedding_pb2_grpc as rpc
from bgem3_emb.servicer import Servicer

logger = logging.getLogger(__name__)


def revision():
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return (result.stdout.strip() or "uncommitted") + ("+dirty" if status.stdout else "")


async def serve(config, host, port):
    engine = Engine(config)
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", config.request_bytes + 65536),
            ("grpc.max_send_message_length", 4 * 1024 * 1024),
        ],
        maximum_concurrent_rpcs=config.queue_capacity + 4,
    )
    rpc.add_EmbeddingServiceServicer_to_server(Servicer(engine, revision()), server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if not bound:
        await engine.close()
        raise RuntimeError("端口不可用")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        await engine.start()
        await server.start()
        logger.info(
            "ready address=%s:%s pid=%s generation=%s revision=%s model=%s device=%s dtype=%s runner=eager",
            host,
            bound,
            os.getpid(),
            engine.generation,
            revision(),
            engine.runtime.model_id,
            config.device,
            config.dtype,
        )
        await stop.wait()
    finally:
        engine.ready = False
        await server.stop(grace=0)
        try:
            await asyncio.wait_for(engine.close(), config.shutdown_seconds)
        except asyncio.TimeoutError:
            # 同步 CUDA kernel 无法通过 Python 线程取消；退出进程交还设备资源。
            logger.critical("shutdown exceeded grace; terminating process")
            os._exit(1)
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)
        logger.info("stopped")
