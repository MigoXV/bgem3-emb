import asyncio
from pathlib import Path

import grpc
import pytest

from bgem3_emb.config import Config
from bgem3_emb.engine import Engine
from bgem3_emb.proto import embedding_pb2 as pb
from bgem3_emb.proto import embedding_pb2_grpc as rpc
from bgem3_emb.servicer import Servicer
from tests.test_engine import FakeRuntime


async def test_grpc_order_health_invalid_deadline_and_shutdown():
    engine = Engine(Config(model_dir=Path(".")), FakeRuntime)
    await engine.start()
    server = grpc.aio.server()
    rpc.add_EmbeddingServiceServicer_to_server(Servicer(engine), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = rpc.EmbeddingServiceStub(channel)
            job = client.Encode(pb.EncodeRequest(texts=["one", "two"]), timeout=1)
            await asyncio.sleep(0.01)
            # Worker 同步执行时，事件循环仍能响应 Metadata。
            metadata = await client.Metadata(pb.MetadataRequest(), timeout=1)
            assert metadata.ready and metadata.active == 1
            engine.runtime.gate.set()
            result = await job
            assert len(result.embeddings) == 2 and result.model_id == "test-model"
            with pytest.raises(grpc.aio.AioRpcError) as invalid:
                await client.Encode(pb.EncodeRequest())
            assert invalid.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            engine.runtime.gate.clear()
            with pytest.raises(grpc.aio.AioRpcError) as expired:
                await client.Encode(pb.EncodeRequest(texts=["late"]), timeout=0.01)
            assert expired.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
            engine.runtime.gate.set()
            await engine.close()
            with pytest.raises(grpc.aio.AioRpcError) as unavailable:
                await client.Encode(pb.EncodeRequest(texts=["stopped"]))
            assert unavailable.value.code() == grpc.StatusCode.UNAVAILABLE
    finally:
        if engine.runtime:
            engine.runtime.gate.set()
        await server.stop(0)
        await engine.close()
