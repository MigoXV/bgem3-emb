import asyncio
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from bgem3_emb.config import Config
from bgem3_emb.engine import Busy, Engine, Unavailable


class FakeRuntime:
    model_id = "test-model"

    def __init__(self, config):
        self.calls = 0
        self.gate = Event()
        self.started = Event()
        self.closed = False
        self.fail = False

    def encode(self, texts, limit, truncate):
        self.calls += 1
        if self.calls > 1:
            self.started.set()
            if not self.gate.wait(3):
                raise RuntimeError("test worker stalled")
        if self.fail:
            raise RuntimeError("injected device failure")
        if not truncate and "long" in texts:
            raise ValueError("too long")
        rows = np.zeros((len(texts), 1024), dtype=np.float32)
        rows[:, 0] = 1
        return SimpleNamespace(
            vectors=rows,
            token_counts=[3] * len(texts),
            truncated=[False] * len(texts),
            batch_sizes=[len(texts)],
            elapsed_ms=1,
        )

    def close(self):
        self.closed = True


async def busy_engine():
    engine = Engine(Config(model_dir=Path("."), queue_capacity=1), FakeRuntime)
    await engine.start()
    first = asyncio.create_task(engine.encode(["first"]))
    assert await asyncio.to_thread(engine.runtime.started.wait, 1)
    return engine, first


def test_configuration_rejects_implicit_device_and_cpu_low_precision():
    for values in (
        {"device": "cuda"},
        {"device": "cpu", "dtype": "bfloat16"},
        {"queue_bytes": 1024, "request_bytes": 2048},
    ):
        with pytest.raises(ValidationError):
            Config(model_dir=Path("."), **values)


async def test_capacity_cancel_and_fifo_do_not_leak_results():
    engine, first = await busy_engine()
    second = asyncio.create_task(engine.encode(["second"]))
    await asyncio.sleep(0)
    try:
        with pytest.raises(Busy):
            await engine.encode(["overflow"])
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        engine.runtime.gate.set()
        result, queue_ms = await second
        assert result.vectors.shape == (1, 1024)
        assert queue_ms >= 0
        assert engine.stats["cancelled"] == 1
        assert engine.stats["completed"] == 1
        assert engine.stats["rejected"] == 1
        assert engine.queued_bytes == 0
    finally:
        engine.runtime.gate.set()
        runtime = engine.runtime
        await engine.close()
    assert runtime.closed and not engine.pending and engine.worker is None


async def test_expired_queued_request_is_not_executed():
    engine, first = await busy_engine()
    try:
        with pytest.raises(TimeoutError):
            await engine.encode(["expired"], timeout=0.001)
        engine.runtime.gate.set()
        await first
        await engine.queue.join()
        assert engine.runtime.calls == 2  # warmup + first，过期项未进入 runtime
    finally:
        engine.runtime.gate.set()
        await engine.close()


async def test_runtime_failure_fails_pending_and_readiness():
    engine, first = await busy_engine()
    engine.runtime.fail = True
    second = asyncio.create_task(engine.encode(["queued"]))
    await asyncio.sleep(0)
    engine.runtime.gate.set()
    for request in (first, second):
        with pytest.raises(Unavailable):
            await request
    assert not engine.ready
    with pytest.raises(Unavailable):
        await engine.encode(["late"])
    await engine.close()


async def test_invalid_payload_and_explicit_truncation_do_not_poison_worker():
    engine = Engine(Config(model_dir=Path(".")), FakeRuntime)
    await engine.start()
    engine.runtime.gate.set()
    try:
        for texts, limit in (([], 32), ([" "], 32), (["a"], 8192), (["a" * 32769], 32)):
            with pytest.raises(ValueError):
                await engine.encode(texts, max_tokens=limit)
        with pytest.raises(ValueError):
            await engine.encode(["long"])
        await engine.encode(["long"], truncate=True)
        assert engine.ready
    finally:
        await engine.close()


async def test_loading_failure_releases_executor():
    def fail(config):
        raise ValueError("bad model")

    engine = Engine(Config(model_dir=Path(".")), fail)
    with pytest.raises(ValueError):
        await engine.start()
    await engine.close()
    assert not engine.ready
