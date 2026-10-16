"""有界 FIFO 调度；一个 executor 线程独占模型和设备。请求不持久化。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic
from uuid import uuid4

from bgem3_emb.config import Config


class Busy(RuntimeError):
    pass


class Unavailable(RuntimeError):
    pass


@dataclass
class Job:
    texts: list[str]
    max_tokens: int
    truncate: bool
    deadline: float
    received: float
    size: int
    future: asyncio.Future


class Engine:
    def __init__(self, config: Config, factory=None):
        self.config = config
        self.factory = factory
        self.generation = uuid4().hex
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bge-device")
        self.queue = asyncio.Queue(config.queue_capacity)
        self.ready = False
        self.runtime = None
        self.worker = None
        self.active = 0
        self.queued_bytes = 0
        self.pending = set()
        self.stats = dict.fromkeys(
            (
                "requests",
                "completed",
                "rejected",
                "cancelled",
                "failed",
                "expired",
                "batches",
                "texts",
            ),
            0,
        )
        self.batch_sizes = {}
        self.last_queue_ms = self.last_inference_ms = 0.0

    async def start(self):
        if self.factory is None:
            from bgem3_emb.runtime import Runtime

            self.factory = Runtime
        loop = asyncio.get_running_loop()
        try:
            self.runtime = await loop.run_in_executor(self.executor, self.factory, self.config)
            await loop.run_in_executor(
                self.executor,
                self.runtime.encode,
                ["向量引擎预热。"],
                32,
                False,
            )
        except BaseException:
            if self.runtime:
                await loop.run_in_executor(self.executor, self.runtime.close)
                self.runtime = None
            self.executor.shutdown(wait=True)
            raise
        self.worker = asyncio.create_task(self._run(), name="embedding-queue")
        self.ready = True

    async def encode(self, texts, max_tokens=0, truncate=False, timeout=None):
        if not self.ready:
            raise Unavailable("engine not ready")
        self.stats["requests"] += 1
        try:
            limit = max_tokens or self.config.max_tokens
            if not texts or len(texts) > self.config.max_texts:
                raise ValueError("invalid number of texts")
            if not 8 <= limit <= self.config.max_tokens:
                raise ValueError("invalid max_tokens")
            if any(not x.strip() or len(x) > self.config.max_text_chars for x in texts):
                raise ValueError("empty or oversized text")
            size = sum(len(x.encode("utf-8")) for x in texts)
            if size > self.config.request_bytes:
                raise ValueError("request bytes exceeded")
        except ValueError:
            self.stats["failed"] += 1
            raise
        if self.queue.full() or self.queued_bytes + size > self.config.queue_bytes:
            self.stats["rejected"] += 1
            raise Busy("embedding queue full")
        now = monotonic()
        ttl = min(
            timeout if timeout is not None else self.config.request_timeout,
            self.config.request_timeout,
        )
        if ttl <= 0:
            self.stats["expired"] += 1
            raise TimeoutError("deadline exceeded")
        future = asyncio.get_running_loop().create_future()
        job = Job(list(texts), limit, truncate, now + ttl, now, size, future)
        self.queue.put_nowait(job)
        self.queued_bytes += size
        self.pending.add(future)
        try:
            return await asyncio.wait_for(asyncio.shield(future), ttl)
        except asyncio.CancelledError:
            self.stats["cancelled"] += 1
            future.cancel()
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            self.stats["expired"] += 1
            future.cancel()
            raise TimeoutError("embedding deadline exceeded") from exc
        finally:
            self.pending.discard(future)
            if future.done() and not future.cancelled():
                future.exception()  # 取消与结果同时到达时也回收异常。

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            job = await self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            self.queued_bytes -= job.size
            try:
                if job.future.done():
                    continue
                if monotonic() >= job.deadline:
                    job.future.set_exception(TimeoutError("queue deadline exceeded"))
                    continue
                self.active = 1
                queue_ms = (monotonic() - job.received) * 1000
                result = await loop.run_in_executor(
                    self.executor,
                    self.runtime.encode,
                    job.texts,
                    job.max_tokens,
                    job.truncate,
                )
                self.stats["batches"] += len(result.batch_sizes)
                self.stats["texts"] += len(job.texts)
                for size in result.batch_sizes:
                    self.batch_sizes[size] = self.batch_sizes.get(size, 0) + 1
                self.last_queue_ms, self.last_inference_ms = queue_ms, result.elapsed_ms
                if not job.future.done():
                    if monotonic() < job.deadline:
                        self.stats["completed"] += 1
                        job.future.set_result((result, queue_ms))
                    else:
                        job.future.set_exception(TimeoutError("inference deadline exceeded"))
            except ValueError as exc:
                self.stats["failed"] += 1
                if not job.future.done():
                    job.future.set_exception(exc)
            except Exception:
                self.stats["failed"] += 1
                # Runtime/设备故障不能让后续请求排队等待失效模型。
                self.ready = False
                if not job.future.done():
                    job.future.set_exception(Unavailable("runtime failed"))
                for future in list(self.pending):
                    if not future.done():
                        future.set_exception(Unavailable("runtime failed"))
                import logging

                logging.getLogger(__name__).exception("embedding runtime failed")
            finally:
                self.active = 0
                self.queue.task_done()

    async def close(self):
        self.ready = False
        for future in list(self.pending):
            if not future.done():
                future.set_exception(Unavailable("engine stopping"))
        if self.worker:
            # 已开始的 kernel 不能硬取消；等待当前工作完成后在同一线程释放设备。
            await self.queue.put(None)
            await self.worker
            self.worker = None
        if self.runtime:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.runtime.close)
            self.runtime = None
        self.executor.shutdown(wait=True)
