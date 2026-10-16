"""gRPC 边界：转换协议、映射错误；不执行模型。"""

import grpc

from bgem3_emb.engine import Busy, Unavailable
from bgem3_emb.proto import embedding_pb2 as pb
from bgem3_emb.proto import embedding_pb2_grpc as rpc


class Servicer(rpc.EmbeddingServiceServicer):
    def __init__(self, engine, revision="unknown"):
        self.engine, self.revision = engine, revision

    async def Encode(self, request, context):
        try:
            result, queue_ms = await self.engine.encode(
                list(request.texts),
                request.max_tokens,
                request.truncate,
                context.time_remaining(),
            )
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Busy as exc:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        except TimeoutError:
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "embedding deadline exceeded")
        except Unavailable:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "embedding engine unavailable")
        return pb.EncodeResponse(
            embeddings=[
                pb.Embedding(values=row.tolist(), token_count=count, truncated=cut)
                for row, count, cut in zip(result.vectors, result.token_counts, result.truncated)
            ],
            model_id=self.engine.runtime.model_id,
            generation=self.engine.generation,
            queue_ms=queue_ms,
            inference_ms=result.elapsed_ms,
        )

    async def Metadata(self, request, context):
        e, c = self.engine, self.engine.config
        return pb.MetadataResponse(
            ready=e.ready,
            model_id=e.runtime.model_id if e.runtime else "",
            generation=e.generation,
            device=c.device,
            dtype=c.dtype,
            runner="torch-eager",
            dimensions=1024,
            max_tokens=c.max_tokens,
            max_texts=c.max_texts,
            max_batch_size=c.batch_size,
            queue_capacity=c.queue_capacity,
            queued=e.queue.qsize(),
            active=e.active,
            queued_bytes=e.queued_bytes,
            queue_byte_capacity=c.queue_bytes,
            batch_sizes=e.batch_sizes,
            revision=self.revision,
            last_queue_ms=e.last_queue_ms,
            last_inference_ms=e.last_inference_ms,
            **e.stats,
        )
