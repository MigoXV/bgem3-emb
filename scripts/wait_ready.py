"""systemd 启动后等待真实 gRPC 就绪。"""
import time
import grpc
from bgem3_emb.proto import embedding_pb2 as pb, embedding_pb2_grpc as rpc
with grpc.insecure_channel('127.0.0.1:50054') as channel:
    client = rpc.EmbeddingServiceStub(channel)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            info = client.Metadata(pb.MetadataRequest(), timeout=2)
            if info.ready and info.device == 'npu:0':
                print(f'ready device={info.device} dtype={info.dtype}')
                break
        except grpc.RpcError:
            pass
        time.sleep(1)
    else:
        raise SystemExit('BGE readiness timeout')
