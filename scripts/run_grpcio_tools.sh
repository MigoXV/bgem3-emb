#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GRPCIO_VERSION="$(poetry run python -c 'import importlib.metadata; print(importlib.metadata.version("grpcio"))')"
poetry run python -m pip install "grpcio-tools==${GRPCIO_VERSION}" mypy-protobuf
poetry run python -m grpc_tools.protoc \
  -I src/bgem3_emb/proto --python_out=src/bgem3_emb/proto \
  --grpc_python_out=src/bgem3_emb/proto --mypy_out=src/bgem3_emb/proto \
  --mypy_grpc_out=src/bgem3_emb/proto src/bgem3_emb/proto/embedding.proto
# 同一协议嵌入两个轻量包时使用相对导入；不修改生成代码的业务逻辑。
sed -i 's/^import embedding_pb2 as/from . import embedding_pb2 as/' src/bgem3_emb/proto/embedding_pb2_grpc.py
sed -i 's/^import embedding_pb2$/from . import embedding_pb2/' src/bgem3_emb/proto/embedding_pb2_grpc.pyi
