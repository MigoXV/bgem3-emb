"""旧 Runtime 构造入口；按设备选择实现。"""

from bgem3_emb.engines import create_runtime
from bgem3_emb.engines.torch_runtime import Encoded, model_identity

__all__ = ["Encoded", "Runtime", "model_identity"]


def Runtime(config):
    return create_runtime(config)
