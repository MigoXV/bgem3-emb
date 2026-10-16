"""调度与平台执行隔离；只加载所选设备的执行实现。"""

from bgem3_emb.engines.scheduler import Busy, Engine, Unavailable

__all__ = ["Busy", "Engine", "Unavailable", "create_runtime"]


def create_runtime(config):
    if config.device == "cpu":
        from bgem3_emb.engines.cpu import CPURuntime

        return CPURuntime(config)
    if config.device.startswith("cuda:"):
        from bgem3_emb.engines.cuda import CUDARuntime

        return CUDARuntime(config)
    raise ValueError(f"不支持的设备: {config.device}")
