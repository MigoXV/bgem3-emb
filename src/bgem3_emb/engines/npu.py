"""Ascend eager 执行；所有设备操作均由模型 Worker 线程调用。"""

import torch

from bgem3_emb.engines.torch_runtime import Runtime


class NPURuntime(Runtime):
    def prepare_device(self, config):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise ValueError("NPU 后端需要匹配 Torch/CANN 的 torch_npu，请检查部署依赖") from exc
        device = torch.device(config.device)
        if not torch.npu.is_available() or device.index >= torch.npu.device_count():
            raise ValueError(f"NPU device 不可用: {device}; 使用容器内逻辑编号")
        torch.npu.set_device(device)
        if config.dtype == "bfloat16" and not torch.npu.is_bf16_supported():
            raise ValueError("设备不支持 bfloat16")
        return device

    def close(self):
        super().close()
        with torch.npu.device(self.device):
            torch.npu.synchronize(self.device)
            torch.npu.empty_cache()
