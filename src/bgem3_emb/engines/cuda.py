"""CUDA 设备验证及资源管理。"""

import torch

from bgem3_emb.engines.torch_runtime import Runtime


class CUDARuntime(Runtime):
    def prepare_device(self, config):
        device = torch.device(config.device)
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise ValueError("CUDA device 不可用")
        torch.cuda.set_device(device)
        if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("设备不支持 bfloat16")
        torch.backends.cuda.matmul.allow_tf32 = False
        return device

    def close(self):
        super().close()
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
