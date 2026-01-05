# SPDX-License-Identifier: Apache-2.0
"""
Intel XPU Platform support for SGLang Diffusion.
This file provides platform abstraction for Intel XPU (GPU) devices.
"""

import os
from functools import lru_cache
from typing import Any

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def device_id_to_physical_device_id(device_id: int) -> int:
    """Convert logical device ID to physical device ID based on ZE_AFFINITY_MASK."""
    if "ZE_AFFINITY_MASK" in os.environ:
        device_ids = os.environ["ZE_AFFINITY_MASK"].split(",")
        if device_ids == [""]:
            msg = (
                "ZE_AFFINITY_MASK is set to empty string, which means"
                " XPU support is disabled."
            )
            raise RuntimeError(msg)
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


class XpuPlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    device_control_env_var: str = "ZE_AFFINITY_MASK"  # Intel GPU environment variable

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        """
        XPU doesn't have a direct equivalent to CUDA compute capability.
        We return a placeholder capability based on device generation.
        """
        try:
            # XPU doesn't expose compute capability like CUDA
            # Return a generic capability (major=1, minor=0) for compatibility
            return DeviceCapability(major=1, minor=0)
        except Exception:
            return None

    @classmethod
    @lru_cache(maxsize=8)
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """Check if the device has the specified capability."""
        try:
            return bool(super().has_device_capability(capability, device_id))
        except RuntimeError:
            return False

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of the XPU device."""
        try:
            return str(torch.xpu.get_device_name(device_id))
        except Exception as e:
            logger.warning(f"Failed to get XPU device name: {e}")
            return "Unknown XPU Device"

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get the UUID of the XPU device."""
        # XPU doesn't provide UUID through PyTorch API yet
        # Use device_id as fallback
        return f"XPU-{device_id}"

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get the total memory of the XPU device in bytes."""
        try:
            return int(torch.xpu.get_device_properties(device_id).total_memory)
        except Exception as e:
            logger.warning(f"Failed to get XPU device memory: {e}")
            return 0

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """Check if async output processing is supported on XPU."""
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable graph mode. "
                "Since enforce-eager is enabled, async output processor cannot be used"
            )
            return False
        # XPU doesn't support CUDA graphs yet, so async output is limited
        return False

    @classmethod
    def log_warnings(cls) -> None:
        """Log XPU-specific warnings."""
        pass

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """Get current memory usage on XPU device."""
        try:
            torch.xpu.reset_peak_memory_stats(device)
            return float(torch.xpu.max_memory_allocated(device))
        except Exception as e:
            logger.warning(f"Failed to get XPU memory usage: {e}")
            return 0.0

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        """
        Get available GPU memory on XPU device.
        
        Returns:
            float: Available memory in GiB.
        """
        if empty_cache:
            torch.xpu.empty_cache()

        try:
            # Get free and total memory
            free_gpu_memory, total_memory = torch.xpu.mem_get_info(device_id)
        except Exception as e:
            logger.warning(f"Failed to get XPU memory info: {e}")
            # Fallback: estimate based on total memory and current usage
            try:
                total_memory = cls.get_device_total_memory(device_id)
                used_memory = torch.xpu.memory_allocated(device_id)
                free_gpu_memory = total_memory - used_memory
            except Exception:
                return 0.0

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="xpu")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """
        Set the seed of each random module for XPU.
        """
        import random

        import numpy as np

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if hasattr(torch.xpu, "manual_seed_all"):
                torch.xpu.manual_seed_all(seed)

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        """
        Get the attention backend class for XPU.
        
        XPU currently only supports Torch SDPA backend.
        Flash Attention and other CUDA-specific backends are not available.
        """
        # Log the requested backend
        if selected_backend is not None:
            logger.info(f"Requested attention backend: {selected_backend}")
        
        # XPU-specific backends that are not supported
        unsupported_backends = [
            AttentionBackendEnum.FA,
            AttentionBackendEnum.FA2,
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.SAGE_ATTN_3,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN,
            AttentionBackendEnum.VMOBA_ATTN,
        ]
        
        if selected_backend in unsupported_backends:
            logger.warning(
                f"{selected_backend.name} is not supported on XPU. "
                "Falling back to Torch SDPA backend."
            )
            selected_backend = AttentionBackendEnum.TORCH_SDPA
        
        # AIter is also CUDA/ROCm specific
        if selected_backend == AttentionBackendEnum.AITER:
            logger.warning(
                "AIter backend is not supported on XPU. "
                "Falling back to Torch SDPA backend."
            )
            selected_backend = AttentionBackendEnum.TORCH_SDPA
        
        # Default to SDPA for XPU
        if selected_backend is None or selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend for XPU.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        
        # Fallback to SDPA for any other unhandled case
        logger.warning(
            f"Unhandled backend {selected_backend}, falling back to SDPA."
        )
        return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device communicator class for XPU distributed communication.
        
        Returns the XPU communicator that uses oneCCL backend through PyTorch.
        """
        logger.info("Using XPU communicator with oneCCL backend.")
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.xpu_communicator.XpuCommunicator"
