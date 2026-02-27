# SPDX-License-Identifier: Apache-2.0
"""
Intel XPU Platform support for SGLang Diffusion.
This file provides platform abstraction for Intel XPU (GPU) devices.
"""

import os
from functools import lru_cache
from typing import Any

import torch

from sglang.multimodal_gen import envs
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
    def get_local_torch_device(cls) -> torch.device:
        return torch.device(f"xpu:{envs.LOCAL_RANK}")

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        """
        Query XPU device capability via sgl_kernel.

        Uses ``torch.ops.sgl_kernel.query_device`` when available, which maps
        Intel GPU architectures to (major, minor) pairs:
          - Xe2 (BMG-G21): (2, 0)
          - (more architectures to be added)

        Falls back to a generic (1, 0) if sgl_kernel is not installed or the
        query fails.
        """
        try:
            major, minor = torch.ops.sgl_kernel.query_device.default(device_id)
            return DeviceCapability(major=int(major), minor=int(minor))
        except Exception:
            # sgl_kernel not loaded or unsupported architecture – fall back
            logger.warning(
                "sgl_kernel.query_device not available; returning generic "
                "XPU capability (1, 0). Install sgl-kernel-xpu for real "
                "device capability detection."
            )
            return DeviceCapability(major=1, minor=0)

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

    # Backend class paths
    _XPU_FLASH_ATTN_CLS = (
        "sglang.multimodal_gen.runtime.layers.attention.backends"
        ".xpu_flash_attn.XpuFlashAttentionBackend"
    )
    _SDPA_BACKEND_CLS = (
        "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
    )

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        """
        Get the attention backend class for XPU.

        By default, uses the XPU flash-attention backend backed by
        ``sgl-kernel-xpu`` (cutlass-based ``torch.ops.sgl_kernel.fwd``).
        When sgl-kernel is unavailable, the backend internally falls back to
        ``torch.nn.functional.scaled_dot_product_attention``.

        Use ``--attention-backend TORCH_SDPA`` to force the pure SDPA path.
        """
        # Explicit SDPA request
        if selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend for XPU (explicitly requested).")
            return cls._SDPA_BACKEND_CLS

        # Non-XPU/SDPA backends are not supported on XPU
        if (
            selected_backend is not None
            and selected_backend != AttentionBackendEnum.XPU_FLASH_ATTN
        ):
            logger.warning(
                f"{selected_backend.name} is not supported on XPU. "
                "Falling back to XPU flash-attention backend."
            )

        logger.info(
            "Using XPU flash-attention backend "
            "(sgl-kernel cutlass flash attention with SDPA fallback)."
        )
        return cls._XPU_FLASH_ATTN_CLS

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device communicator class for XPU distributed communication.
        
        Returns the XPU communicator that uses oneCCL backend through PyTorch.
        """
        logger.info("Using XPU communicator with oneCCL backend.")
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.xpu_communicator.XpuCommunicator"
