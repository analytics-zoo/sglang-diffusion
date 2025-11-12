# SPDX-License-Identifier: Apache-2.0
"""
Intel XPU Platform support for SGLang Diffusion.
This file provides platform abstraction for Intel XPU (GPU) devices.
"""

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class XpuPlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    device_control_env_var: str = "ZE_AFFINITY_MASK"  # Intel GPU environment variable

    @classmethod
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
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of the XPU device."""
        try:
            return str(torch.xpu.get_device_name(device_id))
        except Exception as e:
            logger.warning(f"Failed to get XPU device name: {e}")
            return "Unknown XPU Device"

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get the UUID of the XPU device."""
        # XPU doesn't provide UUID through PyTorch API yet
        # Use device_id as fallback
        return f"XPU-{device_id}"

    @classmethod
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
            AttentionBackendEnum.FA3,
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.SAGE_ATTN_THREE,
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
