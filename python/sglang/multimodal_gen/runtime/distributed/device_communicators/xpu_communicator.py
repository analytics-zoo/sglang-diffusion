# SPDX-License-Identifier: Apache-2.0
"""
Intel XPU Communicator for SGLang Diffusion.

This module provides distributed communication support for Intel XPU devices
using PyTorch's built-in distributed communication primitives with XCCL backend.
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class XpuCommunicator(DeviceCommunicatorBase):
    """
    Communicator for Intel XPU devices using oneCCL backend.
    
    Intel XPU uses the XCCL (Collective Communications Library) backend through
    PyTorch's distributed communication interface. Unlike NVIDIA NCCL, XCCL
    is directly integrated into PyTorch for XPU devices.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        """
        Initialize XPU communicator.
        
        Args:
            cpu_group: CPU process group for control-plane communication
            device: XPU device (e.g., torch.device('xpu:0'))
            device_group: XPU process group for data-plane communication
            unique_name: Unique identifier for this communicator
        """
        super().__init__(cpu_group, device, device_group, unique_name)
        
        # Verify we're on XPU device
        if device is not None and device.type != "xpu":
            logger.warning(
                f"XpuCommunicator initialized with non-XPU device: {device}. "
                "This may cause unexpected behavior."
            )
        
        # Check if XCCL backend is available
        if device_group is not None:
            backend = dist.get_backend(device_group)
            logger.info(
                f"XpuCommunicator initialized with backend: {backend}, "
                f"world_size: {self.world_size}, rank: {self.rank}"
            )
            if backend not in ["xccl", "gloo"]:
                logger.warning(
                    f"Expected 'xccl' or 'gloo' backend for XPU, got: {backend}. "
                    "Communication may not work as expected."
                )

    def all_reduce(
        self, input_: torch.Tensor, op: ReduceOp | None = None
    ) -> torch.Tensor:
        """
        Perform all-reduce operation on XPU.
        
        Args:
            input_: Input tensor to reduce
            op: Reduction operation (default: SUM)
            
        Returns:
            Reduced tensor (in-place operation)
        """
        if op is None:
            op = ReduceOp.SUM
            
        # Verify tensor is on XPU device
        assert input_.device.type == "xpu", (
            f"Input tensor must be on XPU device, got: {input_.device}"
        )
        
        # Perform all-reduce using device group
        dist.all_reduce(input_, op=op, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Perform all-gather operation on XPU.
        
        Args:
            input_: Input tensor to gather
            dim: Dimension along which to concatenate gathered tensors
            
        Returns:
            Concatenated tensor from all ranks
        """
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive
            dim += input_.dim()
        
        input_size = input_.size()
        
        # Allocate output tensor
        output_tensor = torch.empty(
            (self.world_size,) + input_size, 
            dtype=input_.dtype, 
            device=input_.device
        )
        
        # All-gather into tensor
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        
        # Reshape to concatenate along specified dimension
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1:]
        )
        
        return output_tensor

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        Gather tensors from all ranks to a destination rank.
        
        Args:
            input_: Input tensor to gather
            dst: Destination rank (local rank within group)
            dim: Dimension along which to concatenate
            
        Returns:
            Gathered tensor at destination rank, None at other ranks
        """
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()
        
        # XPU gather implementation using all_gather
        # (similar to vLLM's approach due to potential issues with direct gather)
        input_size = input_.size()
        output_tensor = torch.empty(
            (self.world_size,) + input_size,
            dtype=input_.dtype,
            device=input_.device
        )
        
        # All-gather
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        
        if self.rank_in_group == dst:
            # Reshape and return at destination
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1:]
            )
            return output_tensor
        else:
            return None

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> None:
        """
        Broadcast tensor from source rank to all ranks.
        
        Args:
            input_: Tensor to broadcast (modified in-place)
            src: Source rank (local rank within group)
        """
        dist.broadcast(input_, src=self.ranks[src], group=self.device_group)

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """
        Send tensor to destination rank (point-to-point communication).
        
        Args:
            tensor: Tensor to send
            dst: Destination rank (local rank, defaults to next rank)
        """
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        
        dist.send(tensor, dst=self.ranks[dst], group=self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """
        Receive tensor from source rank (point-to-point communication).
        
        Args:
            size: Shape of tensor to receive
            dtype: Data type of tensor to receive
            src: Source rank (local rank, defaults to previous rank)
            
        Returns:
            Received tensor
        """
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        dist.recv(tensor, src=self.ranks[src], group=self.device_group)
        return tensor

    def barrier(self) -> None:
        """
        Synchronization barrier across all ranks.
        """
        dist.barrier(group=self.device_group)

    def destroy(self) -> None:
        """
        Cleanup communicator resources.
        
        Note: For XPU with PyTorch distributed, cleanup is handled
        automatically by PyTorch's process group management.
        """
        logger.info(
            f"XpuCommunicator destroyed for rank {self.rank} "
            f"(unique_name: {self.unique_name})"
        )
        # No explicit cleanup needed for PyTorch XCCL backend
        pass
