#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test script for Intel XPU distributed communication in SGLang.

This script tests the basic functionality of XPU communicator with oneCCL backend.

Usage:
    # Single GPU test
    python test_xpu_distributed.py
    
    # Multi-GPU test (2 GPUs)
    torchrun --nproc_per_node=2 test_xpu_distributed.py
    
    # Multi-GPU test with explicit backend
    torchrun --nproc_per_node=2 test_xpu_distributed.py --backend xccl
"""

import argparse
import os

import torch
import torch.distributed as dist


def test_xpu_availability():
    """Test if XPU is available."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"XPU available: {torch.xpu.is_available()}")
    
    if torch.xpu.is_available():
        print(f"XPU device count: {torch.xpu.device_count()}")
        for i in range(torch.xpu.device_count()):
            print(f"  Device {i}: {torch.xpu.get_device_name(i)}")
    else:
        print("XPU is not available. Exiting.")
        exit(1)


def test_distributed_init(backend="xccl"):
    """Test distributed initialization with XCCL backend."""
    if not dist.is_initialized():
        # Initialize distributed environment
        dist.init_process_group(backend=backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}] Initialized with backend: {backend}")
    print(f"[Rank {rank}] World size: {world_size}")
    
    # Set device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")
    
    print(f"[Rank {rank}] Using device: {device}")
    
    return rank, world_size, device


def test_all_reduce(rank, world_size, device):
    """Test all-reduce operation."""
    print(f"\n[Rank {rank}] Testing all-reduce...")
    
    # Create a tensor
    tensor = torch.ones(10, device=device) * (rank + 1)
    print(f"[Rank {rank}] Before all-reduce: {tensor[:5].tolist()}")
    
    # All-reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Expected result: sum of (1, 2, 3, ..., world_size)
    expected_sum = sum(range(1, world_size + 1))
    print(f"[Rank {rank}] After all-reduce: {tensor[:5].tolist()}")
    print(f"[Rank {rank}] Expected: {expected_sum}")
    
    # Verify
    assert torch.allclose(tensor, torch.ones(10, device=device) * expected_sum), (
        f"All-reduce failed at rank {rank}"
    )
    print(f"[Rank {rank}] ✓ All-reduce test passed!")


def test_broadcast(rank, world_size, device):
    """Test broadcast operation."""
    print(f"\n[Rank {rank}] Testing broadcast...")
    
    # Create a tensor (only meaningful on rank 0)
    if rank == 0:
        tensor = torch.arange(10, device=device, dtype=torch.float32)
    else:
        tensor = torch.zeros(10, device=device, dtype=torch.float32)
    
    print(f"[Rank {rank}] Before broadcast: {tensor[:5].tolist()}")
    
    # Broadcast from rank 0
    dist.broadcast(tensor, src=0)
    
    print(f"[Rank {rank}] After broadcast: {tensor[:5].tolist()}")
    
    # Verify
    expected = torch.arange(10, device=device, dtype=torch.float32)
    assert torch.allclose(tensor, expected), f"Broadcast failed at rank {rank}"
    print(f"[Rank {rank}] ✓ Broadcast test passed!")


def test_all_gather(rank, world_size, device):
    """Test all-gather operation."""
    print(f"\n[Rank {rank}] Testing all-gather...")
    
    # Create a tensor with rank-specific value
    tensor = torch.ones(5, device=device) * (rank + 1)
    print(f"[Rank {rank}] Input tensor: {tensor.tolist()}")
    
    # Prepare output tensor
    output_tensors = [torch.zeros(5, device=device) for _ in range(world_size)]
    
    # All-gather
    dist.all_gather(output_tensors, tensor)
    
    # Verify
    for i, out_tensor in enumerate(output_tensors):
        expected = torch.ones(5, device=device) * (i + 1)
        assert torch.allclose(out_tensor, expected), (
            f"All-gather failed at rank {rank} for tensor {i}"
        )
    
    print(f"[Rank {rank}] Gathered tensors: {[t[0].item() for t in output_tensors]}")
    print(f"[Rank {rank}] ✓ All-gather test passed!")


def test_send_recv(rank, world_size, device):
    """Test point-to-point send/recv."""
    if world_size < 2:
        print(f"[Rank {rank}] Skipping send/recv test (requires world_size >= 2)")
        return
    
    print(f"\n[Rank {rank}] Testing send/recv...")
    
    if rank == 0:
        # Send to rank 1
        tensor = torch.arange(10, device=device, dtype=torch.float32)
        print(f"[Rank {rank}] Sending: {tensor[:5].tolist()}")
        dist.send(tensor, dst=1)
        print(f"[Rank {rank}] ✓ Send completed!")
    elif rank == 1:
        # Receive from rank 0
        tensor = torch.zeros(10, device=device, dtype=torch.float32)
        dist.recv(tensor, src=0)
        print(f"[Rank {rank}] Received: {tensor[:5].tolist()}")
        
        # Verify
        expected = torch.arange(10, device=device, dtype=torch.float32)
        assert torch.allclose(tensor, expected), "Send/recv failed"
        print(f"[Rank {rank}] ✓ Recv test passed!")


def test_barrier(rank, world_size):
    """Test barrier synchronization."""
    print(f"\n[Rank {rank}] Testing barrier...")
    
    print(f"[Rank {rank}] Before barrier")
    dist.barrier()
    print(f"[Rank {rank}] After barrier")
    print(f"[Rank {rank}] ✓ Barrier test passed!")


def main():
    parser = argparse.ArgumentParser(description="Test XPU distributed communication")
    parser.add_argument(
        "--backend",
        type=str,
        default="xccl",
        choices=["xccl", "gloo"],
        help="Distributed backend to use",
    )
    args = parser.parse_args()
    
    # Test XPU availability
    test_xpu_availability()
    
    # Initialize distributed
    rank, world_size, device = test_distributed_init(backend=args.backend)
    
    try:
        # Run tests
        test_all_reduce(rank, world_size, device)
        test_broadcast(rank, world_size, device)
        test_all_gather(rank, world_size, device)
        test_send_recv(rank, world_size, device)
        test_barrier(rank, world_size)
        
        # Final synchronization
        dist.barrier()
        
        if rank == 0:
            print("\n" + "=" * 60)
            print("✓ All XPU distributed tests passed!")
            print("=" * 60)
    
    except Exception as e:
        print(f"\n[Rank {rank}] ✗ Test failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
