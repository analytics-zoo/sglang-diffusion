#!/usr/bin/env python3
"""
Simple unit test to verify XCCL communication primitives.
Run with: torchrun --nproc_per_node=4 test_xccl_communication.py
"""

import os
import sys
import torch
import torch.distributed as dist
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [Rank %(rank)s] %(message)s')

# import oneccl_bindings_for_pytorch

def setup_logger(rank):
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter(f'[%(asctime)s] [Rank {rank}] %(message)s'))
    return logger


def init_process_group():
    """Initialize the distributed process group."""
    backend = "xccl"  # Use XCCL for Intel XPU
    # dist.init_process_group(
    #     backend=backend
    # )

    # rank = dist.get_rank()
    # world_size = dist.get_world_size()

    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    dist.init_process_group(
        backend="xccl",  # NCCL is highly optimized for NVIDIA GPUs
        init_method=f"tcp://localhost:29500",
        world_size=world_size,
        rank=rank,
    )

    logger = setup_logger(rank)
    logger.info(f"Initialized process group: backend={backend}, rank={rank}, world_size={world_size}")

    # Set device
    torch.xpu.set_device(rank)
    logger.info(f"Set device to xpu:{rank}")

    return rank, world_size, logger


def test_1_basic_tensor_creation(rank, world_size, logger):
    """Test 1: Basic tensor creation on XPU."""
    logger.info("=" * 50)
    logger.info("Test 1: Basic tensor creation")
    
    try:
        tensor = torch.randn(10, 10, device=f'xpu:{rank}')
        logger.info(f"✓ Created tensor on xpu:{rank}, shape={tensor.shape}, dtype={tensor.dtype}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create tensor: {e}")
        return False


def test_2_barrier(rank, world_size, logger):
    """Test 2: Barrier synchronization."""
    logger.info("=" * 50)
    logger.info("Test 2: Barrier synchronization")
    
    try:
        logger.info("Before barrier")
        dist.barrier()
        logger.info("✓ Barrier completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Barrier failed: {e}")
        return False


def test_3_broadcast(rank, world_size, logger):
    """Test 3: Broadcast operation."""
    logger.info("=" * 50)
    logger.info("Test 3: Broadcast")
    
    try:
        tensor = torch.zeros(5, device=f'xpu:{rank}')
        if rank == 0:
            tensor.fill_(42)
            logger.info(f"Rank 0 broadcasting tensor with value: {tensor[0].item()}")
        
        dist.broadcast(tensor, src=0)
        
        expected = 42.0
        if torch.allclose(tensor, torch.full_like(tensor, expected)):
            logger.info(f"✓ Broadcast successful, received value: {tensor[0].item()}")
            return True
        else:
            logger.error(f"✗ Broadcast failed, expected {expected}, got {tensor[0].item()}")
            return False
    except Exception as e:
        logger.error(f"✗ Broadcast failed: {e}")
        return False


def test_4_allreduce(rank, world_size, logger):
    """Test 4: AllReduce operation."""
    logger.info("=" * 50)
    logger.info("Test 4: AllReduce")
    
    try:
        tensor = torch.ones(5, device=f'xpu:{rank}') * (rank + 1)
        logger.info(f"Before allreduce: {tensor[0].item()}")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Expected: sum of (1 + 2 + 3 + 4) = 10 for world_size=4
        expected = sum(range(1, world_size + 1))
        if torch.allclose(tensor, torch.full_like(tensor, float(expected))):
            logger.info(f"✓ AllReduce successful, result: {tensor[0].item()}")
            return True
        else:
            logger.error(f"✗ AllReduce failed, expected {expected}, got {tensor[0].item()}")
            return False
    except Exception as e:
        logger.error(f"✗ AllReduce failed: {e}")
        return False


def test_5_send_recv(rank, world_size, logger):
    """Test 5: Point-to-point send/recv."""
    logger.info("=" * 50)
    logger.info("Test 5: Point-to-point send/recv")
    
    try:
        if rank == 0:
            tensor = torch.tensor([1.0, 2.0, 3.0], device=f'xpu:{rank}')
            logger.info(f"Rank 0 sending tensor: {tensor}")
            dist.send(tensor, dst=1)
            logger.info("✓ Rank 0 sent tensor to rank 1")
            return True
        elif rank == 1:
            tensor = torch.zeros(3, device=f'xpu:{rank}')
            logger.info("Rank 1 waiting to receive tensor from rank 0")
            dist.recv(tensor, src=0)
            logger.info(f"✓ Rank 1 received tensor: {tensor}")
            expected = torch.tensor([1.0, 2.0, 3.0], device=f'xpu:{rank}')
            return torch.allclose(tensor, expected)
        else:
            # Other ranks just wait
            dist.barrier()
            return True
    except Exception as e:
        logger.error(f"✗ Send/Recv failed: {e}")
        return False


def test_6_isend_irecv(rank, world_size, logger):
    """Test 6: Asynchronous point-to-point isend/irecv."""
    logger.info("=" * 50)
    logger.info("Test 6: Asynchronous isend/irecv")
    
    try:
        if rank == 0:
            tensor = torch.tensor([10.0, 20.0, 30.0], device=f'xpu:{rank}')
            logger.info(f"Rank 0 async sending tensor: {tensor}")
            req = dist.isend(tensor, dst=1)
            req.wait()
            logger.info("✓ Rank 0 async send completed")
            return True
        elif rank == 1:
            tensor = torch.zeros(3, device=f'xpu:{rank}')
            logger.info("Rank 1 async receiving tensor from rank 0")
            req = dist.irecv(tensor, src=0)
            req.wait()
            logger.info(f"✓ Rank 1 async received tensor: {tensor}")
            expected = torch.tensor([10.0, 20.0, 30.0], device=f'xpu:{rank}')
            return torch.allclose(tensor, expected)
        else:
            dist.barrier()
            return True
    except Exception as e:
        logger.error(f"✗ Async Send/Recv failed: {e}")
        return False


def test_7_all_to_all(rank, world_size, logger):
    """Test 7: All-to-all collective."""
    logger.info("=" * 50)
    logger.info("Test 7: All-to-all collective")
    
    try:
        # Each rank creates input list of tensors
        input_list = [torch.full((3,), float(rank * 10 + i), device=f'xpu:{rank}') 
                      for i in range(world_size)]
        output_list = [torch.zeros(3, device=f'xpu:{rank}') for _ in range(world_size)]
        
        logger.info(f"Before all_to_all: input_list[0]={input_list[0][0].item()}")
        
        dist.all_to_all(output_list, input_list)
        
        logger.info(f"✓ All-to-all completed: output_list[0]={output_list[0][0].item()}")
        
        # Verify: output_list[i] should contain data from rank i
        for i in range(world_size):
            expected = float(i * 10 + rank)
            if not torch.allclose(output_list[i], torch.full_like(output_list[i], expected)):
                logger.error(f"✗ All-to-all verification failed at index {i}")
                return False
        
        logger.info("✓ All-to-all verification passed")
        return True
    except Exception as e:
        logger.error(f"✗ All-to-all failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_8_all_to_all_single(rank, world_size, logger):
    """Test 8: All-to-all-single collective."""
    logger.info("=" * 50)
    logger.info("Test 8: All-to-all-single collective")

    try:
        # Create a single tensor that will be split
        input_tensor = torch.arange(world_size * 4, dtype=torch.float32, device=f'xpu:{rank}') + rank * 100
        output_tensor = torch.zeros(world_size * 4, dtype=torch.float32, device=f'xpu:{rank}')

        logger.info(f"Before all_to_all_single: input shape={input_tensor.shape}, first element={input_tensor[0].item()}")

        torch.xpu.synchronize()  # Ensure all prior ops complete before collective
        dist.all_to_all_single(output_tensor, input_tensor, group=None)

        logger.info(f"✓ All-to-all-single completed: output first element={output_tensor[0].item()}")
        return True
    except Exception as e:
        logger.error(f"✗ All-to-all-single failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_9_manual_all_to_all(rank, world_size, logger):
    """Test 9: Manual all-to-all using isend/irecv."""
    logger.info("=" * 50)
    logger.info("Test 9: Manual all-to-all using isend/irecv")
    
    try:
        # Create input and output lists
        chunk_size = 1000
        input_list = [torch.full((chunk_size,), float(rank * 10 + i), device=f'xpu:{rank}') 
                      for i in range(world_size)]
        output_list = [torch.zeros(chunk_size, device=f'xpu:{rank}') for _ in range(world_size)]
        
        logger.info("Starting manual all-to-all")
        
        # Local copy
        output_list[rank].copy_(input_list[rank])
        logger.info("Local copy done")
        
        # Send/recv with other ranks
        for i in range(world_size):
            if i == rank:
                continue
            
            send_tag = rank * world_size + i
            recv_tag = i * world_size + rank
            
            logger.info(f"Communicating with rank {i} (send_tag={send_tag}, recv_tag={recv_tag})")
            
            send_req = dist.isend(input_list[i], dst=i, tag=send_tag)
            recv_req = dist.irecv(output_list[i], src=i, tag=recv_tag)
            
            send_req.wait()
            recv_req.wait()
            
            logger.info(f"✓ Communication with rank {i} completed")
        
        logger.info("✓ Manual all-to-all completed successfully")
        
        # Verify results
        for i in range(world_size):
            expected = float(i * 10 + rank)
            if not torch.allclose(output_list[i], torch.full_like(output_list[i], expected)):
                logger.error(f"✗ Manual all-to-all verification failed at index {i}")
                return False
        
        logger.info("✓ Manual all-to-all verification passed")
        return True
    except Exception as e:
        logger.error(f"✗ Manual all-to-all failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    rank, world_size, logger = init_process_group()
    
    if world_size != 4:
        logger.warning(f"This test is designed for 4 processes, but got {world_size}")
    
    tests = [
        # ("Basic Tensor Creation", test_1_basic_tensor_creation),
        # ("Barrier", test_2_barrier),
        # ("Broadcast", test_3_broadcast),
        # ("AllReduce", test_4_allreduce),
        # ("Send/Recv", test_5_send_recv),
        # ("Async Send/Recv", test_6_isend_irecv),
        # ("All-to-All", test_7_all_to_all),
        ("All-to-All-Single", test_8_all_to_all_single),
        ("Manual All-to-All", test_9_manual_all_to_all),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            dist.barrier()  # Sync before each test
            success = test_func(rank, world_size, logger)
            results[test_name] = success
            dist.barrier()  # Sync after each test
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    if rank == 0:
        logger.info("=" * 50)
        logger.info("TEST SUMMARY")
        logger.info("=" * 50)
        for test_name, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            logger.info(f"{status}: {test_name}")
        logger.info("=" * 50)
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
