"""
Fallback implementation for USP all-to-all when backend doesn't support all_to_all_single.
This is needed for XCCL backend which may not implement all_to_all_single or all_to_all.
"""

import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def all_to_all_single_manual(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
) -> torch.Tensor:
    """
    Manual implementation of all_to_all_single using point-to-point send/recv.
    
    This is the ultimate fallback when both all_to_all_single and all_to_all
    are not supported by the backend (e.g., XCCL).
    
    Args:
        output: Output tensor (will be filled)
        input: Input tensor
        output_split_sizes: Optional list of output split sizes
        input_split_sizes: Optional list of input split sizes
        group: Process group
    
    Returns:
        Output tensor
    """
    # CRITICAL: For XCCL backend, even send/recv don't work with custom process groups
    # We must use the default world group (group=None) for communication to work
    global_rank = dist.get_rank()
    global_world_size = dist.get_world_size()
    
    # Get the ranks in the process group
    if group is not None:
        group_rank = dist.get_rank(group)
        group_world_size = dist.get_world_size(group)
        # Get all ranks in this group
        group_ranks = list(range(group_world_size))  # Assuming consecutive ranks starting from 0
    else:
        group_rank = global_rank
        group_world_size = global_world_size
        group_ranks = list(range(group_world_size))
    
    # For simplicity, if the group matches the world, use world communication
    use_world_group = (group_world_size == global_world_size)
    
    if use_world_group:
        comm_group = None  # Use default world group
        rank = global_rank
        world_size = global_world_size
    else:
        comm_group = group
        rank = group_rank
        world_size = group_world_size
    
    # Determine split sizes
    if input_split_sizes is None:
        assert input.numel() % world_size == 0, (
            f"Input tensor size {input.numel()} must be divisible by world_size {world_size}"
        )
        split_size = input.numel() // world_size
        input_split_sizes = [split_size] * world_size
    
    if output_split_sizes is None:
        output_split_sizes = input_split_sizes
    
    # Split input tensor
    input_chunks = []
    offset = 0
    for size in input_split_sizes:
        input_chunks.append(input[offset:offset+size].contiguous())
        offset += size
    
    # Prepare output chunks
    output_chunks = []
    offset = 0
    for size in output_split_sizes:
        output_chunks.append(output[offset:offset+size])
        offset += size
    
    # NO SYNCHRONIZATION! Each rank posts all receives first, then all sends
    # This ensures deadlock-free operation even when ranks arrive at different times
    
    # First, copy own data locally (no communication needed)
    output_chunks[rank].copy_(input_chunks[rank])
    
    # Post ALL receives first (non-blocking)
    recv_requests = []
    for peer_rank in range(world_size):
        if peer_rank == rank:
            continue  # Already copied locally
        try:
            recv_req = dist.irecv(output_chunks[peer_rank], src=peer_rank, group=comm_group)
            recv_requests.append((peer_rank, recv_req))
        except Exception as e:
            raise
    
    # Then post ALL sends (non-blocking)
    send_requests = []
    for peer_rank in range(world_size):
        if peer_rank == rank:
            continue  # Already copied locally
        send_req = dist.isend(input_chunks[peer_rank], dst=peer_rank, group=comm_group)
        send_requests.append((peer_rank, send_req))
    
    # Wait for all receives to complete
    for peer_rank, recv_req in recv_requests:
        recv_req.wait()
    
    # Wait for all sends to complete
    for peer_rank, send_req in send_requests:
        send_req.wait()
    
    return output


def all_to_all_single_fallback(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    use_manual=False,
) -> torch.Tensor:
    """
    Fallback implementation of all_to_all_single using all_to_all (list-based version)
    or manual send/recv if use_manual=True.
    
    This is needed when the backend (e.g., XCCL) doesn't support all_to_all_single.
    
    Args:
        output: Output tensor (will be filled)
        input: Input tensor
        output_split_sizes: Optional list of output split sizes
        input_split_sizes: Optional list of input split sizes
        group: Process group
        use_manual: If True, use manual send/recv implementation instead of all_to_all
    
    Returns:
        Output tensor
    """
    rank = dist.get_rank(group) if group else dist.get_rank()
    world_size = dist.get_world_size(group) if group else dist.get_world_size()
    
    # If manual send/recv requested, use that directly
    if use_manual:
        return all_to_all_single_manual(
            output, input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group
        )
    
    # Determine split sizes
    if input_split_sizes is None:
        assert input.numel() % world_size == 0, (
            f"Input tensor size {input.numel()} must be divisible by world_size {world_size}"
        )
        split_size = input.numel() // world_size
        input_split_sizes = [split_size] * world_size
    
    if output_split_sizes is None:
        output_split_sizes = input_split_sizes
    
    # Split input tensor into chunks
    input_list = []
    offset = 0
    for size in input_split_sizes:
        input_list.append(input[offset:offset+size].contiguous())
        offset += size
    
    # Prepare output list
    output_list = []
    offset = 0
    for size in output_split_sizes:
        output_list.append(output[offset:offset+size])
        offset += size
    
    # Perform all_to_all
    try:
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        dist.all_to_all(output_list, input_list, group=group)
        
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as e:
        return all_to_all_single_manual(
            output, input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group
        )
    
    return output


def all_to_all_single_with_fallback(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
) -> torch.Tensor:
    """
    Try all_to_all_single, fall back to list-based implementation if not supported.
    
    Args:
        output: Output tensor
        input: Input tensor
        output_split_sizes: Optional output split sizes
        input_split_sizes: Optional input split sizes
        group: Process group
    
    Returns:
        Output tensor
    """
    try:
        # Try native all_to_all_single first
        dist.all_to_all_single(
            output, input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group
        )
        return output
    except (NotImplementedError, RuntimeError) as e:
        # Fall back to list-based implementation
        rank = dist.get_rank(group) if group else dist.get_rank()
        logger.warning(
            f"Rank {rank}: all_to_all_single not supported ({type(e).__name__}), "
            f"using fallback implementation with all_to_all"
        )
        return all_to_all_single_fallback(
            output, input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group
        )


def ft_c_all_to_all_single_with_fallback(
    input: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
) -> torch.Tensor:
    """
    Functional collectives style all_to_all_single with fallback.
    
    This mimics torch.distributed._functional_collectives.all_to_all_single
    but provides fallback for unsupported backends like XCCL.
    
    For XCCL, we use manual send/recv implementation since both
    all_to_all_single and all_to_all appear to hang.
    
    Args:
        input: Input tensor
        output_split_sizes: Optional output split sizes
        input_split_sizes: Optional input split sizes
        group: Process group
    
    Returns:
        Output tensor
    """
    rank = dist.get_rank(group) if group else dist.get_rank()
    
    # For XCCL backend, use manual send/recv implementation directly
    # because both ft_c.all_to_all_single and dist.all_to_all hang
    backend = dist.get_backend(group) if group else dist.get_backend()
    
    # Allocate output tensor
    output = torch.empty_like(input)
    
    if backend == 'xccl':
        result = all_to_all_single_manual(
            output, input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group
        )
    else:
        result = all_to_all_single_fallback(
            output, input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            use_manual=False
        )
    
    return result
