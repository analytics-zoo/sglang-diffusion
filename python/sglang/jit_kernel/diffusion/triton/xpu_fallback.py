import torch


# TODO: remove this when triton intel xpu bug is fixed
def fuse_scale_shift_native(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    block_l: int = 128,
    block_c: int = 128,
):
    return x * (1 + scale) + shift
