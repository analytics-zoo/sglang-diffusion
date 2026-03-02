# SPDX-License-Identifier: Apache-2.0
"""
XPU Flash Attention backend for Intel XPU (GPU) devices.

Calls ``torch.ops.sgl_kernel.fwd`` (cutlass-based flash attention kernel)
directly with the correct 27-argument signature.  Falls back to
``torch.nn.functional.scaled_dot_product_attention`` when sgl-kernel is not
available or the device does not support flash attention.

"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: check if sgl-kernel XPU flash attention is usable
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _sgl_kernel_flash_attn_available() -> bool:
    """Return *True* if sgl-kernel XPU flash attention can be used."""
    try:
        from sgl_kernel.flash_attn import is_fa3_supported  # noqa: F401

        # Verify the fwd op is registered
        _ = torch.ops.sgl_kernel.fwd.default
        return is_fa3_supported()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Thin wrapper that calls torch.ops.sgl_kernel.fwd with the correct
# 27-argument signature (matching chunked_prefill.cpp / torch_extension_sycl.cc).
#
# fwd(q, k, v, q_v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, page_table,
#     kv_batch_idx, leftpad_k, rotary_cos, rotary_sin, seqlens_rotary,
#     q_descale, k_descale, v_descale, softmax_scale, sinks,
#     is_causal, window_size_left, window_size_right, softcap,
#     is_rotary_interleaved, scheduler_metadata, num_splits, pack_gqa,
#     sm_margin) -> Tensor[]
# ---------------------------------------------------------------------------
def _flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
) -> torch.Tensor:
    """flash_attn_varlen_func compatible wrapper for sgl-kernel XPU.

    Accepts the same high-level parameters as the upstream
    ``flash_attn_varlen_func`` but calls ``torch.ops.sgl_kernel.fwd``
    directly with the correct 27-argument C++ signature.

    When *cu_seqlens_q* is ``None`` the inputs are assumed to be in the
    dense ``(batch, seq_len, heads, head_dim)`` layout.  They will be
    reshaped to ``(total_q, heads, head_dim)`` and ``cu_seqlens_q`` will
    be constructed automatically.
    """
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )

    # --- handle dense (batch, seqlen, heads, hdim) layout ----------------
    is_dense = cu_seqlens_q is None
    if is_dense:
        assert q.dim() == 4, "expected (batch, seq_len, heads, head_dim)"
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        cu_seqlens_q = (
            torch.arange(0, batch_size + 1, dtype=torch.int32, device=q.device)
            * seqlen_q
        )
        max_seqlen_q = seqlen_q
        # q must be (total_q, heads, head_dim) for the varlen kernel
        q = q.reshape(-1, q.shape[-2], q.shape[-1]).contiguous()

    if cu_seqlens_k is None and k.dim() == 4:
        batch_size_k, seqlen_k = k.shape[0], k.shape[1]
        # For paged k/v format, cu_seqlens_k is per-sequence lengths (batch_size,)
        # NOT cumulative (batch_size+1,).  This matches flash_attn_with_kvcache
        # which sets cu_seqlens_k = cache_seqlens.
        cu_seqlens_k = torch.full(
            (batch_size_k,), seqlen_k, dtype=torch.int32, device=k.device
        )
        # k, v stay as (batch, seq_len, heads_k, head_dim) = paged layout
        # where num_pages = batch, page_size = seq_len
        # The kernel requires page_block_size to be a multiple of 256.
        _PAGE_ALIGN = 256
        if seqlen_k % _PAGE_ALIGN != 0:
            pad_len = _PAGE_ALIGN - (seqlen_k % _PAGE_ALIGN)
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))
    elif cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    batch_size = cu_seqlens_q.numel() - 1

    # page_table: trivial identity mapping (one page per sequence)
    page_table = (
        torch.arange(batch_size, device=q.device, dtype=torch.int32)
        .reshape(batch_size, 1)
        .contiguous()
    )

    # Call 27-arg fwd op  --------------------------------------------------
    out, softmax_lse, *rest = torch.ops.sgl_kernel.fwd.default(
        q,  # 1  Tensor!  q
        k,  # 2  Tensor   k
        v,  # 3  Tensor   v
        qv,  # 4  Tensor?  q_v
        cu_seqlens_q,  # 5  Tensor   cu_seqlens_q
        cu_seqlens_k,  # 6  Tensor   cu_seqlens_k
        max_seqlen_q,  # 7  int      max_seqlen_q
        page_table,  # 8  Tensor   page_table
        None,  # 9  Tensor?  kv_batch_idx
        None,  # 10 Tensor?  leftpad_k
        None,  # 11 Tensor?  rotary_cos
        None,  # 12 Tensor?  rotary_sin
        None,  # 13 Tensor?  seqlens_rotary
        q_descale,  # 14 Tensor?  q_descale
        k_descale,  # 15 Tensor?  k_descale
        v_descale,  # 16 Tensor?  v_descale
        softmax_scale,  # 17 float    softmax_scale
        None,  # 18 Tensor?  sinks
        causal,  # 19 bool     is_causal
        window_size[0],  # 20 int      window_size_left
        window_size[1],  # 21 int      window_size_right
        softcap,  # 22 float    softcap
        False,  # 23 bool     is_rotary_interleaved
        None,  # 24 Tensor?  scheduler_metadata
        num_splits,  # 25 int      num_splits
        pack_gqa,  # 26 bool?    pack_gqa
        sm_margin,  # 27 int      sm_margin
    )

    # Reshape back to dense layout if input was dense
    if is_dense:
        out = out.reshape(batch_size, -1, out.shape[-2], out.shape[-1])

    return out


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
class XpuFlashAttentionBackend(AttentionBackend):
    """XPU attention backend backed by sgl-kernel flash attention."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.INTEL_XPU

    @staticmethod
    def get_impl_cls() -> type[XpuFlashAttentionImpl]:
        return XpuFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        return None


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------
class XpuFlashAttentionImpl(AttentionImpl):
    """
    XPU attention implementation.

    When sgl-kernel flash attention is available (BMG / Xe2 and later), the
    forward pass calls ``sgl_kernel.flash_attn.flash_attn_varlen_func``
    which invokes the cutlass-based ``torch.ops.sgl_kernel.fwd`` kernel.

    Otherwise, it falls back to ``torch.nn.functional.scaled_dot_product_attention``.
    """

    _logged_backend: bool = False  # class-level flag to log backend choice once

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

        self._use_flash = _sgl_kernel_flash_attn_available()
        if not XpuFlashAttentionImpl._logged_backend:
            XpuFlashAttentionImpl._logged_backend = True
            if self._use_flash:
                logger.info(
                    "XPU attention: using sgl-kernel flash attention "
                    "(cutlass-based, torch.ops.sgl_kernel.fwd)"
                )
            else:
                logger.info(
                    "XPU attention: sgl-kernel flash attention not available, "
                    "falling back to torch.nn.functional.scaled_dot_product_attention"
                )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query : Tensor
            ``(batch, seq_len_q, num_heads, head_dim)``
        key : Tensor
            ``(batch, seq_len_k, num_kv_heads, head_dim)``
        value : Tensor
            ``(batch, seq_len_k, num_kv_heads, head_dim)``
        attn_metadata : AttentionMetadata, optional
            Currently unused.

        Returns
        -------
        Tensor
            ``(batch, seq_len_q, num_heads, head_dim)``
        """
        if self._use_flash:
            return self._forward_flash(query, key, value)
        return self._forward_sdpa(query, key, value)

    # ------------------------------------------------------------------ #
    # sgl-kernel flash attention path (via _flash_attn_varlen_func)
    # ------------------------------------------------------------------ #
    def _forward_flash(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Call the cutlass-based ``torch.ops.sgl_kernel.fwd`` kernel via
        our ``_flash_attn_varlen_func`` wrapper (27-arg signature).

        Input layout: ``(batch, seq_len, heads, head_dim)``.
        ``_flash_attn_varlen_func`` handles reshaping to varlen format
        internally when ``cu_seqlens_q=None``.
        """
        output = _flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=query.shape[1],
            max_seqlen_k=key.shape[1],
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return output

    # ------------------------------------------------------------------ #
    # SDPA fallback path
    # ------------------------------------------------------------------ #
    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # SDPA expects (batch, heads, seq_len, head_dim)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        attn_kwargs = {
            "attn_mask": None,
            "dropout_p": self.dropout,
            "is_causal": self.causal,
            "scale": self.softmax_scale,
        }
        if q.shape[1] != k.shape[1]:
            attn_kwargs["enable_gqa"] = True

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, **attn_kwargs)
        # back to (batch, seq_len, heads, head_dim)
        return out.transpose(1, 2)
