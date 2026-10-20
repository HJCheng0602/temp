"""Opt-in, model-local GEMM dispatch hooks for the SM120 trial."""

from .dense_fast import (
    DenseDispatch,
    build_wo_a_extension,
    install_dense_mm_mxfp8,
    install_wo_a,
    warmup_wo_a,
    wo_a_bmm,
)

__all__ = [
    "DenseDispatch",
    "build_wo_a_extension",
    "install_dense_mm_mxfp8",
    "install_wo_a",
    "warmup_wo_a",
    "wo_a_bmm",
]
