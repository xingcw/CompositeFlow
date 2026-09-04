"""JAX/TPU implementation of Composite Flow Matching.

Sets float32 matmul precision at import: TPUs default to bfloat16, which costs
4e-3 relative accuracy against the reference. See docs/BACKEND.md.
"""

import os

import jax

jax.config.update(
    "jax_default_matmul_precision",
    os.environ.get("COMP_FLOW_MATMUL_PRECISION", "highest"),
)
