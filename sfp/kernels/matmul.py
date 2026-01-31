import functools
from typing import Callable

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import numpy as np


def naive_matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, n_steps):
  with jax.named_scope('Init Scratch'):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

  with jax.named_scope('Run Dot'):
    acc_ref[...] += jnp.dot(
        x_ref[...], y_ref[...], preferred_element_type=jnp.float32
    )

  with jax.named_scope('Flush to HBM'):
    @pl.when(pl.program_id(2) == n_steps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


def naive_matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
) -> Callable:
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(naive_matmul_kernel, n_steps=k // bk),
      # Need Scalar Prefetch to allocate scratch space
      grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(
          # These map to no-op on single device
          # Useful in megacore, where arbitrary basically tells
          # Mosaic that it's not free to parallelize that grid
          # dim. Esp. here, where the reduction axis accumulates
          # into scratch. We need to impose that ordering constraint
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)


m, k, n = 4096, 4096, 4096
k1, k2 = jax.random.split(jax.random.key(0), 2)
x = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16)
y = jax.random.normal(k2, (k, n), dtype=jnp.bfloat16)

jitted = jax.jit(naive_matmul, static_argnames=['bm', 'bn', 'bk'])
compiled = jitted.lower(x, y, bm=512, bk=1024, bn=1024).compile({'xla_enable_transpose_trace': True})
result = compiled(x, y)
result.block_until_ready()

np.testing.assert_array_almost_equal(x @ y, naive_matmul(x, y))

with jax.profiler.trace('./traces'):
   for _ in range(100):
    result = compiled(x, y)
   result.block_until_ready()

