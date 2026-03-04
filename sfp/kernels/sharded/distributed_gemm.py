import jax

from sfp.kernels.sharded.primitives import make_matmul, make_fused_ag_gemm, make_ag_1D, make_ar_1D


def jax_pallas_gemm(lhs, rhs):
  rhs_full = jax.lax.all_gather(rhs, 'x', axis=0, tiled=True)
  this_y = jax.lax.axis_index('y')
  y_len = jax.lax.axis_size('y')
  k, n = rhs_full.shape
  k_block = k // y_len

  w_slice = jax.lax.dynamic_slice(rhs_full, (this_y * k_block, 0), (k_block, n))
  local_out = make_matmul(lhs, w_slice, bm=512, bk=1024, bn=1024)
  return jax.lax.psum(local_out, 'y')


def ag_gemm_ar_serial(lhs, rhs):
    this_y = jax.lax.axis_index('y')
    y_len = jax.lax.axis_size('y')
    ag = make_ag_1D(rhs)

    k_full, n = ag.shape
    k_block = k_full // y_len

    a_slice = jax.lax.dynamic_slice(ag, (this_y * k_block, 0), (k_block, n))
    gemm = make_matmul(lhs, a_slice, bm=512, bk=1024, bn=1024)
    return make_ar_1D(gemm)


def fused_ag_gemm_ar(lhs, rhs):
    partial = make_fused_ag_gemm(lhs, rhs)
    return make_ar_1D(partial)
