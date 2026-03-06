import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def xla_matmul_1(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    # First we want to all_gather the data
    with jax.named_scope('lhs all_gather'):
        lhs_full = jax.lax.all_gather(lhs, 'y', axis=1, tiled=True)
    with jax.named_scope('rhs all_gather'):
        rhs_full = jax.lax.all_gather(rhs, 'x', axis=0, tiled=True) # gather w along x
    # Then we want to compute on the data
    # Since we did two full all gathers to start, regular GEMMs
    with jax.named_scope('gemm'):
        local_out = lhs_full @ rhs_full
    return local_out

def xla_matmul_2(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    # All gather the weights over x so that each device contains full copy
    with jax.named_scope('rhs all_gather'):
        rhs_full = jax.lax.all_gather(rhs, 'x', axis=0, tiled=True)
    """
    Once we have the full weight matrix on each device, we know that our y_idx
    will let us pluck out columns of inputs, but we need to make sure
    that we're plucking out the appropriate rows of the weights
    """
    this_y = jax.lax.axis_index('y')
    y_len = jax.lax.axis_size('y')
    k, n = rhs_full.shape
    k_block = k // y_len

    # Using the y-ring axis to determined which col stripe of weights to compute locally
    rhs_slice = jax.lax.dynamic_slice(rhs_full, (this_y * k_block, 0), (k_block, n))
    with jax.named_scope('gemm'):
        local_out = lhs @ rhs_slice
    # All Reduce over the y-ring to accumulate partial results
    with jax.named_scope('all reduce'):
        out = jax.lax.psum(local_out, 'y')
    return out

def xla_matmul_3(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """
    Use some higher precision numerics to demonstrate accumulation order (fp32)
    """
    with jax.named_scope('rhs all_gather'):
        rhs_full = jax.lax.all_gather(rhs, 'x', axis=0, tiled=True)
    this_y = jax.lax.axis_index('y')
    y_len = jax.lax.axis_size('y')
    k, n = rhs_full.shape
    k_block = k // y_len
    rhs_slice = jax.lax.dynamic_slice(rhs_full, (this_y * k_block, 0), (k_block, n))

    with jax.named_scope('f32 gemm'):
        local_out = jax.lax.dot_general(
            lhs, rhs_slice,
            dimension_numbers=(((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
    with jax.named_scope('all reduce'):
        out = jax.lax.psum(local_out, 'y')
    return out

def pperm_xla_matmul(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
  """
  In this scenario, we realize that we actaully don't need to send the full
  data both ways. That is, in our simple square, we recognize that only the
  off-diagonal devices _need_ the data
  """
  this_x = jax.lax.axis_index('x')
  # Swap the shards along the x mesh axis
  with jax.named_scope('ppermute'):
    rhs_remote = jax.lax.ppermute(rhs, 'x', perm=[(0, 1), (1, 0)])

  # Build the shards
  with jax.named_scope('build rhs'):
    w_full = jax.lax.cond(
        this_x == 0,
        lambda _: jax.lax.concatenate([rhs, rhs_remote], dimension=0),
        lambda _: jax.lax.concatenate([rhs_remote, rhs], dimension=0),
        operand=None
    ) # (K, N)

  this_y  = jax.lax.axis_index('y')
  y_len = jax.lax.axis_size('y')

  k, n = w_full.shape
  k_block = k // y_len

  w_slice = jax.lax.dynamic_slice(w_full, (this_y * k_block, 0), (k_block, n))
  with jax.named_scope('local gemm'):
    local_out = lhs @ w_slice
  with jax.named_scope('all reduce'):
    out = jax.lax.psum(local_out, 'y')
  return out

def pperm_xla_matmul2(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
  """
  In this scenario, we realize that we actaully don't need to send the full
  data both ways. That is, in our simple square, we recognize that only the
  off-diagonal devices _need_ the data
  """
  this_x = jax.lax.axis_index('x')
  this_y  = jax.lax.axis_index('y')
  # Swap the shards along the x mesh axis
  with jax.named_scope('ppermute'):
    rhs_remote = jax.lax.ppermute(rhs, 'x', perm=[(0, 1), (1, 0)])

  with jax.named_scope('build rhs'):
    w_slice = jax.lax.select(this_x == this_y, rhs, rhs_remote)
  
  with jax.named_scope('local gemm'):
    out = jax.lax.psum(lhs @ w_slice, 'y')

  return out
