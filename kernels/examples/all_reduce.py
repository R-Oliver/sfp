import functools
import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#

P = jax.sharding.PartitionSpec
num_devices = jax.local_device_count()

partition = P(None, 'x')
mesh = jax.make_mesh((num_devices,), ('x',))
sharding = jax.sharding.NamedSharding(mesh, partition)

input_arr = jax.random.uniform(jax.random.key(0), shape=(8, 128 * num_devices))
input_arr = jax.device_put(input_arr, sharding)


def local_barrier(left_neighbor, right_neighbor, double_barrier=True):
  """Performs a barrier with neighbors on the global barrier semaphore.

  Optionally performs a second barrier, which prevents a potential race
  when reusing the same collective_id across kernel invocations.
  """
  barrier_sem = pltpu.get_barrier_semaphore()
  for neighbor in [left_neighbor, right_neighbor]:
    pltpu.semaphore_signal(
      barrier_sem,
      inc=1,
      device_id=(neighbor,),
      device_id_type=pltpu.DeviceIdType.MESH,
    )
  pltpu.semaphore_wait(barrier_sem, 2)
  if double_barrier:
    # The double-barrier prevents a race condition where one neighbor can
    # re-enter the kernel again on a subsequent call and increment the
    # barrier semaphore a second time. This would unblock the current device
    # even if the other neighbor is not ready yet.
    # To implement a double-barrier, we stack-allocate a second REGULAR
    # semaphore using run_scoped.
    @functools.partial(pl.run_scoped,
                       second_barrier=pltpu.SemaphoreType.REGULAR)
    def _(second_barrier):
      for neighbor in [left_neighbor, right_neighbor]:
        pltpu.semaphore_signal(
          second_barrier,
          inc=1,
          device_id=(neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
      pltpu.semaphore_wait(second_barrier, 2)


def all_reduce_kernel(
    x_ref,
    o_ref,
    hbm_scratch,
    copy_sem,
    remote_recv_sem,
    remote_send_sem,
    capacity_sem,
    receive_scratch,
):
  outer_step = pl.program_id(0)
  working_slot = lax.rem(outer_step, 2)
  receiving_slot = 1 - working_slot

  my_id = lax.axis_index('x')
  right_neighbor = lax.rem(my_id + 1, num_devices)
  left_neighbor = lax.rem(my_id - 1 + num_devices, num_devices)

  @pl.when(outer_step == 0)
  def _():
    # Barrier with both neighbors at the start, since we will be
    # communicating with both.
    local_barrier(left_neighbor, right_neighbor)

    # Initialize o_ref, acc_scratch, and hbm_scratch.
    o_ref[...] = jnp.zeros_like(o_ref)
    receive_scratch[...] = jnp.zeros_like(receive_scratch)
    initial_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref,
        dst_ref=hbm_scratch.at[working_slot],
        send_sem=remote_send_sem,
        recv_sem=remote_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
    )
    initial_copy.start()
    initial_copy.wait()

  # Signal to our left neighbor that we are ready to receive.
  # Without this signal, our left neighbor can be >=1 iteration ahead,
  # meaning it could write into our working slot.
  pltpu.semaphore_signal(
      capacity_sem,
      inc=1,
      device_id=(left_neighbor,),
      device_id_type=pltpu.DeviceIdType.MESH,
  )

  # Copy the partial result our left neighbor sent to us into VMEM for
  # computation.
  local_copy = pltpu.make_async_copy(
      src_ref=hbm_scratch.at[working_slot],
      dst_ref=receive_scratch,
      sem=copy_sem,
  )
  local_copy.start()

  # Block until our right neighbor is ready to receive.
  pltpu.semaphore_wait(capacity_sem, 1)
  # Pass the value to our right neighbor.
  remote_copy = pltpu.make_async_remote_copy(
      src_ref=hbm_scratch.at[working_slot],
      dst_ref=hbm_scratch.at[receiving_slot],
      send_sem=remote_send_sem,
      recv_sem=remote_recv_sem,
      device_id=(right_neighbor,),
      device_id_type=pltpu.DeviceIdType.MESH,
  )
  remote_copy.start()
  # Finish local copy and accumulate while remote_copy is happening.
  local_copy.wait()
  o_ref[...] += receive_scratch[...]
  # Block until remote copy finishes.
  remote_copy.wait()


out_shape = (
    jax.ShapeDtypeStruct((8, 128), jnp.float32),
    # We allocate the double-buffer as a Pallas output so that it is
    # resident in HBM.
    jax.ShapeDtypeStruct((2, 8, 128), jnp.float32),  # hbm_scratch
)

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=0,
    in_specs=[
        # Our input lives in VMEM
        pl.BlockSpec(memory_space=pltpu.VMEM),
    ],
    out_specs=[
        # Our output lives in VMEM
        pl.BlockSpec(memory_space=pltpu.VMEM),
        # Our double-buffer lives in HBM
        pl.BlockSpec(memory_space=pl.ANY),
    ],
    grid=(num_devices,),
    scratch_shapes=(
        [pltpu.SemaphoreType.DMA] * 3
        + [pltpu.SemaphoreType.REGULAR]  # capacity_sem
        + [pltpu.VMEM((8, 128), jnp.float32)]  # receive_scratch
    ),
)

kernel = pl.pallas_call(
    all_reduce_kernel,
    out_shape=out_shape,
    grid_spec=grid_spec,
    compiler_params=pltpu.CompilerParams(collective_id=0),
)

pallas_result = jax.jit(
    jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_vma=False,
    )
)(input_arr)
pallas_result = jax.block_until_ready(pallas_result)[0]


def lax_sum(x):
  return lax.psum(x, 'x')


xla_result = jax.jit(
    jax.shard_map(
        lax_sum, mesh=mesh, in_specs=P(None, 'x'), out_specs=P(None, 'x')
    )
)(input_arr)

print('Input = ', input_arr[0, ::128])
print('Pallas result = ', pallas_result[0, ::128])
print('lax.psum result = ', xla_result[0, ::128])
difference = jnp.mean(jnp.abs(pallas_result - xla_result))
print('Difference |Pallas - lax.psum| = ', difference)

