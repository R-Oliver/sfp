import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _gemm_kernel(x_ref, y_ref, o_ref, scratch_ref, *, n_steps):
  # Zero scratch buffer
  with jax.named_scope('Zero Scratch Buffer'):
    @pl.when(pl.program_id(2) == 0)
    def _init_scratch():
      scratch_ref[...] = jnp.zeros_like(scratch_ref)

  # Compute dot
  with jax.named_scope('Compute GEMM'):
    scratch_ref[...] += jnp.dot(
      x_ref[...],
      y_ref[...],
      preferred_element_type=jnp.float32
    )

  # Flush to HBM
  with jax.named_scope('Flush to HBM'):
    @pl.when(pl.program_id(2) == n_steps - 1)
    def _flush_scratch():
      o_ref[...] = scratch_ref[...].astype(o_ref.dtype)


def make_matmul(
  x: jax.Array,
  y: jax.Array,
  *,
  bm: int = 128,
  bk: int = 128,
  bn: int = 128,
):
  m, k = x.shape
  _, n = y.shape

  grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=0,
    grid=(m//bm, n//bn, k//bk),
    in_specs=[
      pl.BlockSpec((bm, bk), lambda i,j,k: (i, k)),
      pl.BlockSpec((bk, bn), lambda i,j,k: (k, j))
    ],
    out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
    scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)]
  )

  return pl.pallas_call(
    functools.partial(_gemm_kernel, n_steps=k//bk),
    grid_spec=grid_spec,
    out_shape=jax.ShapeDtypeStruct((m, n), dtype=jnp.bfloat16)
  )(x, y)


def _emit_gemm_pipeline(x_ref, w_ref, o_ref, *, bm, bk, bn):
    """
    Emit a tiled GEMM pipeline.
    All refs are HBM. emit_pipeline handles VMEM tiling + double-buffering.
    """
    m, k_dim = x_ref.shape
    _, n = w_ref.shape
    grid = (m // bm, n // bn, k_dim // bk)

    def body(x_vmem, w_vmem, o_vmem, accum):
        @pl.when(pl.program_id(2) == 0)
        def _():
            accum[...] = jnp.zeros_like(accum)

        accum[...] += jnp.dot(
            x_vmem[...], w_vmem[...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
        def _():
            o_vmem[...] = accum[...].astype(o_vmem.dtype)

    @functools.partial(pl.run_scoped, accum=pltpu.VMEM((bm, bn), jnp.float32))
    def _(accum):
        pltpu.emit_pipeline(
            functools.partial(body, accum=accum),
            grid=grid,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        )(x_ref, w_ref, o_ref)


def all_gather_kernel_1D(
  input_ref, output_ref,
  local_send_sem, send_sem, recv_sem,
):
  """
  input_ref: shard local data
  output_ref: out shard

  local_send_sem: allocates a semaphore for the local HBM copy
  send_sem: semaphore for the RDMA push
  recv_sem: semaphore for our local data
  """
  # Ignoring Barrier Sem for now

  pid = pl.program_id(0)
  shard_height = input_ref.shape[0]

  # Get neighbors
  x_len = jax.lax.axis_size('x')
  this_x = jax.lax.axis_index('x')
  this_y = jax.lax.axis_index('y')
  right_device_x = jax.lax.rem(this_x + 1, x_len)

  # Recall: This is the _destination_ copy slot
  """
  Imagine that you're in a length 4 ring at position 0
    - Iter 1: You pass your data (say row 0) to the right
      - You receieve row 1 from your right, and row 9 from your left
    - Iter 2: You have (0, 1, X, 3)
      - If you're sending right; You need to send row 3
      - B/c Device 1 has (0, 1, 2, X)
      - Basically, imagine passing what you just received from your left, to the right
  """
  copy_slot_xright = jax.lax.rem(this_x - pid, x_len)

  # We're just copying within our HBM to a bigger HBM memory
  with jax.named_scope("Local HBM Copy"):
    @pl.when(pl.program_id(0) == 0)
    def _copy_local_to_local():
      local_hbm_copy = pltpu.make_async_copy(
        src_ref=input_ref,
        dst_ref=output_ref.at[pl.ds(this_x * shard_height, shard_height), :],
        sem=local_send_sem
      )
      with jax.named_scope("Local Copy Start"):
        local_hbm_copy.start()
      with jax.named_scope("Local Copy Wait"):
        local_hbm_copy.wait()

  right_dma = pltpu.make_async_remote_copy(
    src_ref=output_ref.at[pl.ds(copy_slot_xright * shard_height, shard_height), :],
    dst_ref=output_ref.at[pl.ds(copy_slot_xright * shard_height, shard_height), :],
    send_sem=send_sem,
    recv_sem=recv_sem,
    device_id=(right_device_x, this_y),
    device_id_type=pltpu.DeviceIdType.MESH,
  )

  with jax.named_scope("Right DMA Start"):
    right_dma.start()
  with jax.named_scope("Right DMA Wait"):
    right_dma.wait()


def make_ag_1D(x):
  rows, cols = x.shape
  x_len = jax.lax.axis_size('x')

  grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=0,
    # The pipeline for 2x2 mesh is one hop
    grid=(1,),
    in_specs=[
      # Our input reference is just a chunk in HBM
      pl.BlockSpec(memory_space=pl.ANY)
    ],
    # Our output reference will be another chunk in HBM
    out_specs=pl.BlockSpec(memory_space=pl.ANY),
    scratch_shapes=(
      [pltpu.SemaphoreType.DMA] * 2 # local_copy_op, send_sem
      + [pltpu.SemaphoreType.DMA] * 1 # These are our recv_sems. For 2x2, we only need 1 of them
    )
  )

  out_shape = jax.ShapeDtypeStruct((rows * x_len, cols), dtype=jnp.bfloat16)

  return pl.pallas_call(
    all_gather_kernel_1D,
    grid_spec=grid_spec,
    out_shape=out_shape
  )(x)


def all_gather_kernel_bidi(
  input_ref, output_ref,
  local_send_sem, send_sem_right, send_sem_left,
  recv_sem_right, recv_sem_left
):
  """
  input_ref: shard local data
  output_ref: out shard

  local_send_sem: allocates a semaphore for the local HBM copy
  send_sem: semaphore for the RDMA push
  recv_sem: semaphore for our local data
  """

  pid = pl.program_id(0)

  shard_height = input_ref.shape[0]

  this_x = jax.lax.axis_index('x')
  this_y = jax.lax.axis_index('y')
  x_len = jax.lax.axis_size('x')
  right_device_x = jax.lax.rem(this_x + 1, x_len)
  # Don't want negatives
  left_device_x = jax.lax.rem(this_x - 1 + x_len, x_len)

  # This accounts for the offset when data is being sent both ways
  copy_slot_right = jax.lax.rem(this_x - pid + x_len, x_len)
  copy_slot_left = jax.lax.rem(this_x + pid, x_len)

  # We're just copying within our HBM to a bigger HBM memory
  @pl.when(pid == 0)
  def _prologue_sends():
    local_copy = pltpu.make_async_copy(
    src_ref=input_ref,
    dst_ref=output_ref.at[pl.ds(this_x * shard_height, shard_height), :],
    sem=local_send_sem
  )

    with jax.named_scope('Local Copy Start'):
      local_copy.start()
    with jax.named_scope('Local Copy Wait'):
      local_copy.wait()

  right_dma = pltpu.make_async_remote_copy(
    # Next kernel iter depends on completion of left/right DMAs
    src_ref=output_ref.at[pl.ds(copy_slot_right * shard_height, shard_height), :],
    dst_ref=output_ref.at[pl.ds(copy_slot_right * shard_height, shard_height), :],
    send_sem=send_sem_right,
    recv_sem=recv_sem_right,
    device_id=(right_device_x, this_y),
    device_id_type=pltpu.DeviceIdType.MESH,
  )

  left_dma = pltpu.make_async_remote_copy(
    src_ref=output_ref.at[pl.ds(copy_slot_left * shard_height, shard_height), :],
    dst_ref=output_ref.at[pl.ds(copy_slot_left * shard_height, shard_height), :],
    send_sem=send_sem_left,
    recv_sem=recv_sem_left,
    device_id=(left_device_x, this_y),
    device_id_type=pltpu.DeviceIdType.MESH
  )

  with jax.named_scope('Right DMA Start'):
    right_dma.start()
  with jax.named_scope('Left DMA Start'):
    left_dma.start()
  with jax.named_scope('Right DMA Wait'):
    right_dma.wait()
  with jax.named_scope('Left DMA Wait'):
    left_dma.wait()

  @pl.when(pl.program_id(0) == pl.num_programs(0) - 1)
  def _epilogue():
    right_dma = pltpu.make_async_remote_copy(
      # Next kernel iter depends on completion of left/right DMAs
      src_ref=output_ref.at[pl.ds(copy_slot_right * shard_height, shard_height), :],
      dst_ref=output_ref.at[pl.ds(copy_slot_right * shard_height, shard_height), :],
      send_sem=send_sem_right,
      recv_sem=recv_sem_right,
      device_id=(right_device_x, this_y),
      device_id_type=pltpu.DeviceIdType.MESH,
    )

    with jax.named_scope('Epilogue DMA Start'):
      right_dma.start()
    with jax.named_scope('Epilogue DMA Wait'):
      right_dma.wait()

def make_ag(x):
  x_len = jax.lax.axis_size('x')
  m, k = x.shape

  grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=0,
    # Sort of assuming evenness
    grid=(x_len // 2,),
    in_specs=[
      pl.BlockSpec(memory_space=pl.ANY)
    ],
    out_specs=pl.BlockSpec(memory_space=pl.ANY),
    # This will be an error if you need more semaphores for more neighbors
    scratch_shapes=(
      [pltpu.SemaphoreType.DMA] * 3 # local_copy_op, send_sem_left, send_sem_right
      + [pltpu.SemaphoreType.DMA] * 2 # recv right/left
    )
  )

  out_shape = jax.ShapeDtypeStruct((x_len * m, k), dtype=jnp.bfloat16)

  return pl.pallas_call(
    all_gather_kernel_bidi,
    grid_spec=grid_spec,
    out_shape=out_shape
  )(x)


def all_reduce_kernel_1D(
    local_hbm_ref, output_ref,
    send_sem, recv_sem, copy_sem,
    local_scratch, recv_scratch
):
    y_len = jax.lax.axis_size('y')
    this_x = jax.lax.axis_index('x')
    this_y = jax.lax.axis_index('y')
    right_device_y = jax.lax.rem(this_y + 1, y_len)

    local_copy = pltpu.make_async_copy(
    src_ref=local_hbm_ref,
    dst_ref=local_scratch,
    sem=copy_sem
    )

    with jax.named_scope("Local Copy"):
        local_copy.start()
        local_copy.wait()

    # This will copy our HBM tile into either:
    #  - Remote HBM Tile
    #    - Right now our GEMM works on HBM, so this will be easier temorarily
    #  - Remote VMEM tile (Memory pressure)
    send_ref = local_scratch
    for _ in range(y_len - 1):
        right_dma = pltpu.make_async_remote_copy(
            src_ref=send_ref,
            dst_ref=recv_scratch,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(this_x, right_device_y),
            device_id_type=pltpu.DeviceIdType.MESH
        )

        with jax.named_scope("Right DMA Start"):
            right_dma.start()
        with jax.named_scope("Right DMA Wait"):
            right_dma.wait()

        # Add in VMEM, write back to HBM
        with jax.named_scope("Add Remote Data to Local"):
            local_scratch[...] = local_scratch[...] + recv_scratch[...]

        with jax.named_scope("Write remote data to send slot"):
          send_ref = recv_scratch

    out_copy = pltpu.make_async_copy(
          src_ref=local_scratch,
          dst_ref=output_ref,
          sem=copy_sem
      )

    with jax.named_scope("Copy Out Start"):
        out_copy.start()
    with jax.named_scope("Copy Out Wait"):
        out_copy.wait()

def make_ar_1D(x, bm=1024, bn=1024):
    m_local, n_local = x.shape

    grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=(m_local // bm, n_local // bn),
      in_specs=[
          pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
      scratch_shapes=(
          [pltpu.SemaphoreType.DMA] * 3 # send_sem, recv_sem, copy_sem
          + [pltpu.VMEM((bm, bn), jnp.bfloat16)] # local_scratch
          + [pltpu.VMEM((bm, bn), jnp.bfloat16)] # recv_scratch
          )
    )

    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

    return pl.pallas_call(
        all_reduce_kernel_1D,
        grid_spec=grid_spec,
        out_shape=out_shape,
    )(x)


def fused_ag_gemm_kernel(
    input_ref,          # HBM: inputs shard (m_local, k_local)
    weight_ref,         # HBM: weights shard (k_local, n)
    output_ref,         # HBM: GEMM output (m_local, n)
    recv_weight_ref,    # HBM: workspace for received weights (k_local, n)
    send_sem, recv_sem, # RDMA semaphores
):
    """
    Fused AG + GEMM via collective permute.

    Outer kernel manages weight exchange (RDMA along x-ring).
    Inner pipelines handle the tiled matmul.

    On half the devices (x_idx == y_idx) the local weights are
    already correct, so GEMM runs entirely overlapped with RDMA.
    On the other half (x_idx != y_idx) we wait for the remote
    chunk and then recompute
    """
    this_x = jax.lax.axis_index('x')
    this_y = jax.lax.axis_index('y')
    x_len = jax.lax.axis_size('x')
    right_neighbor = jax.lax.rem(this_x + 1, x_len)

    BM, BK, BN = 512, 1024, 1024

    # Each device sends its shard right and receives from its left neighbor
    rdma = pltpu.make_async_remote_copy(
        src_ref=weight_ref,
        dst_ref=recv_weight_ref,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=(right_neighbor, this_y),
        device_id_type=pltpu.DeviceIdType.MESH,
    )

    with jax.named_scope("Start weight exchange DMA"):
        rdma.start()

    # When x_idx == y_idx this is correct
    # When x_idx != y_idx this is wasted compute
    # MXU busy while ICI transfers the weight shard we need.
    with jax.named_scope("On Diagonal GEMM"):
        @pl.when(this_x == this_y)
        def _on_diag_gemm():
            _emit_gemm_pipeline(input_ref, weight_ref, output_ref, bm=BM, bk=BK, bn=BN)

    with jax.named_scope("Wait for Weight Exchange"):
        rdma.wait()

    # Recompute with correct weights
    # Causes long wait on 2 of our devices (diagonals waiting)
    with jax.named_scope("Off Diagonal GEMM"):
        @pl.when(this_x != this_y)
        def _():
            _emit_gemm_pipeline(input_ref, recv_weight_ref, output_ref, bm=BM, bk=BK, bn=BN)


def make_fused_ag_gemm(lhs, rhs):
    m_local, k_local = lhs.shape
    _, n = rhs.shape
    x_len = jax.lax.axis_size('x')

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(x_len // 2,),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),  # inputs
            pl.BlockSpec(memory_space=pl.ANY),  # weights
        ],
        # Two outputs: real output + HBM workspace for received weights
        out_specs=[
            pl.BlockSpec(memory_space=pl.ANY),  # GEMM output
            pl.BlockSpec(memory_space=pl.ANY),  # recv weight buffer
        ],
        scratch_shapes=(
            [pltpu.SemaphoreType.DMA]  # send_sem
            + [pltpu.SemaphoreType.DMA] # recv_sem
        ),
    )

    out_shape = [
        jax.ShapeDtypeStruct((m_local, n), lhs.dtype),   # GEMM output
        jax.ShapeDtypeStruct((k_local, n), rhs.dtype),  # recv workspace
    ]

    results = pl.pallas_call(
        fused_ag_gemm_kernel,
        grid_spec=grid_spec,
        out_shape=out_shape,
    )(lhs, rhs)

    return results[0]  # discard the workspace
