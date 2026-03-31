import sys
from pathlib import Path

MESH_X, MESH_Y = 4, 4

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "sharded_results" / f"v5e_{MESH_X}x{MESH_Y}" / "xla_baseline"
HLO_DIR = OUTPUT_DIR / "hlo"
LLO_DIR = OUTPUT_DIR / "llo"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HLO_DIR.mkdir(parents=True, exist_ok=True)
LLO_DIR.mkdir(parents=True, exist_ok=True)

import jax

jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from sfp.utils import profile

is_host0 = jax.process_index() == 0

if is_host0:
    print(f"Process {jax.process_index()}/{jax.process_count()} | "
          f"Local devices: {jax.local_device_count()} | "
          f"Total devices: {jax.device_count()}")
    print(f"Mesh: {MESH_X}x{MESH_Y}")

mesh = jax.make_mesh((MESH_X, MESH_Y), ("x", "y"))

m, k, n = 16384, 16384, 8192
inp_sharding = NamedSharding(mesh, P("x", "y"))
w_sharding = NamedSharding(mesh, P("x", None))

@jax.jit
def create_data(key):
    k1, k2 = jax.random.split(key)
    inputs = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16)
    weights = jax.random.normal(k2, (k, n), dtype=jnp.bfloat16)
    return jax.device_put(inputs, inp_sharding), jax.device_put(weights, w_sharding)

inputs, weights = create_data(jax.random.key(0))


@jax.jit
def matmul(x, y):
    return jnp.matmul(x, y)


# Warmup (triggers compilation, HLO/LLO dumps land here)
if is_host0:
    print("Compiling...", end=" ", flush=True)
result = matmul(inputs, weights)
result.block_until_ready()
if is_host0:
    print("done.")

# Profile — all hosts write directly to GCS when GCS_BUCKET is set
gcs_prefix = f"v5e_{MESH_X}x{MESH_Y}/xla_baseline"
if is_host0:
    print("Profiling...", end=" ", flush=True)
with profile(name="xla_4x4_baseline", gcs_prefix=gcs_prefix) as trace_loc:
    result = matmul(inputs, weights)
    result.block_until_ready()
if is_host0:
    print(f"done. Trace at {trace_loc}")
