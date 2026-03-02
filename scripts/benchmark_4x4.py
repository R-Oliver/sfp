"""
Naive JAX matmul on a 4x4 mesh (v5e-16, 2 hosts x 8 devices).
Captures HLO/LLO dumps + a JAX profile trace, uploads to GCS.

Requires GCS_BUCKET env var on the VM (config.local.sh isn't committed):
  export GCS_BUCKET=gs://your-bucket-name

Run on all workers simultaneously:
  gcloud compute tpus tpu-vm ssh $VM_NAME \
      --zone=$ZONE \
      --worker=all \
      --command="cd sfp && uv run python scripts/benchmark_4x4.py"
"""
import os
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

os.environ["XLA_FLAGS"] = (
    f"--xla_dump_hlo_as_text "
    f"--xla_dump_to={HLO_DIR} "
    f"--xla_dump_hlo_pass_re=spmd-partitioner|collective-pipeliner|async-collective|latency-hiding "
)

os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={LLO_DIR} "
    f"--xla_jf_dump_hlo_text=true "
    f"--xla_jf_dump_llo_text=true "
    f"--xla_jf_dump_llo_html=false "
    f"--xla_jf_dump_llo_static_gaps=true "
    f"--xla_jf_emit_annotations=true "
    f"--xla_jf_debug_level=2 "
    f"--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true "
)

import jax

jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from sfp.utils import profile, upload_to_gcs

is_host0 = jax.process_index() == 0

if is_host0:
    print(f"Process {jax.process_index()}/{jax.process_count()} | "
          f"Local devices: {jax.local_device_count()} | "
          f"Total devices: {jax.device_count()}")
    print(f"Mesh: {MESH_X}x{MESH_Y}")

mesh = jax.make_mesh((MESH_X, MESH_Y), ("x", "y"))

m, k, n = 16384, 16384, 8192
k1, k2 = jax.random.split(jax.random.key(0), 2)
inputs = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16)
weights = jax.random.normal(k2, (k, n), dtype=jnp.bfloat16)

inputs = jax.device_put(inputs, NamedSharding(mesh, P("x", "y")))
weights = jax.device_put(weights, NamedSharding(mesh, P("x", None)))


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

# Profile
if is_host0:
    print("Profiling...", end=" ", flush=True)
with profile(name="xla_4x4_baseline") as trace_path:
    result = matmul(inputs, weights)
    result.block_until_ready()
if is_host0:
    print(f"done. Trace at {trace_path}")

# Upload artifacts to GCS (host 0 only)
if is_host0:
    gcs_prefix = f"v5e_{MESH_X}x{MESH_Y}/xla_baseline"
    try:
        gcs_uri = upload_to_gcs(trace_path, prefix=gcs_prefix)
        print(f"Trace uploaded to {gcs_uri}")
        gcs_uri = upload_to_gcs(HLO_DIR, prefix=gcs_prefix)
        print(f"HLO uploaded to {gcs_uri}")
        gcs_uri = upload_to_gcs(LLO_DIR, prefix=gcs_prefix)
        print(f"LLO uploaded to {gcs_uri}")
    except (ValueError, FileNotFoundError) as e:
        print(f"GCS upload skipped: {e}", file=sys.stderr)
        print(f"Set GCS_BUCKET env var to enable upload.")
        print(f"Artifacts saved locally at {OUTPUT_DIR}")
