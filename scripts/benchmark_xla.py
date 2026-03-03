"""
Dumbest possible sharded matmul — just let XLA do its thing.
Captures HLO/LLO dumps + a JAX profile trace, uploads to GCS.

Requires GCS_BUCKET env var on the VM (config.local.sh isn't committed):
  export GCS_BUCKET=gs://your-bucket-name

Run (single-host, e.g. v5e-4):
  uv run python scripts/benchmark_xla.py 2 2

Run (multi-host, e.g. v5e-16):
  gcloud compute tpus tpu-vm ssh $VM_NAME \
      --zone=$ZONE --worker=all \
      --command="cd sfp && uv run python scripts/benchmark_xla.py 4 4"

Run (multi-host, e.g. v5e-64):
  gcloud compute tpus tpu-vm ssh $VM_NAME \
      --zone=$ZONE --worker=all \
      --command="cd sfp && uv run python scripts/benchmark_xla.py 8 8"
"""
import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="XLA baseline sharded matmul benchmark")
parser.add_argument("mesh_x", type=int, help="Mesh dimension along x")
parser.add_argument("mesh_y", type=int, help="Mesh dimension along y")
parser.add_argument("--m", type=int, default=16384)
parser.add_argument("--k", type=int, default=16384)
parser.add_argument("--n", type=int, default=8192)
args = parser.parse_args()

MESH_X, MESH_Y = args.mesh_x, args.mesh_y

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "sharded_results" / f"v5e_{MESH_X}x{MESH_Y}" / "xla_baseline"
HLO_DIR = OUTPUT_DIR / "hlo"
LLO_DIR = OUTPUT_DIR / "llo"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HLO_DIR.mkdir(parents=True, exist_ok=True)
LLO_DIR.mkdir(parents=True, exist_ok=True)

"""
NOTE: Keeping this for reference
This fails with:

> F0303 05:37:58.574267   14193 llo_dumper.cc:471] Check failed: file::GetContents(path, &contents, file::Defaults()) is OK
> (NOT_FOUND: open failed for /home/reed/g3     /platforms/xla/service/jellyfish/tool_data/vmem_report_header.tmpl: No such file or directory

On the cloud VM. Works on Colab for older versions of jax/jaxlib/libtpu (0.7.2 / 0.0.21)

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
"""

import jax

# Safe on single-host (auto-detects single process) and required on multi-host.
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from sfp.utils import profile, upload_to_gcs

is_host0 = jax.process_index() == 0

if is_host0:
    print(f"Process {jax.process_index()}/{jax.process_count()} | "
          f"Local devices: {jax.local_device_count()} | "
          f"Total devices: {jax.device_count()}")
    print(f"Mesh: {MESH_X}x{MESH_Y} | Shape: ({args.m},{args.k}) @ ({args.k},{args.n})")

mesh = jax.make_mesh((MESH_X, MESH_Y), ("x", "y"))

k1, k2 = jax.random.split(jax.random.key(0), 2)
inputs = jax.random.normal(k1, (args.m, args.k), dtype=jnp.bfloat16)
weights = jax.random.normal(k2, (args.k, args.n), dtype=jnp.bfloat16)

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
with profile(name=f"xla_{MESH_X}x{MESH_Y}_baseline") as trace_path:
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
        # Keeping for posterity
        # gcs_uri = upload_to_gcs(HLO_DIR, prefix=gcs_prefix)
        # print(f"HLO uploaded to {gcs_uri}")
        # gcs_uri = upload_to_gcs(LLO_DIR, prefix=gcs_prefix)
        # print(f"LLO uploaded to {gcs_uri}")
    except (ValueError, FileNotFoundError) as e:
        print(f"GCS upload skipped: {e}", file=sys.stderr)
        print(f"Set GCS_BUCKET env var to enable upload.")
        print(f"Artifacts saved locally at {OUTPUT_DIR}")
