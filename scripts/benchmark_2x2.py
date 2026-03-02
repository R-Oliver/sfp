import os
import sys
from pathlib import Path

MESH_X, MESH_Y = 2, 2

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
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from sfp.utils import profile, upload_to_gcs

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
print("Compiling...", end=" ", flush=True)
result = matmul(inputs, weights)
result.block_until_ready()
print("done.")

# Profile
print("Profiling...", end=" ", flush=True)
with profile(name="xla_baseline") as trace_path:
    result = matmul(inputs, weights)
    result.block_until_ready()
print(f"done. Trace at {trace_path}")

# Upload artifacts to GCS
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
