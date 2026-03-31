import argparse
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

from sfp.utils import profile

is_host0 = jax.process_index() == 0

if is_host0:
    print(f"Process {jax.process_index()}/{jax.process_count()} | "
          f"Local devices: {jax.local_device_count()} | "
          f"Total devices: {jax.device_count()}")
    print(f"Mesh: {MESH_X}x{MESH_Y} | Shape: ({args.m},{args.k}) @ ({args.k},{args.n})")

mesh = jax.make_mesh((MESH_X, MESH_Y), ("x", "y"))

inp_sharding = NamedSharding(mesh, P("x", "y"))
w_sharding = NamedSharding(mesh, P("x", None))

@jax.jit
def create_data(key):
    k1, k2 = jax.random.split(key)
    inputs = jax.random.normal(k1, (args.m, args.k), dtype=jnp.bfloat16)
    weights = jax.random.normal(k2, (args.k, args.n), dtype=jnp.bfloat16)
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
with profile(name=f"xla_{MESH_X}x{MESH_Y}_baseline", gcs_prefix=gcs_prefix) as trace_loc:
    result = matmul(inputs, weights)
    result.block_until_ready()
if is_host0:
    print(f"done. Trace at {trace_loc}")
