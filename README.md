# sfp

Minimal infrastructure for TPU kernel development with Pallas.

## Setup

1. Configure GCP project:
   ```bash
   echo 'export GCP_PROJECT="your-project-id"' > infra/config.local.sh
   ```

2. Create a GCS bucket for traces:
   ```bash
   gsutil mb gs://your-bucket-name
   ```

## Usage

```bash
# Create a TPU VM (v5e-1 default, spot pricing, 3h TTL)
./infra/create-vm.sh

# Or specify a configuration
./infra/create-vm.sh --type v6e-4

# Connect
./infra/connect.sh

# On VM: clone repo and install deps
./scripts/setup-vm.sh

# Run kernels
uv run python kernels/my_kernel.py

# Tear down
./infra/delete-vm.sh
```

## Profiling

```python
import jax

jax.profiler.start_trace("./traces")
# kernel code
jax.profiler.stop_trace()
```

View with `xprof ./traces --port=8791` or drag into [Perfetto](https://ui.perfetto.dev).
