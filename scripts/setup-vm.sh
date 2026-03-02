#!/bin/bash
# Setup script for TPU VM - run once after cloning the repo
#
# Single-host:
#   ./scripts/setup-vm.sh
#
# Multi-host:
#   gcloud compute tpus tpu-vm ssh $VM_NAME \
#       --zone=$ZONE --worker=all \
#       --command="cd sfp && ./scripts/setup-vm.sh"
set -euo pipefail

# Detect multi-host environment
WORKER_ID="${TPU_WORKER_ID:-0}"
WORKER_COUNT="${TPU_WORKER_HOSTNAMES:+multi}"
if [[ -n "$WORKER_COUNT" ]]; then
    HOSTS=$(echo "$TPU_WORKER_HOSTNAMES" | tr ',' '\n' | wc -l | tr -d ' ')
    echo "=== SFP VM Setup (worker ${WORKER_ID}/${HOSTS}) ==="
else
    echo "=== SFP VM Setup (single-host) ==="
fi
echo ""

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[worker ${WORKER_ID}] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to current shell (the install script adds to shell config)
    export PATH="$HOME/.local/bin:$PATH"
    echo ""
fi

echo "[worker ${WORKER_ID}] uv version: $(uv --version)"
echo ""

# Sync dependencies (including TPU extras)
echo "[worker ${WORKER_ID}] Syncing dependencies..."
uv sync --extra tpu
echo ""

# Verify JAX installation and TPU access (local devices only)
echo "[worker ${WORKER_ID}] Verifying JAX and local TPU devices..."
uv run python -c "
import jax
print(f'JAX version: {jax.__version__}')
devices = jax.local_devices()
print(f'Local devices: {len(devices)}')
for d in devices:
    print(f'  {d}')
"
echo ""

# Multi-host: verify distributed init works (only meaningful when all workers run)
if [[ -n "$WORKER_COUNT" ]]; then
    echo "[worker ${WORKER_ID}] Verifying distributed JAX..."
    uv run python -c "
import jax
jax.distributed.initialize()
print(f'Process {jax.process_index()}/{jax.process_count()} | '
      f'Local: {jax.local_device_count()} | Total: {jax.device_count()}')
"
    echo ""
fi

# Check GCS access (if configured)
if command -v gsutil &> /dev/null; then
    echo "[worker ${WORKER_ID}] gsutil available for GCS access"
else
    echo "[worker ${WORKER_ID}] Note: gsutil not found. Install gcloud SDK for GCS trace storage."
fi

echo ""
echo "=== Setup Complete (worker ${WORKER_ID}) ==="
echo ""
