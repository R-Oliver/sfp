#!/bin/bash
# Setup script for TPU VM - run once after cloning the repo
set -euo pipefail

echo "=== SFP VM Setup ==="
echo ""

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to current shell (the install script adds to shell config)
    export PATH="$HOME/.local/bin:$PATH"
    echo ""
fi

echo "uv version: $(uv --version)"
echo ""

# Sync dependencies (including TPU extras)
echo "Syncing dependencies..."
uv sync --extra tpu
echo ""

# Verify JAX installation and TPU access
echo "Verifying JAX and TPU..."
uv run python -c "
import jax
print(f'JAX version: {jax.__version__}')
devices = jax.devices()
print(f'Devices: {len(devices)}')
for d in devices:
    print(f'  {d}')
"
echo ""

# Check GCS access (if configured)
if command -v gsutil &> /dev/null; then
    echo "gsutil available for GCS access"
else
    echo "Note: gsutil not found. Install gcloud SDK for GCS trace storage."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
