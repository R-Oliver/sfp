#!/bin/bash
# Create a TPU VM for kernel development
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Defaults
TPU_TYPE="$DEFAULT_TPU_TYPE"
USE_SPOT="$DEFAULT_SPOT"
TTL="$DEFAULT_TTL"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Create a TPU VM for kernel development.

Options:
    --type TYPE     TPU type: v5e-1, v5e-4, v5e-8, v6e-1, v6e-4, v6e-8 (default: $DEFAULT_TPU_TYPE)
    --on-demand     Use on-demand pricing instead of spot (more expensive, guaranteed)
    --ttl DURATION  Auto-shutdown after DURATION (default: $DEFAULT_TTL, e.g., 2h, 4h, 30m)
    --no-ttl        Disable auto-shutdown (be careful!)
    -h, --help      Show this help message

Examples:
    $(basename "$0")                     # Default: v5e-1, spot, 3h TTL
    $(basename "$0") --type v5e-4        # 4-chip v5e for distributed work
    $(basename "$0") --type v6e-4        # 4-chip Trillium
    $(basename "$0") --ttl 6h            # Longer session
    $(basename "$0") --on-demand --no-ttl # Guaranteed, no auto-shutdown
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            TPU_TYPE="$2"
            shift 2
            ;;
        --on-demand)
            USE_SPOT="false"
            shift
            ;;
        --ttl)
            TTL="$2"
            shift 2
            ;;
        --no-ttl)
            TTL=""
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Validate TPU type
if ! validate_tpu_type "$TPU_TYPE"; then
    exit 1
fi

ACCELERATOR_TYPE=$(get_accelerator_type "$TPU_TYPE")
TPU_ZONE=$(get_zone_for_type "$TPU_TYPE")

echo "Creating TPU VM..."
echo "  Name: $VM_NAME"
echo "  Type: $TPU_TYPE ($ACCELERATOR_TYPE)"
echo "  Zone: $TPU_ZONE"
echo "  Spot: $USE_SPOT"
if [[ -n "$TTL" ]]; then
    echo "  TTL: $TTL (auto-shutdown)"
else
    echo "  TTL: disabled (manual shutdown required)"
fi
echo ""

# Build startup script for TTL
STARTUP_SCRIPT=""
if [[ -n "$TTL" ]]; then
    # Convert TTL to seconds
    TTL_SECONDS=$(echo "$TTL" | awk '
        /h$/ { print int($0) * 3600; next }
        /m$/ { print int($0) * 60; next }
        /s$/ { print int($0); next }
        { print int($0) * 3600 }  # Default to hours
    ')
    STARTUP_SCRIPT="#!/bin/bash
echo 'TTL shutdown scheduled for $TTL ($TTL_SECONDS seconds)'
(sleep $TTL_SECONDS && sudo shutdown -h now 'TTL expired') &"
fi

# Build gcloud command
GCLOUD_CMD=(
    gcloud compute tpus tpu-vm create "$VM_NAME"
    --zone="$TPU_ZONE"
    --accelerator-type="$ACCELERATOR_TYPE"
    --version="tpu-ubuntu2204-base"
    --project="$GCP_PROJECT"
)

if [[ "$USE_SPOT" == "true" ]]; then
    GCLOUD_CMD+=(--spot)
fi

if [[ -n "$STARTUP_SCRIPT" ]]; then
    GCLOUD_CMD+=(--metadata="startup-script=$STARTUP_SCRIPT")
fi

# Create the VM
echo "Running: ${GCLOUD_CMD[*]}"
echo ""
"${GCLOUD_CMD[@]}"

# Save state for connect/delete
echo "$TPU_ZONE" > "$VM_STATE_FILE"

echo ""
echo "VM created successfully!"
echo ""
echo "=== Connection Info ==="
echo ""
echo "Terminal SSH:"
echo "  ./infra/connect.sh"
echo ""
echo "VS Code Remote SSH:"
echo "  1. Get external IP: gcloud compute tpus tpu-vm describe $VM_NAME --zone=$TPU_ZONE --format='value(networkEndpoints[0].accessConfig.externalIp)'"
echo "  2. Add to ~/.ssh/config:"
echo "     Host $VM_NAME"
echo "       HostName <external-ip>"
echo "       User $(whoami)"
echo "       ForwardAgent yes"
echo "  3. Connect in VS Code: Remote-SSH: Connect to Host -> $VM_NAME"
echo ""
echo "Once connected, run:"
echo "  git clone <your-repo> && cd sfp && ./scripts/setup-vm.sh"
echo ""
if [[ -n "$TTL" ]]; then
    echo "Auto-shutdown in: $TTL"
fi
