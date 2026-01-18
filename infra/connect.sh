#!/bin/bash
# SSH to TPU VM with agent forwarding
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

XPROF="false"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

SSH to the TPU VM with agent forwarding enabled.

Options:
    --xprof         Enable port forwarding for XProf (8791)
    -h, --help      Show this help message

Examples:
    $(basename "$0")           # Basic SSH with agent forwarding
    $(basename "$0") --xprof   # SSH + XProf port forward
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --xprof)
            XPROF="true"
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

# Get zone from state file (written by create-vm.sh)
if [[ -f "$VM_STATE_FILE" ]]; then
    TPU_ZONE=$(cat "$VM_STATE_FILE")
else
    TPU_ZONE="$GCP_ZONE"
    echo "Warning: No VM state file found. Using default zone: $TPU_ZONE" >&2
fi

# Build port forwarding arguments
PORT_FORWARDS=""
if [[ "$XPROF" == "true" ]]; then
    PORT_FORWARDS="-L 8791:localhost:8791"
    echo "XProf will be available at: http://localhost:8791"
    echo ""
fi

echo "Connecting to $VM_NAME with SSH agent forwarding..."
echo "(Your local SSH keys will work for git on the VM)"
echo ""

if [[ -n "$PORT_FORWARDS" ]]; then
    exec gcloud compute tpus tpu-vm ssh "$VM_NAME" \
        --zone="$TPU_ZONE" \
        --project="$GCP_PROJECT" \
        -- -A "$PORT_FORWARDS"
else
    exec gcloud compute tpus tpu-vm ssh "$VM_NAME" \
        --zone="$TPU_ZONE" \
        --project="$GCP_PROJECT" \
        -- -A
fi
