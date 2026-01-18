#!/bin/bash
# Delete a TPU VM
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

FORCE="false"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Delete the TPU VM.

Options:
    --force, -f     Skip confirmation prompt
    -h, --help      Show this help message
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE="true"
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

echo "Deleting TPU VM: $VM_NAME"
echo "  Zone: $TPU_ZONE"
echo ""

if [[ "$FORCE" != "true" ]]; then
    read -p "Are you sure? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

gcloud compute tpus tpu-vm delete "$VM_NAME" \
    --zone="$TPU_ZONE" \
    --project="$GCP_PROJECT" \
    --quiet

# Clean up state file
rm -f "$VM_STATE_FILE"

echo ""
echo "VM deleted: $VM_NAME"
