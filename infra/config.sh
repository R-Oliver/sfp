#!/bin/bash
# SFP Infrastructure Configuration
#
#   export GCP_PROJECT="my-project-id"
#   export GCP_ZONE="us-central1-a"
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source local overrides if they exist
if [[ -f "${SCRIPT_DIR}/config.local.sh" ]]; then
    source "${SCRIPT_DIR}/config.local.sh"
fi

# GCP Project Settings (defaults, overridden by config.local.sh or env vars)
export GCP_PROJECT="${GCP_PROJECT:-your-project-id}"
export GCP_ZONE="${GCP_ZONE:-us-central1-a}"
export GCP_REGION="${GCP_REGION:-us-central1}"

# GCS Bucket for traces and artifacts
export GCS_BUCKET="${GCS_BUCKET:-gs://sfp-${GCP_PROJECT}}"

# VM Settings
export VM_NAME="${VM_NAME:-sfp-tpu}"
export DEFAULT_TTL="3h"  # Default auto-shutdown time
export VM_STATE_FILE="${SCRIPT_DIR}/.vm-state"

# Default TPU type
export DEFAULT_TPU_TYPE="v5e-1"

# Spot/preemptible settings
export DEFAULT_SPOT="true"

# Helper function to get TPU accelerator type
get_accelerator_type() {
    local type="${1:-$DEFAULT_TPU_TYPE}"
    case "$type" in
        v5e-1) echo "v5litepod-1" ;;
        v5e-4) echo "v5litepod-4" ;;
        v5e-8) echo "v5litepod-8" ;;
        v6e-1) echo "v6e-1" ;;
        v6e-4) echo "v6e-4" ;;
        v6e-8) echo "v6e-8" ;;
        *) echo "$type" ;;
    esac
}

# Helper function to get zone for TPU type (v5e and v6e are in different zones)
get_zone_for_type() {
    local type="${1:-$DEFAULT_TPU_TYPE}"
    case "$type" in
        v5e-*) echo "us-central1-a" ;;
        v6e-*) echo "us-central1-b" ;;
        *) echo "$GCP_ZONE" ;;
    esac
}

# Helper function to validate TPU type
validate_tpu_type() {
    local type="$1"
    case "$type" in
        v5e-1|v5e-4|v5e-8|v6e-1|v6e-4|v6e-8)
            return 0
            ;;
        *)
            echo "Unknown TPU type: $type" >&2
            echo "Available types: v5e-1, v5e-4, v5e-8, v6e-1, v6e-4, v6e-8" >&2
            return 1
            ;;
    esac
}
