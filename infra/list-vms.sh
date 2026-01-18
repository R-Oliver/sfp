#!/bin/bash
# List running TPU VMs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

ALL_ZONES="false"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

List TPU VMs in your project.

Options:
    --all-zones     List VMs across all zones (slower)
    -h, --help      Show this help message

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all-zones)
            ALL_ZONES="true"
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

echo "TPU VMs in project: $GCP_PROJECT"
echo ""

if [[ "$ALL_ZONES" == "true" ]]; then
    echo "(Searching all zones...)"
    gcloud compute tpus tpu-vm list \
        --project="$GCP_PROJECT" \
        --format="table(name,zone,acceleratorType,state,networkEndpoints[0].accessConfig.externalIp:label=EXTERNAL_IP)"
else
    echo "(Zone: $GCP_ZONE)"
    echo ""
    gcloud compute tpus tpu-vm list \
        --zone="$GCP_ZONE" \
        --project="$GCP_PROJECT" \
        --format="table(name,acceleratorType,state,networkEndpoints[0].accessConfig.externalIp:label=EXTERNAL_IP)"
fi
