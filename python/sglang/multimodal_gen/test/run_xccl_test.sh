#!/bin/bash
# Script to run XCCL communication tests

set -e

echo "========================================"
echo "XCCL Communication Test Suite"
echo "========================================"
echo ""

# Set environment variables for debugging
export CCL_LOG_LEVEL=info
# export CCL_ATL_TRANSPORT=ofi
# export FI_PROVIDER=tcp  # Use TCP for testing, change to 'verbs' for InfiniBand

echo "Environment variables:"
echo "  CCL_LOG_LEVEL=$CCL_LOG_LEVEL"
echo "  CCL_ATL_TRANSPORT=$CCL_ATL_TRANSPORT"
echo "  FI_PROVIDER=$FI_PROVIDER"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run with torchrun
echo "Running test with 4 processes..."
echo ""

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    "${SCRIPT_DIR}/test_xccl_communication.py"

echo ""
echo "========================================"
echo "Test completed"
echo "========================================"
