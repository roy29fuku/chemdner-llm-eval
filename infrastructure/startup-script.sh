#!/bin/bash
# Startup script for BioBERT training instance

set -e

echo "=== Starting setup ==="

# Create working directory
mkdir -p /home/jupyter/chemdner
cd /home/jupyter/chemdner

# Install Python dependencies
pip install --upgrade pip
pip install transformers datasets evaluate seqeval accelerate torch

# Verify GPU is available
echo "=== Checking GPU ==="
nvidia-smi

echo "=== Setup complete ==="
echo "Ready for training. Upload your code and data, then run:"
echo "  python src/biobert/train.py"
