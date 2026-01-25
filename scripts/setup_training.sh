#!/bin/bash
# Setup script for BioBERT training environment
# Run this locally to set up the GCP instance

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-chemdner-biobert-training}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-biobert-chemdner-training}"

echo "=== CHEMDNER BioBERT Training Setup ==="
echo "Project: ${PROJECT_ID}"
echo "Zone: ${ZONE}"
echo "Instance: ${INSTANCE_NAME}"

# Check if gcloud is configured
if ! gcloud auth list 2>/dev/null | grep -q "ACTIVE"; then
    echo "Error: Not authenticated with gcloud. Run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project ${PROJECT_ID}

# Check if instance exists
if gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} &>/dev/null; then
    echo "Instance ${INSTANCE_NAME} already exists"
else
    echo "Instance not found. Please run terraform apply first."
    exit 1
fi

# Get instance IP
INSTANCE_IP=$(gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "Instance IP: ${INSTANCE_IP}"

# Create remote directories
echo "Creating remote directories..."
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="mkdir -p ~/chemdner/{src,data,outputs}"

# Upload source code
echo "Uploading source code..."
gcloud compute scp --recurse src/biobert ${INSTANCE_NAME}:~/chemdner/src/ --zone=${ZONE}
gcloud compute scp src/data_loader.py ${INSTANCE_NAME}:~/chemdner/src/ --zone=${ZONE}
gcloud compute scp src/evaluate.py ${INSTANCE_NAME}:~/chemdner/src/ --zone=${ZONE}

# Upload data
echo "Uploading data (this may take a while)..."
gcloud compute scp --recurse data/chemdner_corpus ${INSTANCE_NAME}:~/chemdner/data/ --zone=${ZONE}

# Install dependencies on remote
echo "Installing dependencies on remote..."
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
    pip install --upgrade pip
    pip install transformers datasets evaluate seqeval accelerate torch
"

# Verify GPU
echo "Verifying GPU..."
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="nvidia-smi"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start training, run:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
echo "  cd ~/chemdner && python -m src.biobert.train"
echo ""
echo "Or use scripts/run_training.sh"
