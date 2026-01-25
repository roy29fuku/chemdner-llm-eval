#!/bin/bash
# Download training results from GCP instance

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-chemdner-biobert-training}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-biobert-chemdner-training}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-outputs/exp008_biobert}"

echo "=== Downloading Results ==="
echo "From: ${INSTANCE_NAME}:~/chemdner/outputs/exp008_biobert/"
echo "To: ${LOCAL_OUTPUT_DIR}/"

# Create local directory
mkdir -p ${LOCAL_OUTPUT_DIR}

# Download results
gcloud compute scp --recurse \
    ${INSTANCE_NAME}:~/chemdner/outputs/exp008_biobert/* \
    ${LOCAL_OUTPUT_DIR}/ \
    --zone=${ZONE} \
    --project=${PROJECT_ID}

echo ""
echo "=== Download Complete ==="
echo ""
echo "Files downloaded to ${LOCAL_OUTPUT_DIR}/"
ls -la ${LOCAL_OUTPUT_DIR}/

echo ""
echo "To evaluate, run:"
echo "  docker compose run --rm chemdner python src/evaluate.py --input-dir ${LOCAL_OUTPUT_DIR}"

# Optional: Also download model (large)
read -p "Download trained model? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading model (this may take a while)..."
    mkdir -p ${LOCAL_OUTPUT_DIR}/model
    gcloud compute scp --recurse \
        ${INSTANCE_NAME}:~/chemdner/outputs/exp008_biobert/model/* \
        ${LOCAL_OUTPUT_DIR}/model/ \
        --zone=${ZONE} \
        --project=${PROJECT_ID}
    echo "Model downloaded to ${LOCAL_OUTPUT_DIR}/model/"
fi
