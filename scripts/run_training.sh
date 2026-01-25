#!/bin/bash
# Run BioBERT training on GCP instance
# Can be run locally (connects to GCP) or on the GCP instance directly

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-chemdner-biobert-training}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-biobert-chemdner-training}"

# Training parameters
MODEL_NAME="${MODEL_NAME:-dmis-lab/biobert-v1.1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"

# Check if running on GCP or locally
if [[ -d "/home/jupyter/chemdner" ]]; then
    # Running on GCP instance
    echo "Running on GCP instance..."
    cd /home/jupyter/chemdner

    # Run training
    python -m src.biobert.train \
        --model-name "${MODEL_NAME}" \
        --data-dir "data/chemdner_corpus" \
        --output-dir "outputs/exp008_biobert" \
        --learning-rate ${LEARNING_RATE} \
        --batch-size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --max-length ${MAX_LENGTH}

    # Run inference
    echo "Running inference..."
    python -m src.biobert.inference \
        --model-path "outputs/exp008_biobert/model" \
        --data-path "data/chemdner_corpus/evaluation.ner.ssplit.token4.BIO" \
        --output-path "outputs/exp008_biobert/predictions_biobert.json"

    echo "Training and inference complete!"
    echo "Results in: outputs/exp008_biobert/"

else
    # Running locally - connect to GCP
    echo "Connecting to GCP instance..."

    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="
        cd ~/chemdner

        echo 'Starting training...'
        python -m src.biobert.train \
            --model-name '${MODEL_NAME}' \
            --data-dir 'data/chemdner_corpus' \
            --output-dir 'outputs/exp008_biobert' \
            --learning-rate ${LEARNING_RATE} \
            --batch-size ${BATCH_SIZE} \
            --epochs ${EPOCHS} \
            --max-length ${MAX_LENGTH}

        echo 'Running inference...'
        python -m src.biobert.inference \
            --model-path 'outputs/exp008_biobert/model' \
            --data-path 'data/chemdner_corpus/evaluation.ner.ssplit.token4.BIO' \
            --output-path 'outputs/exp008_biobert/predictions_biobert.json'

        echo 'Complete!'
    "

    echo ""
    echo "Training complete. Download results with:"
    echo "  ./scripts/download_results.sh"
fi
