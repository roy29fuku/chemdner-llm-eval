"""
Inference script for BioBERT NER on CHEMDNER.

Generates predictions in the same format as evaluate.py expects.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm

from .config import Config, default_config
from .data_prep import parse_bio_file, get_sentences_from_bio


def predict_entities(
    model,
    tokenizer,
    tokens: List[str],
    id2label: Dict[int, str],
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """
    Predict entities for a single sentence.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        tokens: List of tokens
        id2label: ID to label mapping
        device: Device to run on

    Returns:
        List of predicted entities
    """
    # Tokenize
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt"
    )

    # Move to device
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Get word IDs for alignment
    word_ids = encoding.word_ids() if hasattr(encoding, 'word_ids') else None

    # If word_ids not available, recreate
    if word_ids is None:
        encoding_list = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=256,
        )
        word_ids = encoding_list.word_ids()

    # Convert predictions back to original tokens
    pred_labels = []
    previous_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != previous_word_id:
            pred_id = predictions[0][idx].item()
            pred_labels.append(id2label[pred_id])
        previous_word_id = word_id

    # Ensure we have labels for all tokens
    while len(pred_labels) < len(tokens):
        pred_labels.append("O")
    pred_labels = pred_labels[:len(tokens)]

    # Extract entities from predictions
    text = " ".join(tokens)
    entities = []
    current_entity_tokens = []
    current_start_idx = 0
    char_offset = 0

    for i, (token, label) in enumerate(zip(tokens, pred_labels)):
        token_start = char_offset

        if label == "B-CHEMICAL":
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                entities.append({
                    "text": entity_text,
                    "type": "CHEMICAL",
                    "start": current_start_idx,
                    "end": current_start_idx + len(entity_text)
                })
            current_entity_tokens = [token]
            current_start_idx = token_start
        elif label == "I-CHEMICAL" and current_entity_tokens:
            current_entity_tokens.append(token)
        else:
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                entities.append({
                    "text": entity_text,
                    "type": "CHEMICAL",
                    "start": current_start_idx,
                    "end": current_start_idx + len(entity_text)
                })
                current_entity_tokens = []

        char_offset += len(token) + 1

    # Handle entity at end
    if current_entity_tokens:
        entity_text = " ".join(current_entity_tokens)
        entities.append({
            "text": entity_text,
            "type": "CHEMICAL",
            "start": current_start_idx,
            "end": current_start_idx + len(entity_text)
        })

    return entities


def run_inference(
    model_path: str,
    data_path: str,
    output_path: str,
    device: str = "cuda",
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Run inference on test set and save predictions.

    Args:
        model_path: Path to trained model
        data_path: Path to test BIO file
        output_path: Path to save predictions
        device: Device to run on
        batch_size: Batch size (currently unused, processes one at a time)

    Returns:
        List of results with gold and predicted entities
    """
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Get id2label from model config
    id2label = model.config.id2label

    print(f"Loading test data from {data_path}")
    sentences = get_sentences_from_bio(Path(data_path))
    print(f"Loaded {len(sentences)} sentences")

    # Process sentences
    results = []
    for i, sent in enumerate(tqdm(sentences, desc="Predicting")):
        tokens = sent["tokens"]
        gold_entities = sent["entities"]

        pred_entities = predict_entities(
            model, tokenizer, tokens, id2label, device
        )

        results.append({
            "document_id": f"sent_{i}",
            "text": sent["text"],
            "gold_entities": gold_entities,
            "predicted_entities": pred_entities
        })

    # Save predictions
    print(f"Saving predictions to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Quick evaluation
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for result in results:
        gold_set = {e["text"].lower() for e in result["gold_entities"]}
        pred_set = {e["text"].lower() for e in result["predicted_entities"]}

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nQuick Evaluation:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run BioBERT inference on CHEMDNER")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--data-path", type=str,
                        default="data/chemdner_corpus/evaluation.ner.ssplit.token4.BIO",
                        help="Path to test BIO file")
    parser.add_argument("--output-path", type=str,
                        default="outputs/exp008_biobert/predictions_biobert.json",
                        help="Output path for predictions")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        device=device
    )


if __name__ == "__main__":
    main()
