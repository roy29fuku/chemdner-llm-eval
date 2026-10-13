"""
BioBERT fine-tuning script for CHEMDNER NER.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    set_seed,
)
import evaluate

from .config import Config, default_config
from .data_prep import load_and_prepare_data


def compute_metrics(eval_preds, label_list: List[str]):
    """Compute seqeval metrics."""
    metric = evaluate.load("seqeval")

    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens) and convert to labels
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        pred_labels = []
        true_labels_seq = []

        for pred, lab in zip(prediction, label):
            if lab != -100:
                pred_labels.append(label_list[pred])
                true_labels_seq.append(label_list[lab])

        true_predictions.append(pred_labels)
        true_labels.append(true_labels_seq)

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def train_model(config: Optional[Config] = None):
    """Train BioBERT for NER."""
    if config is None:
        config = default_config

    # Set seed for reproducibility
    set_seed(config.seed)

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)

    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    print(f"Loading model: {config.model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id,
    )

    print(f"Loading data from {config.data_dir}")
    train_dataset, dev_dataset, test_dataset = load_and_prepare_data(
        config.data_dir,
        tokenizer,
        config.label2id,
        config.max_seq_length,
        config.train_file,
        config.dev_file,
        config.test_file,
    )

    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    # Label list for metrics
    label_list = [config.id2label[i] for i in range(config.num_labels)]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=config.fp16 and torch.cuda.is_available(),
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=config.seed,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, label_list),
    )

    print("Starting training...")
    train_result = trainer.train()

    # Save model
    print(f"Saving model to {config.model_save_dir}")
    trainer.save_model(config.model_save_dir)
    tokenizer.save_pretrained(config.model_save_dir)

    # Evaluate on dev set
    print("Evaluating on dev set...")
    dev_metrics = trainer.evaluate()
    print(f"Dev metrics: {dev_metrics}")

    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_dataset)
    print(f"Test metrics: {test_metrics}")

    # Save metrics
    metrics = {
        "train": {
            "loss": train_result.training_loss,
        },
        "dev": dev_metrics,
        "test": test_metrics,
        "config": {
            "model_name": config.model_name,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "max_seq_length": config.max_seq_length,
        }
    }

    metrics_path = Path(config.output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")
    print("\nTraining complete!")
    print(f"Test F1: {test_metrics.get('eval_f1', 'N/A'):.4f}")

    return trainer, metrics


def main():
    parser = argparse.ArgumentParser(description="Train BioBERT for CHEMDNER NER")
    parser.add_argument("--model-name", type=str, default="dmis-lab/biobert-v1.1",
                        help="Pretrained model name")
    parser.add_argument("--data-dir", type=str, default="data/chemdner_corpus",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/exp008_biobert",
                        help="Output directory")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Create config from args
    config = Config(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_save_dir=f"{args.output_dir}/model",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_seq_length=args.max_length,
        seed=args.seed,
    )

    train_model(config)


if __name__ == "__main__":
    main()
