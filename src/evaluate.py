"""
Evaluation script for CHEMDNER predictions.
Calculates Precision, Recall, and F1 score.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
import pandas as pd


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def extract_entity_set(entities: List[Dict], normalize: bool = True) -> Set[str]:
    """Extract set of entity texts from entity list."""
    texts = set()
    for e in entities:
        text = e.get("text", "")
        if normalize:
            text = normalize_text(text)
        if text:
            texts.add(text)
    return texts


def calculate_metrics(
    gold_entities: Set[str],
    predicted_entities: Set[str]
) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1.

    Args:
        gold_entities: Set of gold standard entity texts
        predicted_entities: Set of predicted entity texts

    Returns:
        Dict with precision, recall, f1
    """
    if not predicted_entities:
        fn = len(gold_entities)
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": fn
        }

    true_positives = gold_entities & predicted_entities
    tp = len(true_positives)
    fp = len(predicted_entities - gold_entities)
    fn = len(gold_entities - predicted_entities)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }


def evaluate_results(results: List[Dict]) -> Dict[str, any]:
    """
    Evaluate a list of prediction results.

    Args:
        results: List of results with gold_entities and predicted_entities

    Returns:
        Evaluation metrics (micro and macro averaged)
    """
    # Per-document metrics for macro average
    doc_metrics = []

    # Aggregated counts for micro average
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for result in results:
        gold = extract_entity_set(result["gold_entities"])
        pred = extract_entity_set(result["predicted_entities"])

        metrics = calculate_metrics(gold, pred)
        doc_metrics.append(metrics)

        total_tp += metrics["true_positives"]
        total_fp += metrics["false_positives"]
        total_fn += metrics["false_negatives"]

    # Micro-averaged metrics (recommended for NER)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro-averaged metrics
    macro_precision = sum(m["precision"] for m in doc_metrics) / len(doc_metrics)
    macro_recall = sum(m["recall"] for m in doc_metrics) / len(doc_metrics)
    macro_f1 = sum(m["f1"] for m in doc_metrics) / len(doc_metrics)

    return {
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "counts": {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn
        },
        "num_documents": len(results)
    }


def print_comparison_table(all_metrics: Dict[str, Dict]):
    """Print a comparison table of model metrics."""
    print("\n" + "=" * 70)
    print("CHEMDNER Evaluation Results")
    print("=" * 70)

    # Create DataFrame for nice table output
    rows = []
    for model, metrics in all_metrics.items():
        rows.append({
            "Model": model,
            "Micro-P": f"{metrics['micro']['precision']:.4f}",
            "Micro-R": f"{metrics['micro']['recall']:.4f}",
            "Micro-F1": f"{metrics['micro']['f1']:.4f}",
            "Macro-F1": f"{metrics['macro']['f1']:.4f}",
            "TP": metrics['counts']['true_positives'],
            "FP": metrics['counts']['false_positives'],
            "FN": metrics['counts']['false_negatives']
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print("\n" + "=" * 70)


def analyze_errors(results: List[Dict], limit: int = 5):
    """Analyze and print error examples."""
    print("\n" + "-" * 50)
    print("Error Analysis (sample)")
    print("-" * 50)

    count = 0
    for result in results:
        gold = extract_entity_set(result["gold_entities"])
        pred = extract_entity_set(result["predicted_entities"])

        fn = gold - pred  # Missed entities
        fp = pred - gold  # Wrong predictions

        if fn or fp:
            print(f"\nDocument: {result['document_id']}")
            if fn:
                print(f"  Missed (FN): {list(fn)[:5]}")
            if fp:
                print(f"  Wrong (FP): {list(fp)[:5]}")
            count += 1

        if count >= limit:
            break


def main():
    parser = argparse.ArgumentParser(description="CHEMDNER Evaluation")
    parser.add_argument("--input-dir", type=str, default="outputs",
                        help="Directory with prediction files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for metrics (JSON)")
    parser.add_argument("--analyze-errors", action="store_true",
                        help="Print error analysis")
    parser.add_argument("--pattern", type=str, default="predictions_*.json",
                        help="Glob pattern for prediction files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    prediction_files = list(input_dir.glob(args.pattern))

    if not prediction_files:
        print(f"No prediction files found in {input_dir}")
        return

    all_metrics = {}
    all_results = {}

    for pred_file in prediction_files:
        model_name = pred_file.stem.replace("predictions_", "").replace("_", ".")
        print(f"Evaluating {model_name}...")

        with open(pred_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        metrics = evaluate_results(results)
        all_metrics[model_name] = metrics
        all_results[model_name] = results

    # Print comparison table
    print_comparison_table(all_metrics)

    # Error analysis
    if args.analyze_errors:
        for model, results in all_results.items():
            print(f"\n{'='*50}")
            print(f"Error Analysis for {model}")
            analyze_errors(results)

    # Save metrics
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
