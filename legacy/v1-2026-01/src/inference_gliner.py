"""
GLiNER-BioMed inference for CHEMDNER chemical NER.
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from gliner import GLiNER

from data_loader import load_chemdner


# Entity labels to extract
CHEMICAL_LABELS = ["chemical", "chemical compound", "drug", "molecule"]


def extract_entities_gliner(model: GLiNER, text: str, labels: list[str], threshold: float = 0.5) -> list[dict]:
    """
    Extract chemical entities using GLiNER.

    Args:
        model: GLiNER model
        text: Input text
        labels: Entity labels to extract
        threshold: Confidence threshold

    Returns:
        List of extracted entities
    """
    entities = model.predict_entities(text, labels, threshold=threshold)

    # Convert to standard format
    result = []
    for entity in entities:
        result.append({
            "text": entity["text"],
            "type": entity["label"].upper(),
            "start": entity["start"],
            "end": entity["end"],
            "score": entity["score"]
        })

    return result


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """
    Remove duplicate entities (same text, overlapping spans).
    Keep the one with highest score.
    """
    # Group by text
    by_text = {}
    for entity in entities:
        text = entity["text"].lower()
        if text not in by_text or entity.get("score", 0) > by_text[text].get("score", 0):
            by_text[text] = entity

    return list(by_text.values())


def run_inference(
    model_name: str = "Ihor/gliner-biomed-base-v1.0",
    split: str = "test",
    limit: int | None = None,
    output_dir: str = "outputs/exp007_gliner",
    threshold: float = 0.5,
    labels: list[str] | None = None,
    data_dir: str | None = None,
):
    """
    Run GLiNER inference on CHEMDNER dataset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {model_name}")
    model = GLiNER.from_pretrained(model_name)

    # Load dataset
    print(f"Loading CHEMDNER {split} split...")
    samples = load_chemdner(split=split, limit=limit, data_dir=data_dir)
    print(f"Loaded {len(samples)} documents")

    # Use provided labels or default
    entity_labels = labels or CHEMICAL_LABELS
    print(f"Entity labels: {entity_labels}")
    print(f"Threshold: {threshold}")

    # Run inference
    results = []
    total_entities = 0

    for sample in tqdm(samples, desc="Processing"):
        doc_id = sample["document_id"]
        text = sample["text"]
        ground_truth = sample["entities"]

        # Extract entities
        predicted = extract_entities_gliner(model, text, entity_labels, threshold)

        # Deduplicate
        predicted = deduplicate_entities(predicted)

        total_entities += len(predicted)

        results.append({
            "document_id": doc_id,
            "text": text,
            "gold_entities": ground_truth,
            "predicted_entities": predicted
        })

    # Save results
    model_short = model_name.split("/")[-1].replace("-", "_")
    output_file = output_path / f"predictions_{model_short}_th{threshold}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Total documents: {len(results)}")
    print(f"Total predicted entities: {total_entities}")
    print(f"Average entities per doc: {total_entities / len(results):.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GLiNER inference for CHEMDNER")
    parser.add_argument("--model", default="Ihor/gliner-biomed-base-v1.0",
                        help="GLiNER model name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output-dir", default="outputs/exp007_gliner",
                        help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Entity labels to extract")
    parser.add_argument("--data-dir", default=None, help="Data directory override")

    args = parser.parse_args()

    run_inference(
        model_name=args.model,
        split=args.split,
        limit=args.limit,
        output_dir=args.output_dir,
        threshold=args.threshold,
        labels=args.labels,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
