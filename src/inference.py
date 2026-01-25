"""
LLM inference for CHEMDNER chemical entity recognition.
"""
import argparse
import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from data_loader import load_chemdner

load_dotenv()


SYSTEM_PROMPT = """You are an expert in chemical named entity recognition.
Your task is to extract all chemical entity mentions from the given biomedical text.

Chemical entities include:
- Trivial names (e.g., aspirin, caffeine)
- Systematic names (e.g., 2-acetoxybenzoic acid)
- Abbreviations (e.g., ATP, DNA)
- Molecular formulas (e.g., H2O, C6H12O6)
- Chemical family names (e.g., steroids, amino acids)
- Identifiers (e.g., CAS numbers, IUPAC names)

Output a JSON array of chemical entities found in the text.
Each entity should have:
- "text": the exact text of the chemical mention as it appears
- "type": one of "TRIVIAL", "SYSTEMATIC", "ABBREVIATION", "FORMULA", "FAMILY", "IDENTIFIER", "MULTIPLE"

Only output the JSON array, no other text."""


USER_PROMPT_TEMPLATE = """Extract all chemical entities from the following text:

{text}

Output JSON array:"""


def extract_json_from_response(response: str) -> List[Dict]:
    """Extract JSON array from LLM response."""
    # Try to find JSON array in response
    try:
        # First try direct parse
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array with regex
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def run_inference(
    samples: List[Dict[str, Any]],
    model: str,
    client: OpenAI
) -> List[Dict[str, Any]]:
    """
    Run inference on samples using the specified model.

    Args:
        samples: List of samples with text
        model: OpenAI model name
        client: OpenAI client

    Returns:
        List of results with predictions
    """
    results = []

    # GPT-5.x uses max_completion_tokens instead of max_tokens
    use_completion_tokens = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")

    for sample in tqdm(samples, desc=f"Running {model}"):
        try:
            # Build request params
            request_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=sample["text"])}
                ],
                "temperature": 0,
            }

            # Use appropriate token parameter based on model
            if use_completion_tokens:
                request_params["max_completion_tokens"] = 4096
            else:
                request_params["max_tokens"] = 4096

            response = client.chat.completions.create(**request_params)

            response_text = response.choices[0].message.content
            predicted_entities = extract_json_from_response(response_text)

            results.append({
                "document_id": sample["document_id"],
                "text": sample["text"],
                "gold_entities": sample["entities"],
                "predicted_entities": predicted_entities,
                "raw_response": response_text,
                "model": model
            })

        except Exception as e:
            print(f"Error processing {sample['document_id']}: {e}")
            results.append({
                "document_id": sample["document_id"],
                "text": sample["text"],
                "gold_entities": sample["entities"],
                "predicted_entities": [],
                "error": str(e),
                "model": model
            })

    return results


def main():
    JST = timezone(timedelta(hours=9))
    start_time = datetime.now(JST)
    print(f"{'='*50}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')} JST")
    print(f"{'='*50}\n")

    parser = argparse.ArgumentParser(description="CHEMDNER LLM Inference")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--models", nargs="+", default=["gpt-4.1", "gpt-5.2"],
                        help="Models to evaluate")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    args = parser.parse_args()

    # Setup
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=api_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"Loading CHEMDNER dataset (split={args.split}, limit={args.limit})...")
    samples = load_chemdner(split=args.split, limit=args.limit)
    print(f"Loaded {len(samples)} samples")

    # Run inference for each model
    all_results = {}
    for model in args.models:
        print(f"\n{'='*50}")
        print(f"Running inference with {model}")
        print(f"{'='*50}")

        results = run_inference(samples, model, client)
        all_results[model] = results

        # Save results
        output_file = output_dir / f"predictions_{model.replace('.', '_')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {output_file}")

    end_time = datetime.now(JST)
    elapsed = end_time - start_time

    print("\nInference complete!")
    print(f"Run evaluation with: python src/evaluate.py")
    print(f"\n{'='*50}")
    print(f"Started at:  {start_time.strftime('%Y-%m-%d %H:%M:%S')} JST")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')} JST")
    print(f"Elapsed:     {elapsed}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
