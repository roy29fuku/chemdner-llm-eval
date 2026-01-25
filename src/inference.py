"""
Parallel LLM inference for CHEMDNER chemical entity recognition.
Version 2: Improved prompts with few-shot examples following CHEMDNER guidelines.
"""
import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
import tiktoken
from tqdm.asyncio import tqdm_asyncio

from data_loader import load_chemdner


# Pricing per 1M tokens (USD)
MODEL_PRICING = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.1": {"input": 1.00, "output": 8.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}

load_dotenv()

JST = timezone(timedelta(hours=9))

# Improved system prompt based on CHEMDNER annotation guidelines
SYSTEM_PROMPT = """You are an expert in chemical named entity recognition following the CHEMDNER annotation guidelines.

## Task
Extract chemical entity mentions from biomedical text. Only extract entities whose chemical structure can be determined or inferred.

## What to INCLUDE:
- Trivial/common names: aspirin, caffeine, morphine, sildenafil
- Systematic IUPAC names: 2-acetoxybenzoic acid, 1,3,7-trimethylxanthine
- Abbreviations of chemicals: ATP, ADP, NAD+, NADH, cAMP, NO (nitric oxide)
- Molecular formulas: H2O, C6H12O6, NaCl, MgCl2
- Drug names: saquinavir, ritonavir, darunavir, metformin
- Specific chemical identifiers: CAS numbers, SMILES

## What to EXCLUDE (very important):
- Proteins and enzymes: CYP3A4, P-glycoprotein, kinase, synthase
- Genes: BRCA1, eNOS, p53
- General drug classes: "drugs", "medications", "compounds"
- Non-specific chemical families: "phytochemicals", "metabolites", "antioxidants"
- Biological entities: cells, bacteria, viruses
- Plant/food names: garlic, ginger, turmeric (unless referring to a specific extract compound)
- Transporters: OATP, MCT1, PepT1

## Output Format
Output a JSON array. Each entity should have:
- "text": exact text as it appears in the document
- "type": one of "TRIVIAL", "SYSTEMATIC", "ABBREVIATION", "FORMULA", "IDENTIFIER"

Only output the JSON array, nothing else."""


# Few-shot examples (from CHEMDNER training set - NOT from evaluation set)
FEW_SHOT_EXAMPLES = [
    {
        # From training set: haloperidol example
        "text": "Chronic administration of haloperidol increased Dpp6 expression in mouse brains. DPP6 is an auxiliary subunit of Kv4.",
        "output": [
            {"text": "haloperidol", "type": "TRIVIAL"}
        ],
        "explanation": "haloperidol is a drug. Dpp6/DPP6 and Kv4 are proteins, so excluded."
    },
    {
        # From training set: chromium and malondialdehyde example
        "text": "The objective was to find out the toxic effects of chromium on tannery workers. Blood malondialdehyde levels were measured.",
        "output": [
            {"text": "chromium", "type": "TRIVIAL"},
            {"text": "malondialdehyde", "type": "TRIVIAL"}
        ],
        "explanation": "Both chromium (element) and malondialdehyde (chemical compound) are included."
    },
    {
        # From training set: PAH and enzyme example
        "text": "Concentrations of polycyclic aromatic hydrocarbons (PAHs) were quantified. Catalase and glutathione peroxidase activities were also measured.",
        "output": [
            {"text": "polycyclic aromatic hydrocarbons", "type": "FAMILY"},
            {"text": "PAHs", "type": "ABBREVIATION"},
            {"text": "glutathione", "type": "TRIVIAL"}
        ],
        "explanation": "Catalase and peroxidase are enzymes (excluded). glutathione is a chemical compound."
    },
    {
        # From training set: RAP (rapamycin) and kinase example
        "text": "RAP inhibits mTOR activity. MNK1 kinase activity in MM cell lines was measured by a mitogen-activated protein kinase-dependent mechanism.",
        "output": [
            {"text": "RAP", "type": "ABBREVIATION"}
        ],
        "explanation": "RAP (rapamycin) is a drug. mTOR, MNK1, kinase are all proteins/enzymes, so excluded."
    }
]


def estimate_cost(samples: List[Dict], models: List[str]) -> Dict[str, Any]:
    """
    Estimate the cost of running inference.

    Returns:
        Dict with token counts and estimated costs per model
    """
    # Use cl100k_base encoding (used by GPT-4 and later)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback: estimate 4 chars per token
        encoding = None

    # Build a sample prompt to estimate system + few-shot tokens
    sample_user_prompt = build_user_prompt("Sample text for estimation.")
    system_tokens = len(encoding.encode(SYSTEM_PROMPT)) if encoding else len(SYSTEM_PROMPT) // 4
    few_shot_tokens = len(encoding.encode(sample_user_prompt)) if encoding else len(sample_user_prompt) // 4

    # Estimate input tokens for all samples
    total_text_tokens = 0
    for sample in samples:
        if encoding:
            total_text_tokens += len(encoding.encode(sample["text"]))
        else:
            total_text_tokens += len(sample["text"]) // 4

    # Per-request overhead (system + few-shot)
    overhead_per_request = system_tokens + few_shot_tokens - 50  # subtract the sample text
    total_input_tokens = total_text_tokens + (overhead_per_request * len(samples))

    # Estimate output tokens (assume ~200 tokens per response on average)
    estimated_output_tokens_per_sample = 200
    total_output_tokens = estimated_output_tokens_per_sample * len(samples)

    # Calculate costs per model
    costs = {}
    total_cost = 0.0

    for model in models:
        pricing = MODEL_PRICING.get(model, {"input": 2.00, "output": 10.00})
        input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
        model_cost = input_cost + output_cost
        costs[model] = {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": model_cost
        }
        total_cost += model_cost

    return {
        "num_samples": len(samples),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "costs_per_model": costs,
        "total_cost": total_cost
    }


def print_cost_estimate(estimate: Dict[str, Any], models: List[str]):
    """Print cost estimate in a formatted way."""
    print(f"\n{'='*60}")
    print("COST ESTIMATE")
    print(f"{'='*60}")
    print(f"Number of samples: {estimate['num_samples']}")
    print(f"Estimated input tokens:  {estimate['total_input_tokens']:,}")
    print(f"Estimated output tokens: {estimate['total_output_tokens']:,}")
    print()
    print("Cost breakdown per model:")
    print("-" * 40)

    for model in models:
        cost = estimate['costs_per_model'][model]
        print(f"  {model}:")
        print(f"    Input:  ${cost['input_cost']:.4f}")
        print(f"    Output: ${cost['output_cost']:.4f}")
        print(f"    Total:  ${cost['total_cost']:.4f}")

    print("-" * 40)
    print(f"TOTAL ESTIMATED COST: ${estimate['total_cost']:.4f}")
    print(f"{'='*60}\n")


def build_user_prompt(text: str) -> str:
    """Build user prompt with few-shot examples."""
    examples_text = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"""
Example {i}:
Input: "{ex['text']}"
Output: {json.dumps(ex['output'])}
Note: {ex['explanation']}
"""

    return f"""Here are some examples:
{examples_text}

Now extract chemical entities from the following text:

{text}

Output JSON array:"""


def extract_json_from_response(response: str) -> List[Dict]:
    """Extract JSON array from LLM response."""
    if not response:
        return []
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


async def process_single_sample(
    sample: Dict[str, Any],
    model: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Process a single sample with rate limiting via semaphore."""
    async with semaphore:
        try:
            use_completion_tokens = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")

            request_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(sample["text"])}
                ],
                "temperature": 0,
            }

            if use_completion_tokens:
                request_params["max_completion_tokens"] = 4096
            else:
                request_params["max_tokens"] = 4096

            response = await client.chat.completions.create(**request_params)

            response_text = response.choices[0].message.content
            predicted_entities = extract_json_from_response(response_text)

            return {
                "document_id": sample["document_id"],
                "text": sample["text"],
                "gold_entities": sample["entities"],
                "predicted_entities": predicted_entities,
                "raw_response": response_text,
                "model": model
            }

        except Exception as e:
            return {
                "document_id": sample["document_id"],
                "text": sample["text"],
                "gold_entities": sample["entities"],
                "predicted_entities": [],
                "error": str(e),
                "model": model
            }


async def run_inference_parallel(
    samples: List[Dict[str, Any]],
    model: str,
    client: AsyncOpenAI,
    max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """Run inference on samples using parallel requests."""
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        process_single_sample(sample, model, client, semaphore)
        for sample in samples
    ]

    results = await tqdm_asyncio.gather(*tasks, desc=f"Running {model} (parallel={max_concurrent})")

    return results


async def main_async():
    start_time = datetime.now(JST)
    print(f"{'='*60}")
    print(f"CHEMDNER Inference v2 (Improved Prompts)")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')} JST")
    print(f"{'='*60}\n")

    parser = argparse.ArgumentParser(description="CHEMDNER LLM Inference v2 (Parallel)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--models", nargs="+", default=["gpt-4.1", "gpt-5.2"],
                        help="Models to evaluate")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--parallel", type=int, default=10,
                        help="Max concurrent requests per model")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for output files")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = AsyncOpenAI(api_key=api_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading CHEMDNER dataset (split={args.split}, limit={args.limit})...")
    samples = load_chemdner(split=args.split, limit=args.limit)
    print(f"Loaded {len(samples)} samples")
    print(f"Parallel requests: {args.parallel}")

    # Estimate and display cost
    cost_estimate = estimate_cost(samples, args.models)
    print_cost_estimate(cost_estimate, args.models)

    # Confirm execution
    if not args.yes:
        response = input("Proceed with inference? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    print("\nStarting inference...\n")

    all_results = {}
    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Running inference with {model}")
        print(f"{'='*60}")

        model_start = datetime.now(JST)
        results = await run_inference_parallel(samples, model, client, args.parallel)
        model_end = datetime.now(JST)

        all_results[model] = results

        output_file = output_dir / f"predictions_{model.replace('.', '_')}{args.suffix}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        model_elapsed = model_end - model_start
        print(f"Saved results to {output_file}")
        print(f"{model} completed in {model_elapsed}")

    end_time = datetime.now(JST)
    elapsed = end_time - start_time

    print("\nInference complete!")
    print(f"Run evaluation with: python src/evaluate.py --input-dir outputs --pattern '*{args.suffix}.json'")
    print(f"\n{'='*60}")
    print(f"Started at:  {start_time.strftime('%Y-%m-%d %H:%M:%S')} JST")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')} JST")
    print(f"Elapsed:     {elapsed}")
    print(f"{'='*60}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
