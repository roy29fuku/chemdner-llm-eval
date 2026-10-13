"""
Parallel LLM inference for CHEMDNER chemical entity recognition.
Version 2: Improved prompts with few-shot examples following CHEMDNER guidelines.
Version 3: Added Responses API support for GPT-5.2.
"""
import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Literal

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


# MRC (Machine Reading Comprehension) system prompt
# Based on BioBERT-MRC approach that achieved F1=92.92% on CHEMDNER
SYSTEM_PROMPT_MRC = """You are a chemical entity extraction system. Answer the following question about the given text.

## Question
What are all the chemical entity mentions in this text? A chemical entity is any mention whose chemical structure can be determined or inferred.

## Types of chemical entities to find:
1. TRIVIAL: Common/trivial names (aspirin, caffeine, morphine, haloperidol)
2. SYSTEMATIC: IUPAC names (2-acetoxybenzoic acid, 1,3,7-trimethylxanthine)
3. ABBREVIATION: Chemical abbreviations (ATP, NAD+, NO, PAHs)
4. FORMULA: Molecular formulas (H2O, NaCl, C6H12O6)
5. FAMILY: Chemical families (polycyclic aromatic hydrocarbons, fatty acids)

## NOT chemical entities (do not include):
- Proteins/enzymes: CYP3A4, kinase, synthase, P-glycoprotein
- Genes: BRCA1, p53, eNOS
- Transporters: OATP, MCT1
- Vague terms: "drugs", "compounds", "metabolites"

## Answer format
List each chemical entity found as a JSON object with "text" (exact match from input) and "type".
Output only a JSON array. If no chemicals found, output []."""


# Self-verification system prompt
SYSTEM_PROMPT_VERIFY = """You are an expert in distinguishing chemical entities from biological entities.

## Task
Review the extracted entities and REMOVE any that are NOT chemicals according to CHEMDNER guidelines.

## REMOVE these types (they are NOT chemicals):
- Proteins and enzymes: CYP3A4, P-glycoprotein, kinase, synthase, receptor, transporter
- Genes: BRCA1, eNOS, p53, any gene names
- General/vague terms: "drugs", "compounds", "metabolites", "chemicals"
- Biological entities: cells, bacteria, viruses, organisms
- Plant/food names without specific compound: garlic, ginger (but "allicin" is OK)

## KEEP these types (they ARE chemicals):
- Specific drug names: aspirin, morphine, metformin
- Chemical compounds: malondialdehyde, glutathione
- Elements: chromium, sodium, iron
- Abbreviations of specific chemicals: ATP, NAD+, NO (nitric oxide)
- Molecular formulas: H2O, NaCl

## Output Format
Return a JSON array containing ONLY the entities that should be KEPT.
Keep the same format: [{"text": "...", "type": "..."}, ...]
If all entities should be removed, return an empty array: []"""


# Chain-of-Thought system prompt
SYSTEM_PROMPT_COT = """You are an expert in chemical named entity recognition following the CHEMDNER annotation guidelines.

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

## Chain-of-Thought Process
Before outputting the final answer, think step by step:
1. Scan the text and identify ALL potential chemical-related terms
2. For EACH candidate, ask: "Can I determine or infer its chemical structure?"
3. Check if it's a protein, enzyme, gene, or transporter (if yes, EXCLUDE)
4. Verify the exact text span to extract

## Output Format
First, show your reasoning in <thinking> tags.
Then output the JSON array.

Example output format:
<thinking>
- "aspirin" - drug name, chemical structure known → INCLUDE
- "CYP3A4" - this is an enzyme/protein → EXCLUDE
</thinking>
[{"text": "aspirin", "type": "TRIVIAL"}]"""


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


def build_mrc_prompt(text: str) -> str:
    """Build MRC-style user prompt with few-shot examples."""
    examples_text = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"""
Example {i}:
Text: "{ex['text']}"
Question: What are all the chemical entity mentions in this text?
Answer: {json.dumps(ex['output'])}
Explanation: {ex['explanation']}
"""

    return f"""Here are some examples of how to answer:
{examples_text}

Now answer this question:

Text: "{text}"

Question: What are all the chemical entity mentions in this text?

Answer:"""


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


async def verify_entities(
    text: str,
    entities: List[Dict],
    model: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> List[Dict]:
    """Verify extracted entities and filter out non-chemicals."""
    if not entities:
        return []

    async with semaphore:
        try:
            entity_list = json.dumps(entities, ensure_ascii=False, indent=2)
            user_prompt = f"""Original text:
{text}

Extracted entities to verify:
{entity_list}

Review each entity and return only the ones that are TRUE chemical entities (not proteins, genes, etc.).
Keep the same JSON format with "text" and "type" fields.
Output JSON array:"""

            use_completion_tokens = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")
            request_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_VERIFY},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0,
            }

            if use_completion_tokens:
                request_params["max_completion_tokens"] = 4096
            else:
                request_params["max_tokens"] = 4096

            response = await client.chat.completions.create(**request_params)
            response_text = response.choices[0].message.content
            verified_entities = extract_json_from_response(response_text)

            return verified_entities

        except Exception:
            # On error, return original entities
            return entities


async def process_single_sample_chat(
    sample: Dict[str, Any],
    model: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    use_cot: bool = False,
    use_verify: bool = False,
    use_mrc: bool = False
) -> Dict[str, Any]:
    """Process a single sample using Chat Completions API."""
    async with semaphore:
        try:
            use_completion_tokens = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")

            # Select system prompt and user prompt based on mode
            if use_mrc:
                system_prompt = SYSTEM_PROMPT_MRC
                user_prompt = build_mrc_prompt(sample["text"])
            elif use_cot:
                system_prompt = SYSTEM_PROMPT_COT
                user_prompt = build_user_prompt(sample["text"])
            else:
                system_prompt = SYSTEM_PROMPT
                user_prompt = build_user_prompt(sample["text"])

            request_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
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

            # Store original entities before verification
            original_entities = predicted_entities.copy() if use_verify else None

            return {
                "document_id": sample["document_id"],
                "text": sample["text"],
                "gold_entities": sample["entities"],
                "predicted_entities": predicted_entities,
                "original_entities": original_entities,
                "raw_response": response_text,
                "model": model,
                "api": "chat",
                "cot": use_cot,
                "verify": use_verify,
                "mrc": use_mrc
            }

        except Exception as e:
            return {
                "document_id": sample["document_id"],
                "text": sample["text"],
                "gold_entities": sample["entities"],
                "predicted_entities": [],
                "error": str(e),
                "model": model,
                "api": "chat",
                "cot": use_cot,
                "verify": use_verify,
                "mrc": use_mrc
            }


async def process_single_sample_responses(
    sample: Dict[str, Any],
    model: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    use_cot: bool = False
) -> Dict[str, Any]:
    """Process a single sample using Responses API (better for GPT-5.2)."""
    async with semaphore:
        try:
            # Build full prompt with few-shot examples
            full_input = build_user_prompt(sample["text"])
            system_prompt = SYSTEM_PROMPT_COT if use_cot else SYSTEM_PROMPT

            response = await client.responses.create(
                model=model,
                instructions=system_prompt,
                input=full_input,
            )

            response_text = response.output_text
            predicted_entities = extract_json_from_response(response_text)

            return {
                "document_id": sample["document_id"],
                "text": sample["text"],
                "gold_entities": sample["entities"],
                "predicted_entities": predicted_entities,
                "raw_response": response_text,
                "model": model,
                "api": "responses",
                "cot": use_cot
            }

        except Exception as e:
            return {
                "document_id": sample["document_id"],
                "text": sample["text"],
                "gold_entities": sample["entities"],
                "predicted_entities": [],
                "error": str(e),
                "model": model,
                "api": "responses",
                "cot": use_cot
            }


async def process_single_sample(
    sample: Dict[str, Any],
    model: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    api_type: Literal["chat", "responses"] = "chat",
    use_cot: bool = False,
    use_verify: bool = False,
    use_mrc: bool = False
) -> Dict[str, Any]:
    """Process a single sample with specified API type."""
    if api_type == "responses":
        return await process_single_sample_responses(sample, model, client, semaphore, use_cot)
    else:
        return await process_single_sample_chat(sample, model, client, semaphore, use_cot, use_verify, use_mrc)


async def run_inference_parallel(
    samples: List[Dict[str, Any]],
    model: str,
    client: AsyncOpenAI,
    max_concurrent: int = 10,
    api_type: Literal["chat", "responses"] = "chat",
    use_cot: bool = False,
    use_verify: bool = False,
    use_mrc: bool = False
) -> List[Dict[str, Any]]:
    """Run inference on samples using parallel requests."""
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        process_single_sample(sample, model, client, semaphore, api_type, use_cot, use_verify, use_mrc)
        for sample in samples
    ]

    labels = []
    if api_type == "responses":
        labels.append("responses")
    if use_cot:
        labels.append("CoT")
    if use_verify:
        labels.append("verify")
    if use_mrc:
        labels.append("MRC")
    label_str = f" [{', '.join(labels)}]" if labels else ""
    results = await tqdm_asyncio.gather(*tasks, desc=f"Running {model}{label_str} (parallel={max_concurrent})")

    # If verification is enabled, run second pass to verify entities
    if use_verify:
        print(f"\nRunning verification pass for {model}...")
        verify_tasks = [
            verify_entities(r["text"], r["predicted_entities"], model, client, semaphore)
            for r in results
        ]
        verified_results = await tqdm_asyncio.gather(
            *verify_tasks, desc=f"Verifying {model} (parallel={max_concurrent})"
        )

        # Update results with verified entities
        for i, verified in enumerate(verified_results):
            results[i]["original_entities"] = results[i]["predicted_entities"]
            results[i]["predicted_entities"] = verified

    return results


async def main_async():
    start_time = datetime.now(JST)
    print(f"{'='*60}")
    print(f"CHEMDNER Inference v3 (Responses API Support)")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')} JST")
    print(f"{'='*60}\n")

    parser = argparse.ArgumentParser(description="CHEMDNER LLM Inference v3 (Parallel)")
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
    parser.add_argument("--api", type=str, default="chat", choices=["chat", "responses"],
                        help="API type: 'chat' for Chat Completions, 'responses' for Responses API")
    parser.add_argument("--cot", action="store_true",
                        help="Use Chain-of-Thought prompting")
    parser.add_argument("--verify", action="store_true",
                        help="Use self-verification (2-pass: extract then verify)")
    parser.add_argument("--mrc", action="store_true",
                        help="Use MRC (Machine Reading Comprehension) style prompting")
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
    print(f"API type: {args.api}")
    print(f"Chain-of-Thought: {args.cot}")
    print(f"Self-verification: {args.verify}")
    print(f"MRC mode: {args.mrc}")

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
        labels = []
        if args.api == "responses":
            labels.append("responses")
        if args.cot:
            labels.append("CoT")
        if args.verify:
            labels.append("verify")
        if args.mrc:
            labels.append("MRC")
        label_str = f" [{', '.join(labels)}]" if labels else ""
        print(f"\n{'='*60}")
        print(f"Running inference with {model}{label_str}")
        print(f"{'='*60}")

        model_start = datetime.now(JST)
        results = await run_inference_parallel(samples, model, client, args.parallel, args.api, args.cot, args.verify, args.mrc)
        model_end = datetime.now(JST)

        all_results[model] = results

        # Include API type and CoT in filename
        suffixes = []
        if args.api == "responses":
            suffixes.append("responses")
        if args.cot:
            suffixes.append("cot")
        if args.verify:
            suffixes.append("verify")
        if args.mrc:
            suffixes.append("mrc")
        extra_suffix = f"_{'_'.join(suffixes)}" if suffixes else ""
        output_file = output_dir / f"predictions_{model.replace('.', '_')}{extra_suffix}{args.suffix}.json"
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
