"""
CHEMDNER dataset loader for BIO format data.
"""
from pathlib import Path
from typing import List, Dict, Any


# Default data directory (can be overridden)
DATA_DIR = Path("/app/data/chemdner_corpus")


def parse_bio_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Parse BIO format file.

    Format:
    - Token<TAB>Tag per line
    - Empty line separates sentences/documents
    - Tags: O (Outside), B (Beginning of entity), I (Inside entity)

    Returns:
        List of documents with text and entities
    """
    documents = []
    current_tokens = []
    current_tags = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                # End of sentence/document
                if current_tokens:
                    doc = process_sentence(current_tokens, current_tags)
                    documents.append(doc)
                    current_tokens = []
                    current_tags = []
            else:
                parts = line.split("\t")
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[1]
                    current_tokens.append(token)
                    current_tags.append(tag)
                elif len(parts) == 1:
                    # Sometimes there's only token, assume O tag
                    current_tokens.append(parts[0])
                    current_tags.append("O")

        # Process last document if no trailing newline
        if current_tokens:
            doc = process_sentence(current_tokens, current_tags)
            documents.append(doc)

    return documents


def process_sentence(tokens: List[str], tags: List[str]) -> Dict[str, Any]:
    """
    Process a sentence to extract text and entities.

    Args:
        tokens: List of tokens
        tags: List of BIO tags

    Returns:
        Dict with text and entities
    """
    # Reconstruct text
    text = " ".join(tokens)

    # Extract entities
    entities = []
    current_entity_tokens = []
    current_start_idx = 0

    char_offset = 0
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        token_start = char_offset
        token_end = char_offset + len(token)

        if tag == "B":
            # Start of new entity
            if current_entity_tokens:
                # Save previous entity if exists
                entity_text = " ".join(current_entity_tokens)
                entities.append({
                    "text": entity_text,
                    "type": "CHEMICAL",
                    "start": current_start_idx,
                    "end": current_start_idx + len(entity_text)
                })
            current_entity_tokens = [token]
            current_start_idx = token_start

        elif tag == "I" and current_entity_tokens:
            # Continue entity
            current_entity_tokens.append(token)

        else:
            # Outside or new sentence
            if current_entity_tokens:
                # Save entity
                entity_text = " ".join(current_entity_tokens)
                entities.append({
                    "text": entity_text,
                    "type": "CHEMICAL",
                    "start": current_start_idx,
                    "end": current_start_idx + len(entity_text)
                })
                current_entity_tokens = []

        char_offset = token_end + 1  # +1 for space

    # Handle entity at end of sentence
    if current_entity_tokens:
        entity_text = " ".join(current_entity_tokens)
        entities.append({
            "text": entity_text,
            "type": "CHEMICAL",
            "start": current_start_idx,
            "end": current_start_idx + len(entity_text)
        })

    return {
        "text": text,
        "entities": entities
    }


def merge_sentences_to_documents(sentences: List[Dict], sentences_per_doc: int = 10) -> List[Dict[str, Any]]:
    """
    Merge sentences into pseudo-documents for evaluation.

    The BIO file contains sentences, but we need documents for evaluation.
    We merge multiple sentences together.

    Args:
        sentences: List of parsed sentences
        sentences_per_doc: Number of sentences per document

    Returns:
        List of documents
    """
    documents = []
    current_text_parts = []
    current_entities = []
    current_offset = 0

    for i, sentence in enumerate(sentences):
        # Add sentence text
        sent_text = sentence["text"]
        current_text_parts.append(sent_text)

        # Adjust entity offsets and add
        for entity in sentence["entities"]:
            adjusted_entity = entity.copy()
            adjusted_entity["start"] += current_offset
            adjusted_entity["end"] += current_offset
            current_entities.append(adjusted_entity)

        current_offset += len(sent_text) + 1  # +1 for space

        # Create document when we have enough sentences
        if (i + 1) % sentences_per_doc == 0:
            doc_text = " ".join(current_text_parts)
            documents.append({
                "document_id": f"doc_{len(documents)}",
                "text": doc_text,
                "entities": current_entities
            })
            current_text_parts = []
            current_entities = []
            current_offset = 0

    # Add remaining sentences
    if current_text_parts:
        doc_text = " ".join(current_text_parts)
        documents.append({
            "document_id": f"doc_{len(documents)}",
            "text": doc_text,
            "entities": current_entities
        })

    return documents


def load_chemdner(split: str = "test", limit: int | None = None, data_dir: str | None = None) -> List[Dict[str, Any]]:
    """
    Load CHEMDNER dataset.

    Args:
        split: Dataset split ('train', 'development', 'test')
        limit: Number of documents to load (None for all)
        data_dir: Optional override for data directory

    Returns:
        List of samples with document_id, text, and entities
    """
    base_dir = Path(data_dir) if data_dir else DATA_DIR

    # Map split names to file names
    split_map = {
        "train": "training",
        "training": "training",
        "dev": "development",
        "development": "development",
        "valid": "development",
        "validation": "development",
        "test": "evaluation",
        "evaluation": "evaluation"
    }

    file_prefix = split_map.get(split, split)
    bio_file = base_dir / f"{file_prefix}.ner.ssplit.token4.BIO"

    if not bio_file.exists():
        raise FileNotFoundError(f"BIO file not found: {bio_file}")

    print(f"Loading {split} split from {bio_file}")

    # Parse BIO file
    sentences = parse_bio_file(bio_file)
    print(f"Parsed {len(sentences)} sentences")

    # Merge sentences to documents
    documents = merge_sentences_to_documents(sentences)
    print(f"Created {len(documents)} documents")

    # Apply limit
    if limit:
        documents = documents[:limit]

    return documents


def format_entities_for_evaluation(entities: List[Dict]) -> List[tuple]:
    """
    Format entities for evaluation comparison.
    Returns list of (start, end, text) tuples.
    """
    return [(e["start"], e["end"], e["text"]) for e in entities]


if __name__ == "__main__":
    import sys

    # Allow override of data directory from command line
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None

    # Test loading
    samples = load_chemdner(split="test", limit=5, data_dir=data_dir)
    print(f"\nLoaded {len(samples)} samples")

    for s in samples:
        print(f"\nDoc ID: {s['document_id']}")
        print(f"Text preview: {s['text'][:200]}...")
        print(f"Entities: {len(s['entities'])}")
        for e in s["entities"][:5]:
            print(f"  - '{e['text']}' ({e['type']})")
