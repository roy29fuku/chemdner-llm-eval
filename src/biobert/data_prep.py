"""
Data preparation for BioBERT fine-tuning on CHEMDNER.

Converts BIO format data to Hugging Face datasets format with proper
subword tokenization and label alignment.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class TokenizedExample:
    """A single tokenized example."""
    tokens: List[str]
    ner_tags: List[int]
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


def parse_bio_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Parse BIO format file.

    Format:
    - Token<TAB>Tag per line
    - Empty line separates sentences
    - Tags: O, B, I (CHEMDNER uses simple B/I/O without entity type suffix)

    Returns:
        List of sentences with tokens and tags
    """
    sentences = []
    current_tokens = []
    current_tags = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current_tokens:
                    sentences.append({
                        "tokens": current_tokens,
                        "tags": current_tags
                    })
                    current_tokens = []
                    current_tags = []
            else:
                parts = line.split("\t")
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[1]
                    current_tokens.append(token)
                    # Map B/I/O to B-CHEMICAL/I-CHEMICAL/O
                    if tag == "B":
                        current_tags.append("B-CHEMICAL")
                    elif tag == "I":
                        current_tags.append("I-CHEMICAL")
                    else:
                        current_tags.append("O")
                elif len(parts) == 1:
                    current_tokens.append(parts[0])
                    current_tags.append("O")

        # Handle last sentence
        if current_tokens:
            sentences.append({
                "tokens": current_tokens,
                "tags": current_tags
            })

    return sentences


def align_labels_with_tokens(
    tokens: List[str],
    tags: List[str],
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 256
) -> TokenizedExample:
    """
    Tokenize and align labels with subword tokens.

    For tokens that are split into multiple subwords:
    - First subword gets the original label
    - Subsequent subwords get -100 (ignored in loss)

    Args:
        tokens: Original tokens
        tags: Original NER tags
        tokenizer: Hugging Face tokenizer
        label2id: Label to ID mapping
        max_length: Maximum sequence length

    Returns:
        TokenizedExample with aligned labels
    """
    # Tokenize with is_split_into_words=True
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )

    # Get word IDs for each token
    word_ids = encoding.word_ids()

    # Align labels
    labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special tokens ([CLS], [SEP], [PAD])
            labels.append(-100)
        elif word_id != previous_word_id:
            # First subword of a word - use the original label
            labels.append(label2id[tags[word_id]])
        else:
            # Continuation subword - ignore in loss
            labels.append(-100)
        previous_word_id = word_id

    return TokenizedExample(
        tokens=tokens,
        ner_tags=[label2id[t] for t in tags],
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        labels=labels
    )


class CHEMDNERDataset(Dataset):
    """PyTorch Dataset for CHEMDNER NER."""

    def __init__(
        self,
        filepath: Path,
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

        # Parse BIO file
        sentences = parse_bio_file(filepath)

        # Tokenize and align labels
        self.examples = []
        for sent in sentences:
            example = align_labels_with_tokens(
                sent["tokens"],
                sent["tags"],
                tokenizer,
                label2id,
                max_length
            )
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(example.attention_mask, dtype=torch.long),
            "labels": torch.tensor(example.labels, dtype=torch.long)
        }


def load_and_prepare_data(
    data_dir: str,
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 256,
    train_file: str = "training.ner.ssplit.token4.BIO",
    dev_file: str = "development.ner.ssplit.token4.BIO",
    test_file: str = "evaluation.ner.ssplit.token4.BIO"
) -> Tuple[CHEMDNERDataset, CHEMDNERDataset, CHEMDNERDataset]:
    """
    Load and prepare train/dev/test datasets.

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
    """
    data_path = Path(data_dir)

    train_dataset = CHEMDNERDataset(
        data_path / train_file,
        tokenizer,
        label2id,
        max_length
    )

    dev_dataset = CHEMDNERDataset(
        data_path / dev_file,
        tokenizer,
        label2id,
        max_length
    )

    test_dataset = CHEMDNERDataset(
        data_path / test_file,
        tokenizer,
        label2id,
        max_length
    )

    return train_dataset, dev_dataset, test_dataset


def get_sentences_from_bio(filepath: Path) -> List[Dict[str, Any]]:
    """
    Get sentences with their original tokens and entities for evaluation.

    Returns list of dicts with:
    - tokens: List[str]
    - entities: List[Dict] with text, type, start, end
    """
    sentences = parse_bio_file(filepath)
    results = []

    for sent in sentences:
        tokens = sent["tokens"]
        tags = sent["tags"]
        text = " ".join(tokens)

        # Extract entities
        entities = []
        current_entity_tokens = []
        current_start_idx = 0
        char_offset = 0

        for i, (token, tag) in enumerate(zip(tokens, tags)):
            token_start = char_offset

            if tag == "B-CHEMICAL":
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
            elif tag == "I-CHEMICAL" and current_entity_tokens:
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

        results.append({
            "tokens": tokens,
            "text": text,
            "entities": entities
        })

    return results


if __name__ == "__main__":
    # Test data loading
    from transformers import AutoTokenizer
    from .config import default_config

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(default_config.model_name)

    print(f"Loading data from {default_config.data_dir}...")
    train_ds, dev_ds, test_ds = load_and_prepare_data(
        default_config.data_dir,
        tokenizer,
        default_config.label2id,
        default_config.max_seq_length
    )

    print(f"Train: {len(train_ds)} sentences")
    print(f"Dev: {len(dev_ds)} sentences")
    print(f"Test: {len(test_ds)} sentences")

    # Print a sample
    sample = train_ds[0]
    print(f"\nSample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample labels: {sample['labels'][:20]}...")
