"""
Configuration for BioBERT fine-tuning on CHEMDNER.
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    """Training configuration."""

    # Model
    model_name: str = "dmis-lab/biobert-v1.1"

    # Data
    data_dir: str = "data/chemdner_corpus"
    train_file: str = "training.ner.ssplit.token4.BIO"
    dev_file: str = "development.ner.ssplit.token4.BIO"
    test_file: str = "evaluation.ner.ssplit.token4.BIO"

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5
    max_seq_length: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Training settings
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100

    # Output
    output_dir: str = "outputs/exp008_biobert"
    model_save_dir: str = "outputs/exp008_biobert/model"

    # Label mapping
    label2id: dict = field(default_factory=lambda: {
        "O": 0,
        "B-CHEMICAL": 1,
        "I-CHEMICAL": 2,
    })
    id2label: dict = field(default_factory=lambda: {
        0: "O",
        1: "B-CHEMICAL",
        2: "I-CHEMICAL",
    })

    # Seed for reproducibility
    seed: int = 42

    @property
    def num_labels(self) -> int:
        return len(self.label2id)

    def get_train_path(self) -> Path:
        return Path(self.data_dir) / self.train_file

    def get_dev_path(self) -> Path:
        return Path(self.data_dir) / self.dev_file

    def get_test_path(self) -> Path:
        return Path(self.data_dir) / self.test_file


# Default configuration instance
default_config = Config()
