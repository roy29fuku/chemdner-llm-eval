#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from chem_ner import ChemicalMentionRecognizer, RecognizerConfig, predict_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-annotations", default="data/training.annotations.txt")
    ap.add_argument("--test-abstracts", default="data/test.abstracts.txt")
    ap.add_argument("--output", default="output/predictions.tsv")
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args()

    config = RecognizerConfig(min_count=args.min_count)
    recognizer = ChemicalMentionRecognizer.from_training(args.train_annotations, config=config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for pmid, section, start, end in predict_file(recognizer, args.test_abstracts):
            out.write(f"{pmid}\t{section}:{start}:{end}\n")


if __name__ == "__main__":
    main()

