The scripts generate chemical mention spans directly from the supplied article text. All chemical-name supervision comes from the supplied training and development annotations.

`ner.py` trains an offset-preserving BIES token classifier using a publicly available biomedical model pretrained on unlabeled text. It uses overlapping context windows and constrained decoding. `case_ner.py` additionally learns capitalization features and the supplied chemical mention classes, then marginalizes class scores to obtain BIES scores.

Neural model comparisons reserve 500 development articles, chosen by shuffling with seed 1234. The other 6,500 articles provide supervision. `stacking.py` compares regularized linear and shallow tree classifiers using five-fold cross-validation grouped by article. It combines proposed spans, model confidence, repeated names, and lexical statistics derived from the 6,500 training articles. `ensemble.py` compares this combination with direct logit averaging.

`final_train.py` retrains the selected models on all 7,000 labeled articles. Epoch choices remain fixed at the values selected on the reserved articles. Final inference uses the trained models and the selected ensemble configuration; no test annotations are available or used.

Initial public checkpoints:
- [BiomedBERT base, abstracts and full text](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
- [BiomedBERT large, abstracts](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract)
- [BioLinkBERT large, abstracts and citation links](https://huggingface.co/michiyasunaga/BioLinkBERT-large)

Training runs use Python 3.10 and PyTorch 2.9.1 with CUDA 12.9 on one NVIDIA L4. Other package versions are listed in requirements.txt. The provenance manifest records hashes of the supplied data. No language model inference APIs or externally trained chemical recognizers are used.

Check the submitted file with:

    python3 /workspace/check_format.py /workspace/output/predictions.tsv
