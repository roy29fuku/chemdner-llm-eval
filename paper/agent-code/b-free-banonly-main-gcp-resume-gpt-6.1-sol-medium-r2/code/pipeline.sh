#!/bin/bash
set -e
while kill -0 2492 2>/dev/null; do sleep 10; done
python3 ~/code/select_average.py ~/runs/validation > ~/runs/validation_average.log 2>&1
python3 ~/code/ner.py --mode predict --data ~/data --model ~/runs/validation/average --out ~/runs/validation/average_results --batch 24 --split development > ~/runs/validation_average_dev.log 2>&1
python3 ~/code/ner.py --mode predict --data ~/data --model ~/runs/validation/average --out ~/runs/validation/average_results --batch 24 --split test > ~/runs/validation_average_test.log 2>&1
python3 ~/code/ner.py --mode predict --data ~/data --model ~/runs/validation/best --out ~/runs/validation --batch 24 --split test > ~/runs/validation_test.log 2>&1
python3 ~/code/ner.py --data ~/data --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract --out ~/runs/validation2 --epochs 6 --batch 24 --seed 137 > ~/runs/validation2.log 2>&1
python3 ~/code/select_average.py ~/runs/validation2 > ~/runs/validation2_average.log 2>&1
python3 ~/code/ner.py --mode predict --data ~/data --model ~/runs/validation2/average --out ~/runs/validation2/average_results --batch 24 --split development > ~/runs/validation2_average_dev.log 2>&1
python3 ~/code/ner.py --mode predict --data ~/data --model ~/runs/validation2/average --out ~/runs/validation2/average_results --batch 24 --split test > ~/runs/validation2_average_test.log 2>&1
python3 ~/code/ner.py --mode predict --data ~/data --model ~/runs/validation2/best --out ~/runs/validation2 --batch 24 --split test > ~/runs/validation2_test.log 2>&1
python3 ~/code/ner.py --data ~/data --train training,development --dev none --out ~/runs/full1 --epochs 6 --batch 24 --seed 42 > ~/runs/full1.log 2>&1
python3 ~/code/ner.py --mode predict --data ~/data --model ~/runs/full1/epoch6 --out ~/runs/full1 --batch 24 --split test > ~/runs/full1_test.log 2>&1
python3 ~/code/average.py ~/runs/full1/epoch4 ~/runs/full1/epoch5 ~/runs/full1/epoch6 --out ~/runs/full1/average > ~/runs/full1_average.log 2>&1
python3 ~/code/ner.py --mode predict --data ~/data --model ~/runs/full1/average --out ~/runs/full1/average_results --batch 24 --split test > ~/runs/full1_average_test.log 2>&1
python3 ~/code/ner.py --data ~/data --train training,development --dev none --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract --out ~/runs/full2 --epochs 6 --batch 24 --seed 137 > ~/runs/full2.log 2>&1
python3 ~/code/ner.py --mode predict --data ~/data --model ~/runs/full2/epoch6 --out ~/runs/full2 --batch 24 --split test > ~/runs/full2_test.log 2>&1
python3 ~/code/average.py ~/runs/full2/epoch4 ~/runs/full2/epoch5 ~/runs/full2/epoch6 --out ~/runs/full2/average > ~/runs/full2_average.log 2>&1
python3 ~/code/ner.py --mode predict --data ~/data --model ~/runs/full2/average --out ~/runs/full2/average_results --batch 24 --split test > ~/runs/full2_average_test.log 2>&1
echo PIPELINE_DONE
