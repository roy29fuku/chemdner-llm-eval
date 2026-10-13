
## How the predictions are evaluated

Your predictions will be compared with hidden gold annotations of the test articles. A predicted mention is correct only if its PMID, section, start and end all match a gold mention exactly. The score is the F1 score (the harmonic mean of precision and recall) over all mentions in the test articles. Aim for the highest score you can within the time limit.
