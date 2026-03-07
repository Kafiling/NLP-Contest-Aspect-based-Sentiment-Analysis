cd /Users/kafiling/Projects/NLP-Contest1
.venv/bin/python roberta_absa_v3.py
Using device: cpu
Train rows: 3156 | Unique IDs: 2584
Test  rows: 461
Text cleaning applied (lowercase + whitespace normalisation)
  Sample: but the staff was so horrible to us.
Split -> train: 2680 rows | dev: 476 rows
  Dev aspect distribution:
food                       158
anecdotes/miscellaneous    145
service                     77
ambience                    56
price                       40
Sentiment training rows after augmentation: 4226
  Polarity distribution after augmentation:
positive    1603
negative    1216
neutral     1011
conflict     396

Loading tokenizer: roberta-base
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Sentiment class weights: positive=0.42 | negative=1.10 | neutral=1.99 | conflict=5.08

Loading RoBERTa backbone: roberta-base
Loading weights: 100%|█████████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 7685.09it/s]
RobertaModel LOAD REPORT from: roberta-base
Key                             | Status     | 
--------------------------------+------------+-
lm_head.dense.weight            | UNEXPECTED | 
lm_head.layer_norm.weight       | UNEXPECTED | 
lm_head.bias                    | UNEXPECTED | 
roberta.embeddings.position_ids | UNEXPECTED | 
lm_head.dense.bias              | UNEXPECTED | 
lm_head.layer_norm.bias         | UNEXPECTED | 
pooler.dense.weight             | MISSING    | 
pooler.dense.bias               | MISSING    | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING       :those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

=== Joint Training (shared encoder) ===
  Epoch  Train Loss    Dev Loss    Dev F1     Time  ETA
  ------------------------------------------------------------
      1      1.7186      1.3490    0.5673     361s  84.2min
      2      0.9953      1.2797    0.6778     348s  76.8min
      3      0.7356      1.1914    0.7120     352s  70.7min
      4      0.6272      1.3307    0.7280     351s  64.7min
      5      0.5663      1.2683    0.7429     386s  59.9min
      6      0.5357      1.3034    0.7560     384s  54.5min
      7      0.5151      1.3058    0.7547     361s  48.4min
      8      0.4906      1.3753    0.7469     357s  42.3min
      9      0.4826      1.4188    0.7500     358s  36.2min
     10      0.4700      1.3989    0.7559     360s  30.1min
     11      0.4691      1.3959    0.7705     367s  24.1min
     12      0.4671      1.4114    0.7633     372s  18.1min
     13      0.4674      1.4037    0.7738     366s  12.1min
     14      0.4635      1.3997    0.7733     366s  6.1min
     15      0.4643      1.4140    0.7692     367s  0.0min

Best overall dev F1 (default threshold): 0.7738
Loss curve saved → /Users/kafiling/Projects/NLP-Contest1/Resource/roberta_v3_loss_curve.png

=== Tuning per-aspect thresholds on dev set ===
  Threshold tuning per aspect:
    food                         thr=0.35  F1=0.9486
    service                      thr=0.60  F1=0.9091
    ambience                     thr=0.95  F1=0.8544
    price                        thr=0.60  F1=0.9500
    anecdotes/miscellaneous      thr=0.10  F1=0.8664

=== Final Dev Evaluation (tuned thresholds) ===
  Aspect     F1: 0.9057
  Sentiment  F1: 0.8529
  Overall    F1: 0.7725
Saved dev predictions → /Users/kafiling/Projects/NLP-Contest1/Resource/roberta_v3_dev_pred.csv

=== Predicting on Test Set ===
Saved 588 rows → /Users/kafiling/Projects/NLP-Contest1/Resource/roberta_v3_test_pred.csv
     id           aspectCategory  polarity
0   899                     food   neutral
1  1349  anecdotes/miscellaneous  positive
2   934                     food  negative
3  2199                     food  positive
4   188                     food  negative
5   188                  service  positive
6   188                 ambience  positive
7  1748                     food  positive
8  1748                  service  positive
9   949                     food  negative


# Evaluate dev predictions         
cd /Users/kafiling/Projects/NLP-Contest1/Resource
../.venv/bin/python evaluate.py contest1_train.csv roberta_v3_dev_pred.csv
zsh: command not found: #
=== CLASSIFICATION : ASPECT ===
                class name  precision  recall  F1-score  support
0                     food      0.908   0.994     0.949      158
1                    price      0.950   0.950     0.950       40
2                  service      0.909   0.909     0.909       77
3                 ambience      0.917   0.786     0.846       56
4  anecdotes/miscellaneous      0.821   0.917     0.866      145
5                MACRO AVG      0.901   0.911     0.904      476
6                MICRO AVG      0.884   0.929     0.906      476 

=== CLASSIFICATION : SENTIMENT ===
  class name  precision  recall  F1-score  support
0   positive      0.903   0.939     0.920      228
1   negative      0.817   0.826     0.822       92
2    neutral      0.792   0.623     0.697       61
3   conflict      0.581   0.545     0.562       33
4  MACRO AVG      0.773   0.733     0.750      414
5  MICRO AVG      0.846   0.836     0.841      414 

=== CLASSIFICATION : OVERALL ===
              precision  recall  F1-score  support
0  MICRO AVG      0.754   0.792     0.773      476 

(.venv) kafiling@Panthawits-MacBook-Pro Resource % # Verify test predictions have correct IDs
../.venv/bin/python check_id.py contest1_test.csv roberta_v3_test_pred.csv
zsh: command not found: #
All good! Sentences in your predicted file are labeled!
