Using device: cpu
Train rows: 3156 | Unique IDs: 2584
Test  rows: 461
Split -> train: 2680 rows | dev: 476 rows
Sentiment training rows after augmentation: 4239
  Polarity distribution after augmentation:
positive    1598
negative    1210
neutral     1002
conflict     429

Loading tokenizer: roberta-base
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Sentiment class weights: positive=0.42 | negative=1.11 | neutral=2.01 | conflict=4.69

Loading RoBERTa backbone: roberta-base
Loading weights: 100%|███████████████████████████████████████████| 197/197 [00:00<00:00, 7491.32it/s]
RobertaModel LOAD REPORT from: roberta-base
Key                             | Status     | 
--------------------------------+------------+-
lm_head.dense.bias              | UNEXPECTED | 
lm_head.layer_norm.weight       | UNEXPECTED | 
lm_head.layer_norm.bias         | UNEXPECTED | 
lm_head.bias                    | UNEXPECTED | 
lm_head.dense.weight            | UNEXPECTED | 
roberta.embeddings.position_ids | UNEXPECTED | 
pooler.dense.weight             | MISSING    | 
pooler.dense.bias               | MISSING    | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING       :those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

=== Joint Training (shared encoder) ===
  Epoch  1/15 | loss=1.7042 | overall-dev F1=0.4273 | 365s | ETA 85.1min
  Epoch  2/15 | loss=0.9632 | overall-dev F1=0.6667 | 361s | ETA 78.6min
  Epoch  3/15 | loss=0.7113 | overall-dev F1=0.6904 | 361s | ETA 72.4min
  Epoch  4/15 | loss=0.6008 | overall-dev F1=0.6954 | 359s | ETA 66.3min
  Epoch  5/15 | loss=0.5504 | overall-dev F1=0.7060 | 361s | ETA 60.2min
  Epoch  6/15 | loss=0.5148 | overall-dev F1=0.7097 | 366s | ETA 54.3min
  Epoch  7/15 | loss=0.4947 | overall-dev F1=0.7150 | 360s | ETA 48.2min
  Epoch  8/15 | loss=0.4789 | overall-dev F1=0.7254 | 361s | ETA 42.2min
  Epoch  9/15 | loss=0.4656 | overall-dev F1=0.7461 | 361s | ETA 36.2min
  Epoch 10/15 | loss=0.4610 | overall-dev F1=0.7387 | 361s | ETA 30.1min
  Epoch 11/15 | loss=0.4492 | overall-dev F1=0.7306 | 364s | ETA 24.1min
  Epoch 12/15 | loss=0.4510 | overall-dev F1=0.7344 | 361s | ETA 18.1min
  Epoch 13/15 | loss=0.4545 | overall-dev F1=0.7349 | 361s | ETA 12.1min
  Epoch 14/15 | loss=0.4478 | overall-dev F1=0.7431 | 363s | ETA 6.0min
  Epoch 15/15 | loss=0.4453 | overall-dev F1=0.7398 | 360s | ETA 0.0min

Best overall dev F1 (default threshold): 0.7461

=== Tuning per-aspect thresholds on dev set ===
  Threshold tuning per aspect:
    food                         thr=0.10  F1=0.9329
    service                      thr=0.80  F1=0.9091
    ambience                     thr=0.75  F1=0.8571
    price                        thr=0.85  F1=0.9391
    anecdotes/miscellaneous      thr=0.50  F1=0.8543

=== Final Dev Evaluation (tuned thresholds) ===
  Aspect     F1: 0.8961
  Sentiment  F1: 0.8337
  Overall    F1: 0.7471
Saved dev predictions → /Users/kafiling/Projects/NLP-Contest1/Resource/roberta_v2_dev_pred.csv

=== Predicting on Test Set ===
Saved 571 rows → /Users/kafiling/Projects/NLP-Contest1/Resource/roberta_v2_test_pred.csv
     id           aspectCategory  polarity
0   899                     food   neutral
1  1349  anecdotes/miscellaneous  positive
2   934                     food  negative
3  2199                     food  positive
4   188                     food  negative
5   188                  service  positive
6   188                 ambience  negative
7  1748                     food  positive
8  1748                  service  positive
9   949                  service  negative



# Evaluate dev
cd Resource
../.venv/bin/python evaluate.py contest1_train.csv roberta_v2_dev_pred.csv

# Validate test
../.venv/bin/python check_id.py contest1_test.csv roberta_v2_test_pred.csv
cd: no such file or directory: Resource
=== CLASSIFICATION : ASPECT ===
                class name  precision  recall  F1-score  support
0                     food      0.903   0.965     0.933      144
1                    price      0.947   0.931     0.939       58
2                  service      0.938   0.870     0.902       69
3                 ambience      0.938   0.789     0.857       57
4  anecdotes/miscellaneous      0.838   0.872     0.854      148
5                MACRO AVG      0.913   0.885     0.897      476
6                MICRO AVG      0.895   0.897     0.896      476 

=== CLASSIFICATION : SENTIMENT ===
  class name  precision  recall  F1-score  support
0   positive      0.925   0.861     0.892      230
1   negative      0.748   0.802     0.774       96
2    neutral      0.641   0.651     0.646       63
3   conflict      0.520   0.542     0.531       24
4  MACRO AVG      0.708   0.714     0.711      413
5  MICRO AVG      0.810   0.797     0.803      413 

=== CLASSIFICATION : OVERALL ===
              precision  recall  F1-score  support
0  MICRO AVG      0.746   0.748     0.747      476 

All good! Sentences in your predicted file are labeled!