=== Training BERT Sentiment Classifier ===
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████| 199/199 [00:00<00:00, 8060.59it/s]
BertModel LOAD REPORT from: bert-base-uncased
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Epoch  1 | loss=0.9755 | overall-dev F1=0.6528
  Epoch  2 | loss=0.5958 | overall-dev F1=0.6653
  Epoch  3 | loss=0.3782 | overall-dev F1=0.6839
  Epoch  4 | loss=0.2434 | overall-dev F1=0.6715
  Epoch  5 | loss=0.1493 | overall-dev F1=0.6839
  Epoch  6 | loss=0.1002 | overall-dev F1=0.7005
  Epoch  7 | loss=0.0676 | overall-dev F1=0.7067
  Epoch  8 | loss=0.0443 | overall-dev F1=0.7109
  Epoch  9 | loss=0.0241 | overall-dev F1=0.7047
  Epoch 10 | loss=0.0135 | overall-dev F1=0.7047
Best overall dev F1: 0.7109

=== Final Dev Evaluation ===
Dev Overall F1: 0.7109
Saved dev predictions to /Users/kafiling/Projects/NLP-Contest1/Resource/bert_dev_pred.csv

=== Predicting on Test Set ===
Saved 582 rows to /Users/kafiling/Projects/NLP-Contest1/Resource/bert_test_pred.csv
     id           aspectCategory  polarity
0   899                     food   neutral
1  1349                 ambience  positive
2  1349  anecdotes/miscellaneous  positive
3   934                     food  negative
4  2199                     food  positive
5   188                     food  negative
6   188                  service  negative
7   188                 ambience  negative
8  1748                     food  positive
9  1748                  service  positive
zsh: command not found: #
=== CLASSIFICATION : ASPECT ===
                class name  precision  recall  F1-score  support
0                     food      0.887   0.931     0.908      144
1                    price      0.844   0.931     0.885       58
2                  service      0.859   0.884     0.871       69
3                 ambience      0.793   0.807     0.800       57
4  anecdotes/miscellaneous      0.848   0.831     0.840      148
5                MACRO AVG      0.846   0.877     0.861      476
6                MICRO AVG      0.855   0.878     0.866      476 

=== CLASSIFICATION : SENTIMENT ===
  class name  precision  recall  F1-score  support
0   positive      0.874   0.878     0.876      230
1   negative      0.737   0.760     0.749       96
2    neutral      0.621   0.571     0.595       63
3   conflict      0.435   0.417     0.426       24
4  MACRO AVG      0.667   0.657     0.661      413
5  MICRO AVG      0.781   0.777     0.779      413 

=== CLASSIFICATION : OVERALL ===
              precision  recall  F1-score  support
0  MICRO AVG      0.701   0.721     0.711      476 