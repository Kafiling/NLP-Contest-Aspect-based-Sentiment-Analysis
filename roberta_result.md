# Evaluate dev
cd Resource
../.venv/bin/python evaluate.py contest1_train.csv roberta_dev_pred.csv

# Validate test
../.venv/bin/python check_id.py contest1_test.csv roberta_test_pred.csv
cd: no such file or directory: Resource
=== CLASSIFICATION : ASPECT ===
                class name  precision  recall  F1-score  support
0                     food      0.898   0.979     0.937      144
1                    price      0.902   0.948     0.924       58
2                  service      0.899   0.899     0.899       69
3                 ambience      0.785   0.895     0.836       57
4  anecdotes/miscellaneous      0.891   0.831     0.860      148
5                MACRO AVG      0.875   0.910     0.891      476
6                MICRO AVG      0.882   0.908     0.894      476 

=== CLASSIFICATION : SENTIMENT ===
  class name  precision  recall  F1-score  support
0   positive      0.893   0.909     0.901      230
1   negative      0.765   0.781     0.773       96
2    neutral      0.645   0.635     0.640       63
3   conflict      0.389   0.292     0.333       24
4  MACRO AVG      0.673   0.654     0.662      413
5  MICRO AVG      0.803   0.801     0.802      413 

=== CLASSIFICATION : OVERALL ===
              precision  recall  F1-score  support
0  MICRO AVG      0.724   0.746     0.735      476 

All good! Sentences in your predicted file are labeled!