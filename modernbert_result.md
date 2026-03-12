 ../.venv/bin/python evaluate.py contest1_train.csv modernbert_dev_pred.csv
=== CLASSIFICATION : ASPECT ===
                class name  precision  recall  F1-score  support
0                     food      0.872   0.944     0.907      144
1                    price      0.898   0.914     0.906       58
2                  service      0.900   0.913     0.906       69
3                 ambience      0.811   0.754     0.782       57
4  anecdotes/miscellaneous      0.849   0.838     0.844      148
5                MACRO AVG      0.866   0.873     0.869      476
6                MICRO AVG      0.866   0.880     0.873      476 

=== CLASSIFICATION : SENTIMENT ===
  class name  precision  recall  F1-score  support
0   positive      0.869   0.891     0.880      230
1   negative      0.780   0.740     0.759       96
2    neutral      0.562   0.571     0.567       63
3   conflict      0.346   0.375     0.360       24
4  MACRO AVG      0.639   0.644     0.642      413
5  MICRO AVG      0.770   0.777     0.773      413 

=== CLASSIFICATION : OVERALL ===
              precision  recall  F1-score  support
0  MICRO AVG      0.700   0.712     0.706      476 