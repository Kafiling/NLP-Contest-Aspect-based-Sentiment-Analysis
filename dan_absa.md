../.venv/bin/python evaluate.py contest1_train.csv dev_pred.csv
=== CLASSIFICATION : ASPECT ===
                class name  precision  recall  F1-score  support
0                     food      0.842   0.812     0.827      144
1                    price      0.829   0.586     0.687       58
2                  service      0.714   0.797     0.753       69
3                 ambience      0.643   0.474     0.545       57
4  anecdotes/miscellaneous      0.733   0.818     0.773      148
5                MACRO AVG      0.752   0.697     0.717      476
6                MICRO AVG      0.763   0.744     0.753      476 

=== CLASSIFICATION : SENTIMENT ===
  class name  precision  recall  F1-score  support
0   positive      0.708   0.778     0.741      230
1   negative      0.474   0.573     0.519       96
2    neutral      0.410   0.254     0.314       63
3   conflict      0.000   0.000     0.000       24
4  MACRO AVG      0.398   0.401     0.393      413
5  MICRO AVG      0.613   0.605     0.609      413 

=== CLASSIFICATION : OVERALL ===
              precision  recall  F1-score  support
0  MICRO AVG      0.494   0.481     0.487      476 


# Word2Vec
../.venv/bin/python evaluate.py contest1_train.csv w2v_dev_pred.csv


=== CLASSIFICATION : ASPECT ===
                class name  precision  recall  F1-score  support
0                     food      0.743   0.924     0.824      144
1                    price      0.902   0.638     0.747       58
2                  service      0.765   0.754     0.759       69
3                 ambience      0.500   0.474     0.486       57
4  anecdotes/miscellaneous      0.748   0.804     0.775      148
5                MACRO AVG      0.732   0.719     0.718      476
6                MICRO AVG      0.735   0.773     0.753      476 

=== CLASSIFICATION : SENTIMENT ===
  class name  precision  recall  F1-score  support
0   positive      0.763   0.757     0.760      230
1   negative      0.582   0.594     0.588       96
2    neutral      0.408   0.492     0.446       63
3   conflict      0.250   0.208     0.227       24
4  MACRO AVG      0.501   0.513     0.505      413
5  MICRO AVG      0.633   0.646     0.640      413 

=== CLASSIFICATION : OVERALL ===
              precision  recall  F1-score  support
0  MICRO AVG      0.489   0.515     0.502      476 

# GloVe
../.venv/bin/python evaluate.py contest1_train.csv glove_dev_pred.csv
=== CLASSIFICATION : ASPECT ===
                class name  precision  recall  F1-score  support
0                     food      0.821   0.889     0.853      144
1                    price      0.854   0.707     0.774       58
2                  service      0.779   0.870     0.822       69
3                 ambience      0.655   0.632     0.643       57
4  anecdotes/miscellaneous      0.829   0.818     0.823      148
5                MACRO AVG      0.787   0.783     0.783      476
6                MICRO AVG      0.801   0.811     0.806      476 

=== CLASSIFICATION : SENTIMENT ===
  class name  precision  recall  F1-score  support
0   positive      0.765   0.791     0.778      230
1   negative      0.565   0.542     0.553       96
2    neutral      0.457   0.508     0.481       63
3   conflict      0.000   0.000     0.000       24
4  MACRO AVG      0.447   0.460     0.453      413
5  MICRO AVG      0.660   0.644     0.652      413 

=== CLASSIFICATION : OVERALL ===
              precision  recall  F1-score  support
0  MICRO AVG      0.539   0.546     0.543      476 