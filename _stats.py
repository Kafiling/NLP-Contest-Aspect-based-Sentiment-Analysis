import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DEV_SPLIT = 0.15
ASPECTS = ['food', 'service', 'ambience', 'price', 'anecdotes/miscellaneous']
SENTIMENTS = ['positive', 'negative', 'neutral', 'conflict']

train_df = pd.read_csv('Resource/contest1_train.csv')
test_df  = pd.read_csv('Resource/contest1_test.csv')

uids = train_df['id'].unique()
tids, dids = train_test_split(uids, test_size=DEV_SPLIT, random_state=SEED)
tr = train_df[train_df['id'].isin(tids)]
dv = train_df[train_df['id'].isin(dids)]

has_pol = 'polarity' in test_df.columns
has_asp = 'aspectCategory' in test_df.columns

print("SENTIMENT | train | dev | test")
for s in SENTIMENTS:
    tst = test_df['polarity'].eq(s).sum() if has_pol else 'N/A'
    print(f"  {s:<12} {tr['polarity'].eq(s).sum():>5}  {dv['polarity'].eq(s).sum():>4}  {tst}")

print("\nASPECT | train | dev | test")
for a in ASPECTS:
    tst = test_df['aspectCategory'].eq(a).sum() if has_asp else 'N/A'
    print(f"  {a:<28} {tr['aspectCategory'].eq(a).sum():>5}  {dv['aspectCategory'].eq(a).sum():>4}  {tst}")

print(f"\nTOTAL rows  {len(tr):>5}  {len(dv):>4}  {len(test_df)}")
print(f"TOTAL ids   {tr['id'].nunique():>5}  {dv['id'].nunique():>4}  {test_df['id'].nunique()}")
