"""
RoBERTa v2 — Joint Training + Class Weights + Per-class Threshold Tuning
                + Data Augmentation + Label Smoothing

Improvements over v1:
  1. Joint training      — single shared RoBERTa encoder with two heads trained
                           simultaneously (aspect + sentiment loss combined each step)
  2. Class weights       — inverse-frequency weights on CrossEntropyLoss to fix
                           under-represented sentiments (esp. 'conflict')
  3. Per-class threshold — grid search over [0.10, 0.95] per aspect on dev set
                           to maximise per-aspect F1 independently
  4. Data augmentation   — oversample minority sentiment classes in training set
                           before building the sentiment DataLoader
  5. Label smoothing     — CrossEntropyLoss(label_smoothing=0.1) on sentiment head
"""

import os
import time
import random
from itertools import cycle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
ROBERTA_MODEL     = 'roberta-base'
MAX_LEN           = 128
EPOCHS            = 15        # joint training epochs
BATCH_SIZE        = 16
LR                = 2e-5
WARMUP_RATIO      = 0.06
WEIGHT_DECAY      = 0.01
DROPOUT           = 0.1
DEV_SPLIT         = 0.15
ASP_LOSS_WEIGHT   = 1.0       # λ for aspect loss in joint objective
SENT_LOSS_WEIGHT  = 1.0       # λ for sentiment loss in joint objective
OVERSAMPLE_FACTOR = 3         # max copies added per minority class

ASPECTS    = ['food', 'service', 'ambience', 'price', 'anecdotes/miscellaneous']
SENTIMENTS = ['positive', 'negative', 'neutral', 'conflict']
SENT2ID    = {s: i for i, s in enumerate(SENTIMENTS)}
ID2SENT    = {i: s for s, i in SENT2ID.items()}

DATA_DIR     = os.path.join(os.path.dirname(__file__), 'Resource')
TRAIN_FILE   = os.path.join(DATA_DIR, 'contest1_train.csv')
TEST_FILE    = os.path.join(DATA_DIR, 'contest1_test.csv')
OUT_FILE     = os.path.join(DATA_DIR, 'roberta_v2_test_pred.csv')
DEV_OUT_FILE = os.path.join(DATA_DIR, 'roberta_v2_dev_pred.csv')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── Improvement 4: Data augmentation (oversample minority sentiments) ─────────
def augment_sentiment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate rows whose polarity class appears fewer times than the majority
    class, up to OVERSAMPLE_FACTOR extra copies.  Only the sentiment training
    split is augmented (aspect dataset is unaffected).
    """
    counts   = df['polarity'].value_counts()
    majority = counts.max()
    parts    = [df]
    for sent in SENTIMENTS:
        cnt = counts.get(sent, 0)
        if cnt == 0:
            continue
        avg_cnt = len(df) / len(SENTIMENTS)
        if cnt < avg_cnt:
            minority_df = df[df['polarity'] == sent]
            n_copies    = min(OVERSAMPLE_FACTOR - 1,
                              max(1, int(majority / cnt) - 1))
            for _ in range(n_copies):
                parts.append(minority_df)
    augmented = pd.concat(parts, ignore_index=True)
    return augmented.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ── Datasets ──────────────────────────────────────────────────────────────────
class AspectDataset(Dataset):
    """One sample per unique sentence; multi-label aspect vector."""
    def __init__(self, df: pd.DataFrame, tokenizer):
        grouped = df.groupby('id').agg(
            text=('text', 'first'),
            aspects=('aspectCategory', list)
        ).reset_index()

        enc = tokenizer(
            grouped['text'].tolist(),
            truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors='pt'
        )
        self.input_ids      = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        self.labels = torch.tensor(
            [[1.0 if a in aspects else 0.0 for a in ASPECTS]
             for aspects in grouped['aspects']],
            dtype=torch.float
        )

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels':         self.labels[idx],
        }


class SentimentDataset(Dataset):
    """One sample per (sentence, aspect) pair; sentence-pair encoding."""
    def __init__(self, df: pd.DataFrame, tokenizer):
        enc = tokenizer(
            df['text'].tolist(),
            df['aspectCategory'].tolist(),
            truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors='pt'
        )
        self.input_ids      = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        self.labels = torch.tensor(
            df['polarity'].map(SENT2ID).tolist(), dtype=torch.long
        )

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels':         self.labels[idx],
        }


# ── Improvement 1: Joint model — shared encoder, two heads ───────────────────
class JointRoBERTa(nn.Module):
    """
    Single RoBERTa backbone shared between aspect and sentiment tasks.
    forward_aspect   : single-sentence input  → aspect logits (5)
    forward_sentiment: sentence-pair input    → sentiment logits (4)
    """
    def __init__(self, roberta, n_aspects: int, n_sentiments: int, dropout: float):
        super().__init__()
        self.roberta   = roberta
        self.dropout   = nn.Dropout(dropout)
        self.asp_head  = nn.Linear(roberta.config.hidden_size, n_aspects)
        self.sent_head = nn.Linear(roberta.config.hidden_size, n_sentiments)

    def _encode(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(out.last_hidden_state[:, 0, :])

    def forward_aspect(self, input_ids, attention_mask):
        return self.asp_head(self._encode(input_ids, attention_mask))

    def forward_sentiment(self, input_ids, attention_mask):
        return self.sent_head(self._encode(input_ids, attention_mask))


# ── Helpers ───────────────────────────────────────────────────────────────────
def to_device(batch: dict) -> dict:
    return {k: v.to(DEVICE) for k, v in batch.items()}


# ── Improvement 1: Joint training step ────────────────────────────────────────
def train_epoch_joint(model, asp_loader, sent_loader,
                      optimizer, scheduler, asp_crit, sent_crit):
    """
    Each step: one aspect mini-batch + one sentiment mini-batch.
    Shorter loader is cycled so both contribute every step.
    """
    model.train()
    total_loss = 0.0
    n_steps    = max(len(asp_loader), len(sent_loader))

    if len(asp_loader) >= len(sent_loader):
        asp_iter  = iter(asp_loader)
        sent_iter = cycle(sent_loader)
    else:
        asp_iter  = cycle(asp_loader)
        sent_iter = iter(sent_loader)

    for _ in range(n_steps):
        asp_batch  = to_device(next(asp_iter))
        sent_batch = to_device(next(sent_iter))

        asp_labels  = asp_batch.pop('labels')
        sent_labels = sent_batch.pop('labels')

        optimizer.zero_grad()
        asp_logits  = model.forward_aspect(**asp_batch)
        sent_logits = model.forward_sentiment(**sent_batch)

        asp_loss  = asp_crit(asp_logits, asp_labels)
        sent_loss = sent_crit(sent_logits, sent_labels)
        loss      = ASP_LOSS_WEIGHT * asp_loss + SENT_LOSS_WEIGHT * sent_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / n_steps


# ── Improvement 3: Per-class threshold tuning ─────────────────────────────────
def tune_thresholds(model, tokenizer, dev_df,
                    grid=np.arange(0.10, 0.96, 0.05)) -> list:
    """
    For each aspect independently, find the sigmoid threshold on dev set that
    maximises per-aspect F1, then return the list of 5 thresholds.
    """
    model.eval()
    dev_unique = dev_df.drop_duplicates('id')
    gold_asp   = set(zip(dev_df['id'], dev_df['aspectCategory']))

    # Collect probabilities once
    all_ids   = dev_unique['id'].tolist()
    all_probs = []
    with torch.no_grad():
        for _, row in dev_unique.iterrows():
            enc = tokenizer(
                row['text'], truncation=True, padding='max_length',
                max_length=MAX_LEN, return_tensors='pt'
            )
            p = torch.sigmoid(
                model.forward_aspect(
                    input_ids=enc['input_ids'].to(DEVICE),
                    attention_mask=enc['attention_mask'].to(DEVICE)
                )
            ).cpu().numpy()[0]
            all_probs.append(p)
    all_probs = np.array(all_probs)          # (n, 5)

    best_thresholds = []
    print("  Threshold tuning per aspect:")
    for asp_idx, asp_name in enumerate(ASPECTS):
        gold_specific = {(did, a) for did, a in gold_asp if a == asp_name}
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred_specific = {
                (did, asp_name)
                for did, probs in zip(all_ids, all_probs)
                if probs[asp_idx] >= t
            }
            tp   = len(pred_specific & gold_specific)
            prec = tp / len(pred_specific) if pred_specific else 0.0
            rec  = tp / len(gold_specific) if gold_specific  else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds.append(float(best_t))
        print(f"    {asp_name:<28} thr={best_t:.2f}  F1={best_f1:.4f}")
    return best_thresholds


# ── Evaluation helpers ────────────────────────────────────────────────────────
def evaluate_overall_f1(gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    gold_set = set(zip(gold_df['id'], gold_df['aspectCategory'], gold_df['polarity']))
    pred_set = set(zip(pred_df['id'], pred_df['aspectCategory'], pred_df['polarity']))
    if not pred_set:
        return 0.0
    prec = len(gold_set & pred_set) / len(pred_set)
    rec  = len(gold_set & pred_set) / len(gold_set)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def aspect_dev_f1(model, tokenizer, dev_df, thresholds: list) -> float:
    model.eval()
    gold_asp   = set(zip(dev_df['id'], dev_df['aspectCategory']))
    pred_asp   = set()
    dev_unique = dev_df.drop_duplicates('id')
    with torch.no_grad():
        for _, row in dev_unique.iterrows():
            enc = tokenizer(
                row['text'], truncation=True, padding='max_length',
                max_length=MAX_LEN, return_tensors='pt'
            )
            probs = torch.sigmoid(
                model.forward_aspect(
                    input_ids=enc['input_ids'].to(DEVICE),
                    attention_mask=enc['attention_mask'].to(DEVICE)
                )
            ).cpu().numpy()[0]
            preds = [ASPECTS[i] for i, p in enumerate(probs) if p >= thresholds[i]]
            if not preds:
                preds = [ASPECTS[int(np.argmax(probs))]]
            for asp in preds:
                pred_asp.add((row['id'], asp))
    tp   = len(gold_asp & pred_asp)
    prec = tp / len(pred_asp) if pred_asp else 0.0
    rec  = tp / len(gold_asp) if gold_asp  else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, tokenizer, texts, ids, thresholds: list) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for doc_id, text in zip(ids, texts):
            # ---- Aspect ----
            enc = tokenizer(
                text, truncation=True, padding='max_length',
                max_length=MAX_LEN, return_tensors='pt'
            )
            probs = torch.sigmoid(
                model.forward_aspect(
                    input_ids=enc['input_ids'].to(DEVICE),
                    attention_mask=enc['attention_mask'].to(DEVICE)
                )
            ).cpu().numpy()[0]

            pred_aspects = [ASPECTS[i] for i, p in enumerate(probs)
                            if p >= thresholds[i]]
            if not pred_aspects:
                pred_aspects = [ASPECTS[int(np.argmax(probs))]]

            # ---- Sentiment per aspect ----
            for asp in pred_aspects:
                s_enc = tokenizer(
                    text, asp, truncation=True, padding='max_length',
                    max_length=MAX_LEN, return_tensors='pt'
                )
                logits = model.forward_sentiment(
                    input_ids=s_enc['input_ids'].to(DEVICE),
                    attention_mask=s_enc['attention_mask'].to(DEVICE)
                )
                pred_sent = ID2SENT[int(logits.argmax(1).cpu())]
                rows.append({'id': doc_id, 'aspectCategory': asp, 'polarity': pred_sent})

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ---- Load data ----
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    print(f"Train rows: {len(train_df)} | Unique IDs: {train_df['id'].nunique()}")
    print(f"Test  rows: {len(test_df)}")

    # ---- Train / dev split by unique sentence ID ----
    unique_ids = train_df['id'].unique()
    train_ids, dev_ids = train_test_split(unique_ids, test_size=DEV_SPLIT,
                                          random_state=SEED)
    tr_df  = train_df[train_df['id'].isin(train_ids)].reset_index(drop=True)
    dev_df = train_df[train_df['id'].isin(dev_ids)].reset_index(drop=True)
    print(f"Split -> train: {len(tr_df)} rows | dev: {len(dev_df)} rows")

    # ---- Improvement 4: Augment sentiment training data ----
    tr_sent_aug = augment_sentiment_data(tr_df)
    print(f"Sentiment training rows after augmentation: {len(tr_sent_aug)}")
    print("  Polarity distribution after augmentation:")
    print(tr_sent_aug['polarity'].value_counts().to_string(header=False))

    # ---- Tokenizer ----
    print(f"\nLoading tokenizer: {ROBERTA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)

    # ---- Datasets / DataLoaders ----
    asp_ds_train  = AspectDataset(tr_df, tokenizer)
    sent_ds_train = SentimentDataset(tr_sent_aug, tokenizer)   # augmented

    asp_dl_train  = DataLoader(asp_ds_train,  BATCH_SIZE, shuffle=True)
    sent_dl_train = DataLoader(sent_ds_train, BATCH_SIZE, shuffle=True)

    # ---- Improvement 2: Class weights for sentiment ----
    counts = tr_df['polarity'].value_counts()
    total  = len(tr_df)
    class_weights = torch.tensor(
        [total / (len(SENTIMENTS) * counts.get(s, 1)) for s in SENTIMENTS],
        dtype=torch.float
    ).to(DEVICE)
    print(f"\nSentiment class weights: "
          + " | ".join(f"{s}={w:.2f}" for s, w in zip(SENTIMENTS, class_weights.tolist())))

    # ---- Build joint model ----
    print(f"\nLoading RoBERTa backbone: {ROBERTA_MODEL}")
    roberta    = AutoModel.from_pretrained(ROBERTA_MODEL)
    model      = JointRoBERTa(roberta, len(ASPECTS), len(SENTIMENTS), DROPOUT).to(DEVICE)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)

    n_steps_per_epoch = max(len(asp_dl_train), len(sent_dl_train))
    total_steps       = n_steps_per_epoch * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )

    # ---- Improvement 2+5: Loss functions ----
    asp_crit  = nn.BCEWithLogitsLoss()
    sent_crit = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1              # Improvement 5
    )

    # ── Joint training loop ───────────────────────────────────────────────────
    print("\n=== Joint Training (shared encoder) ===")
    #  Use a fixed default threshold for dev monitoring during training;
    #  final thresholds are tuned after training completes.
    default_thresholds = [0.4] * len(ASPECTS)
    best_f1     = -1.0
    best_state  = None
    epoch_times = []

    for epoch in range(1, EPOCHS + 1):
        t0   = time.time()
        loss = train_epoch_joint(model, asp_dl_train, sent_dl_train,
                                 optimizer, scheduler, asp_crit, sent_crit)
        # Quick dev evaluation
        dev_unique  = dev_df.drop_duplicates('id')
        dev_pred_df = predict(model, tokenizer,
                              dev_unique['text'].tolist(),
                              dev_unique['id'].tolist(),
                              default_thresholds)
        f1      = evaluate_overall_f1(dev_df, dev_pred_df)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        eta = sum(epoch_times) / len(epoch_times) * (EPOCHS - epoch)
        print(f"  Epoch {epoch:2d}/{EPOCHS} | loss={loss:.4f} | "
              f"overall-dev F1={f1:.4f} | {elapsed:.0f}s | ETA {eta/60:.1f}min")
        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"\nBest overall dev F1 (default threshold): {best_f1:.4f}")

    # ── Improvement 3: Per-class threshold tuning ─────────────────────────────
    print("\n=== Tuning per-aspect thresholds on dev set ===")
    thresholds = tune_thresholds(model, tokenizer, dev_df)

    # ── Final dev evaluation with tuned thresholds ────────────────────────────
    print("\n=== Final Dev Evaluation (tuned thresholds) ===")
    dev_unique     = dev_df.drop_duplicates('id')
    dev_pred_final = predict(model, tokenizer,
                             dev_unique['text'].tolist(),
                             dev_unique['id'].tolist(),
                             thresholds)
    final_f1 = evaluate_overall_f1(dev_df, dev_pred_final)
    asp_f1   = aspect_dev_f1(model, tokenizer, dev_df, thresholds)

    # Sentiment F1 (on gold-aspect rows only)
    gold_asp_in_dev = dev_df[['id', 'aspectCategory']].drop_duplicates()
    merged = gold_asp_in_dev.merge(dev_pred_final, on=['id', 'aspectCategory'], how='inner')
    gold_merged = dev_df.merge(
        merged[['id', 'aspectCategory', 'polarity']],
        on=['id', 'aspectCategory']
    )
    if len(gold_merged):
        from sklearn.metrics import f1_score
        sent_f1 = f1_score(
            dev_df.set_index(['id', 'aspectCategory'])['polarity'].reindex(
                pd.MultiIndex.from_frame(merged[['id', 'aspectCategory']])
            ).values,
            merged['polarity'].values,
            average='micro', zero_division=0,
            labels=SENTIMENTS
        )
    else:
        sent_f1 = 0.0

    print(f"  Aspect     F1: {asp_f1:.4f}")
    print(f"  Sentiment  F1: {sent_f1:.4f}")
    print(f"  Overall    F1: {final_f1:.4f}")
    dev_pred_final.to_csv(DEV_OUT_FILE, index=False)
    print(f"Saved dev predictions → {DEV_OUT_FILE}")

    # ── Test set prediction ───────────────────────────────────────────────────
    print("\n=== Predicting on Test Set ===")
    test_pred = predict(model, tokenizer,
                        test_df['text'].tolist(),
                        test_df['id'].tolist(),
                        thresholds)
    test_pred.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(test_pred)} rows → {OUT_FILE}")
    print(test_pred.head(10))


if __name__ == '__main__':
    main()
