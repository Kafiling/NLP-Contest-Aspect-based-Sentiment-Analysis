"""
Fine-tuned DeBERTa-v3-base Classifier for Aspect-Based Sentiment Analysis (ABSA)

Architecture:
  Model 1 — Aspect Classifier:
      Input : sentence
      Encode: [CLS] sentence [SEP]  →  DeBERTa-v3  →  [CLS] representation
      Head  : Linear(768 → 5)  +  BCEWithLogitsLoss  (multi-label)

  Model 2 — Sentiment Classifier:
      Input : sentence + aspect name pair
      Encode: [CLS] sentence [SEP] aspect_name [SEP]  →  DeBERTa-v3  →  [CLS]
      Head  : Linear(768 → 4)  +  CrossEntropyLoss

Note: DeBERTa-v3 does NOT use token_type_ids.

Prediction:
  For every test sentence:
    1. Run Aspect Classifier → aspects present (sigmoid threshold)
    2. For each aspect → run Sentiment Classifier → polarity
    3. Emit rows (id, aspectCategory, polarity)
"""

import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2Model, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
DEBERTA_MODEL   = 'microsoft/deberta-v3-base'
MAX_LEN         = 128
EPOCHS_ASPECT   = 10
EPOCHS_SENT     = 10
BATCH_SIZE      = 16
LR              = 2e-5
WARMUP_RATIO    = 0.1
DROPOUT         = 0.1
ASPECT_THRESH   = 0.4
DEV_SPLIT       = 0.15

ASPECTS    = ['food', 'service', 'ambience', 'price', 'anecdotes/miscellaneous']
SENTIMENTS = ['positive', 'negative', 'neutral', 'conflict']
ASPECT2ID  = {a: i for i, a in enumerate(ASPECTS)}
SENT2ID    = {s: i for i, s in enumerate(SENTIMENTS)}
ID2SENT    = {i: s for s, i in SENT2ID.items()}

DATA_DIR     = os.path.join(os.path.dirname(__file__), 'Resource')
TRAIN_FILE   = os.path.join(DATA_DIR, 'contest1_train.csv')
TEST_FILE    = os.path.join(DATA_DIR, 'contest1_test.csv')
OUT_FILE     = os.path.join(DATA_DIR, 'deberta_test_pred.csv')
DEV_OUT_FILE = os.path.join(DATA_DIR, 'deberta_dev_pred.csv')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── Datasets ─────────────────────────────────────────────────────────────────
class AspectDataset(Dataset):
    """
    One sample per unique sentence.
    Label = binary vector (multi-label) over 5 aspects.
    DeBERTa does not return token_type_ids — excluded here.
    """
    def __init__(self, df, tokenizer):
        grouped = df.groupby('id').agg(
            text=('text', 'first'),
            aspects=('aspectCategory', list)
        ).reset_index()

        enc = tokenizer(
            grouped['text'].tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors='pt'
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
    """
    One sample per (sentence, aspect) pair.
    Input formatted as: sentence [SEP] aspect_name
    """
    def __init__(self, df, tokenizer):
        enc = tokenizer(
            df['text'].tolist(),
            df['aspectCategory'].tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors='pt'
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


# ── DeBERTa Models ────────────────────────────────────────────────────────────
class DeBERTaAspect(nn.Module):
    """Multi-label aspect classifier using [CLS] from DeBERTa-v3."""
    def __init__(self, deberta, n_aspects, dropout):
        super().__init__()
        self.deberta = deberta
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(deberta.config.hidden_size, n_aspects)

    def forward(self, input_ids, attention_mask):
        out = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :].float())
        return self.fc(cls)


class DeBERTaSentiment(nn.Module):
    """Sentiment classifier conditioned on aspect via sentence-pair encoding."""
    def __init__(self, deberta, n_sentiments, dropout):
        super().__init__()
        self.deberta = deberta
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(deberta.config.hidden_size, n_sentiments)

    def forward(self, input_ids, attention_mask):
        out = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :].float())
        return self.fc(cls)


# ── Training helpers ──────────────────────────────────────────────────────────
def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch  = to_device(batch)
        labels = batch.pop('labels')
        optimizer.zero_grad()
        logits = model(**batch)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Evaluation helpers ────────────────────────────────────────────────────────
def evaluate_overall_f1(gold_df, pred_df):
    gold_set = set(zip(gold_df['id'], gold_df['aspectCategory'], gold_df['polarity']))
    pred_set = set(zip(pred_df['id'], pred_df['aspectCategory'], pred_df['polarity']))
    if not pred_set:
        return 0.0
    precision = len(gold_set & pred_set) / len(pred_set)
    recall    = len(gold_set & pred_set) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def aspect_dev_f1(asp_model, tokenizer, dev_df, threshold=ASPECT_THRESH):
    asp_model.eval()
    gold_asp   = set(zip(dev_df['id'], dev_df['aspectCategory']))
    pred_asp   = set()
    dev_unique = dev_df.drop_duplicates('id')
    with torch.no_grad():
        for _, row in dev_unique.iterrows():
            enc = tokenizer(
                row['text'], truncation=True, padding='max_length',
                max_length=MAX_LEN, return_tensors='pt'
            )
            input_ids      = enc['input_ids'].to(DEVICE)
            attention_mask = enc['attention_mask'].to(DEVICE)
            probs = torch.sigmoid(
                asp_model(input_ids=input_ids, attention_mask=attention_mask)
            ).cpu().numpy()[0]
            preds = [ASPECTS[i] for i, p in enumerate(probs) if p >= threshold]
            if not preds:
                preds = [ASPECTS[int(np.argmax(probs))]]
            for asp in preds:
                pred_asp.add((row['id'], asp))
    tp   = len(gold_asp & pred_asp)
    prec = tp / len(pred_asp) if pred_asp else 0
    rec  = tp / len(gold_asp) if gold_asp else 0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(asp_model, sent_model, tokenizer, texts, ids, threshold=ASPECT_THRESH):
    asp_model.eval()
    sent_model.eval()
    rows = []

    with torch.no_grad():
        for doc_id, text in zip(ids, texts):
            # ---- Aspect prediction ----
            enc = tokenizer(
                text, truncation=True, padding='max_length',
                max_length=MAX_LEN, return_tensors='pt'
            )
            input_ids      = enc['input_ids'].to(DEVICE)
            attention_mask = enc['attention_mask'].to(DEVICE)
            probs = torch.sigmoid(
                asp_model(input_ids=input_ids, attention_mask=attention_mask)
            ).cpu().numpy()[0]

            predicted_aspects = [ASPECTS[i] for i, p in enumerate(probs) if p >= threshold]
            if not predicted_aspects:
                predicted_aspects = [ASPECTS[int(np.argmax(probs))]]

            # ---- Sentiment prediction per aspect ----
            for asp in predicted_aspects:
                s_enc = tokenizer(
                    text, asp,
                    truncation=True, padding='max_length',
                    max_length=MAX_LEN, return_tensors='pt'
                )
                s_input_ids      = s_enc['input_ids'].to(DEVICE)
                s_attention_mask = s_enc['attention_mask'].to(DEVICE)
                s_logits  = sent_model(input_ids=s_input_ids, attention_mask=s_attention_mask)
                pred_sent = ID2SENT[int(s_logits.argmax(1).cpu())]
                rows.append({'id': doc_id, 'aspectCategory': asp, 'polarity': pred_sent})

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ---- Load data ----
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    print(f"Train rows: {len(train_df)} | Unique train IDs: {train_df['id'].nunique()}")
    print(f"Test  rows: {len(test_df)}")

    # ---- Train / dev split by unique ID ----
    unique_ids = train_df['id'].unique()
    train_ids, dev_ids = train_test_split(unique_ids, test_size=DEV_SPLIT, random_state=SEED)
    tr_df  = train_df[train_df['id'].isin(train_ids)].reset_index(drop=True)
    dev_df = train_df[train_df['id'].isin(dev_ids)].reset_index(drop=True)
    print(f"Split -> train: {len(tr_df)} rows | dev: {len(dev_df)} rows")

    # ---- Tokenizer ----
    print(f"\nLoading tokenizer: {DEBERTA_MODEL}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(DEBERTA_MODEL)

    # ── Aspect model ─────────────────────────────────────────────────────────
    print("\n=== Training DeBERTa-v3 Aspect Classifier ===")
    asp_ds_train = AspectDataset(tr_df,  tokenizer)
    asp_ds_dev   = AspectDataset(dev_df, tokenizer)
    asp_dl_train = DataLoader(asp_ds_train, BATCH_SIZE, shuffle=True)
    asp_dl_dev   = DataLoader(asp_ds_dev,  BATCH_SIZE, shuffle=False)

    asp_deberta = DebertaV2Model.from_pretrained(DEBERTA_MODEL)
    asp_model   = DeBERTaAspect(asp_deberta, len(ASPECTS), DROPOUT).to(DEVICE)
    asp_optim   = torch.optim.AdamW(asp_model.parameters(), lr=LR, weight_decay=0.01)
    total_asp_steps = len(asp_dl_train) * EPOCHS_ASPECT
    asp_sched = get_linear_schedule_with_warmup(
        asp_optim,
        num_warmup_steps=int(WARMUP_RATIO * total_asp_steps),
        num_training_steps=total_asp_steps
    )
    asp_crit = nn.BCEWithLogitsLoss()

    best_asp_state = None
    best_asp_f1    = -1.0
    asp_epoch_times = []

    for epoch in range(1, EPOCHS_ASPECT + 1):
        t0   = time.time()
        loss = train_epoch(asp_model, asp_dl_train, asp_optim, asp_sched, asp_crit)
        f1   = aspect_dev_f1(asp_model, tokenizer, dev_df)
        elapsed = time.time() - t0
        asp_epoch_times.append(elapsed)
        avg_t = sum(asp_epoch_times) / len(asp_epoch_times)
        eta   = avg_t * (EPOCHS_ASPECT - epoch)
        print(f"  Epoch {epoch:2d}/{EPOCHS_ASPECT} | loss={loss:.4f} | aspect-dev F1={f1:.4f} "
              f"| {elapsed:.0f}s/ep | ETA {eta/60:.1f}min")
        if f1 > best_asp_f1:
            best_asp_f1 = f1
            best_asp_state = {k: v.clone() for k, v in asp_model.state_dict().items()}

    asp_model.load_state_dict(best_asp_state)
    print(f"Best aspect dev F1: {best_asp_f1:.4f}")

    # ── Sentiment model ───────────────────────────────────────────────────────
    print("\n=== Training DeBERTa-v3 Sentiment Classifier ===")
    sent_ds_train = SentimentDataset(tr_df,  tokenizer)
    sent_ds_dev   = SentimentDataset(dev_df, tokenizer)
    sent_dl_train = DataLoader(sent_ds_train, BATCH_SIZE, shuffle=True)
    sent_dl_dev   = DataLoader(sent_ds_dev,  BATCH_SIZE, shuffle=False)

    sent_deberta = DebertaV2Model.from_pretrained(DEBERTA_MODEL)
    sent_model   = DeBERTaSentiment(sent_deberta, len(SENTIMENTS), DROPOUT).to(DEVICE)
    sent_optim   = torch.optim.AdamW(sent_model.parameters(), lr=LR, weight_decay=0.01)
    total_sent_steps = len(sent_dl_train) * EPOCHS_SENT
    sent_sched = get_linear_schedule_with_warmup(
        sent_optim,
        num_warmup_steps=int(WARMUP_RATIO * total_sent_steps),
        num_training_steps=total_sent_steps
    )
    sent_crit = nn.CrossEntropyLoss()

    best_sent_state = None
    best_overall_f1 = -1.0
    sent_epoch_times = []

    for epoch in range(1, EPOCHS_SENT + 1):
        t0   = time.time()
        loss = train_epoch(sent_model, sent_dl_train, sent_optim, sent_sched, sent_crit)
        dev_unique  = dev_df.drop_duplicates('id')
        dev_pred_df = predict(asp_model, sent_model, tokenizer,
                              dev_unique['text'].tolist(),
                              dev_unique['id'].tolist())
        f1 = evaluate_overall_f1(dev_df, dev_pred_df)
        elapsed = time.time() - t0
        sent_epoch_times.append(elapsed)
        avg_t = sum(sent_epoch_times) / len(sent_epoch_times)
        eta   = avg_t * (EPOCHS_SENT - epoch)
        print(f"  Epoch {epoch:2d}/{EPOCHS_SENT} | loss={loss:.4f} | overall-dev F1={f1:.4f} "
              f"| {elapsed:.0f}s/ep | ETA {eta/60:.1f}min")
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_sent_state = {k: v.clone() for k, v in sent_model.state_dict().items()}

    sent_model.load_state_dict(best_sent_state)
    print(f"Best overall dev F1: {best_overall_f1:.4f}")

    # ── Final dev evaluation ──────────────────────────────────────────────────
    print("\n=== Final Dev Evaluation ===")
    dev_unique     = dev_df.drop_duplicates('id')
    dev_pred_final = predict(asp_model, sent_model, tokenizer,
                             dev_unique['text'].tolist(),
                             dev_unique['id'].tolist())
    final_f1 = evaluate_overall_f1(dev_df, dev_pred_final)
    print(f"Dev Overall F1: {final_f1:.4f}")
    dev_pred_final.to_csv(DEV_OUT_FILE, index=False)
    print(f"Saved dev predictions to {DEV_OUT_FILE}")

    # ── Predict on test set ───────────────────────────────────────────────────
    print("\n=== Predicting on Test Set ===")
    test_pred = predict(asp_model, sent_model, tokenizer,
                        test_df['text'].tolist(),
                        test_df['id'].tolist())
    test_pred.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(test_pred)} rows to {OUT_FILE}")
    print(test_pred.head(10))


if __name__ == '__main__':
    main()
