"""
Fine-tuned BERT Classifier for Aspect-Based Sentiment Analysis (ABSA)

Architecture:
  Model 1 — Aspect Classifier:
      Input : sentence
      Encode: [CLS] sentence [SEP]  →  BERT  →  [CLS] representation
      Head  : Linear(768 → 5)  +  BCEWithLogitsLoss  (multi-label)

  Model 2 — Sentiment Classifier:
      Input : sentence + aspect name pair
      Encode: [CLS] sentence [SEP] aspect_name [SEP]  →  BERT  →  [CLS]
      Head  : Linear(768 → 4)  +  CrossEntropyLoss

Prediction:
  For every test sentence:
    1. Run Aspect Classifier → aspects present (sigmoid threshold)
    2. For each aspect → run Sentiment Classifier → polarity
    3. Emit rows (id, aspectCategory, polarity)
"""

import os
import re
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
BERT_MODEL      = 'bert-base-uncased'
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
OUT_FILE     = os.path.join(DATA_DIR, 'bert_test_pred.csv')
DEV_OUT_FILE = os.path.join(DATA_DIR, 'bert_dev_pred.csv')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── Datasets ─────────────────────────────────────────────────────────────────
class AspectDataset(Dataset):
    """
    One sample per unique sentence.
    Label = binary vector (multi-label) over 5 aspects.
    """
    def __init__(self, df, tokenizer):
        grouped = df.groupby('id').agg(
            text=('text', 'first'),
            aspects=('aspectCategory', list)
        ).reset_index()

        self.encodings = tokenizer(
            grouped['text'].tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        self.labels = []
        for aspects in grouped['aspects']:
            vec = [1.0 if a in aspects else 0.0 for a in ASPECTS]
            self.labels.append(vec)
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'labels':         self.labels[idx],
        }


class SentimentDataset(Dataset):
    """
    One sample per (sentence, aspect) pair.
    Input formatted as: sentence [SEP] aspect_name
    """
    def __init__(self, df, tokenizer):
        sentences = df['text'].tolist()
        aspects   = df['aspectCategory'].tolist()

        self.encodings = tokenizer(
            sentences,
            aspects,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        self.labels = torch.tensor(
            df['polarity'].map(SENT2ID).tolist(), dtype=torch.long
        )

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'labels':         self.labels[idx],
        }


# ── BERT Models ───────────────────────────────────────────────────────────────
class BERTAspect(nn.Module):
    """Multi-label aspect classifier using [CLS] from BERT."""
    def __init__(self, bert, n_aspects, dropout):
        super().__init__()
        self.bert    = bert
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(bert.config.hidden_size, n_aspects)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.fc(cls)


class BERTSentiment(nn.Module):
    """Sentiment classifier conditioned on aspect via sentence-pair encoding."""
    def __init__(self, bert, n_sentiments, dropout):
        super().__init__()
        self.bert    = bert
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(bert.config.hidden_size, n_sentiments)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.fc(cls)


# ── Training helpers ──────────────────────────────────────────────────────────
def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def train_epoch(model, loader, optimizer, scheduler, criterion, is_multilabel=False):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = to_device(batch)
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


# ── Evaluation helper ─────────────────────────────────────────────────────────
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


# ── Prediction ─────────────────────────────────────────────────────────────── 
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
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = asp_model(**enc)                            # (1, n_aspects)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]     # (n_aspects,)

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
                s_enc = {k: v.to(DEVICE) for k, v in s_enc.items()}
                s_logits   = sent_model(**s_enc)                 # (1, n_sentiments)
                pred_sent  = ID2SENT[int(s_logits.argmax(1).cpu())]
                rows.append({'id': doc_id, 'aspectCategory': asp, 'polarity': pred_sent})

    return pd.DataFrame(rows)


# ── Dev aspect-only F1 ────────────────────────────────────────────────────────
def aspect_dev_f1(asp_model, tokenizer, dev_df, threshold=ASPECT_THRESH):
    asp_model.eval()
    gold_asp = set(zip(dev_df['id'], dev_df['aspectCategory']))
    pred_asp = set()
    dev_unique = dev_df.drop_duplicates('id')
    with torch.no_grad():
        for _, row in dev_unique.iterrows():
            enc = tokenizer(
                row['text'], truncation=True, padding='max_length',
                max_length=MAX_LEN, return_tensors='pt'
            )
            enc   = {k: v.to(DEVICE) for k, v in enc.items()}
            probs = torch.sigmoid(asp_model(**enc)).cpu().numpy()[0]
            preds = [ASPECTS[i] for i, p in enumerate(probs) if p >= threshold]
            if not preds:
                preds = [ASPECTS[int(np.argmax(probs))]]
            for asp in preds:
                pred_asp.add((row['id'], asp))
    tp   = len(gold_asp & pred_asp)
    prec = tp / len(pred_asp) if pred_asp else 0
    rec  = tp / len(gold_asp) if gold_asp else 0
    return 2*prec*rec/(prec+rec) if prec+rec else 0


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
    print(f"\nLoading tokenizer: {BERT_MODEL}")
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

    # ── Aspect model ─────────────────────────────────────────────────────────
    print("\n=== Training BERT Aspect Classifier ===")
    asp_ds_train = AspectDataset(tr_df,  tokenizer)
    asp_ds_dev   = AspectDataset(dev_df, tokenizer)
    asp_dl_train = DataLoader(asp_ds_train, BATCH_SIZE, shuffle=True)
    asp_dl_dev   = DataLoader(asp_ds_dev,  BATCH_SIZE, shuffle=False)

    asp_bert  = BertModel.from_pretrained(BERT_MODEL)
    asp_model = BERTAspect(asp_bert, len(ASPECTS), DROPOUT).to(DEVICE)
    asp_optim = torch.optim.AdamW(asp_model.parameters(), lr=LR, weight_decay=0.01)
    total_asp_steps = len(asp_dl_train) * EPOCHS_ASPECT
    asp_sched = get_linear_schedule_with_warmup(
        asp_optim,
        num_warmup_steps=int(WARMUP_RATIO * total_asp_steps),
        num_training_steps=total_asp_steps
    )
    asp_crit  = nn.BCEWithLogitsLoss()

    best_asp_state = None
    best_asp_f1    = -1.0

    for epoch in range(1, EPOCHS_ASPECT + 1):
        loss = train_epoch(asp_model, asp_dl_train, asp_optim, asp_sched, asp_crit)
        f1   = aspect_dev_f1(asp_model, tokenizer, dev_df)
        print(f"  Epoch {epoch:2d} | loss={loss:.4f} | aspect-dev F1={f1:.4f}")
        if f1 > best_asp_f1:
            best_asp_f1 = f1
            best_asp_state = {k: v.clone() for k, v in asp_model.state_dict().items()}

    asp_model.load_state_dict(best_asp_state)
    print(f"Best aspect dev F1: {best_asp_f1:.4f}")

    # ── Sentiment model ───────────────────────────────────────────────────────
    print("\n=== Training BERT Sentiment Classifier ===")
    sent_ds_train = SentimentDataset(tr_df,  tokenizer)
    sent_ds_dev   = SentimentDataset(dev_df, tokenizer)
    sent_dl_train = DataLoader(sent_ds_train, BATCH_SIZE, shuffle=True)
    sent_dl_dev   = DataLoader(sent_ds_dev,  BATCH_SIZE, shuffle=False)

    sent_bert  = BertModel.from_pretrained(BERT_MODEL)
    sent_model = BERTSentiment(sent_bert, len(SENTIMENTS), DROPOUT).to(DEVICE)
    sent_optim = torch.optim.AdamW(sent_model.parameters(), lr=LR, weight_decay=0.01)
    total_sent_steps = len(sent_dl_train) * EPOCHS_SENT
    sent_sched = get_linear_schedule_with_warmup(
        sent_optim,
        num_warmup_steps=int(WARMUP_RATIO * total_sent_steps),
        num_training_steps=total_sent_steps
    )
    sent_crit  = nn.CrossEntropyLoss()

    best_sent_state  = None
    best_overall_f1  = -1.0

    for epoch in range(1, EPOCHS_SENT + 1):
        loss = train_epoch(sent_model, sent_dl_train, sent_optim, sent_sched, sent_crit)
        # Full overall F1 on dev
        dev_unique  = dev_df.drop_duplicates('id')
        dev_pred_df = predict(asp_model, sent_model, tokenizer,
                              dev_unique['text'].tolist(),
                              dev_unique['id'].tolist())
        f1 = evaluate_overall_f1(dev_df, dev_pred_df)
        print(f"  Epoch {epoch:2d} | loss={loss:.4f} | overall-dev F1={f1:.4f}")
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
