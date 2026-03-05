"""
Deep Averaging Network (DAN) for Aspect-Based Sentiment Analysis (ABSA)

Architecture:
  - Model 1 (Aspect): text -> avg word embeddings -> FC layers -> 5 binary outputs
  - Model 2 (Sentiment): [text_avg_emb ; aspect_emb] -> FC layers -> 4-class output

Prediction pipeline:
  For each test sentence:
    1. Predict which aspects are present (threshold on sigmoid)
    2. For each predicted aspect, predict its sentiment
    3. Output rows: (id, aspectCategory, polarity)
"""

import os
import re
import sys
import random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
EMBED_DIM       = 128
HIDDEN_DIM      = 256
DROPOUT         = 0.3
EPOCHS_ASPECT   = 30
EPOCHS_SENT     = 30
BATCH_SIZE      = 32
LR              = 1e-3
ASPECT_THRESH   = 0.4       # sigmoid threshold for aspect presence
DEV_SPLIT       = 0.15      # fraction of train used as dev

ASPECTS    = ['food', 'service', 'ambience', 'price', 'anecdotes/miscellaneous']
SENTIMENTS = ['positive', 'negative', 'neutral', 'conflict']
ASPECT2ID  = {a: i for i, a in enumerate(ASPECTS)}
SENT2ID    = {s: i for i, s in enumerate(SENTIMENTS)}
ID2SENT    = {i: s for s, i in SENT2ID.items()}

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'Resource')
TRAIN_FILE = os.path.join(DATA_DIR, 'contest1_train.csv')
TEST_FILE  = os.path.join(DATA_DIR, 'contest1_test.csv')
OUT_FILE     = os.path.join(DATA_DIR, 'test_pred.csv')
DEV_OUT_FILE = os.path.join(DATA_DIR, 'dev_pred.csv')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── Text helpers ─────────────────────────────────────────────────────────────
def tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return text.split()


def build_vocab(texts, min_freq: int = 1):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int = 80):
    ids = [vocab.get(t, 1) for t in tokenize(text)[:max_len]]
    if not ids:
        ids = [1]
    return ids


def pad_batch(sequences):
    """Pad a list of id-lists to the same length, return LongTensor."""
    max_len = max(len(s) for s in sequences)
    padded = [s + [0] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)


# ── Datasets ─────────────────────────────────────────────────────────────────
class AspectDataset(Dataset):
    """
    Each sample = one unique sentence.
    Label = binary vector over ASPECTS (multi-label).
    """
    def __init__(self, df, vocab):
        self.vocab = vocab
        # Group by id to collect all aspects per sentence
        grouped = df.groupby('id').agg(
            text=('text', 'first'),
            aspects=('aspectCategory', list)
        ).reset_index()
        self.texts   = grouped['text'].tolist()
        self.labels  = []
        for aspects in grouped['aspects']:
            vec = [1.0 if a in aspects else 0.0 for a in ASPECTS]
            self.labels.append(vec)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab)
        return ids, torch.tensor(self.labels[idx], dtype=torch.float)


class SentimentDataset(Dataset):
    """
    Each sample = (sentence, aspect) pair with a sentiment label.
    """
    def __init__(self, df, vocab):
        self.vocab  = vocab
        self.texts   = df['text'].tolist()
        self.aspects = df['aspectCategory'].map(ASPECT2ID).tolist()
        self.labels  = df['polarity'].map(SENT2ID).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab)
        return ids, self.aspects[idx], self.labels[idx]


def aspect_collate(batch):
    seqs, labels = zip(*batch)
    return pad_batch(seqs), torch.stack(labels)


def sentiment_collate(batch):
    seqs, aspects, labels = zip(*batch)
    return pad_batch(seqs), torch.tensor(aspects, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ── DAN Models ───────────────────────────────────────────────────────────────
class DANAspect(nn.Module):
    """
    DAN for multi-label aspect detection.
    Input : padded token ids  (batch, seq_len)
    Output: raw logits        (batch, n_aspects)
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_aspects, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_aspects),
        )

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)          # (B, L, 1)
        emb  = self.embedding(x)                        # (B, L, E)
        avg  = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, E)
        return self.fc(avg)


class DANSentiment(nn.Module):
    """
    DAN for sentiment classification, conditioned on aspect.
    Concatenates averaged text embedding with a learned aspect embedding.
    Input : padded token ids (B, L), aspect ids (B,)
    Output: raw logits       (B, n_sentiments)
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_aspects, n_sentiments, dropout):
        super().__init__()
        self.embedding        = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.aspect_embedding = nn.Embedding(n_aspects, embed_dim)
        in_dim = embed_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_sentiments),
        )

    def forward(self, x, aspect_ids):
        mask = (x != 0).float().unsqueeze(-1)
        emb  = self.embedding(x)
        avg  = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        asp  = self.aspect_embedding(aspect_ids)
        return self.fc(torch.cat([avg, asp], dim=-1))


# ── Training helpers ──────────────────────────────────────────────────────────
def train_aspect_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, labels in loader:
        x, labels = x.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_sentiment_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, aspects, labels in loader:
        x, aspects, labels = x.to(DEVICE), aspects.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x, aspects)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Evaluation (mirrors evaluate.py logic) ────────────────────────────────────
def evaluate_overall_f1(gold_df, pred_df):
    """Compute micro overall F1 matching (id, aspectCategory, polarity) tuples."""
    gold_set = set(zip(gold_df['id'], gold_df['aspectCategory'], gold_df['polarity']))
    pred_set = set(zip(pred_df['id'], pred_df['aspectCategory'], pred_df['polarity']))
    if not pred_set:
        return 0.0
    precision = len(gold_set & pred_set) / len(pred_set)
    recall    = len(gold_set & pred_set) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(aspect_model, sent_model, texts, ids, vocab, threshold=ASPECT_THRESH):
    aspect_model.eval()
    sent_model.eval()
    rows = []
    with torch.no_grad():
        for doc_id, text in zip(ids, texts):
            # ---- Aspect prediction ----
            enc = encode(text, vocab)
            x   = pad_batch([enc]).to(DEVICE)
            logits = aspect_model(x)                     # (1, n_aspects)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]  # (n_aspects,)

            predicted_aspects = [ASPECTS[i] for i, p in enumerate(probs) if p >= threshold]

            # Fall back to top-1 if nothing clears the threshold
            if not predicted_aspects:
                predicted_aspects = [ASPECTS[int(np.argmax(probs))]]

            # ---- Sentiment prediction for each aspect ----
            for asp in predicted_aspects:
                asp_id = torch.tensor([ASPECT2ID[asp]], dtype=torch.long).to(DEVICE)
                s_logits = sent_model(x, asp_id)          # (1, n_sentiments)
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

    # ---- Train / dev split (by unique sentence ID) ----
    unique_ids = train_df['id'].unique()
    train_ids, dev_ids = train_test_split(unique_ids, test_size=DEV_SPLIT, random_state=SEED)
    tr_df  = train_df[train_df['id'].isin(train_ids)].reset_index(drop=True)
    dev_df = train_df[train_df['id'].isin(dev_ids)].reset_index(drop=True)
    print(f"Split -> train: {len(tr_df)} rows | dev: {len(dev_df)} rows")

    # ---- Vocabulary (built only on training texts) ----
    vocab = build_vocab(tr_df['text'].tolist(), min_freq=1)
    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # ── Aspect model ─────────────────────────────────────────────────────────
    print("\n=== Training Aspect Classifier ===")
    asp_train_ds = AspectDataset(tr_df,  vocab)
    asp_dev_ds   = AspectDataset(dev_df, vocab)
    asp_train_dl = DataLoader(asp_train_ds, BATCH_SIZE, shuffle=True,  collate_fn=aspect_collate)
    asp_dev_dl   = DataLoader(asp_dev_ds,  BATCH_SIZE, shuffle=False, collate_fn=aspect_collate)

    asp_model  = DANAspect(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, len(ASPECTS), DROPOUT).to(DEVICE)
    asp_optim  = torch.optim.Adam(asp_model.parameters(), lr=LR, weight_decay=1e-5)
    asp_crit   = nn.BCEWithLogitsLoss()
    asp_sched  = torch.optim.lr_scheduler.StepLR(asp_optim, step_size=10, gamma=0.5)

    best_asp_state = None
    best_asp_f1    = -1.0

    for epoch in range(1, EPOCHS_ASPECT + 1):
        loss = train_aspect_model(asp_model, asp_train_dl, asp_optim, asp_crit)
        asp_sched.step()
        if epoch % 5 == 0:
            # Quick dev check: aspect-level F1 (no sentiment model needed)
            asp_model.eval()
            gold_asp = set(zip(dev_df['id'], dev_df['aspectCategory']))
            pred_asp = set()
            dev_unique = dev_df.drop_duplicates('id')
            with torch.no_grad():
                for _, row in dev_unique.iterrows():
                    enc   = encode(row['text'], vocab)
                    x     = pad_batch([enc]).to(DEVICE)
                    probs = torch.sigmoid(asp_model(x)).cpu().numpy()[0]
                    preds = [ASPECTS[i] for i, p in enumerate(probs) if p >= ASPECT_THRESH]
                    if not preds:
                        preds = [ASPECTS[int(np.argmax(probs))]]
                    for asp in preds:
                        pred_asp.add((row['id'], asp))
            tp   = len(gold_asp & pred_asp)
            prec = tp / len(pred_asp) if pred_asp else 0
            rec  = tp / len(gold_asp) if gold_asp else 0
            f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
            print(f"  Epoch {epoch:3d} | loss={loss:.4f} | aspect-dev F1={f1:.4f}")
            if f1 > best_asp_f1:
                best_asp_f1 = f1
                best_asp_state = {k: v.clone() for k, v in asp_model.state_dict().items()}

    asp_model.load_state_dict(best_asp_state)
    print(f"Best aspect dev F1: {best_asp_f1:.4f}")

    # ── Sentiment model ───────────────────────────────────────────────────────
    print("\n=== Training Sentiment Classifier ===")
    sent_train_ds = SentimentDataset(tr_df,  vocab)
    sent_dev_ds   = SentimentDataset(dev_df, vocab)
    sent_train_dl = DataLoader(sent_train_ds, BATCH_SIZE, shuffle=True,  collate_fn=sentiment_collate)
    sent_dev_dl   = DataLoader(sent_dev_ds,  BATCH_SIZE, shuffle=False, collate_fn=sentiment_collate)

    sent_model = DANSentiment(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, len(ASPECTS), len(SENTIMENTS), DROPOUT).to(DEVICE)
    sent_optim = torch.optim.Adam(sent_model.parameters(), lr=LR, weight_decay=1e-5)
    sent_crit  = nn.CrossEntropyLoss()
    sent_sched = torch.optim.lr_scheduler.StepLR(sent_optim, step_size=10, gamma=0.5)

    best_sent_state = None
    best_overall_f1 = -1.0

    for epoch in range(1, EPOCHS_SENT + 1):
        loss = train_sentiment_model(sent_model, sent_train_dl, sent_optim, sent_crit)
        sent_sched.step()
        if epoch % 5 == 0:
            # Full overall F1 on dev set
            dev_unique   = dev_df.drop_duplicates('id')
            dev_pred_df  = predict(asp_model, sent_model,
                                   dev_unique['text'].tolist(),
                                   dev_unique['id'].tolist(),
                                   vocab)
            f1 = evaluate_overall_f1(dev_df, dev_pred_df)
            print(f"  Epoch {epoch:3d} | loss={loss:.4f} | overall-dev F1={f1:.4f}")
            if f1 > best_overall_f1:
                best_overall_f1 = f1
                best_sent_state = {k: v.clone() for k, v in sent_model.state_dict().items()}

    sent_model.load_state_dict(best_sent_state)
    print(f"Best overall dev F1: {best_overall_f1:.4f}")

    # ── Final dev evaluation ──────────────────────────────────────────────────
    print("\n=== Final Dev Evaluation ===")
    dev_unique     = dev_df.drop_duplicates('id')
    dev_pred_final = predict(asp_model, sent_model,
                             dev_unique['text'].tolist(),
                             dev_unique['id'].tolist(), vocab)
    final_f1 = evaluate_overall_f1(dev_df, dev_pred_final)
    print(f"Dev Overall F1: {final_f1:.4f}")
    dev_pred_final.to_csv(DEV_OUT_FILE, index=False)
    print(f"Saved dev predictions to {DEV_OUT_FILE}")

    # ── Predict on test set ───────────────────────────────────────────────────
    print("\n=== Predicting on Test Set ===")
    test_pred = predict(asp_model, sent_model,
                        test_df['text'].tolist(),
                        test_df['id'].tolist(),
                        vocab)
    test_pred.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(test_pred)} rows to {OUT_FILE}")
    print(test_pred.head(10))


if __name__ == '__main__':
    main()
