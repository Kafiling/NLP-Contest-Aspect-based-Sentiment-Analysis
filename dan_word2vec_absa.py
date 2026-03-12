"""
Deep Averaging Network (DAN) for Aspect-Based Sentiment Analysis (ABSA)
Embedding: Word2Vec trained on the training corpus (gensim)

Key difference from dan_absa.py:
  - Word embeddings are initialised from Word2Vec vectors trained on the
    training-set sentences (vector_size=100, window=5, sg=1 / skip-gram).
  - OOV words get a small random uniform vector.
  - Embeddings remain trainable (fine-tuned during DAN training).

Architecture:
  - Model 1 (Aspect):    text → avg w2v embeddings → FC layers → 5 binary outputs
  - Model 2 (Sentiment): [text_avg; aspect_emb] → FC layers → 4-class output
"""

import os
import re
import random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_DIM       = 100       # Word2Vec vector size
HIDDEN_DIM      = 256
DROPOUT         = 0.3
EPOCHS_ASPECT   = 30
EPOCHS_SENT     = 30
BATCH_SIZE      = 32
LR              = 1e-3
ASPECT_THRESH   = 0.4
DEV_SPLIT       = 0.15

# Word2Vec hyperparams
W2V_WINDOW   = 5
W2V_MIN_CNT  = 1
W2V_WORKERS  = 4
W2V_EPOCHS   = 10
W2V_SG       = 1    # 1 = skip-gram, 0 = CBOW

ASPECTS    = ['food', 'service', 'ambience', 'price', 'anecdotes/miscellaneous']
SENTIMENTS = ['positive', 'negative', 'neutral', 'conflict']
ASPECT2ID  = {a: i for i, a in enumerate(ASPECTS)}
SENT2ID    = {s: i for i, s in enumerate(SENTIMENTS)}
ID2SENT    = {i: s for s, i in SENT2ID.items()}

DATA_DIR     = os.path.join(os.path.dirname(__file__), 'Resource')
TRAIN_FILE   = os.path.join(DATA_DIR, 'contest1_train.csv')
TEST_FILE    = os.path.join(DATA_DIR, 'contest1_test.csv')
OUT_FILE     = os.path.join(DATA_DIR, 'w2v_test_pred.csv')
DEV_OUT_FILE = os.path.join(DATA_DIR, 'w2v_dev_pred.csv')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── Text helpers ──────────────────────────────────────────────────────────────
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
    return ids if ids else [1]


def pad_batch(sequences):
    max_len = max(len(s) for s in sequences)
    padded  = [s + [0] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)


# ── Word2Vec embedding matrix ─────────────────────────────────────────────────
def train_word2vec(texts):
    """Train a skip-gram Word2Vec model on tokenised training sentences."""
    sentences = [tokenize(t) for t in texts]
    print(f"Training Word2Vec (sg={W2V_SG}, dim={EMBED_DIM}, window={W2V_WINDOW}, "
          f"epochs={W2V_EPOCHS}) on {len(sentences)} sentences…")
    model = Word2Vec(
        sentences,
        vector_size=EMBED_DIM,
        window=W2V_WINDOW,
        min_count=W2V_MIN_CNT,
        workers=W2V_WORKERS,
        epochs=W2V_EPOCHS,
        sg=W2V_SG,
        seed=SEED,
    )
    return model


def build_embedding_matrix_w2v(vocab: dict, w2v_model) -> np.ndarray:
    """
    Map each vocab word to its Word2Vec vector.
    OOV words (not in Word2Vec vocab) get a small random vector.
    <PAD> (index 0) stays all-zeros.
    """
    vocab_size = len(vocab)
    matrix     = np.zeros((vocab_size, EMBED_DIM), dtype=np.float32)
    found = 0
    rng   = np.random.default_rng(SEED)
    for word, idx in vocab.items():
        if idx == 0:        # <PAD> stays zero
            continue
        if word in w2v_model.wv:
            matrix[idx] = w2v_model.wv[word]
            found += 1
        else:
            matrix[idx] = rng.uniform(-0.1, 0.1, EMBED_DIM).astype(np.float32)
    print(f"Embedding matrix: {found}/{vocab_size} words found in Word2Vec "
          f"({found/vocab_size*100:.1f}%)")
    return matrix


# ── Datasets ──────────────────────────────────────────────────────────────────
class AspectDataset(Dataset):
    def __init__(self, df, vocab):
        grouped = df.groupby('id').agg(
            text=('text', 'first'),
            aspects=('aspectCategory', list)
        ).reset_index()
        self.texts  = grouped['text'].tolist()
        self.labels = [
            [1.0 if a in aspects else 0.0 for a in ASPECTS]
            for aspects in grouped['aspects']
        ]
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return encode(self.texts[idx], self.vocab), \
               torch.tensor(self.labels[idx], dtype=torch.float)


class SentimentDataset(Dataset):
    def __init__(self, df, vocab):
        self.vocab   = vocab
        self.texts   = df['text'].tolist()
        self.aspects = df['aspectCategory'].map(ASPECT2ID).tolist()
        self.labels  = df['polarity'].map(SENT2ID).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return encode(self.texts[idx], self.vocab), \
               self.aspects[idx], self.labels[idx]


def aspect_collate(batch):
    seqs, labels = zip(*batch)
    return pad_batch(seqs), torch.stack(labels)


def sentiment_collate(batch):
    seqs, aspects, labels = zip(*batch)
    return (pad_batch(seqs),
            torch.tensor(aspects, dtype=torch.long),
            torch.tensor(labels,  dtype=torch.long))


# ── DAN Models with pretrained embeddings ────────────────────────────────────
class DANAspect(nn.Module):
    def __init__(self, pretrained_weights: np.ndarray, hidden_dim, n_aspects, dropout):
        super().__init__()
        vocab_size, embed_dim = pretrained_weights.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(pretrained_weights, dtype=torch.float)
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_aspects),
        )

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        emb  = self.embedding(x)
        avg  = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(avg)


class DANSentiment(nn.Module):
    def __init__(self, pretrained_weights: np.ndarray, hidden_dim,
                 n_aspects, n_sentiments, dropout):
        super().__init__()
        vocab_size, embed_dim = pretrained_weights.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(pretrained_weights, dtype=torch.float)
        )
        self.aspect_embedding = nn.Embedding(n_aspects, embed_dim)
        in_dim = embed_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
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
    total = 0.0
    for x, labels in loader:
        x, labels = x.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), labels)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_sentiment_model(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x, aspects, labels in loader:
        x, aspects, labels = x.to(DEVICE), aspects.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x, aspects), labels)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate_overall_f1(gold_df, pred_df):
    gold = set(zip(gold_df['id'], gold_df['aspectCategory'], gold_df['polarity']))
    pred = set(zip(pred_df['id'], pred_df['aspectCategory'], pred_df['polarity']))
    if not pred:
        return 0.0
    p = len(gold & pred) / len(pred)
    r = len(gold & pred) / len(gold)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def predict(asp_model, sent_model, texts, ids, vocab, threshold=ASPECT_THRESH):
    asp_model.eval(); sent_model.eval()
    rows = []
    with torch.no_grad():
        for doc_id, text in zip(ids, texts):
            x     = pad_batch([encode(text, vocab)]).to(DEVICE)
            probs = torch.sigmoid(asp_model(x)).cpu().numpy()[0]
            preds = [ASPECTS[i] for i, p in enumerate(probs) if p >= threshold]
            if not preds:
                preds = [ASPECTS[int(np.argmax(probs))]]
            for asp in preds:
                asp_id   = torch.tensor([ASPECT2ID[asp]], dtype=torch.long).to(DEVICE)
                pred_s   = ID2SENT[int(sent_model(x, asp_id).argmax(1).cpu())]
                rows.append({'id': doc_id, 'aspectCategory': asp, 'polarity': pred_s})
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    print(f"Train rows: {len(train_df)} | Unique IDs: {train_df['id'].nunique()}")
    print(f"Test  rows: {len(test_df)}")

    uids = train_df['id'].unique()
    tids, dids = train_test_split(uids, test_size=DEV_SPLIT, random_state=SEED)
    tr_df  = train_df[train_df['id'].isin(tids)].reset_index(drop=True)
    dev_df = train_df[train_df['id'].isin(dids)].reset_index(drop=True)
    print(f"Split -> train: {len(tr_df)} | dev: {len(dev_df)}")

    # ── Build vocab + train Word2Vec ──────────────────────────────────────────
    vocab    = build_vocab(tr_df['text'].tolist())
    w2v      = train_word2vec(tr_df['text'].tolist())
    emb_mat  = build_embedding_matrix_w2v(vocab, w2v)
    print(f"Vocab size: {len(vocab)} | Embedding matrix: {emb_mat.shape}")

    # ── Aspect model ──────────────────────────────────────────────────────────
    print("\n=== Training Aspect Classifier (Word2Vec) ===")
    asp_tr_dl  = DataLoader(AspectDataset(tr_df,  vocab), BATCH_SIZE,
                             shuffle=True,  collate_fn=aspect_collate)
    asp_model  = DANAspect(emb_mat, HIDDEN_DIM, len(ASPECTS), DROPOUT).to(DEVICE)
    asp_optim  = torch.optim.Adam(asp_model.parameters(), lr=LR, weight_decay=1e-5)
    asp_crit   = nn.BCEWithLogitsLoss()
    asp_sched  = torch.optim.lr_scheduler.StepLR(asp_optim, step_size=10, gamma=0.5)

    best_asp_state, best_asp_f1 = None, -1.0
    for epoch in range(1, EPOCHS_ASPECT + 1):
        loss = train_aspect_model(asp_model, asp_tr_dl, asp_optim, asp_crit)
        asp_sched.step()
        if epoch % 5 == 0:
            asp_model.eval()
            gold_asp = set(zip(dev_df['id'], dev_df['aspectCategory']))
            pred_asp = set()
            dev_unique = dev_df.drop_duplicates('id')
            with torch.no_grad():
                for _, row in dev_unique.iterrows():
                    x     = pad_batch([encode(row['text'], vocab)]).to(DEVICE)
                    probs = torch.sigmoid(asp_model(x)).cpu().numpy()[0]
                    preds = [ASPECTS[i] for i, p in enumerate(probs) if p >= ASPECT_THRESH]
                    if not preds:
                        preds = [ASPECTS[int(np.argmax(probs))]]
                    for asp in preds:
                        pred_asp.add((row['id'], asp))
            tp   = len(gold_asp & pred_asp)
            prec = tp / len(pred_asp) if pred_asp else 0
            rec  = tp / len(gold_asp) if gold_asp  else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
            print(f"  Epoch {epoch:3d} | loss={loss:.4f} | aspect-dev F1={f1:.4f}")
            if f1 > best_asp_f1:
                best_asp_f1   = f1
                best_asp_state = {k: v.clone() for k, v in asp_model.state_dict().items()}

    asp_model.load_state_dict(best_asp_state)
    print(f"Best aspect dev F1: {best_asp_f1:.4f}")

    # ── Sentiment model ───────────────────────────────────────────────────────
    print("\n=== Training Sentiment Classifier (Word2Vec) ===")
    sent_tr_dl  = DataLoader(SentimentDataset(tr_df,  vocab), BATCH_SIZE,
                              shuffle=True,  collate_fn=sentiment_collate)
    sent_model  = DANSentiment(emb_mat, HIDDEN_DIM, len(ASPECTS),
                               len(SENTIMENTS), DROPOUT).to(DEVICE)
    sent_optim  = torch.optim.Adam(sent_model.parameters(), lr=LR, weight_decay=1e-5)
    sent_crit   = nn.CrossEntropyLoss()
    sent_sched  = torch.optim.lr_scheduler.StepLR(sent_optim, step_size=10, gamma=0.5)

    best_sent_state, best_overall_f1 = None, -1.0
    for epoch in range(1, EPOCHS_SENT + 1):
        loss = train_sentiment_model(sent_model, sent_tr_dl, sent_optim, sent_crit)
        sent_sched.step()
        if epoch % 5 == 0:
            dev_unique  = dev_df.drop_duplicates('id')
            dev_pred_df = predict(asp_model, sent_model,
                                  dev_unique['text'].tolist(),
                                  dev_unique['id'].tolist(), vocab)
            f1 = evaluate_overall_f1(dev_df, dev_pred_df)
            print(f"  Epoch {epoch:3d} | loss={loss:.4f} | overall-dev F1={f1:.4f}")
            if f1 > best_overall_f1:
                best_overall_f1  = f1
                best_sent_state  = {k: v.clone() for k, v in sent_model.state_dict().items()}

    sent_model.load_state_dict(best_sent_state)
    print(f"Best overall dev F1: {best_overall_f1:.4f}")

    # ── Final evaluation + predictions ───────────────────────────────────────
    print("\n=== Final Dev Evaluation ===")
    dev_unique     = dev_df.drop_duplicates('id')
    dev_pred_final = predict(asp_model, sent_model,
                             dev_unique['text'].tolist(),
                             dev_unique['id'].tolist(), vocab)
    print(f"Dev Overall F1: {evaluate_overall_f1(dev_df, dev_pred_final):.4f}")
    dev_pred_final.to_csv(DEV_OUT_FILE, index=False)
    print(f"Saved dev predictions → {DEV_OUT_FILE}")

    print("\n=== Predicting on Test Set ===")
    test_pred = predict(asp_model, sent_model,
                        test_df['text'].tolist(),
                        test_df['id'].tolist(), vocab)
    test_pred.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(test_pred)} rows → {OUT_FILE}")
    print(test_pred.head(10))


if __name__ == '__main__':
    main()
