"""
ConfidenceLSTM — BiLSTM confidence classifier for interview answers.

Architecture (exact spec):
  - Embedding: nn.Embedding(vocab_size, 100) initialised with GloVe 100d weights
  - BiLSTM: 2 layers, hidden=128, bidirectional, dropout=0.3 between layers
  - Mean pooling over sequence (more stable than last hidden state)
  - FC stack: Linear(256→64) → ReLU → Linear(64→32) → ReLU → Linear(32→4)
  - Output: 4-class softmax (high=0, moderate=1, low=2, anxious=3)

Key design decision from spec:
  "Confidence scoring is independent of content — a wrong answer delivered
   confidently scores High; a correct answer with heavy hedging may score Low."

Vocabulary:
  - Built from training corpus at train time, saved as vocab.json
  - Special tokens: <PAD>=0, <UNK>=1
  - At inference: tokenise → lookup → pad/truncate → forward
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Label mapping ──────────────────────────────────────────────────────────────
# Index → label name (matches spec and answers table CHECK constraint)
IDX_TO_LABEL = {0: "high", 1: "moderate", 2: "low", 3: "anxious"}
LABEL_TO_IDX = {v: k for k, v in IDX_TO_LABEL.items()}


# ── Model ──────────────────────────────────────────────────────────────────────

class ConfidenceLSTM(nn.Module):
    """
    BiLSTM with GloVe embeddings for confidence classification.
    Labels: 0=high, 1=moderate, 2=low, 3=anxious
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,   # GloVe 100d
        hidden: int = 128,
        n_classes: int = 4,
        pad_idx: int = 0,
    ):
        super().__init__()

        # Embedding layer — weights initialised with GloVe at train time
        # padding_idx=0 ensures <PAD> token contributes zero gradient
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # BiLSTM: 2 layers, bidirectional → output dim = hidden * 2 = 256
        # dropout=0.3 applied between LSTM layers (not after final layer)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.dropout = nn.Dropout(0.3)

        # FC stack: 256 → 64 → 32 → 4
        self.fc1 = nn.Linear(hidden * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token index tensor of shape [batch, seq_len]

        Returns:
            logits of shape [batch, 4]
        """
        # Embed tokens → [batch, seq_len, 100]
        emb = self.dropout(self.embedding(x))

        # BiLSTM → [batch, seq_len, 256]
        out, _ = self.lstm(emb)

        # Mean pool over the sequence dimension.
        # More stable than using the final hidden state — handles variable-length
        # transcripts without masking. Spec explicitly specifies mean pooling.
        pooled = out.mean(dim=1)  # [batch, 256]

        # FC stack with ReLU activations
        h = F.relu(self.fc1(pooled))  # [batch, 64]
        h = F.relu(self.fc2(h))       # [batch, 32]
        return self.fc3(h)            # [batch, 4] — raw logits


# ── GloVe loader ───────────────────────────────────────────────────────────────

def load_glove_embeddings(
    glove_path: str,
    vocab: dict[str, int],
    embed_dim: int = 100,
) -> torch.Tensor:
    """
    Load GloVe 100d embeddings for words in vocab.
    Words not in GloVe are left as random normal (Xavier) vectors.

    Args:
        glove_path: path to glove.6B.100d.txt
        vocab:      word → index mapping (built by build_vocab())
        embed_dim:  embedding dimension (must match GloVe file)

    Returns:
        weight tensor of shape [vocab_size, embed_dim] for nn.Embedding
    """
    vocab_size = len(vocab)
    # Xavier uniform initialisation for words not in GloVe
    weights = torch.zeros(vocab_size, embed_dim)
    nn.init.xavier_uniform_(weights)

    # <PAD> token is always zero vector
    weights[0] = torch.zeros(embed_dim)

    loaded = 0
    logger.info(f"Loading GloVe embeddings from: {glove_path}")

    try:
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in vocab:
                    idx = vocab[word]
                    vector = torch.tensor([float(v) for v in parts[1:]], dtype=torch.float)
                    if vector.shape[0] == embed_dim:
                        weights[idx] = vector
                        loaded += 1
    except FileNotFoundError:
        logger.warning(
            f"GloVe file not found at {glove_path}. "
            "Embeddings will be randomly initialised."
        )

    coverage = (loaded / max(vocab_size - 2, 1)) * 100  # exclude PAD and UNK
    logger.info(f"GloVe coverage: {loaded}/{vocab_size} words ({coverage:.1f}%)")
    return weights


# ── Vocabulary utilities ───────────────────────────────────────────────────────

SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1}


def build_vocab(
    transcripts: list[str],
    min_freq: int = 2,
    max_vocab: int = 30_000,
) -> dict[str, int]:
    """
    Build a word → index vocabulary from a list of transcripts.

    Args:
        transcripts: raw text strings from training set
        min_freq:    minimum word frequency to include (filters rare noise)
        max_vocab:   maximum vocabulary size (most frequent words first)

    Returns:
        vocab dict with <PAD>=0, <UNK>=1, then words by frequency
    """
    from collections import Counter
    counts: Counter = Counter()
    for text in transcripts:
        counts.update(_tokenise(text))

    # Keep only words meeting minimum frequency threshold
    eligible = [(word, freq) for word, freq in counts.items() if freq >= min_freq]
    # Sort by frequency descending, then alphabetically for determinism
    eligible.sort(key=lambda x: (-x[1], x[0]))
    # Truncate to max_vocab - 2 (leaving room for PAD and UNK)
    eligible = eligible[: max_vocab - 2]

    vocab = dict(SPECIAL_TOKENS)  # start with PAD=0, UNK=1
    for idx, (word, _) in enumerate(eligible, start=2):
        vocab[word] = idx

    logger.info(f"Vocabulary built: {len(vocab)} tokens (min_freq={min_freq})")
    return vocab


def save_vocab(vocab: dict[str, int], path: str) -> None:
    """Save vocab dict to JSON file alongside the model weights."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    logger.info(f"Vocabulary saved to: {path}")


def load_vocab(path: str) -> dict[str, int]:
    """Load vocab dict from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    logger.info(f"Vocabulary loaded: {len(vocab)} tokens from {path}")
    return vocab


# ── Tokenisation ───────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokeniser.
    Lowercases, strips punctuation except apostrophes (for contractions),
    splits on whitespace. Matches how annotators processed transcripts.
    """
    # Lowercase
    text = text.lower()
    # Preserve apostrophes for contractions (i'm, don't, etc.)
    # Remove all other punctuation
    text = re.sub(r"[^\w\s']", " ", text)
    # Normalise whitespace
    tokens = text.split()
    return tokens


import re  # noqa: E402 — re used in _tokenise above


def encode(
    text: str,
    vocab: dict[str, int],
    max_length: int = 300,
) -> list[int]:
    """
    Tokenise text and map tokens to vocabulary indices.
    Truncates to max_length. UNK index (1) for out-of-vocabulary words.
    """
    tokens = _tokenise(text)[:max_length]
    unk_idx = vocab.get("<UNK>", 1)
    return [vocab.get(tok, unk_idx) for tok in tokens]


def encode_batch(
    texts: list[str],
    vocab: dict[str, int],
    max_length: int = 300,
) -> torch.Tensor:
    """
    Encode a batch of texts and pad to the longest sequence in the batch.
    Returns a LongTensor of shape [batch, max_seq_len].
    """
    encoded = [encode(t, vocab, max_length) for t in texts]
    max_len = max(len(seq) for seq in encoded) if encoded else 1
    pad_idx = vocab.get("<PAD>", 0)
    # Pad shorter sequences
    padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in encoded]
    return torch.tensor(padded, dtype=torch.long)
