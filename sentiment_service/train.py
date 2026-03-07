"""
Training script for ConfidenceLSTM.

Training config (from spec):
  - Embeddings: GloVe 100d (pre-trained, fine-tuned during training)
  - BiLSTM: hidden=128, 2 layers, bidirectional, dropout=0.3
  - Optimizer: Adam (lr=1e-3)
  - Epochs: up to 20 with early stopping (patience=3)
  - Batch size: 32
  - Loss: CrossEntropyLoss (4 classes)
  - Max sequence length: 300 tokens

Training data format (JSONL):
  {"transcript": "I think recursion is... um... when a function calls itself",
   "label": 2}
  Labels: 0=high, 1=moderate, 2=low, 3=anxious

Usage:
  python sentiment_service/train.py \
    --train_path data/confidence_train.jsonl \
    --val_path   data/confidence_val.jsonl \
    --glove_path models/glove.6B.100d.txt \
    --output_dir models/
"""
import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import (
    ConfidenceLSTM,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    build_vocab,
    encode_batch,
    encode,
    load_glove_embeddings,
    save_vocab,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────

class ConfidenceDataset(Dataset):
    """
    Loads confidence-labelled transcripts from JSONL.
    Expected fields: "transcript" (str) and "label" (int 0-3 OR string).
    """

    def __init__(self, jsonl_path: str, vocab: dict[str, int], max_length: int = 300):
        self.vocab = vocab
        self.max_length = max_length
        self.records: list[dict] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping line {line_no} — JSON parse error")
                    continue

                if "transcript" not in rec or "label" not in rec:
                    logger.warning(f"Skipping line {line_no} — missing fields")
                    continue

                # Accept both integer labels (0-3) and string labels
                label = rec["label"]
                if isinstance(label, str):
                    if label not in LABEL_TO_IDX:
                        logger.warning(f"Skipping line {line_no} — unknown label '{label}'")
                        continue
                    label = LABEL_TO_IDX[label]

                if label not in IDX_TO_LABEL:
                    logger.warning(f"Skipping line {line_no} — label {label} out of range 0-3")
                    continue

                self.records.append({"transcript": rec["transcript"], "label": label})

        logger.info(f"Loaded {len(self.records)} examples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        indices = encode(rec["transcript"], self.vocab, self.max_length)
        return {
            "indices": torch.tensor(indices, dtype=torch.long),
            "label":   torch.tensor(rec["label"], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Custom collate function: pad sequences to the longest in the batch.
    This is more memory-efficient than global max_length padding.
    """
    max_len = max(item["indices"].shape[0] for item in batch)
    pad_idx = 0  # <PAD>

    padded_indices = []
    labels = []
    for item in batch:
        seq = item["indices"]
        pad_len = max_len - seq.shape[0]
        padded_indices.append(
            torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(item["label"])

    return {
        "indices": torch.stack(padded_indices),   # [batch, max_len]
        "label":   torch.stack(labels),           # [batch]
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def evaluate_accuracy(
    model: ConfidenceLSTM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Compute validation loss and accuracy.
    Returns (avg_loss, accuracy_fraction).
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["indices"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(loader) if loader else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def train(
    train_path: str,
    val_path: str,
    glove_path: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_length: int = 300,
    min_freq: int = 2,
    hidden: int = 128,
    early_stop_patience: int = 3,
    seed: int = 42,
) -> None:
    """Main training loop for ConfidenceLSTM."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    # ── Build vocabulary from training transcripts ────────────────────
    logger.info("Building vocabulary from training data…")
    raw_transcripts: list[str] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_transcripts.append(json.loads(line)["transcript"])
                except (json.JSONDecodeError, KeyError):
                    pass

    vocab = build_vocab(raw_transcripts, min_freq=min_freq)

    # Save vocab alongside model weights (required at inference time)
    vocab_path = str(Path(output_dir) / "vocab.json")
    save_vocab(vocab, vocab_path)

    # ── Datasets and loaders ──────────────────────────────────────────
    train_ds = ConfidenceDataset(train_path, vocab, max_length)
    val_ds   = ConfidenceDataset(val_path,   vocab, max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = ConfidenceLSTM(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden=hidden,
        n_classes=4,
    ).to(device)

    # Initialise embeddings with GloVe
    glove_weights = load_glove_embeddings(glove_path, vocab, embed_dim=100)
    model.embedding.weight.data.copy_(glove_weights.to(device))
    logger.info("GloVe embeddings loaded into model")

    # ── Optimiser and loss ────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──────────────────────────────────────────────────
    output_path = Path(output_dir) / "confidence_lstm.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            x = batch["indices"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Clip gradients — important for RNNs to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            epoch_loss += loss.item()

            if step % 50 == 0:
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(train_loader)} "
                    f"| Loss {epoch_loss / step:.4f}"
                )

        val_loss, val_acc = evaluate_accuracy(model, val_loader, device)
        elapsed = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {epoch_loss / len(train_loader):.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc * 100:.1f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
            logger.info(f"✓ New best val loss: {val_loss:.4f} → saved to {output_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{early_stop_patience})")
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

    logger.info(f"Training complete. Best model saved to: {output_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConfidenceLSTM")
    parser.add_argument("--train_path",  required=True)
    parser.add_argument("--val_path",    required=True)
    parser.add_argument("--glove_path",  required=True, help="Path to glove.6B.100d.txt")
    parser.add_argument("--output_dir",  default="models/")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max_length",  type=int,   default=300)
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--patience",    type=int,   default=3)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        glove_path=args.glove_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        hidden=args.hidden,
        early_stop_patience=args.patience,
        seed=args.seed,
    )
