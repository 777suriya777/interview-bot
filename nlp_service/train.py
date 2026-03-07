"""
Fine-tuning script for InterviewBERTScorer.

Training config (from paper / spec):
  - Base model:   bert-base-uncased
  - Optimizer:    AdamW (lr=2e-5, weight_decay=0.01)
  - Scheduler:    Linear warmup (10% steps) → linear decay to 0
  - Batch size:   16 (gradient accumulation × 2 if GPU VRAM < 8 GB)
  - Epochs:       3–5 with early stopping (patience=2 on val loss)
  - Max seq len:  512 tokens
  - Loss:         CrossEntropyLoss per head, summed (equal weight)
  - Dropout:      0.1 on pooler output (BERT default)

Training data format (JSONL, one record per line):
  {
    "question":     "Explain hash tables and collision handling.",
    "answer":       "A hash table maps keys to values...",
    "content":      3,
    "relevance":    3,
    "completeness": 2,
    "accuracy":     4
  }

Usage:
  python nlp_service/train.py \
    --train_path data/train.jsonl \
    --val_path   data/val.jsonl \
    --output_dir models/ \
    --epochs 5 \
    --batch_size 16
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from model import InterviewBERTScorer

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────

class InterviewQADataset(Dataset):
    """
    Loads question-answer pairs from a JSONL file.
    Tokenises each pair as: [CLS] question [SEP] answer [SEP]
    Returns input_ids, attention_mask, token_type_ids, and label tensors.
    """

    DIMS = ["content", "relevance", "completeness", "accuracy"]

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: list[dict] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_no} — JSON parse error: {e}")
                    continue

                # Validate required fields
                if not all(k in record for k in ["question", "answer"] + self.DIMS):
                    logger.warning(f"Skipping line {line_no} — missing fields")
                    continue

                # Validate score ranges
                if any(not (0 <= record[d] <= 4) for d in self.DIMS):
                    logger.warning(f"Skipping line {line_no} — score out of range 0-4")
                    continue

                self.records.append(record)

        logger.info(f"Loaded {len(self.records)} examples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]

        # Tokenise the question-answer pair as a sentence pair.
        # token_type_ids=0 for question tokens, 1 for answer tokens.
        enc = self.tokenizer(
            r["question"],
            r["answer"],
            max_length=self.max_length,
            truncation=True,         # truncate answer if pair exceeds 512 tokens
            padding="max_length",    # pad shorter sequences to max_length
            return_tensors="pt",
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # [seq_len]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [seq_len]
            "token_type_ids": enc["token_type_ids"].squeeze(0),  # [seq_len]
            # Integer labels for CrossEntropyLoss
            "label_content":      torch.tensor(r["content"],      dtype=torch.long),
            "label_relevance":    torch.tensor(r["relevance"],    dtype=torch.long),
            "label_completeness": torch.tensor(r["completeness"], dtype=torch.long),
            "label_accuracy":     torch.tensor(r["accuracy"],     dtype=torch.long),
        }


# ── Training ───────────────────────────────────────────────────────────────────

def compute_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    criterion: nn.CrossEntropyLoss,
) -> torch.Tensor:
    """
    Sum the CrossEntropyLoss across all 4 heads.
    Equal weighting per spec — each dimension contributes equally.
    """
    loss = (
        criterion(logits["content"],      batch["label_content"])
        + criterion(logits["relevance"],    batch["label_relevance"])
        + criterion(logits["completeness"], batch["label_completeness"])
        + criterion(logits["accuracy"],     batch["label_accuracy"])
    )
    return loss


def evaluate_val_loss(
    model: InterviewBERTScorer,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    """Compute average validation loss over the full validation set."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )
            loss = compute_loss(logits, batch, criterion)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train(
    train_path: str,
    val_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    warmup_ratio: float = 0.1,
    early_stop_patience: int = 2,
    gradient_accumulation_steps: int = 1,
    pretrained_name: str = "bert-base-uncased",
    seed: int = 42,
) -> None:
    """
    Main training loop.
    Saves best checkpoint to output_dir/bert_scorer.pt
    """
    # Reproducibility
    torch.manual_seed(seed)

    # Device — use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Tokeniser & Datasets ──────────────────────────────────────────
    logger.info(f"Loading tokeniser: {pretrained_name}")
    tokenizer = BertTokenizer.from_pretrained(pretrained_name)

    train_ds = InterviewQADataset(train_path, tokenizer, max_length)
    val_ds   = InterviewQADataset(val_path,   tokenizer, max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # ── Model ─────────────────────────────────────────────────────────
    logger.info(f"Initialising InterviewBERTScorer from {pretrained_name}")
    model = InterviewBERTScorer(pretrained_name).to(device)

    # ── Optimiser — AdamW with weight decay ──────────────────────────
    # Exclude bias and LayerNorm params from weight decay (standard BERT practice)
    no_decay = {"bias", "LayerNorm.weight"}
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)

    # ── Scheduler — linear warmup then linear decay to 0 ─────────────
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    logger.info(f"Total training steps: {total_steps} | Warmup steps: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # ── Training Loop ──────────────────────────────────────────────────
    output_path = Path(output_dir) / "bert_scorer.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )

            loss = compute_loss(logits, batch, criterion)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            # Only step optimizer every N batches
            if step % gradient_accumulation_steps == 0:
                # Gradient clipping — prevents exploding gradients during fine-tuning
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log every 50 steps
            if step % 50 == 0:
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(train_loader)} "
                    f"| Loss {epoch_loss / step:.4f} "
                    f"| LR {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate_val_loss(model, val_loader, criterion, device)
        elapsed = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{epochs} complete | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ── Early stopping & checkpoint ───────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save only the model state dict (not the full model object)
            # so we can load it with model.load_state_dict() at inference time
            torch.save(model.state_dict(), output_path)
            logger.info(f"✓ New best val loss: {val_loss:.4f} → saved to {output_path}")
        else:
            patience_counter += 1
            logger.info(
                f"Val loss did not improve ({patience_counter}/{early_stop_patience})"
            )
            if patience_counter >= early_stop_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best val loss: {best_val_loss:.4f}"
                )
                break

    logger.info(f"Training complete. Best model saved to: {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune InterviewBERTScorer on labelled QA pairs"
    )
    parser.add_argument("--train_path",  required=True, help="Path to train.jsonl")
    parser.add_argument("--val_path",    required=True, help="Path to val.jsonl")
    parser.add_argument("--output_dir",  default="models/", help="Where to save bert_scorer.pt")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--max_length",  type=int,   default=512)
    parser.add_argument("--grad_accum",  type=int,   default=1,
                        help="Gradient accumulation steps (use 2 if GPU VRAM < 8 GB)")
    parser.add_argument("--patience",    type=int,   default=2,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--seed",        type=int,   default=42)

    args = parser.parse_args()
    train(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        early_stop_patience=args.patience,
        gradient_accumulation_steps=args.grad_accum,
        seed=args.seed,
    )
