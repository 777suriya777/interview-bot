"""
InterviewBERTScorer — BERT-base fine-tuned for interview answer evaluation.

Architecture:
  - Base: bert-base-uncased (110M params)
  - Input: [CLS] question [SEP] answer [SEP]  (max 512 tokens)
  - 4 independent linear heads on the [CLS] pooler output
  - Each head: Linear(768, 5) → 5-class softmax (scores 0–4)
  - During inference: argmax of logits → integer score per dimension

The 4 dimensions are scored independently so a model can learn, for example,
that an answer can be highly relevant (on-topic) but incomplete (missing depth).
"""
import torch
import torch.nn as nn
from transformers import BertModel


class InterviewBERTScorer(nn.Module):
    """
    BERT-base with 4 independent classification heads.
    Each head predicts one scoring dimension (0–4).
    """

    # Map head name → column index (used when loading label tensors)
    DIMENSIONS = ["content", "relevance", "completeness", "accuracy"]

    def __init__(self, pretrained_name: str = "bert-base-uncased"):
        super().__init__()

        # Load the pre-trained BERT encoder
        # During fine-tuning all BERT weights are updated (not frozen)
        self.bert = BertModel.from_pretrained(pretrained_name)
        hidden = self.bert.config.hidden_size  # 768 for bert-base

        # Dropout on the pooler output — prevents over-reliance on a single
        # [CLS] dimension and matches BERT fine-tuning best practice (p=0.1)
        self.dropout = nn.Dropout(0.1)

        # Four independent scoring heads — one per evaluation dimension.
        # Using separate Linear layers (not shared) lets each head develop
        # its own decision boundary for its specific rubric.
        self.head_content      = nn.Linear(hidden, 5)
        self.head_relevance    = nn.Linear(hidden, 5)
        self.head_completeness = nn.Linear(hidden, 5)
        self.head_accuracy     = nn.Linear(hidden, 5)

    def forward(
        self,
        input_ids: torch.Tensor,        # [batch, seq_len]
        attention_mask: torch.Tensor,   # [batch, seq_len]
        token_type_ids: torch.Tensor,   # [batch, seq_len]
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict of raw logits (not softmaxed) for each dimension.
        Shape of each tensor: [batch, 5]

        During training use CrossEntropyLoss directly on these logits.
        During inference use .argmax(dim=-1) to get the predicted score.
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # pooler_output is the [CLS] token representation after a linear + tanh.
        # Shape: [batch, 768]
        cls = self.dropout(out.pooler_output)

        return {
            "content":      self.head_content(cls),       # [batch, 5]
            "relevance":    self.head_relevance(cls),     # [batch, 5]
            "completeness": self.head_completeness(cls),  # [batch, 5]
            "accuracy":     self.head_accuracy(cls),      # [batch, 5]
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> dict[str, int]:
        """
        Convenience method for single-sample inference.
        Returns a dict of integer scores (0–4) per dimension.
        Does NOT compute gradients.
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
        return {
            dim: int(logits[dim].argmax(dim=-1).item())
            for dim in self.DIMENSIONS
        }


# Feedback text generation is in feedback.py (torch-free module)
# Import from there: from feedback import build_feedback_summary
