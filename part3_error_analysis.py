import torch
import numpy as np
import gc
import random
from typing import List
from dataclasses import dataclass

# ============================
# Data structure
# ============================

@dataclass
class FactExample:
    fact: str
    passages: List[dict]
    label: str  # "S" or "NS" (IR mapped to NS upstream)

# ============================
# Base checker
# ============================

class FactChecker:
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise NotImplementedError

# ============================
# Simple baselines
# ============================

class AlwaysEntailFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"

class RandomFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return random.choice(["S", "NS"])

# ============================
# Word-overlap checker (Part 1 style)
# ============================

def _tokenize(text: str):
    return [t.lower() for t in text.split()]

class WordRecallThresholdFactChecker(FactChecker):
    """Predict S if at least one passage has word-recall >= threshold."""
    def __init__(self, overlap_threshold: float = 0.3):
        self.threshold = overlap_threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_toks = set(_tokenize(fact))
        if not fact_toks:
            return "NS"
        for p in passages:
            p_toks = set(_tokenize(p.get("text", "")))
            if not p_toks:
                continue
            overlap = len(fact_toks & p_toks) / len(fact_toks)
            if overlap >= self.threshold:
                return "S"
        return "NS"

# ============================
# Entailment model wrapper (Part 2)
# ============================

class EntailmentModel:
    """Thin wrapper around a HuggingFace NLI model."""
    def __init__(self, model, tokenizer, use_cuda: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.use_cuda = bool(use_cuda)
        if self.use_cuda:
            self.model.to("cuda")
        self.model.eval()

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        """Return P(entailment | premise, hypothesis)."""
        if not premise or not hypothesis:
            return 0.0

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        if self.use_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

        # For MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli:
        # 0: entailment, 1: neutral, 2: contradiction
        ent_prob = float(probs[0].item())

        # cleanup
        del inputs, outputs, logits, probs
        gc.collect()

        return ent_prob

class EntailmentFactChecker(FactChecker):
    """Sentence-level max-entailment over retrieved passages."""
    def __init__(self, ent_model: EntailmentModel, entailment_threshold: float = 0.7):
        self.ent_model = ent_model
        self.threshold = entailment_threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        best = 0.0
        for p in passages:
            text = p.get("text", "")
            for sent in text.split("."):
                sent = sent.strip()
                if not sent:
                    continue
                prob = self.ent_model.entailment_prob(sent, fact)
                if prob > best:
                    best = prob
        return "S" if best >= self.threshold else "NS"

# ============================
# Optional: ensemble checker
# ============================

class EnsembleFactChecker(FactChecker):
    """Combine word-overlap and entailment: predict S if either says S."""
    def __init__(self, word_checker: WordRecallThresholdFactChecker,
                 entail_checker: EntailmentFactChecker):
        self.word_checker = word_checker
        self.entail_checker = entail_checker

    def predict(self, fact: str, passages: List[dict]) -> str:
        if self.word_checker.predict(fact, passages) == "S":
            return "S"
        if self.entail_checker.predict(fact, passages) == "S":
            return "S"
        return "NS"

# ============================
# Optional: parsing-based stub
# ============================

class DependencyRecallThresholdFactChecker(FactChecker):
    """Simple stub reusing word-overlap with a lower threshold."""
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        wr = WordRecallThresholdFactChecker(overlap_threshold=self.threshold)
        return wr.predict(fact, passages)
