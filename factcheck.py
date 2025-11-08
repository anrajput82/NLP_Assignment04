# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re

# -------------------------
# Lightweight utilities
# -------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "of", "at",
    "by", "for", "to", "in", "on", "from", "up", "down", "over", "under",
    "into", "about", "than", "then", "so", "such", "as", "is", "am", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "he", "she", "they", "them", "his", "her", "their", "we", "you",
    "i", "me", "my", "our", "us"
}


def simple_tokenize(text: str):
    """
    Very lightweight tokenizer:
    - lowercase
    - keep alphanumeric word pieces
    - drop common stopwords
    """
    tokens = re.findall(r"\b\w+\b", (text or "").lower())
    return [t for t in tokens if t not in STOPWORDS]


def recall_overlap(fact_tokens, ctx_tokens):
    """
    Word recall of fact w.r.t. context:
        | overlap(fact, ctx) | / | unique(fact) |
    """
    fact_set = set(fact_tokens)
    if not fact_set:
        return 0.0
    ctx_set = set(ctx_tokens)
    overlap = fact_set & ctx_set
    return float(len(overlap)) / float(len(fact_set))


def split_sentences(text: str):
    """
    Simple sentence splitter for noisy passages.
    """
    if not text:
        return []
    parts = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in parts if s.strip()]


# -------------------------
# Core data structure
# -------------------------

class FactExample:
    """
    Container for one fact-checking instance.

    fact:     atomic fact text
    passages: list of {"title": str, "text": str}
    label:    "S", "NS", or "IR"
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return f"FactExample(fact={self.fact!r}, label={self.label!r}, passages={self.passages!r})"


# -------------------------
# Entailment model wrapper
# -------------------------

class EntailmentModel:
    """
    Thin wrapper around a HuggingFace NLI model.

    Assumes label order: [entailment, neutral, contradiction]
    (true for MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.to("cpu")
        self.model.eval()

    def check_entailment(self, premise: str, hypothesis: str):
        """
        Run the NLI model and return:
            {
                "entailment": p_ent,
                "neutral": p_neu,
                "contradiction": p_con
            }
        """
        if not premise or not hypothesis:
            return {
                "entailment": 0.0,
                "neutral": 1.0,
                "contradiction": 0.0,
            }

        with torch.no_grad():
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # Avoid memory buildup
        del inputs, outputs, logits
        gc.collect()

        return {
            "entailment": float(probs[0]),
            "neutral": float(probs[1]),
            "contradiction": float(probs[2]),
        }


# -------------------------
# Base + baselines
# -------------------------

class FactChecker(object):
    """
    Base class for all fact checkers.
    """
    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Must return "S" or "NS".
        """
        raise Exception("Use a subclass of FactChecker.")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return np.random.choice(["S", "NS"])


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


# -------------------------
# Part 1: Word overlap
# -------------------------

class WordRecallThresholdFactChecker(FactChecker):
    """
    Word-overlap baseline.

    Strategy:
    - Tokenize fact and each passage.
    - Compute recall of fact tokens w.r.t. passage tokens.
    - Take max recall over passages.
    - Predict "S" if max_recall >= threshold else "NS".
    """

    def __init__(self, threshold: float = 0.35):
        # This threshold can be tuned on dev for >=75% accuracy.
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = simple_tokenize(fact)
        if not fact_tokens or not passages:
            return "NS"

        best_recall = 0.0
        for p in passages:
            text = p.get("text", "")
            if not text:
                continue
            ctx_tokens = simple_tokenize(text)
            if not ctx_tokens:
                continue
            r = recall_overlap(fact_tokens, ctx_tokens)
            if r > best_recall:
                best_recall = r

        return "S" if best_recall >= self.threshold else "NS"


# -------------------------
# Part 2: Entailment
# -------------------------

class EntailmentFactChecker(FactChecker):
    """
    Entailment-based fact checker with lexical pruning.

    Steps:
    1. Coarse prune passages by low word overlap.
    2. Split remaining passages into sentences.
    3. Coarse prune sentences by low overlap.
    4. Run NLI on (premise = sentence, hypothesis = fact).
    5. Take max entailment probability.
    6. Predict "S" if:
         best_ent >= entailment_threshold
         AND best_ent >= contradiction probability.
    """

    def __init__(
        self,
        ent_model: EntailmentModel,
        overlap_prune_threshold: float = 0.08,
        sent_overlap_threshold: float = 0.10,
        entailment_threshold: float = 0.60,
    ):
        self.ent_model = ent_model
        self.overlap_prune_threshold = overlap_prune_threshold
        self.sent_overlap_threshold = sent_overlap_threshold
        self.entailment_threshold = entailment_threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact = (fact or "").strip()
        if not fact or not passages:
            return "NS"

        fact_tokens = simple_tokenize(fact)
        if not fact_tokens:
            return "NS"

        best_ent = 0.0
        best_margin = -1.0

        for p in passages:
            text = p.get("text", "")
            if not text:
                continue

            # Passage-level pruning
            ptoks = simple_tokenize(text)
            if ptoks:
                passage_recall = recall_overlap(fact_tokens, ptoks)
                if passage_recall < self.overlap_prune_threshold:
                    continue

            # Sentence-level
            for sent in split_sentences(text):
                if not sent:
                    continue
                stoks = simple_tokenize(sent)
                if not stoks:
                    continue

                sent_recall = recall_overlap(fact_tokens, stoks)
                if sent_recall < self.sent_overlap_threshold:
                    continue

                scores = self.ent_model.check_entailment(sent, fact)
                ent = scores["entailment"]
                con = scores["contradiction"]
                margin = ent - con

                if ent > best_ent:
                    best_ent = ent
                    best_margin = margin

        if best_ent >= self.entailment_threshold and best_margin >= 0.0:
            return "S"
        else:
            return "NS"


# -------------------------
# Optional: Dependency-based
# -------------------------

class DependencyRecallThresholdFactChecker(FactChecker):
    """
    Optional: dependency-based overlap checker.

    Uses dependency triples (head, dep, child).
    Robust to missing spaCy model:
      - tries 'en_core_web_sm'
      - falls back to blank('en') (then returns empty deps).
    """

    def __init__(self, threshold: float = 0.3):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = spacy.blank("en")
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact = (fact or "").strip()
        if not fact or not passages:
            return "NS"

        fact_rels = self.get_dependencies(fact)
        if not fact_rels:
            return "NS"

        best_recall = 0.0
        for p in passages:
            text = p.get("text", "")
            if not text:
                continue
            ctx_rels = self.get_dependencies(text)
            if not ctx_rels:
                continue
            overlap = fact_rels & ctx_rels
            recall = float(len(overlap)) / float(len(fact_rels))
            if recall > best_recall:
                best_recall = recall

        return "S" if best_recall >= self.threshold else "NS"

    def get_dependencies(self, sent: str):
        """
        Extract (head, dep, child) relations, filtering out
        uninformative dependency labels.
        """
        sent = (sent or "").strip()
        if not sent:
            return set()

        if "parser" not in self.nlp.pipe_names:
            return set()

        doc = self.nlp(sent)
        relations = set()
        ignore_dep = {
            "punct", "ROOT", "root", "det", "case",
            "aux", "auxpass", "dep", "cop", "mark"
        }

        for token in doc:
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            head = token.head.lemma_ if token.head.pos_ == "VERB" else token.head.text
            child = token.lemma_ if token.pos_ == "VERB" else token.text
            relations.add((head, token.dep_, child))

        return relations

