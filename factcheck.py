# factcheck.py
import torch
from typing import List
import numpy as np
import spacy
import gc
import re

# =========================
# Tokenization utilities
# =========================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "of", "at",
    "by", "for", "to", "in", "on", "from", "up", "down", "over", "under",
    "into", "about", "than", "then", "so", "such", "as", "is", "am", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "he", "she", "they", "them", "his", "her", "their", "we", "you",
    "i", "me", "my", "our", "us"
}


def bow_tokenize(text: str):
    """
    Tokenizer for the word-overlap model.
    - lowercase
    - keep alphanumeric tokens
    - DO NOT remove stopwords
    """
    return re.findall(r"\b\w+\b", (text or "").lower())


def content_tokenize(text: str):
    """
    Tokenizer for lexical pruning in entailment / parsing models.
    - lowercase
    - keep alphanumeric tokens
    - remove simple stopwords
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


# =========================
# Core container
# =========================

class FactExample:
    """
    fact:     fact string
    passages: list of {"title": str, "text": str}
    label:    "S", "NS", or "IR"
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return f"FactExample(fact={self.fact!r}, label={self.label!r}, passages={self.passages!r})"


# =========================
# Entailment model wrapper
# =========================

class EntailmentModel:
    """
    Thin wrapper around HuggingFace NLI model.

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
        Run NLI and return:
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

        del inputs, outputs, logits
        gc.collect()

        return {
            "entailment": float(probs[0]),
            "neutral": float(probs[1]),
            "contradiction": float(probs[2]),
        }


# =========================
# Base + simple baselines
# =========================

class FactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Use a subclass of FactChecker.")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return np.random.choice(["S", "NS"])


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


# =========================
# Part 1: Word Overlap
# =========================

class WordRecallThresholdFactChecker(FactChecker):
    """
    Bag-of-words overlap baseline.

    Implementation:
    - Case-insensitive unigram overlap (no stopword removal).
    - For each fact, compute recall of its tokens w.r.t. each passage.
    - Use the maximum recall over passages as score.
    - Predict "S" if score >= threshold, else "NS".
    """

    def __init__(self, threshold: float = 0.32):
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = bow_tokenize(fact)
        if not fact_tokens or not passages:
            return "NS"

        best_recall = 0.0
        for p in passages:
            text = p.get("text", "") or ""
            if not text:
                continue
            ctx_tokens = bow_tokenize(text)
            if not ctx_tokens:
                continue
            r = recall_overlap(fact_tokens, ctx_tokens)
            if r > best_recall:
                best_recall = r

        return "S" if best_recall >= self.threshold else "NS"


# =========================
# Part 2: Entailment-based
# =========================

class EntailmentFactChecker(FactChecker):
    """
    Entailment-based fact checker with lexical pruning.

    Steps:
      1. Coarse prune passages using content-word recall.
      2. Split remaining passages into sentences.
      3. Coarse prune sentences using content-word recall.
      4. Run NLI on (premise = sentence, hypothesis = fact).
      5. Use max entailment probability for decision.
    """

    def __init__(
        self,
        ent_model: EntailmentModel,
        overlap_prune_threshold: float = 0.05,
        sent_overlap_threshold: float = 0.08,
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

        fact_tokens = content_tokenize(fact)
        if not fact_tokens:
            return "NS"

        best_ent = 0.0
        best_margin = -1.0

        for p in passages:
            text = p.get("text", "") or ""
            if not text:
                continue

            # Passage-level pruning
            ptoks = content_tokenize(text)
            if ptoks:
                passage_recall = recall_overlap(fact_tokens, ptoks)
                if passage_recall < self.overlap_prune_threshold:
                    continue

            # Sentence-level pruning + NLI
            for sent in split_sentences(text):
                if not sent:
                    continue
                stoks = content_tokenize(sent)
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


# =========================
# Optional: Dependency-based
# =========================

class DependencyRecallThresholdFactChecker(FactChecker):
    """
    Optional syntactic-overlap model.

    Uses dependency triples (head, dep, child) and computes
    recall of fact's triples within passage triples.

    Robust to missing spaCy model: falls back to blank English.
    """

    def __init__(self, threshold: float = 0.30):
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
            text = p.get("text", "") or ""
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
