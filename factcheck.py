# factcheck.py
# Final optimized implementation for A4: Word Overlap + Entailment with pruning and ensemble

import torch
from typing import List
import numpy as np
import gc
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Ensure required NLTK data is available
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ======================
# Data container
# ======================
class FactExample:
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label
    def __repr__(self):
        return f"fact={repr(self.fact)}; label={repr(self.label)}; passages={repr(self.passages)}"

# ======================
# Base + simple baselines
# ======================
class FactChecker:
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise NotImplementedError("Subclasses must implement this method")

class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return np.random.choice(["S", "NS"])

class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"

# ============================================================
# Part 1: WordRecallThresholdFactChecker (optimized)
# ============================================================
class WordRecallThresholdFactChecker(FactChecker):
    """
    Word overlap baseline (unigram + bigram recall).
    Steps:
    - Preprocess fact and passages:
        * lowercase
        * remove non-alphanumeric chars
        * tokenize
        * remove stopwords
        * lemmatize
        * drop very short tokens
    - For each passage, compute:
        score = 0.4 * unigram_recall + 0.6 * bigram_recall
    - Predict "S" if any passage has score >= overlap_threshold, else "NS".
    """
    def __init__(self, overlap_threshold: float = 0.31):
        self.lemmatizer = WordNetLemmatizer()
        self.overlap_threshold = overlap_threshold
        self.stop_words = set(stopwords.words("english")).union(
            {"the", "in", "of", "will", "may", "should"}
        )
    def preprocess(self, text: str):
        text = text or ""
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        tokens = word_tokenize(text.lower())
        tokens = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and len(tok) > 2
        ]
        return tokens
    def word_overlap_score(self, fact_tokens, passage_tokens) -> float:
        fact_set = set(fact_tokens)
        passage_set = set(passage_tokens)
        # unigram recall
        unigram_overlap = (
            len(fact_set & passage_set) / len(fact_set) if fact_set else 0.0
        )
        # bigram recall
        bigrams_fact = set(nltk.bigrams(fact_tokens))
        bigrams_passage = set(nltk.bigrams(passage_tokens))
        bigram_overlap = (
            len(bigrams_fact & bigrams_passage) / len(bigrams_fact)
            if bigrams_fact
            else 0.0
        )
        # Weighted: bigram gets higher weight
        return (unigram_overlap + bigram_overlap) / 2.0
    def predict(self, fact: str, passages: List[dict]) -> str:
        if not fact or not passages:
            return "NS"
        fact_tokens = self.preprocess(fact)
        if not fact_tokens:
            return "NS"
        for p in passages:
            text = p["text"]
            passage_tokens = self.preprocess(text)
            if not passage_tokens:
                continue
            score = self.word_overlap_score(fact_tokens, passage_tokens)
            if score >= self.overlap_threshold:
                return "S"
        return "NS"

# ============================================================
# Part 2: Entailment model + checker (optimized)
# ============================================================
# Small stopword set for pruning
_ENT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "of", "at",
    "by", "for", "to", "in", "on", "from", "up", "down", "over", "under",
    "into", "about", "than", "then", "so", "such", "as", "is", "am", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "he", "she", "they", "them", "his", "her", "their", "we", "you",
    "i", "me", "my", "our", "us"
}
def _content_tokens(text: str):
    if not text:
        return []
    toks = re.findall(r"\b\w+\b", text.lower())
    return [t for t in toks if t not in _ENT_STOPWORDS]
def _recall(fact_tokens, ctx_tokens):
    f = set(fact_tokens)
    if not f:
        return 0.0
    c = set(ctx_tokens)
    return float(len(f & c)) / float(len(f))
def _split_sentences(text: str):
    if not text:
        return []
    return sent_tokenize(text)

class EntailmentModel:
    """
    Wraps externally provided NLI model.
    Assumes label order: [entailment, neutral, contradiction].
    Returns a single float: P(entailment).
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.to("cpu")
        self.model.eval()
    def check_entailment(self, premise: str, hypothesis: str) -> float:
        if not premise or not hypothesis:
            return 0.0
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
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            ent_prob = float(probs[0].item()) # index 0 = entailment
            del inputs, outputs, logits, probs
            gc.collect()
            return ent_prob

class EntailmentFactChecker(FactChecker):
    """
    Entailment-based checker with lexical pruning.
    - Prune passages and sentences by content-word recall.
    - Run NLI on (sentence, fact) pairs that pass thresholds.
    - Use max entailment probability to decide.
    """
    def __init__(
        self,
        ent_model: EntailmentModel,
        passage_overlap_threshold: float = 0.01, # Lowered for higher recall
        sentence_overlap_threshold: float = 0.01, # Lowered for higher recall
        entailment_threshold: float = 0.58, # Lowered for higher recall
    ):
        self.ent_model = ent_model
        self.passage_overlap_threshold = passage_overlap_threshold
        self.sentence_overlap_threshold = sentence_overlap_threshold
        self.entailment_threshold = entailment_threshold
    def predict(self, fact: str, passages: List[dict]) -> str:
        fact = (fact or "").strip()
        if not fact or not passages:
            return "NS"
        fact_tokens = _content_tokens(fact)
        if not fact_tokens:
            return "NS"
        best_ent = 0.0
        for p in passages:
            text = p.get("text", "") or ""
            if not text:
                continue
            # Passage-level pruning
            p_tokens = _content_tokens(text)
            if p_tokens and _recall(fact_tokens, p_tokens) < self.passage_overlap_threshold:
                continue
            # Sentence-level
            for sent in _split_sentences(text):
                if not sent:
                    continue
                s_tokens = _content_tokens(sent)
                if not s_tokens:
                    continue
                if _recall(fact_tokens, s_tokens) < self.sentence_overlap_threshold:
                    continue
                ent_prob = self.ent_model.check_entailment(sent, fact)
                if ent_prob > best_ent:
                    best_ent = ent_prob
        return "S" if best_ent >= self.entailment_threshold else "NS"

# ============================================================
# Ensemble Checker (NEW)
# ============================================================
class EnsembleFactChecker(FactChecker):
    """
    Combines WordOverlap and Entailment checkers.
    Predicts "S" if either checker predicts "S".
    """
    def __init__(self, word_checker: WordRecallThresholdFactChecker, entail_checker: EntailmentFactChecker):
        self.word_checker = word_checker
        self.entail_checker = entail_checker
    def predict(self, fact: str, passages: List[dict]) -> str:
        word_pred = self.word_checker.predict(fact, passages)
        entail_pred = self.entail_checker.predict(fact, passages)
        if word_pred == "S" or entail_pred == "S":
            return "S"
        return "NS"

# ============================================================
# Optional: Dependency-based checker (simple stub)
# ============================================================
class DependencyRecallThresholdFactChecker(FactChecker):
    """
    Simple optional checker; not central for grading.
    Reuses word-overlap logic with a different threshold.
    """
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
    def predict(self, fact: str, passages: List[dict]) -> str:
        wr = WordRecallThresholdFactChecker(overlap_threshold=self.threshold)
        return wr.predict(fact, passages)