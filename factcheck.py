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
        # Simple random baseline
        return np.random.choice(["S", "NS"])


class AlwaysSupportedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class AlwaysNotSupportedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "NS"


# ============================
# Word Overlap Threshold Checker
# ============================
class WordRecallThresholdFactChecker(FactChecker):
    """
    Uses preprocessed word overlap (unigram + bigram recall) between fact and passages.
    Classifies as "S" if max overlap score >= threshold, else "NS".
    """

    def __init__(self, overlap_threshold: float = 0.18):
        self.overlap_threshold = overlap_threshold
        self.lemmatizer = WordNetLemmatizer()
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

        # Weighted: bigram more important than unigram
        return 0.6 * bigram_overlap + 0.4 * unigram_overlap

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact = (fact or "").strip()
        if not fact or not passages:
            return "NS"

        fact_tokens = self.preprocess(fact)
        if not fact_tokens:
            return "NS"

        max_score = 0.0
        for p in passages:
            text = p.get("text", "") or ""
            if not text:
                continue
            passage_tokens = self.preprocess(text)
            if not passage_tokens:
                continue
            score = self.word_overlap_score(fact_tokens, passage_tokens)
            if score > max_score:
                max_score = score

        return "S" if max_score >= self.overlap_threshold else "NS"


# ============================
# Shared helpers for entailment
# ============================
# Content-word stoplist for pruning
_ENT_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "for", "of", "and", "or", "to", "is", "are",
    "was", "were", "be", "been", "being", "by", "with", "as", "that", "this", "it",
    "from", "but", "about", "into", "over", "after", "such", "no", "nor", "not",
    "too", "very", "can", "will", "just", "than", "then", "so", "if", "there",
    "their", "its", "also", "any", "all", "some", "one", "two", "three", "many",
    "most", "other", "more", "these", "those", "he", "she", "they", "them", "his",
    "her", "their", "we", "you", "i", "me", "my", "our", "us"
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


# ============================
# Entailment Model Wrapper
# ============================
class EntailmentModel:
    """
    Wraps externally provided NLI model.
    Assumes label order: [entailment, neutral, contradiction].
    Returns a single float: P(entailment).
    """

    def __init__(self, model, tokenizer, use_cuda: bool = False):
        # Store tokenizer
        self.tokenizer = tokenizer
        # Select device based on flag and availability
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # Move model to device and set eval mode
        self.model = model.to(self.device)
        self.model.eval()

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """Return entailment probability P(entailment | premise, hypothesis)."""
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
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

            # Index 0 is assumed to be 'entailment'
            ent_prob = float(probs[0].item())

            # Cleanup for memory safety
            del inputs, outputs, logits, probs
            gc.collect()

            return ent_prob


# ============================
# Entailment-based Checker
# ============================
class EntailmentFactChecker(FactChecker):
    """
    Entailment-based checker with lexical pruning.
    - Prune passages and sentences by content-word recall.
    - Run NLI on (sentence, fact) pairs that pass thresholds.
    - Use max en
