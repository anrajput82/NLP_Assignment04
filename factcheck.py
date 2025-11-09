# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str) -> str:
        """
        Run the NLI model on (premise, hypothesis) and return the predicted textual label.

        Steps:
          1. Encode (premise, hypothesis) with tokenizer.
          2. Run model (optionally on CUDA).
          3. Argmax over logits.
          4. Map id -> label using model.config.id2label when available,
             otherwise fall back to a standard mapping.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            if self.cuda:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits

            pred_id = int(torch.argmax(logits, dim=-1).item())

            id2label = getattr(self.model.config, "id2label", None)
            if isinstance(id2label, dict) and pred_id in id2label:
                label = id2label[pred_id]
            else:
                # Fallback ordering (many MNLI-style models use this or a variant)
                fallback = {
                    0: "entailment",
                    1: "neutral",
                    2: "contradiction",
                }
                label = fallback.get(pred_id, "neutral")

        # Clean up to reduce mem usage in Colab/autograder
        del inputs, outputs, logits
        gc.collect()

        return str(label)


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    """
    Baseline using word recall of the fact in passages.

    If the best (max) recall across passages exceeds threshold -> "S", else "NS".
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @staticmethod
    def _tokenize(text: str):
        # Simple, assignment-safe tokenizer
        return [t.lower() for t in re.findall(r"\w+", text)]

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = self._tokenize(fact)
        fact_vocab = set(fact_tokens)

        if not fact_vocab:
            return "NS"

        max_recall = 0.0
        for p in passages:
            passage_text = p.get("text", "")
            passage_tokens = self._tokenize(passage_text)
            passage_vocab = set(passage_tokens)

            if not passage_vocab:
                continue

            overlap = fact_vocab & passage_vocab
            recall = len(overlap) / float(len(fact_vocab))
            if recall > max_recall:
                max_recall = recall

        return "S" if max_recall >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    """
    Uses the NLI model:
      - For each passage: check entailment(passage, fact)
      - If any is entailment -> "S"
      - Otherwise -> "NS"
    """
    def __init__(self, ent_model: EntailmentModel):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        for p in passages:
            passage_text = p.get("text", "")
            if not passage_text:
                continue
            label = self.ent_model.check_entailment(passage_text, fact)
            if str(label).lower().startswith("entail"):
                return "S"
        return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    """
    Optional: dependency-based recall heuristic.
    """
    def __init__(self, threshold: float = 0.5):
        self.nlp = spacy.load('en_core_web_sm')
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_rels = self.get_dependencies(fact)
        if not fact_rels:
            return "NS"

        needed = len(fact_rels)
        for p in passages:
            passage_text = p.get("text", "")
            if not passage_text:
                continue
            passage_rels = self.get_dependencies(passage_text)
            if not passage_rels:
                continue
            overlap = fact_rels & passage_rels
            recall = len(overlap) / float(needed)
            if recall >= self.threshold:
                return "S"
        return "NS"

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        processed_sent = self.nlp(sent)
        relations = set()
        ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']

        for token in processed_sent:
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)

        return relations
