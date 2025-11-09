import json
import argparse
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from factcheck import FactExample, EntailmentModel, EntailmentFactChecker


# ==========================
# Helpers to load data
# ==========================

def load_passages(passages_path: str) -> Dict[str, List[dict]]:
    """
    Load mapping from fact sentence -> retrieved passages.

    Assumes each line in passages_bm25_ChatGPT_humfacts.jsonl has:
      {
        "name": ...,
        "sent": <fact-string>,
        "passages": [ { "title": ..., "text": ... }, ... ]
      }
    """
    fact_to_passages = {}
    with open(passages_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sent = entry.get("sent")
            if not sent:
                continue
            fact_to_passages[sent] = entry.get("passages", [])
    return fact_to_passages


def create_fact_examples(
    labeled_facts_path: str,
    fact_to_passage_dict: Dict[str, List[dict]],
) -> List[FactExample]:
    """
    Build FactExample objects from the human-labeled dev file.

    We use `human-atomic-facts` and their labels.
    IR is treated as NS for evaluation, same as in the assignment code.
    """
    examples: List[FactExample] = []

    with open(labeled_facts_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    for line in all_lines:
        if not line.strip():
            continue
        item = json.loads(line)

        annotations = item.get("annotations")
        if not annotations:
            continue

        for ann in annotations:
            human_facts = ann.get("human-atomic-facts")
            if not human_facts:
                continue

            for fact_obj in human_facts:
                fact_text = fact_obj.get("text")
                gold_label = fact_obj.get("label")

                if fact_text is None or gold_label is None:
                    continue

                # Map IR -> NS as in the evaluation protocol
                if gold_label == "IR":
                    gold_label = "NS"

                passages = fact_to_passage_dict.get(fact_text, [])

                examples.append(FactExample(fact_text, passages, gold_label))

    return examples


# ==========================
# Run entailment model & collect errors
# ==========================

def run_entailment_and_collect_errors(
    examples: List[FactExample],
    use_cuda: bool = False,
):
    """
    Uses the Part 2 entailment-based checker on all examples.
    Returns:
      - confusion matrix
      - list of false positives
      - list of false negatives
    """
    # Same model as in the assignment starter (Part 2)
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if use_cuda:
        nli_model.to("cuda")

    # IMPORTANT: matches your updated EntailmentModel signature from factcheck.py
    ent_model = EntailmentModel(nli_model, tokenizer)

    # Use your optimized EntailmentFactChecker from factcheck.py
    fact_checker = EntailmentFactChecker(ent_model)

    confusion = {
        "S": {"S": 0, "NS": 0},
        "NS": {"S": 0, "NS": 0},
    }

    false_positives = []  # gold NS, pred S
    false_negatives = []  # gold S, pred NS

    for ex in examples:
        gold = ex.label
        if gold not in ("S", "NS"):
            # We already mapped IR -> NS earlier, so this is just a safeguard
            continue

        pred = fact_checker.predict(ex.fact, ex.passages)
        if pred not in ("S", "NS"):
            # Safety: treat anything weird as NS
            pred = "NS"

        confusion[gold][pred] += 1

        if gold == "NS" and pred == "S":
            false_positives.append(ex)
        elif gold == "S" and pred == "NS":
            false_negatives.append(ex)

    return confusion, false_positives, false_negatives


# ==========================
# Pretty-print helpers
# ==========================

def passage_snippet(passages: List[dict], max_len: int = 280) -> str:
    """
    Extract a short evidence snippet from the first passage.
    This is just to help manual inspection for Part 3.
    """
    if not passages:
        return "(no retrieved passages)"

    text = passages[0].get("text", "") or ""
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + " ..."
    return text


def print_summary_and_samples(
    confusion,
    false_positives: List[FactExample],
    false_negatives: List[FactExample],
    max_samples: int = 10,
):
    print("=== Confusion Matrix (S / NS, with IR mapped to NS) ===")
    print(f"Gold S, Pred S : {confusion['S']['S']}")
    print(f"Gold S, Pred NS: {confusion['S']['NS']}")
    print(f"Gold NS, Pred S: {confusion['NS']['S']}")
    print(f"Gold NS, Pred NS: {confusion['NS']['NS']}")
    print()

    # Limit to at most 10 each as required
    fp_samples = false_positives[:max_samples]
    fn_samples = false_negatives[:max_samples]

    print(f"=== False Positives (pred=S, gold=NS) [showing {len(fp_samples)}] ===")
    for i, ex in enumerate(fp_samples, 1):
        print(f"\n[FP {i}]")
        print(f"(a) Fact: {ex.fact}")
        print(f"(b) Gold label: {ex.label}")
        print(f"(c) Predicted label: S")
        print(f"(evidence) {passage_snippet(ex.passages)}")

    print("\n=== False Negatives (pred=NS, gold=S) [showing {len(fn_samples)}] ===")
    for i, ex in enumerate(fn_samples, 1):
        print(f"\n[FN {i}]")
        print(f"(a) Fact: {ex.fact}")
        print(f"(b) Gold label: {ex.label}")
        print(f"(c) Predicted label: NS")
        print(f"(evidence) {passage_snippet(ex.passages)}")

    print("\n=== Next Steps for Your Writeup ===")
    print("1. Manually read the above FP/FN examples.")
    print("2. Define 2-4 fine-grained error categories based on recurring patterns")
    print("   (e.g., lexical-overlap hallucinations, paraphrase misses, retrieval/entity issues).")
    print("3. Assign each of the ~20 examples to one of these categories.")
    print("4. Count how many errors fall into each category -> report these statistics.")
    print("5. Pick 3 illustrative examples and, for each:")
    print("   (a) include the fact text,")
    print("   (b) gold label,")
    print("   (c) model prediction,")
    print("   (d) chosen fine-grained error label,")
    print("   (e) 1-3 sentences explaining why it fits that category.")


# ==========================
# Main
# ==========================

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels_path",
        type=str,
        default="dev_labeled_ChatGPT.jsonl",
        help="Path to dev_labeled_ChatGPT.jsonl",
    )
    parser.add_argument(
        "--passages_path",
        type=str,
        default="passages_bm25_ChatGPT_humfacts.jsonl",
        help="Path to passages_bm25_ChatGPT_humfacts.jsonl",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Set to 1 if you want to move the NLI model to GPU manually.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    fact_to_passages = load_passages(args.passages_path)
    examples = create_fact_examples(args.labels_path, fact_to_passages)

    confusion, fps, fns = run_entailment_and_collect_errors(
        examples,
        use_cuda=bool(args.cuda),
    )

    print_summary_and_samples(confusion, fps, fns, max_samples=10)
