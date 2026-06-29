# evaluation/benchmark.py
"""
ML evaluation harness for the emotion classifier.

Computes Precision, Recall, F1 (per-class + macro + weighted) on a held-out
test set, generates a confusion matrix PNG, and writes a Markdown report.

Usage:
    python -m evaluation.benchmark
    python -m evaluation.benchmark --samples 500
    python -m evaluation.benchmark --samples 2000 --output-dir ./my_results
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── FIX PATH CONFLICTS BEFORE LOADING HUGGINGFACE DATASETS ────────────
# Local folder name 'datasets/' shadows the third-party HuggingFace 'datasets' library.
# We temporarily remove the project root directory from sys.path to force 
# Python to resolve the actual pip-installed library.
project_root = str(Path(__file__).parent.parent)
original_path = sys.path.copy()

try:
    if project_root in sys.path:
        sys.path.remove(project_root)
    if "" in sys.path:
        sys.path.remove("")
    if "." in sys.path:
        sys.path.remove(".")
    
    from datasets import load_dataset
finally:
    # Safely restore paths so local backend imports work perfectly
    sys.path = original_path

# Ensure backend is cleanly importable when running from project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from evaluation.label_mapper import (
    DATASET_LABELS,
    dataset_label_to_model_label,
    get_eval_label_set,
    normalize_model_prediction,
)
from evaluation.report_generator import generate_markdown_report


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_DATASET = "dair-ai/emotion"
DEFAULT_SAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"
DEFAULT_REPORT_PATH = Path(__file__).parent.parent / "docs" / "EVALUATION.md"


# ──────────────────────────────────────────────────────────────────────
# Pretty terminal output
# ──────────────────────────────────────────────────────────────────────
def _print_header(text: str) -> None:
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def _print_metric(label: str, value: float, fmt: str = ".4f") -> None:
    print(f"  {label:.<40} {value:{fmt}}")


# ──────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────
def load_eval_dataset(
    dataset_name: str, sample_count: int, seed: int
) -> Tuple[List[str], List[str]]:
    """
    Load evaluation samples from HuggingFace datasets.

    Returns:
        (texts, ground_truth_labels)
    """
    _print_header(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="test")
    print(f"  Full test set size: {len(ds)}")

    # Deterministic shuffle + take
    random.seed(seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:sample_count]

    texts: List[str] = []
    labels: List[str] = []
    for i in indices:
        sample = ds[i]
        text = sample["text"].strip()
        if not text:
            continue
        label_id = sample["label"]
        try:
            canonical_label = dataset_label_to_model_label(label_id)
        except ValueError:
            continue
        texts.append(text)
        labels.append(canonical_label)

    print(f"  Loaded {len(texts)} evaluation samples")
    print(f"  Label distribution: {dict(Counter(labels))}")
    return texts, labels


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────
def run_inference(texts: List[str]) -> List[str]:
    """Run the production classifier on every text. Returns predictions."""
    _print_header("Running model inference (this may take a few minutes on CPU)")

    from backend.models.classifier import emotion_classifier
    if not emotion_classifier.is_loaded:
        print("  Loading model...")
        emotion_classifier.load()

    predictions: List[str] = []
    start = time.time()

    for text in tqdm(texts, desc="  Predicting", unit="sample", ncols=80):
        try:
            result = emotion_classifier.predict(text)
            predictions.append(result["emotion"].lower().strip())
        except Exception as e:
            print(f"\n  ⚠️  Inference failed on sample: {e}")
            predictions.append("error")

    elapsed = time.time() - start
    print(f"\n  Total inference time: {elapsed:.1f}s "
          f"({elapsed / len(texts) * 1000:.1f}ms/sample)")
    return predictions


# ──────────────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────────────
def compute_metrics(
    ground_truth: List[str], predictions: List[str]
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    _print_header("Computing metrics")

    label_set = get_eval_label_set()

    # Filter out samples where model predicted excluded labels
    filtered_gt: List[str] = []
    filtered_pred: List[str] = []
    excluded_count = 0
    for gt, pred in zip(ground_truth, predictions):
        norm_pred = normalize_model_prediction(pred)
        if norm_pred is None:
            # Model predicted disgust/neutral — count as wrong (misclassification)
            filtered_gt.append(gt)
            filtered_pred.append("__excluded__")
            excluded_count += 1
        else:
            filtered_gt.append(gt)
            filtered_pred.append(norm_pred)

    print(f"  Samples where model predicted excluded label: {excluded_count}")

    # Core metrics — only over the label_set
    accuracy = accuracy_score(filtered_gt, filtered_pred)
    macro_f1 = f1_score(
        filtered_gt, filtered_pred,
        labels=label_set, average="macro", zero_division=0,
    )
    weighted_f1 = f1_score(
        filtered_gt, filtered_pred,
        labels=label_set, average="weighted", zero_division=0,
    )

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        filtered_gt, filtered_pred,
        labels=label_set, zero_division=0,
    )
    per_class = {
        label_set[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(label_set))
    }

    # Crisis-specific metrics (sadness + fear treated as crisis signals)
    crisis_labels = {"sadness", "fear"}
    crisis_gt = [1 if g in crisis_labels else 0 for g in filtered_gt]
    crisis_pred = [1 if p in crisis_labels else 0 for p in filtered_pred]
    crisis_p, crisis_r, crisis_f, _ = precision_recall_fscore_support(
        crisis_gt, crisis_pred, average="binary", zero_division=0
    )

    _print_metric("Accuracy", accuracy)
    _print_metric("Macro F1", macro_f1)
    _print_metric("Weighted F1", weighted_f1)
    _print_metric("Crisis Recall (sadness + fear)", crisis_r)
    _print_metric("Crisis Precision (sadness + fear)", crisis_p)

    # Most-confused pair
    cm = confusion_matrix(filtered_gt, filtered_pred, labels=label_set)
    confusion_summary = _analyze_confusion(cm, label_set)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "crisis_precision": float(crisis_p),
        "crisis_recall": float(crisis_r),
        "crisis_f1": float(crisis_f),
        "labels_used": label_set,
        "excluded_predictions": excluded_count,
        "confusion_matrix": cm.tolist(),
        "confusion_summary": confusion_summary,
    }


def _analyze_confusion(cm: np.ndarray, labels: List[str]) -> str:
    """Find the most-confused class pair and produce a markdown blurb."""
    np.fill_diagonal(cm, 0)  # ignore correct predictions
    if cm.sum() == 0:
        return "✅ No confusions detected (perfect classification)."

    flat_idx = cm.argmax()
    i, j = divmod(flat_idx, cm.shape[1])
    count = int(cm[i, j])
    total_errors = int(cm.sum())
    pct = count / total_errors * 100 if total_errors else 0

    return (
        f"**Most confused pair:** `{labels[i]}` misclassified as "
        f"`{labels[j]}` ({count} times, {pct:.1f}% of all errors)."
    )


# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    cm_data: List[List[int]],
    labels: List[str],
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    cm = np.array(cm_data)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix — Emotion Classifier", fontsize=14, pad=15)
    plt.ylabel("Ground Truth", fontsize=11)
    plt.xlabel("Predicted", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved confusion matrix → {output_path}")


def plot_per_class_f1(
    per_class: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Save a bar chart of per-class F1 scores."""
    labels = sorted(per_class.keys())
    f1_scores = [per_class[l]["f1"] for l in labels]
    precisions = [per_class[l]["precision"] for l in labels]
    recalls = [per_class[l]["recall"] for l in labels]

    x = np.arange(len(labels))
    width = 0.27

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, precisions, width, label="Precision", color="#4C72B0")
    plt.bar(x,         recalls,    width, label="Recall",    color="#55A868")
    plt.bar(x + width, f1_scores,  width, label="F1",        color="#C44E52")

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-Class Precision / Recall / F1", fontsize=13, pad=10)
    plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved per-class chart → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────
def save_predictions_csv(
    texts: List[str],
    ground_truth: List[str],
    predictions: List[str],
    output_path: Path,
) -> None:
    """Save per-sample predictions for debugging / inspection."""
    df = pd.DataFrame({
        "text": texts,
        "ground_truth": ground_truth,
        "predicted": predictions,
        "correct": [g == p for g, p in zip(ground_truth, predictions)],
    })
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"  ✅ Saved predictions CSV → {output_path}")


def save_metrics_json(metrics: Dict[str, Any], output_path: Path) -> None:
    """Save raw metrics JSON for programmatic access."""
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ Saved metrics JSON → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the emotion classifier on a labeled test set."
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="Number of test samples to evaluate (default: 1000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="HuggingFace dataset name")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name (for report metadata only)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save results")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH,
                        help="Path to write EVALUATION.md")
    args = parser.parse_args()

    _print_header("🧪 Mental Health Agentic AI — Model Evaluation")
    print(f"  Model:        {args.model}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Samples:      {args.samples}")
    print(f"  Seed:         {args.seed}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Report:       {args.report_path}")

    # ── 1. Load dataset
    texts, ground_truth = load_eval_dataset(
        args.dataset, args.samples, args.seed
    )

    # ── 2. Run inference
    predictions = run_inference(texts)

    # ── 3. Compute metrics
    metrics = compute_metrics(ground_truth, predictions)

    # ── 4. Save artifacts
    _print_header("Saving artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cm_path = args.output_dir / "confusion_matrix.png"
    chart_path = args.output_dir / "per_class_f1.png"
    json_path = args.output_dir / "metrics.json"
    csv_path = args.output_dir / "predictions.csv"

    plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels_used"], cm_path)
    plot_per_class_f1(metrics["per_class"], chart_path)
    save_metrics_json(metrics, json_path)
    save_predictions_csv(texts, ground_truth, predictions, csv_path)

    # ── 5. Generate report
    _print_header("Generating markdown report")
    generate_markdown_report(
        metrics,
        args.report_path,
        model_name=args.model,
        dataset_name=args.dataset,
        sample_count=len(texts),
    )
    print(f"  ✅ Saved report → {args.report_path}")

    # ── 6. Final summary
    _print_header("✨ Evaluation Complete")
    _print_metric("Macro F1", metrics["macro_f1"])
    _print_metric("Weighted F1", metrics["weighted_f1"])
    _print_metric("Accuracy", metrics["accuracy"])
    _print_metric("Crisis Recall", metrics["crisis_recall"])
    print()
    print(f"  📊 Open the report:   {args.report_path}")
    print(f"  🖼️  Confusion matrix:  {cm_path}")
    print()


if __name__ == "__main__":
    main()