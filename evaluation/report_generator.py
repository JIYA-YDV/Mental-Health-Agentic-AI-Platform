# evaluation/report_generator.py
"""
Auto-generates a beautiful EVALUATION.md report from raw metrics.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def generate_markdown_report(
    metrics: Dict[str, Any],
    output_path: Path,
    *,
    model_name: str,
    dataset_name: str,
    sample_count: int,
    confusion_matrix_path: str = "../evaluation/results/confusion_matrix.png",
    per_class_chart_path: str = "../evaluation/results/per_class_f1.png",
) -> None:
    """Write a publication-quality evaluation report to output_path."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    macro_f1 = metrics["macro_f1"]
    weighted_f1 = metrics["weighted_f1"]
    accuracy = metrics["accuracy"]
    per_class = metrics["per_class"]
    confusion_summary = metrics.get("confusion_summary", "")

    # ── Build per-class table ────────────────────────────────────────
    per_class_rows = []
    for emotion, scores in sorted(per_class.items()):
        per_class_rows.append(
            f"| {emotion} "
            f"| {scores['precision']:.3f} "
            f"| {scores['recall']:.3f} "
            f"| {scores['f1']:.3f} "
            f"| {scores['support']} |"
        )
    per_class_table = "\n".join(per_class_rows)

    # ── F1 grade interpretation ──────────────────────────────────────
    if macro_f1 >= 0.85:
        grade, emoji = "Excellent", "🏆"
    elif macro_f1 >= 0.75:
        grade, emoji = "Strong", "✅"
    elif macro_f1 >= 0.65:
        grade, emoji = "Acceptable", "🟡"
    else:
        grade, emoji = "Needs improvement", "🔴"

    # ── Compose markdown ─────────────────────────────────────────────
    markdown = f"""# 📊 Model Evaluation Report

> Auto-generated on **{timestamp}** by `evaluation/benchmark.py`.
> Re-run with: `python -m evaluation.benchmark --samples {sample_count}`

---

## 🎯 Headline Metrics

| Metric | Value | Grade |
|--------|-------|-------|
| **Macro F1** | **{macro_f1:.3f}** | {emoji} {grade} |
| **Weighted F1** | **{weighted_f1:.3f}** | |
| **Accuracy** | **{accuracy:.3f}** | |
| **Sample Count** | {sample_count} | |

---

## 🧪 Methodology

- **Model:** `{model_name}`
- **Dataset:** `{dataset_name}` (test split)
- **Sample size:** {sample_count} examples
- **Evaluation labels:** {", ".join(metrics["labels_used"])}
- **Excluded labels:** `disgust`, `neutral` (not in dataset)
- **Label mapping:** `love` → `joy` (no direct dataset equivalent)
- **Random seed:** 42 (reproducible)

The evaluation uses **macro-averaged F1** as the headline metric to
weight all emotion classes equally regardless of frequency.

---

## 📈 Per-Class Performance

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|---------|
{per_class_table}

![Per-class F1 scores]({per_class_chart_path})

---

## 🧩 Confusion Matrix

![Confusion Matrix]({confusion_matrix_path})

{confusion_summary}

---

## 🚦 Crisis Detection Performance

A subset metric: how reliably does the model identify high-risk emotions
(sadness + fear, often correlated with depression/anxiety)?

| Metric | Value |
|--------|-------|
| **Combined Sadness + Fear Recall** | {metrics.get("crisis_recall", 0):.3f} |
| **Combined Sadness + Fear Precision** | {metrics.get("crisis_precision", 0):.3f} |
| **False Negative Rate** | {1 - metrics.get("crisis_recall", 0):.3f} |

> ⚠️ False negatives (missed crisis signals) are the most dangerous
> error type in mental health applications.

---

## 🔍 Reproducibility

```bash
# Recreate this exact report
python -m evaluation.benchmark --samples {sample_count} --seed 42

# View raw metrics
cat evaluation/results/metrics.json

# Inspect sample predictions
head evaluation/results/predictions.csv