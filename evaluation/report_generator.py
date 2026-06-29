# evaluation/report_generator.py
"""
Auto-generates a publication-quality EVALUATION.md report from raw metrics.

Uses string concatenation instead of one large f-string to avoid parser
issues with nested triple-backtick code blocks.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _grade_for_f1(macro_f1: float) -> tuple:
    """Map macro F1 to a (grade_text, emoji) tuple."""
    if macro_f1 >= 0.85:
        return "Excellent", "🏆"
    if macro_f1 >= 0.75:
        return "Strong", "✅"
    if macro_f1 >= 0.65:
        return "Acceptable", "🟡"
    return "Needs improvement", "🔴"


def _build_per_class_table(per_class: Dict[str, Dict[str, float]]) -> str:
    """Build the markdown table rows for per-class metrics."""
    rows = []
    for emotion in sorted(per_class.keys()):
        scores = per_class[emotion]
        row = "| {emotion} | {p:.3f} | {r:.3f} | {f:.3f} | {s} |".format(
            emotion=emotion,
            p=scores["precision"],
            r=scores["recall"],
            f=scores["f1"],
            s=scores["support"],
        )
        rows.append(row)
    return "\n".join(rows)


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

    # ── Extract values ────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    macro_f1 = metrics["macro_f1"]
    weighted_f1 = metrics["weighted_f1"]
    accuracy = metrics["accuracy"]
    per_class = metrics["per_class"]
    confusion_summary = metrics.get("confusion_summary", "")
    crisis_recall = metrics.get("crisis_recall", 0)
    crisis_precision = metrics.get("crisis_precision", 0)
    labels_used = metrics["labels_used"]
    grade, emoji = _grade_for_f1(macro_f1)
    per_class_table = _build_per_class_table(per_class)
    margin = 1.96 * (0.5 / (sample_count ** 0.5)) * 100 if sample_count else 0

    # ── Build report in sections (no nested triple quotes) ────────────
    sections = []

    # Header
    sections.append("# 📊 Model Evaluation Report\n")
    sections.append(
        "> Auto-generated on **{}** by `evaluation/benchmark.py`.\n".format(timestamp)
    )
    sections.append(
        "> Re-run with: `python -m evaluation.benchmark --samples {}`\n".format(sample_count)
    )
    sections.append("\n---\n\n")

    # Headline metrics
    sections.append("## 🎯 Headline Metrics\n\n")
    sections.append("| Metric | Value | Grade |\n")
    sections.append("|--------|-------|-------|\n")
    sections.append("| **Macro F1** | **{:.3f}** | {} {} |\n".format(macro_f1, emoji, grade))
    sections.append("| **Weighted F1** | **{:.3f}** | |\n".format(weighted_f1))
    sections.append("| **Accuracy** | **{:.3f}** | |\n".format(accuracy))
    sections.append("| **Sample Count** | {} | |\n".format(sample_count))
    sections.append("\n---\n\n")

    # Methodology
    sections.append("## 🧪 Methodology\n\n")
    sections.append("- **Model:** `{}`\n".format(model_name))
    sections.append("- **Dataset:** `{}` (test split)\n".format(dataset_name))
    sections.append("- **Sample size:** {} examples\n".format(sample_count))
    sections.append("- **Evaluation labels:** {}\n".format(", ".join(labels_used)))
    sections.append("- **Excluded labels:** `disgust`, `neutral` (not in dataset)\n")
    sections.append("- **Label mapping:** `love` → `joy` (no direct dataset equivalent)\n")
    sections.append("- **Random seed:** 42 (reproducible)\n\n")
    sections.append(
        "The evaluation uses **macro-averaged F1** as the headline metric to "
        "weight all emotion classes equally regardless of frequency.\n\n"
    )
    sections.append("---\n\n")

    # Per-class table
    sections.append("## 📈 Per-Class Performance\n\n")
    sections.append("| Emotion | Precision | Recall | F1 | Support |\n")
    sections.append("|---------|-----------|--------|-----|---------|\n")
    sections.append(per_class_table + "\n\n")
    sections.append("![Per-class F1 scores]({})\n\n".format(per_class_chart_path))
    sections.append("---\n\n")

    # Confusion matrix
    sections.append("## 🧩 Confusion Matrix\n\n")
    sections.append("![Confusion Matrix]({})\n\n".format(confusion_matrix_path))
    sections.append(confusion_summary + "\n\n")
    sections.append("---\n\n")

    # Crisis metrics
    sections.append("## 🚦 Crisis Detection Performance\n\n")
    sections.append(
        "A subset metric: how reliably does the model identify high-risk "
        "emotions (sadness + fear, often correlated with depression/anxiety)?\n\n"
    )
    sections.append("| Metric | Value |\n")
    sections.append("|--------|-------|\n")
    sections.append("| **Combined Sadness + Fear Recall** | {:.3f} |\n".format(crisis_recall))
    sections.append("| **Combined Sadness + Fear Precision** | {:.3f} |\n".format(crisis_precision))
    sections.append("| **False Negative Rate** | {:.3f} |\n\n".format(1 - crisis_recall))
    sections.append(
        "> ⚠️ False negatives (missed crisis signals) are the most dangerous "
        "error type in mental health applications.\n\n"
    )
    sections.append("---\n\n")

    # Reproducibility — use plain text, NOT nested code blocks
    sections.append("## 🔍 Reproducibility\n\n")
    sections.append("Recreate this exact report:\n\n")
    sections.append("    python -m evaluation.benchmark --samples {} --seed 42\n\n".format(sample_count))
    sections.append("View raw metrics:\n\n")
    sections.append("    cat evaluation/results/metrics.json\n\n")
    sections.append("Inspect sample predictions:\n\n")
    sections.append("    head evaluation/results/predictions.csv\n\n")
    sections.append("---\n\n")

    # Limitations
    sections.append("## 🛠️ Limitations\n\n")
    sections.append("1. **Domain mismatch:** Twitter text differs from clinical/therapeutic text\n")
    sections.append("2. **Label collapse:** `love` mapped to `joy` may inflate joy recall\n")
    sections.append(
        "3. **Excluded labels:** Model's `disgust` and `neutral` predictions "
        "on a 6-class dataset count as misclassifications by default\n"
    )
    sections.append(
        "4. **Sample size:** {} samples gives ±{:.1f}% margin "
        "at 95% confidence for accuracy metrics\n\n".format(sample_count, margin)
    )
    sections.append("---\n\n")

    # Next steps
    sections.append("## 📊 Next Steps\n\n")
    sections.append("- [ ] Evaluate on domain-specific mental health corpus (e.g., DAIC-WOZ)\n")
    sections.append("- [ ] Add temperature calibration for confidence scores\n")
    sections.append("- [ ] Fine-tune on labeled mental health data (planned v1.2)\n")
    sections.append("- [ ] Test against adversarial / paraphrased inputs\n\n")
    sections.append("---\n\n")

    # Footer
    sections.append(
        "*Report generated by the Mental Health Agentic AI Platform "
        "evaluation harness.*\n"
    )

    # ── Write to disk ─────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(sections), encoding="utf-8")