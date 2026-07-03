---
title: Mental Health Agentic AI Platform
emoji: 🧠
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: true
license: mit
short_description: Multi-agent emotion analysis with fine-tuned DistilRoBERTa
---

# 🧠 Mental Health Agentic AI Platform

**Live demo** of a production-grade multi-agent AI platform for mental health intelligence.

Analyzes user text through:
1. 🔍 **Emotion Classification** — Fine-tuned DistilRoBERTa (macro F1 = 0.89)
2. 🚨 **Crisis Detection** — Keyword + confidence-based risk assessment
3. 📚 **Wellness Recommendations** — Curated coping strategies
4. 🔮 **Explainability** — Token-level attribution showing key influencing words

---

## 🎯 Model

Powered by [YDVJIYA/distilroberta-base-finetuned-emotion](https://huggingface.co/YDVJIYA/distilroberta-base-finetuned-emotion)

**Metrics on `dair-ai/emotion` test set (n=2000):**

| Metric | Score |
|--------|-------|
| Macro F1 | **0.89** |
| Weighted F1 | **0.93** |
| Accuracy | **0.93** |
| Crisis Recall (sadness+fear) | **0.96** |

---

## 🔗 Links

- 💻 **Source Code:** [github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform)
- 🤗 **Fine-tuned Model:** [YDVJIYA/distilroberta-base-finetuned-emotion](https://huggingface.co/YDVJIYA/distilroberta-base-finetuned-emotion)
- 📊 **Evaluation Report:** [docs/EVALUATION.md](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform/blob/main/docs/EVALUATION.md)
- 🧪 **Fine-Tuning Report:** [docs/FINE_TUNING.md](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform/blob/main/docs/FINE_TUNING.md)

---

## ⚠️ Important Disclaimer

This is an **AI research tool** and NOT a substitute for professional mental health care. If you are in crisis, please contact:

- 🇺🇸 **988** Suicide & Crisis Lifeline (call or text)
- 🇺🇸 **Text HOME to 741741** (Crisis Text Line)
- 🇮🇳 **iCall India:** 9152987821
- 🌍 [International Directory](https://www.iasp.info/resources/Crisis_Centres/)

---

## 👤 Author

Built by [Jiya Yadav](https://github.com/JIYA-YDV) as a portfolio piece demonstrating end-to-end ML engineering.