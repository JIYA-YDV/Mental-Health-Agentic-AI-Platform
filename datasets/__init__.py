"""
Datasets Package

Storage for raw and processed mental health datasets
used for model training, evaluation, and fine-tuning.

Expected structure:
    datasets/
    ├── raw/              Original unprocessed datasets
    ├── processed/        Cleaned and tokenized datasets
    ├── splits/           Train / validation / test splits
    └── README.md         Dataset documentation and sources

Recommended public datasets:
    - Emotion dataset (dair-ai/emotion) on HuggingFace
    - GoEmotions (Google Research)
    - Mental health Reddit dataset (SMHD)
    - DAIC-WOZ depression corpus

Note:
    Raw datasets are excluded from version control via .gitignore.
    Download instructions available in datasets/README.md
"""