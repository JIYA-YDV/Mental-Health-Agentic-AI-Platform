import importlib.metadata

# List of all packages from your requirements list
packages = [
    "fastapi",
    "starlette",
    "uvicorn",
    "streamlit",
    "pydantic",
    "pydantic-settings",
    "transformers",
    "tokenizers",
    "torch",
    "sentence-transformers",
    "scikit-learn",
    "numpy",
    "pandas",
    "sentencepiece",
    "chromadb",
    "faiss-cpu",
    "shap",
    "prometheus-client",
    "structlog",
    "httpx",
    "aiohttp",
    "requests",
    "python-dotenv",
    "python-multipart",
    "huggingface-hub",
    "groq",
    "pytest",
    "pytest-asyncio",
    "pytest-cov",
    "ruff",
    "seaborn",
    "tqdm",
]

print(f"{'Package Name':<25} | {'Installed Version'}")
print("-" * 45)

for pkg in packages:
  try:
    version = importlib.metadata.version(pkg)
    print(f"{pkg:<25} | {version}")
  except importlib.metadata.PackageNotFoundError:
    print(f"{pkg:<25} | Not Installed")