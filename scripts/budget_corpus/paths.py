from pathlib import Path

# scripts/budget_corpus/paths.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CORPUS = REPO_ROOT / "data" / "corpora" / "federal-budget"
RAW_ROOT = DATA_CORPUS / "raw"
PROCESSED_ROOT = DATA_CORPUS / "processed"
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
