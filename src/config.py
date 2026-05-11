from pathlib import Path
import os

# --------------------------------------------------
# Project roots
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

# --------------------------------------------------
# Final fracture classifier package
# --------------------------------------------------
CLASSIFIER_DIR = MODELS_DIR / "fracture_classifier_resnet50_final"

BEST_MODEL_PATH = CLASSIFIER_DIR / "best_model.pt"
INFERENCE_CONFIG_PATH = CLASSIFIER_DIR / "inference_config.json"
SELECTED_THRESHOLD_PATH = CLASSIFIER_DIR / "selected_threshold.json"
LABEL_MAP_PATH = CLASSIFIER_DIR / "label_map.json"
FEATURE_SCHEMA_PATH = CLASSIFIER_DIR / "feature_schema.json"
METRICS_SUMMARY_PATH = CLASSIFIER_DIR / "metrics_summary.json"

# Optional but useful
TRAIN_CONFIG_PATH = CLASSIFIER_DIR / "train_config.json"
VAL_METRICS_PATH = CLASSIFIER_DIR / "val_metrics_best_threshold.csv"
TEST_METRICS_PATH = CLASSIFIER_DIR / "test_metrics_locked_threshold.csv"

# --------------------------------------------------
# MedGemma / LLM environment settings
# --------------------------------------------------
MEDGEMMA_BACKEND = os.getenv("MEDGEMMA_BACKEND", "unset")
MEDGEMMA_MODEL_NAME = os.getenv("MEDGEMMA_MODEL_NAME", "unset")

# Example optional values for later use
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "")

# --------------------------------------------------
# App settings
# --------------------------------------------------
APP_TITLE = "Fracture Detection + MedGemma Explanation"
DEFAULT_IMAGE_SIZE = 224
MAX_UPLOAD_MB = 10

# --------------------------------------------------
# Required file checks
# --------------------------------------------------
REQUIRED_CLASSIFIER_FILES = {
    "best_model": BEST_MODEL_PATH,
    "inference_config": INFERENCE_CONFIG_PATH,
    "selected_threshold": SELECTED_THRESHOLD_PATH,
    "label_map": LABEL_MAP_PATH,
    "feature_schema": FEATURE_SCHEMA_PATH,
    "metrics_summary": METRICS_SUMMARY_PATH,
}

def get_missing_required_files():
    missing = {}
    for name, path in REQUIRED_CLASSIFIER_FILES.items():
        if not path.exists():
            missing[name] = str(path)
    return missing