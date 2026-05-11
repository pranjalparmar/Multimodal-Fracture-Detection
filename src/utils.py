from pathlib import Path
from src.config import (
    PROJECT_ROOT,
    CLASSIFIER_DIR,
    REQUIRED_CLASSIFIER_FILES,
    get_missing_required_files,
)

def format_path_status(path: Path) -> str:
    exists = path.exists()
    kind = "FILE" if path.is_file() else "DIR" if path.is_dir() else "MISSING"
    return f"[{kind}] {path} :: exists={exists}"

def print_project_status():
    print("=== PROJECT STATUS ===")
    print(format_path_status(PROJECT_ROOT))
    print(format_path_status(CLASSIFIER_DIR))

    print("\n=== REQUIRED CLASSIFIER FILES ===")
    for name, path in REQUIRED_CLASSIFIER_FILES.items():
        exists = path.exists()
        print(f"{name}: {path} :: exists={exists}")

def validate_required_files(raise_error: bool = True):
    missing = get_missing_required_files()

    if not missing:
        print("All required classifier files are present.")
        return True

    print("Missing required classifier files:")
    for name, path in missing.items():
        print(f"- {name}: {path}")

    if raise_error:
        raise FileNotFoundError(
            "Required classifier files are missing. See printed paths above."
        )

    return False