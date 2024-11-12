from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "test_data"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

