import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Auto-generate model artifact if tests are run before run_pipeline.py
from src.config import MODEL_PATH
if not MODEL_PATH.exists():
    from src.train import train_model
    train_model(save_artifacts=True)
