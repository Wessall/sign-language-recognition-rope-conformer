import shutil
from pathlib import Path

ROOT = Path.cwd()
DRY_RUN = False

def log(msg):
    print(msg)

def move(src, dst):
    if not src.exists():
        return
    if DRY_RUN:
        log(f"[DRY MOVE] {src} -> {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

def remove(path):
    if not path.exists():
        return
    if DRY_RUN:
        log(f"[DRY REMOVE] {path}")
    else:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

# =========================
# STEP 1: FLATTEN EXPERIMENT
# =========================
def flatten_exp():
    exp = ROOT / "experiments" / "exp_001"

    for folder in ["data", "metrics", "plots", "predictions", "reports"]:
        move(exp / folder, ROOT / folder)

# =========================
# STEP 2: MOVE COMPARISON
# =========================
def move_comparison():
    move(ROOT / "comparisons/model_comparison_summary.csv",
         ROOT / "comparisons.csv")

# =========================
# STEP 3: KEEP ONLY TFLITE
# =========================
def clean_models():
    move(ROOT / "models/final_model_RoPE_Conformer.tflite",
         ROOT / "models/model.tflite")

    remove(ROOT / "models/rope_conformer")

# =========================
# STEP 4: DELETE HEAVY FILES
# =========================
def remove_heavy():
    remove(ROOT / "experiments")
    remove(ROOT / "archives")

# =========================
# MAIN
# =========================
def main():
    flatten_exp()
    move_comparison()
    clean_models()
    remove_heavy()

if __name__ == "__main__":
    main()