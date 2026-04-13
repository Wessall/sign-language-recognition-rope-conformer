import shutil
from pathlib import Path

ROOT = Path.cwd()
DRY_RUN = False

def log(msg):
    print(msg)

def move(src, dst):
    if DRY_RUN:
        log(f"[DRY] {src} -> {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        log(f"[MOVE] {src} -> {dst}")

# =========================
# STEP 1: create structure
# =========================
def create_structure():
    for p in [
        "models/rope_conformer",
        "experiments/exp_001",
        "experiments/exp_002",
        "comparisons",
        "archives"
    ]:
        path = ROOT / p
        if not DRY_RUN:
            path.mkdir(parents=True, exist_ok=True)

# =========================
# STEP 2: move models
# =========================
def move_models():
    src = ROOT / "models" / "rope_conformer"

    for file in src.glob("*"):
        if file.suffix in [".keras", ".h5", ".json"] or file.name == "RoPE_Conformer_saved_model":
            dst = ROOT / "models" / "rope_conformer" / file.name
            move(file, dst)

# =========================
# STEP 3: move experiments
# =========================
def move_experiment(src_name, exp_name):
    src = ROOT / "models" / src_name

    for folder in ["data", "checkpoints"]:
        path = src / folder
        if path.exists():
            move(path, ROOT / "experiments" / exp_name / folder)

# =========================
# STEP 4: merge outputs
# =========================
def merge_outputs():
    outputs = ROOT / "outputs"

    if not outputs.exists():
        return

    for category in outputs.iterdir():
        for model_folder in category.iterdir():
            exp_name = "exp_001" if "20260412" not in model_folder.name else "exp_002"

            dst = ROOT / "experiments" / exp_name / category.name
            move(model_folder, dst)

# =========================
# STEP 5: clean old folders
# =========================
def cleanup():
    for name in ["models/rope_conformer_20260412_102312", "outputs"]:
        path = ROOT / name
        if path.exists():
            if DRY_RUN:
                log(f"[DRY REMOVE] {path}")
            else:
                shutil.rmtree(path)

# =========================
# MAIN
# =========================
def main():
    create_structure()

    move_experiment("rope_conformer", "exp_001")
    move_experiment("rope_conformer_20260412_102312", "exp_002")

    merge_outputs()
    cleanup()

if __name__ == "__main__":
    main()