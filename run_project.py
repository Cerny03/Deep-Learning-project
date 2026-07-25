import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    # --- Inspection ---
    "src/inspect_data.py",
    "src/inspect_noise.py",
    "src/inspect_baseline.py",
    "src/inspect_autoencoder.py",
    "src/inspect_conv_autoencoder.py",

    # --- Training & Evaluation ---
    "src/train_fc_autoencoder.py",
    "src/evaluate_fc_autoencoder.py",
    "src/train_conv_autoencoder.py",
    "src/evaluate_all_models.py",
    "src/evaluate_noise_robustness.py",
    "src/evaluate_unseen_corruptions.py",

    # --- Analysis ---
    "src/analyze_latent_representations.py",
    "src/analyze_latent_structure.py",
]


def run_script(script_path: Path, index: int, total: int):
    print("\n" + "=" * 70)
    print(f"[{index}/{total}] Running {script_path.name}")
    print("=" * 70)

    start = time.perf_counter()

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True
    )

    elapsed = time.perf_counter() - start
    print(f"Completed in {elapsed:.1f} seconds")


def main():
    print("Project root:", PROJECT_ROOT)
    print("Python executable:", sys.executable)

    total = len(SCRIPTS)

    for i, relative in enumerate(SCRIPTS, start=1):
        script_path = PROJECT_ROOT / relative

        if not script_path.exists():
            print(f"ERROR: Script not found: {script_path}")
            sys.exit(1)

        try:
            run_script(script_path, i, total)
        except subprocess.CalledProcessError as e:
            print("\n" + "=" * 70)
            print("PIPELINE STOPPED — Script failed")
            print("=" * 70)
            print("Command:", " ".join(e.cmd))
            print("Return code:", e.returncode)
            sys.exit(e.returncode)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
