"""Train a small CPU MLP baseline on the fixed, leakage-safe Stage 3 assets."""

from pathlib import Path

from classical_training import RESULTS_DIR, main as train_main
from classical_utils import generate_classical_plots


def main() -> None:
    train_main("mlp")
    svm_results = RESULTS_DIR / "svm_results.csv"
    mlp_results = RESULTS_DIR / "mlp_results.csv"
    if svm_results.is_file() and mlp_results.is_file():
        generate_classical_plots(
            svm_results,
            mlp_results,
            RESULTS_DIR / "classical_comparison.csv",
            RESULTS_DIR / "plots",
        )
        print(f"Wrote classical comparison and plots under {RESULTS_DIR}")


if __name__ == "__main__":
    main()