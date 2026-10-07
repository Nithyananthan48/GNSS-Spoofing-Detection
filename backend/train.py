"""
Standalone Training Script
===========================
Trains the Bi-GRU model from the command line.

Usage:
    python train.py --dataset datasets/synthetic_gnss_dataset.csv

If no dataset is specified, generates and uses the synthetic dataset.
"""
import argparse
import sys
from pathlib import Path

# Ensure backend package is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ml.trainer import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train the GNSS Spoofing Bi-GRU model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to GNSS CSV dataset. If not provided, synthetic dataset is generated.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Force regeneration of the synthetic dataset.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset

    if dataset_path is None or args.generate:
        synthetic_path = Path(__file__).parent / "datasets" / "synthetic_gnss_dataset.csv"
        if not synthetic_path.exists() or args.generate:
            print("Generating synthetic GNSS dataset...")
            sys.path.insert(0, str(Path(__file__).parent / "datasets"))
            from datasets.generate_synthetic import generate_gnss_dataset
            generate_gnss_dataset(
                n_normal=3000,
                n_spoofed=2000,
                output_file=str(synthetic_path),
            )
        dataset_path = str(synthetic_path)
        print(f"Using dataset: {dataset_path}")

    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        sys.exit(1)

    print("=" * 60)
    print("GNSS Spoofing Detection - Bi-GRU Training")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print()

    result = train_model(dataset_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    bigru = result.get("bigru_metrics", {})
    print(f"  Accuracy:   {bigru.get('accuracy', 0):.4f} ({bigru.get('accuracy', 0)*100:.2f}%)")
    print(f"  Precision:  {bigru.get('precision', 0):.4f}")
    print(f"  Recall:     {bigru.get('recall', 0):.4f}")
    print(f"  F1 Score:   {bigru.get('f1_score', 0):.4f}")
    print(f"  ROC-AUC:    {bigru.get('roc_auc', 0):.4f}")

    print("\nBaseline Model Comparison:")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10} {'ROC-AUC':>10}")
    print("-" * 60)
    for model_name, metrics in result.get("baseline_metrics", {}).items():
        if "error" not in metrics:
            print(
                f"  {model_name:<23} "
                f"{metrics.get('accuracy', 0):>10.4f} "
                f"{metrics.get('f1_score', 0):>10.4f} "
                f"{metrics.get('roc_auc', 0):>10.4f}"
            )
    print(
        f"  {'Bi-GRU (Proposed)':<23} "
        f"{bigru.get('accuracy', 0):>10.4f} "
        f"{bigru.get('f1_score', 0):>10.4f} "
        f"{bigru.get('roc_auc', 0):>10.4f}"
    )

    print(f"\nModel saved to: trained_models/bigru_model.keras")
    print(f"Scaler saved to: trained_models/scaler.pkl")
    print(f"Metadata: trained_models/model_metadata.json")


if __name__ == "__main__":
    main()
