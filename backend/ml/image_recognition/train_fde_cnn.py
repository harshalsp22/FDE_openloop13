"""
FDE CNN Training Script
========================
Run this ONCE after downloading the DermNet dataset from Kaggle.

──────────────────────────────────────────────────────────────
QUICK START
──────────────────────────────────────────────────────────────
# 1. Install the Kaggle CLI and authenticate
pip install kaggle --break-system-packages
# Place kaggle.json in ~/.kaggle/  (download from kaggle.com → Account → API)

# 2. Download DermNet (~500 MB)
kaggle datasets download -d shubhamgoel27/dermnet
unzip dermnet.zip -d /path/to/dermnet

# 3. Install ML deps
pip install tensorflow pillow matplotlib scikit-learn --break-system-packages

# 4. Train (uses GPU if available, otherwise CPU)
python backend/ml/image_recognition/train_fde_cnn.py --dataset /path/to/dermnet

# 5. Check dataset structure first (dry-run)
python backend/ml/image_recognition/train_fde_cnn.py --dataset /path/to/dermnet --check-only
──────────────────────────────────────────────────────────────

EXPECTED DATASET STRUCTURE:
    /path/to/dermnet/
        train/
            Drug Eruptions/           ← ~500 images
            Eczema Photos/            ← ~1500 images
            Psoriasis Pictures .../
            ... (23 category folders)
        test/
            Drug Eruptions/
            ...

TRAINING OUTPUT:
    backend/models/fde_cnn_model.h5     ← trained Keras weights
    backend/models/model_meta.json      ← class mappings + accuracy metrics
"""

import argparse
import sys
from pathlib import Path

# Ensure the backend root is on sys.path so imports resolve
_BACKEND = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_BACKEND))

from ml.image_recognition.fde_cnn_model import train_model, MODEL_PATH, FDE_CLASS_MAPPING


# ─────────────────────────────────────────────────────────────────────────────
def _validate_dataset(dataset_path: Path) -> tuple[list, list]:
    train_dir = dataset_path / "train"
    test_dir  = dataset_path / "test"

    if not train_dir.exists() or not test_dir.exists():
        print(f"❌ Expected train/ and test/ subdirectories in: {dataset_path}")
        print("   Make sure you've extracted the DermNet zip correctly.")
        sys.exit(1)

    train_classes = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    test_classes  = sorted([d for d in test_dir.iterdir()  if d.is_dir()])
    return train_classes, test_classes


def _print_dataset_report(dataset_path: Path):
    train_classes, test_classes = _validate_dataset(dataset_path)

    print(f"\n{'='*65}")
    print(f"  DermNet Dataset Validation")
    print(f"{'='*65}")
    print(f"  Dataset path    : {dataset_path}")
    print(f"  Train classes   : {len(train_classes)}")
    print(f"  Test  classes   : {len(test_classes)}")

    total_train = 0
    total_test  = 0
    print(f"\n  Class breakdown (train):")

    for cls_dir in train_classes:
        imgs = (list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))
                + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.JPG")))
        fde_cat = FDE_CLASS_MAPPING.get(cls_dir.name, "NOT_FDE")
        marker  = "🔴" if "HIGH" in fde_cat else ("🟡" if "MODERATE" in fde_cat else
                  ("🟠" if "DIFF" in fde_cat else "⚪"))
        print(f"    {marker} {cls_dir.name:<58} {len(imgs):>5} imgs  [{fde_cat}]")
        total_train += len(imgs)

    for cls_dir in test_classes:
        imgs = (list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))
                + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.JPG")))
        total_test += len(imgs)

    print(f"\n  Total train images : {total_train:,}")
    print(f"  Total test  images : {total_test:,}")
    print(f"\n  FDE legend: 🔴 FDE_HIGH  🟡 FDE_MODERATE  🟠 DIFFERENTIAL  ⚪ NOT_FDE")
    print(f"{'='*65}\n")

    return total_train


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train FDE CNN on DermNet dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="Path to extracted DermNet root (contains train/ and test/)"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=30,
        help="Phase 1 training epochs (frozen backbone). Default: 30"
    )
    parser.add_argument(
        "--fine-tune-epochs", "-f", type=int, default=10,
        help="Phase 2 fine-tuning epochs. Default: 10"
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate dataset structure only — do not train"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        sys.exit(1)

    total_train = _print_dataset_report(dataset_path)

    if args.check_only:
        print("✅ Dataset structure is valid. Ready for training.")
        print("   Remove --check-only to start training.")
        return

    # ── Check TensorFlow ──────────────────────────────────────────────────────
    try:
        import tensorflow as tf
        print(f"  TensorFlow version : {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"  GPU detected       : {gpus[0].name}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("  ⚠️  No GPU — training on CPU (expect ~2–4 h for full DermNet).")
            print("     Tip: Use Kaggle Notebooks (free T4 GPU) for faster training.")
    except ImportError:
        print("❌ TensorFlow not installed.")
        print("   Run: pip install tensorflow --break-system-packages")
        sys.exit(1)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n  Training config:")
    print(f"    Phase 1 epochs   : {args.epochs}")
    print(f"    Fine-tune epochs : {args.fine_tune_epochs}")
    print(f"    Output model     : {MODEL_PATH}")
    print(f"\n  Starting training …\n")

    model, history = train_model(
        dataset_path=str(dataset_path),
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
    )

    print(f"\n{'='*65}")
    print(f"  ✅  Training complete!")
    print(f"  Model     : {MODEL_PATH}")
    print(f"  Accuracy  : {history['test']['accuracy']:.4f}")
    print(f"  AUC       : {history['test']['auc']:.4f}")
    print(f"{'='*65}")
    print(f"\n  Next → start the Flask backend:")
    print(f"    cd FDE_System/backend && python app.py")
    print(f"\n  Image prediction endpoint:")
    print(f"    POST http://localhost:5000/api/image/predict_fde  (multipart/form-data, key=file)")


if __name__ == "__main__":
    main()
