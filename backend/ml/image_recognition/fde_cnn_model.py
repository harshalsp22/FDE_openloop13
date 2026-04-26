"""
FDE Image Recognition — CNN Model (DermNet-based)
===================================================
Architecture : Custom CNN + Transfer Learning (MobileNetV2 backbone)
Dataset      : DermNet (https://www.kaggle.com/datasets/shubhamgoel27/dermnet)

FDE-RELEVANT CLASSES (mapped from DermNet 23-category taxonomy):
  HIGH_RISK_FDE:
    - "Drug Eruptions"              → direct FDE
    - "Stevens-Johnson Syndrome"    → severe drug reaction
    - "Bullous Disease Photos"      → blistering (FDE variant)
    - "Vasculitis Photos"           → inflammatory drug reaction
    - "Exanthems and Drug Eruptions"

  DIFFERENTIAL (look-alikes that are NOT FDE):
    - "Psoriasis Pictures Lichen Planus and related diseases"
    - "Urticaria Hives"
    - "Eczema Photos"
    - "Tinea Ringworm Candidiasis and other Fungal Infections"

MODEL PIPELINE:
  Input → Preprocessing → MobileNetV2 backbone → Custom Head → FDE Score
"""

import os
import json
import numpy as np
from pathlib import Path

# ─── FDE-relevant class mapping from DermNet ────────────────────────────────
# Maps DermNet folder names → FDE relevance category
FDE_CLASS_MAPPING = {
    # Direct drug eruption indicators (HIGH confidence FDE)
    "Drug Eruptions":                                                   "FDE_HIGH",
    "Stevens-Johnson Syndrome":                                         "FDE_HIGH",
    "Bullous Disease Photos":                                           "FDE_MODERATE",
    "Vasculitis Photos":                                                "FDE_MODERATE",
    "Exanthems and Drug Eruptions":                                     "FDE_HIGH",

    # Differential diagnoses (require clinical differentiation)
    "Psoriasis Pictures Lichen Planus and related diseases":           "DIFFERENTIAL",
    "Urticaria Hives":                                                  "DIFFERENTIAL",
    "Poison Ivy Photos and other Contact Dermatitis":                  "DIFFERENTIAL",
    "Lupus and other Connective Tissue diseases":                      "DIFFERENTIAL",
    "Systemic Disease":                                                 "DIFFERENTIAL",
    "Scabies Lyme Disease and other Infestations and Bites":          "DIFFERENTIAL",
    "Light Diseases and Disorders of Pigmentation":                    "DIFFERENTIAL",

    # NOT FDE
    "Eczema Photos":                                                    "NOT_FDE",
    "Acne and Rosacea Photos":                                         "NOT_FDE",
    "Atopic Dermatitis Photos":                                        "NOT_FDE",
    "Tinea Ringworm Candidiasis and other Fungal Infections":         "NOT_FDE",
    "Seborrheic Keratoses and other Benign Tumors":                   "NOT_FDE",
    "Warts Molluscum and other Viral Infections":                     "NOT_FDE",
    "Vascular Tumors":                                                  "NOT_FDE",
    "Nail Fungus and other Nail Disease":                              "NOT_FDE",
    "Melanoma Skin Cancer Nevi and Moles":                             "NOT_FDE",
    "Cellulitis Impetigo and other Bacterial Infections":             "NOT_FDE",
}

# Binary label map for training
LABEL_MAP = {
    "FDE_HIGH":     2,
    "FDE_MODERATE": 1,
    "DIFFERENTIAL": 0,
    "NOT_FDE":      0,
}

# Image config
IMG_SIZE    = 224       # MobileNetV2 standard
BATCH_SIZE  = 32
NUM_CLASSES = 3         # mapped FDE risk classes

_BASE_DIR   = Path(__file__).parent.parent.parent
MODEL_PATH  = _BASE_DIR / "models" / "fde_cnn_model.h5"
META_PATH   = _BASE_DIR / "models" / "model_meta.json"


# ─────────────────────────────────────────────────────────────────────────────
def build_fde_cnn(num_classes: int = None, trainable_base: bool = False):
    """
    Build the FDE CNN using MobileNetV2 transfer learning.

    Architecture:
        MobileNetV2 (pretrained ImageNet)  ←  frozen initially
            ↓ GlobalAveragePooling2D
            ↓ Dense(256, ReLU) + BatchNorm + Dropout(0.4)
            ↓ Dense(128, ReLU) + BatchNorm + Dropout(0.3)
            ↓ Dense(num_classes, Softmax)

    Why MobileNetV2?
        Lightweight (3.4 M params), runs on CPU in production, depthwise
        separable convolutions handle texture patterns exceptionally well,
        and ImageNet features transfer strongly to dermatology imagery.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.losses import CategoricalCrossentropy

        n_cls = num_classes or NUM_CLASSES

        # ── Base model ──────────────────────────────────────────────────────
        base = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet"
        )
        base.trainable = trainable_base

        # ── Custom classification head ───────────────────────────────────────
        inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name="gap")(x)

        # Block 1
        x = layers.Dense(256, name="dense_1")(x)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Activation("relu", name="relu_1")(x)
        x = layers.Dropout(0.4, name="drop_1")(x)

        # Block 2
        x = layers.Dense(128, name="dense_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Activation("relu", name="relu_2")(x)
        x = layers.Dropout(0.3, name="drop_2")(x)

        # Output
        outputs = layers.Dense(n_cls, activation="softmax", name="output")(x)

        model = Model(inputs, outputs, name="FDE_CNN_v1")

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=CategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )

        return model

    except ImportError:
        raise ImportError(
            "TensorFlow not installed.\n"
            "  Run: pip install tensorflow pillow matplotlib --break-system-packages"
        )


# ─────────────────────────────────────────────────────────────────────────────
def get_data_generators(dataset_path: str):
    """
    Build train / validation / test ImageDataGenerators from DermNet folder.

    Expected DermNet structure:
        dataset_path/
            train/
                Drug Eruptions/         (~500 images)
                Eczema Photos/          (~1500 images)
                ... (23 category folders)
            test/
                Drug Eruptions/
                ...

    DermNet comes pre-split into train/test.  We carve 20% of train as
    validation via `validation_split`.
    """
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        train_aug = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            shear_range=0.1,
            fill_mode="nearest",
            validation_split=0.2,
        )
        val_aug  = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
        test_aug = ImageDataGenerator(rescale=1.0 / 255)

        train_path = os.path.join(dataset_path, "train")
        test_path  = os.path.join(dataset_path, "test")

        train_gen = train_aug.flow_from_directory(
            train_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=42,
        )
        val_gen = val_aug.flow_from_directory(
            train_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )
        test_gen = test_aug.flow_from_directory(
            test_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )

        return train_gen, val_gen, test_gen

    except ImportError:
        raise ImportError("TensorFlow not installed.")


# ─────────────────────────────────────────────────────────────────────────────
def train_model(dataset_path: str, epochs: int = 30, fine_tune_epochs: int = 10):
    """
    Two-phase training pipeline:
        Phase 1 — Frozen base  : train only the custom head         (30 epochs)
        Phase 2 — Fine-tuning  : unfreeze top 30 MobileNetV2 layers (10 epochs)

    Returns:
        (model, history_dict)
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, test_gen = get_data_generators(dataset_path)
    num_classes_actual = train_gen.num_classes

    print(f"\n{'='*60}")
    print(f"  FDE CNN Training — DermNet Dataset")
    print(f"  Classes detected  : {num_classes_actual}")
    print(f"  Training samples  : {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"{'='*60}\n")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    model = build_fde_cnn(num_classes=num_classes_actual, trainable_base=False)
    model.summary()

    cb_p1 = [
        EarlyStopping(monitor="val_auc", patience=6, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_auc", mode="max"),
    ]

    print("\n── Phase 1: Training custom head (frozen backbone) ──")
    hist1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs, callbacks=cb_p1, verbose=1
    )

    # ── Phase 2: Fine-tune top 30 layers ─────────────────────────────────────
    print("\n── Phase 2: Fine-tuning top 30 MobileNetV2 layers ──")
    base_model = model.get_layer("mobilenetv2_1.00_224")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    cb_p2 = [
        EarlyStopping(monitor="val_auc", patience=4, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-8),
        ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_auc", mode="max"),
    ]

    hist2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=fine_tune_epochs, callbacks=cb_p2, verbose=1
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n── Test-set evaluation ──")
    test_results = model.evaluate(test_gen, verbose=1)

    meta = {
        "class_indices":     train_gen.class_indices,
        "num_classes":       num_classes_actual,
        "img_size":          IMG_SIZE,
        "fde_class_mapping": FDE_CLASS_MAPPING,
        "test_accuracy":     float(test_results[1]),
        "test_auc":          float(test_results[2]),
        "model_path":        str(MODEL_PATH),
        "architecture":      "MobileNetV2 + Custom Dense Head",
        "dataset":           "DermNet (Kaggle: shubhamgoel27/dermnet)",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅  Model saved  → {MODEL_PATH}")
    print(f"✅  Metadata     → {META_PATH}")
    print(f"    Test Accuracy: {test_results[1]:.4f}")
    print(f"    Test AUC     : {test_results[2]:.4f}")

    return model, {
        "phase1": hist1.history,
        "phase2": hist2.history,
        "test":   {"accuracy": test_results[1], "auc": test_results[2]},
    }


# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    """Load trained model from disk. Raises FileNotFoundError if not yet trained."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}.\n"
            "Train first:  python backend/ml/image_recognition/train_fde_cnn.py "
            "--dataset /path/to/dermnet"
        )
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_PATH))


def load_meta() -> dict:
    """Load model metadata (class indices, accuracy, thresholds)."""
    if not META_PATH.exists():
        return {}
    with open(META_PATH) as f:
        return json.load(f)
