# RoPE Conformer — Sign Language Gesture Recognition

A position-aware Bidirectional GRU model with Rotary Position Encoding (RoPE) for skeletal sequence-based sign language recognition, trained on 94,477 sequences spanning 250 gesture classes. This repository contains the full training pipeline, evaluation artifacts, saved model weights, and analysis notebooks.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Per-Class Analysis](#per-class-analysis)
- [Error Analysis](#error-analysis)
- [Project Structure](#project-structure)
- [Saved Artifacts](#saved-artifacts)
- [Notebooks](#notebooks)
- [Reproducing the Experiment](#reproducing-the-experiment)
- [Known Limitations](#known-limitations)

---

## Overview

This model classifies sign language gestures from skeletal landmark sequences. Each input is a variable-length sequence of body and hand pose landmarks, fixed to a maximum of 384 frames, with 118 selected landmark nodes producing 708 feature channels per frame.

The RoPE Conformer extends a standard BiGRU backbone with Rotary Position Encoding — a technique originally developed for large language models. RoPE injects relative temporal position information directly into the attention scores, allowing the model to reason about when in the sequence each motion occurs, not only what that motion is. This produces meaningful gains on signs that differ primarily in timing or motion trajectory, and achieves the highest accuracy, F1, and PR-AUC among all individually evaluated models.

---

## Dataset

| Property              | Value                                               |
|-----------------------|-----------------------------------------------------|
| Total sequences       | 94,477                                              |
| Unique classes        | 250 sign vocabulary                                 |
| Landmark nodes        | 118 selected                                        |
| Feature channels      | 708 per frame                                       |
| Max sequence length   | 384 frames                                          |
| Input tensor shape    | (Batch, 384, 708)                                   |
| Training split        | 75,581 samples (80%) — avg 302.3 per class          |
| Validation split      | 9,448 samples (10%) — avg 37.8 per class            |
| Test split            | 9,344 samples — avg 37.8 per class                  |
| Class balance         | Very balanced — min 30, max 41 per class (test set) |

---

## Model Architecture

| Property               | Value                                      |
|------------------------|--------------------------------------------|
| Core unit              | Bidirectional GRU                          |
| Positional encoding    | Rotary Position Encoding (RoPE)            |
| Gating mechanism       | 3 gates: reset, update, new                |
| Trainable parameters   | ~2M+                                       |
| Saved formats          | SavedModel (.keras) + .h5 weights          |
| Validation-to-test gap | ~0.5% (most stable among all models)      |

The BiGRU backbone processes input sequences in both forward and backward directions. On top of this, RoPE encodes the relative temporal position of each frame directly into the attention mechanism without adding a significant number of parameters. This is the same positional encoding technique used in modern large language models such as LLaMA, applied here to skeletal pose sequences rather than token embeddings.

This design allows the model to distinguish signs that differ in when a motion occurs, not just whether it occurs — a critical capability for pairs such as awake/wake or sleepy/sleep that share nearly identical handshapes and differ only in subtle temporal properties.

---

## Performance

Evaluated on the 9,344-sample held-out test set.

| Metric          | Value  |
|-----------------|--------|
| Accuracy        | 85.08% |
| Top-5 Accuracy  | 94.19% |
| Macro F1        | 0.8491 |
| Weighted F1     | 0.8510 |
| Macro ROC-AUC   | 0.9880 |
| Macro PR-AUC    | 0.8680 |
| Misclassified   | 1,394  |

The model makes 1,394 errors on the 9,344-sample test set — a 23.5% reduction in errors compared to BiLSTM and an 18.3% reduction compared to BiGRU. The PR-AUC of 0.8680 is the most informative aggregate metric here, as it is more sensitive than ROC-AUC to performance on hard boundary classes where predicted probabilities are close together. The top-5 accuracy of 94.19% confirms that the correct label is almost always within the model's top five candidates, which is favorable for candidate-list assistive applications.

---

## Per-Class Analysis

**Best performing classes:**

| Class       | Precision | Recall | F1   |
|-------------|-----------|--------|------|
| callonphone | 1.00      | 0.97   | 0.99 |
| horse       | 0.97      | 1.00   | 0.99 |
| uncle       | 1.00      | 0.97   | 0.99 |
| grandpa     | 1.00      | 1.00   | 1.00 |
| shhh        | 0.95      | 0.98   | 0.96 |
| frog        | 0.97      | 0.97   | 0.97 |
| fireman     | 0.98      | 0.98   | 0.98 |

Two classes reach 100% per-class accuracy for the first time in this evaluation — grandpa and horse. Both involve longer or more motion-rich trajectories that benefit directly from RoPE's temporal position awareness. The fireman class shows the most notable single-class improvement, with F1 jumping from 0.90 (BiGRU) to 0.98 — an 8 percentage point gain.

**Most difficult classes:**

| Class    | Precision | Recall | F1   | Primary Issue                         |
|----------|-----------|--------|------|---------------------------------------|
| awake    | 0.46      | 0.45   | 0.45 | Structural vocabulary confusion       |
| beside   | 0.50      | 0.45   | 0.54 | Spatial preposition, weak trajectory  |
| wake     | 0.50      | 0.50   | 0.50 | Mirrors the awake confusion           |
| pencil   | —         | —      | —    | Handshape overlap regression          |
| sleep    | —         | —      | —    | Timing-based disambiguation           |

Despite being the best overall model, the floor on the worst classes is still above that of BiLSTM. The worst per-class accuracy across the 250-class vocabulary is 44.7% (awake), compared to 31.6% for BiLSTM — meaning RoPE raises the performance floor even on signs it cannot fully solve.

Mean per-class accuracy across all 250 classes: **84.93%**. Approximately 8 classes fall below an F1 of 0.70, down from 12 in BiGRU and 18 in BiLSTM.

---

## Error Analysis

**Top confused pairs on the test set:**

| True Class | Predicted | Count | Root Cause                               |
|------------|-----------|-------|------------------------------------------|
| awake      | wake      | 20    | Identical handshape, subtle timing diff  |
| wake       | awake     | 18    | Symmetric — same underlying issue        |
| bedroom    | bed       | 9     | Compound sign subset overlap             |
| sleepy     | sleep     | 10    | Timing-based disambiguation failure      |
| give       | gift      | 6     | Motion trajectory similarity             |
| lips       | mouth     | 7     | Facial location overlap                  |
| pencil     | pen       | 12    | Fine-grained handshape regression        |
| goose      | duck      | 6     | Animal handshape proximity               |

**The awake/wake pair** (20 and 18 errors respectively) represents a structural vocabulary problem, not a modeling failure. All three models evaluated in this project fail similarly on this pair, confirming that no recurrent architecture can solve it without dedicated data-level intervention. The recommended fix is collecting speed-varied augmented examples of both classes.

**Where RoPE clearly helps.** Compared to BiGRU: lips→mouth drops from 8 to 7 errors, give→gift drops from 10 to 6, bedroom→bed drops from 11 to 9, and beside improves from F1 0.37 to 0.54. These gains are consistent with RoPE's positional awareness helping signs that differ in motion trajectory rather than handshape alone.

**Where RoPE slightly hurts.** The pencil→pen confusion worsens to 12 errors in RoPE compared to 7 in BiGRU. This is the most notable regression across all classes. The pen/pencil distinction relies on fine-grained finger configuration — a feature that positional encoding does not address. A handshape-specific auxiliary loss would be needed to recover this.

---

## Project Structure

```
.
|-- organize_project.py
|-- structure.txt
|-- folders.txt
|-- organization_log.txt
|
+-- archives/
|   +-- RoPE_Conformer_20260412_102312.zip     # Complete packaged model archive
|
+-- comparisons/
|   +-- model_comparison_summary.csv
|
+-- experiments/
|   |
|   +-- exp_001/                               # Primary experiment (full evaluation)
|   |   |
|   |   +-- checkpoints/
|   |   |   |-- RoPE_Conformer.weights.h5      # Full weights snapshot
|   |   |   +-- best/                          # Best checkpoints (ckpt-127, ckpt-140)
|   |   |   +-- last/                          # Final checkpoints (ckpt-126 to ckpt-148)
|   |   |
|   |   +-- data/
|   |   |   |-- train_split.csv
|   |   |   |-- val_split.csv
|   |   |   +-- test_split.csv
|   |   |
|   |   +-- logs/                              # 7 training sessions (April 7–13, 2026)
|   |   |
|   |   +-- metrics/                           # Full evaluation metrics (see below)
|   |   |
|   |   +-- plots/                             # All visualization outputs (see below)
|   |   |
|   |   +-- predictions/
|   |   |   |-- RoPE_Conformer_test_predictions.csv
|   |   |   +-- test_predictions.csv
|   |   |
|   |   +-- reports/
|   |       |-- classification_report.txt
|   |       +-- evaluation_summary.txt
|   |
|   +-- exp_002/                               # Secondary experiment (training only)
|       |
|       +-- checkpoints/
|       |   |-- RoPE_Conformer.weights.h5
|       |   +-- best/                          # Best checkpoint (ckpt-140)
|       |   +-- last/                          # Final checkpoints (ckpt-147, ckpt-148)
|       |
|       +-- data/
|       |   |-- train_split.csv
|       |   |-- val_split.csv
|       |   +-- test_split.csv
|       |
|       +-- logs/                              # 5 training sessions
|       |
|       +-- metrics/
|       |   |-- dataset_report.txt
|       |   |-- model_params.txt
|       |   |-- RoPE_Conformer_test_results.csv
|       |   +-- RoPE_Conformer_training_history.csv
|       |
|       +-- plots/
|       |   |-- data_split_distribution.png
|       |   |-- RoPE_Conformer_architecture.png
|       |   +-- RoPE_Conformer_training_history.png
|       |
|       +-- predictions/
|           +-- RoPE_Conformer_test_predictions.csv
|
+-- models/
    +-- rope_conformer/
        |-- RoPE_Conformer.keras               # Keras saved model (primary)
        |-- RoPE_Conformer_architecture.json   # Architecture config
        |-- RoPE_Conformer_final.h5            # Final weights in h5 format
        |
        +-- RoPE_Conformer_saved_model/        # TensorFlow SavedModel format
            |-- fingerprint.pb
            |-- saved_model.pb
            +-- variables/
                |-- variables.data-00000-of-00001
                +-- variables.index
```

---

## Saved Artifacts

### Metrics — `experiments/exp_001/metrics/`

| File                              | Description                                              |
|-----------------------------------|----------------------------------------------------------|
| `evaluation_summary.csv`          | Top-level metrics: accuracy, F1, AUC, sample counts      |
| `classification_report.csv`       | Per-class precision, recall, F1, and support             |
| `per_class_accuracy.csv`          | Per-class accuracy for all 250 classes                   |
| `pr_auc_per_class.csv`            | Per-class PR-AUC values                                  |
| `roc_auc_per_class.csv`           | Per-class ROC-AUC values                                 |
| `confused_pairs.csv`              | Top confusion pairs with error counts                    |
| `misclassified_samples.csv`       | All 1,394 misclassified samples with confidence scores   |
| `hard_examples.csv`               | Lowest-confidence correct predictions                    |
| `RoPE_Conformer_training_history.csv` | Per-epoch loss and accuracy across all sessions      |
| `RoPE_Conformer_test_results.csv` | Full test set predictions with confidence scores         |
| `inference_performance.csv`       | Latency and throughput benchmarks                        |
| `model_params.txt`                | Parameter count breakdown                                |
| `model_size.csv`                  | Model file size statistics                               |
| `dataset_report.txt`              | Dataset statistics for this experiment                   |

### Plots — `experiments/exp_001/plots/`

| File                              | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `RoPE_Conformer_training_history.png` | Loss and accuracy curves across training epochs |
| `RoPE_Conformer_architecture.png` | Visual diagram of the model architecture         |
| `confusion_matrix_normalized.png` | Row-normalized confusion matrix (250 x 250)      |
| `confusion_matrix_raw.png`        | Raw count confusion matrix                       |
| `confused_pairs.png`              | Bar chart of top confusion pairs                 |
| `per_class_accuracy.png`          | Per-class accuracy distribution                  |
| `metrics_distribution.png`        | Precision, recall, and F1 distributions          |
| `roc_curves.png`                  | Macro and per-class ROC curves                   |
| `precision_recall_curves.png`     | Macro and per-class PR curves                    |
| `data_split_distribution.png`     | Train/val/test class distribution                |

---

## Notebooks

Training and evaluation notebooks are located in the project root notebooks directory.

The training notebook covers data loading from landmark CSV splits, sequence padding and masking, RoPE Conformer model definition with the rotary attention layer, training with callbacks (early stopping, checkpoint saving), and export to SavedModel and .h5 formats.

The evaluation notebook covers loading the saved model, running inference on the test split, computing all evaluation metrics (accuracy, F1, ROC-AUC, PR-AUC), generating confusion matrices, extracting hard examples and misclassified samples, and producing all plots saved under `exp_001/plots/`.

---

## Reproducing the Experiment

Two experiments are recorded in this repository. `exp_001` contains the full evaluation suite and is the primary reference. `exp_002` contains training runs and checkpoints but does not include a full classification report.

**To run evaluation from saved weights:**

1. Load the Keras model from `models/rope_conformer/RoPE_Conformer.keras` or the SavedModel from `models/rope_conformer/RoPE_Conformer_saved_model/`.
2. Run evaluation using the test split at `experiments/exp_001/data/test_split.csv`.
3. Reference reports and metrics are in `experiments/exp_001/metrics/` and `experiments/exp_001/reports/`.

**To resume or retrain from the best checkpoint:**

1. Restore from `experiments/exp_001/checkpoints/best/ckpt-140` or load `experiments/exp_001/checkpoints/RoPE_Conformer.weights.h5` directly.
2. The training data index is at `experiments/exp_001/data/train_split.csv`.
3. Seven training sessions were logged between April 7 and April 13, 2026 under `experiments/exp_001/logs/`.

---

## Known Limitations

**awake / wake confusion.** These two signs share handshape, facial location, and movement onset, differing only in a temporal extension that is insufficiently represented in the current training distribution. All models in this evaluation fail similarly on this pair, confirming it is a data-level problem. Targeted augmentation with speed-varied examples of both classes is the recommended fix.

**pencil / pen regression.** The pencil→pen confusion worsens in this model (12 errors) compared to BiGRU (7 errors). The distinction between these signs relies on fine-grained finger configuration rather than motion trajectory, which RoPE's positional encoding does not address. A handshape-specific auxiliary loss or feature injection would be needed to recover this.

**Single-dataset training.** This model was trained and evaluated on one internal dataset. Generalization to other sign language corpora such as WLASL or MS-ASL has not been validated.

**Deployment complexity.** The RoPE attention layer adds modest complexity relative to a plain BiGRU, which may require additional integration work for constrained deployment environments.
