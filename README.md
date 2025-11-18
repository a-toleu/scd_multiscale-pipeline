# 🔊 Speaker Change Detection (SCD) Pipeline

This repository implements a versatile Speaker Change Detection (SCD) pipeline capable of using various embedding backends (e.g., ECAPA, WavLM) and different detection strategies, including a **Multi-Scale Detector** and a **Cluster-Hysteresis-based Pipeline**.

The core pipeline is written in Python, leveraging `scipy`, `numpy`, and optional deep learning libraries (`torch`, `speechbrain`, `transformers`) for advanced embeddings.

## ✨ Features

* **Modular Embeddings:** Supports `logmel` (baseline), `mfcc`, and deep speaker embeddings like **ECAPA**, **WavLM**, and **Wav2Vec2**.
* **Detection Backends:**
    * `multiscale_only`: Boundary detection based on calculating embedding jump scores across multiple time scales and fusing the results via voting/clustering.
    * `cluster_hysteresis`: A pipeline combining single-scale embedding jump scores with spectral flux for initial segment creation, followed by **Agglomerative Clustering** of segments and a **Hysteresis** decoding to finalize boundaries.
    * `viterbi_only`: (Placeholder, but referenced in config for potential future Viterbi decoding).
* **Batch Processing:** Includes a script (`run_batch_experiments.py`) to process an entire dataset defined in a CSV file and generate a timing report.
* **Experimentation:** The `run_all.sh` script is set up to run and evaluate specific configurations (e.g., multiscale vs. cluster-hysteresis baseline).

## 🛠️ Installation

### Prerequisites

You need **Python 3.8+** and access to standard audio/signal processing libraries. For deep learning backends, **PyTorch** and the `huggingface/transformers` library are required.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/scd_multiscale-pipeline.git](https://github.com/YourUsername/scd_multiscale-pipeline.git)
    cd scd_multiscale-pipeline
    ```

2.  **Install core dependencies:**
    ```bash
    pip install numpy scipy pandas scikit-learn matplotlib
    ```

3.  **Install deep learning dependencies (for ECAPA, WavLM, etc.):**
    ```bash
    pip install torch torchaudio
    # For ECAPA-TDNN:
    pip install speechbrain
    # For WavLM/Wav2Vec2:
    pip install transformers
    ```
    *Note: The script includes fallbacks to `logmel` if deep learning models cannot be loaded.*

## ⚙️ Configuration (SCDConfig)

The main configuration is managed by the `SCDConfig` dataclass in `scd_pipeline.py`. Key parameters you might want to adjust include:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `embed` | Embedding feature type: `logmel`, `mfcc`, `ecapa`, `wavlm`, etc. | `logmel` |
| `backend` | Detection method: `multiscale_only`, `cluster_hysteresis`, or `viterbi_only`. | `cluster_hysteresis` |
| `block_win_s`, `block_hop_s` | Window/hop size for the block embeddings (seconds). | `0.5`, `0.25` |
| `scales` | Tuple of window lengths for the `multiscale_only` backend. | `(0.4, 0.8, 1.6)` |
| `hysteresis_high`, `hysteresis_low` | Thresholds for the final boundary decoding in `cluster_hysteresis`. | `0.7`, `0.3` |
| `agglom_distance_threshold` | Distance threshold for Agglomerative Clustering in the pipeline. | `0.5` |

## 🚀 Running Experiments

### 1. Prepare your Dataset

You need a CSV file (e.g., `ami_dev.csv`) with an **`audio_path`** column pointing to the WAV files. You may also need a **ground truth boundary file** for evaluation (e.g., RTTM or a CSV with boundaries).

### 2. Batch Processing

Use `run_batch_experiments.py` to run the SCD pipeline over your dataset. The script includes a feature to add white noise for robustness testing (`x_noisy = add_white_noise_level(x, noise_db=10)`).

```bash
python3 run_batch_experiments.py \
    --ref-csv path/to/your/dataset.csv \
    --outdir results_ecapa_test \
    --embed ecapa \
    --backend cluster_hysteresis \
    --block-win 0.8 \
    --block-hop 0.4 \
    --peak-quantile 0.75 \
    --min-seg 1.2
