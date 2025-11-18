#!/bin/bash
set -e

echo "============================================================"
echo "开始运行所有 SCD 管线实验..."
echo "============================================================"

# --- 配置 ---

BATCH_SCRIPT="run_batch_experiments.py"
EVAL_SCRIPT="eval_from_refcsv.py"
REF_CSV="ami_dev.csv"
EMBED_FEATURE="ecapa"

COMMON_PARAMS=(
    --embed "$EMBED_FEATURE"
    --min-seg 1.2
    --peak-quantile 0.7 #
    --peak-min-dist 0.5
)


OUTDIR_B2="results_mutiscale_noise_high/"
echo ""
echo "[INFO] --------------------------------------------------"
echo "[INFO] 正在运行: mutiscale..."
python3 "$BATCH_SCRIPT" \
    --ref-csv "$REF_CSV" \
    --outdir "$OUTDIR_B2" \
    --embed ecapa \
    --backend multiscale_only \
    --scales 0.4 0.8 \
    "${COMMON_PARAMS[@]}"

echo "[INFO] 正在评估: mutiscale..."
python3 "$EVAL_SCRIPT" \
    --pred "$OUTDIR_B2/pred_fileid_boundary.csv" \
    --ref-csv "$REF_CSV" \
    --tol 0.5 | tee "$OUTDIR_B2/test_eval_tol0.5.txt"


# --- 实验 3: Baseline 1 (单尺度 J + Viterbi J) ---
OUTDIR_B1="results_pipline_noise_high"
echo ""
echo "[INFO] --------------------------------------------------"
echo "[INFO] 正在运行: Baseline 1 (单尺度 + SCDpipline)..."
python3 "$BATCH_SCRIPT" \
    --ref-csv "$REF_CSV" \
    --outdir "$OUTDIR_B1" \
    --embed "ecapa" \
    --backend cluster_hysteresis \
    "${COMMON_PARAMS[@]}"

echo "[INFO] 正在评估: Baseline 1..."
python3 "$EVAL_SCRIPT" \
    --pred "$OUTDIR_B1/pred_fileid_boundary.csv" \
    --ref-csv "$REF_CSV" \
    --tol 0.5 | tee "$OUTDIR_B1/test_eval_tol0.5.txt"


