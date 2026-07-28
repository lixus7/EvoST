#!/bin/bash
# ===========================================================================
# A2TTA on PEMS05.
# ---------------------------------------------------------------------------
# Backbone : Online-AN per-year .pkl checkpoints (oneline_st_an_pems05-<seed>/<year>/*.pkl)
#            Falls back to retrain_st_pems05 if missing.
# Method   : freeze backbone + add a FiLM calibrator. Process test windows
#            chronologically, update on delayed labels every 64 windows, and
#            use a disposable per-window local clone for tta_ctx_local.
# ---------------------------------------------------------------------------
# Output   : log/PEMS05/a2tta_*-<seed>/*.log  (per-year metrics)
#            run_logs/a2tta_lite_results.csv  (per year × method × seed × horizon)
# Usage    :
#   cd eac/
#   bash scripts/a2tta_lite_pems05_run.sh                    # full matrix, default GPU
#   GPU=2 SEEDS="51" bash scripts/a2tta_lite_pems05_run.sh   # single seed on GPU 2
#   FAST_DEV_RUN=1 bash scripts/a2tta_lite_pems05_run.sh     # debug: 1 year, 4 batches
#   METHODS="tta_ctx_local" bash scripts/a2tta_lite_pems05_run.sh  # only Ours
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."  # → eac/

# ---------------------------------------------------------------------------
# Background mode: NOHUP=1 bash scripts/a2tta_lite_pems05_run.sh
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    LOG_FILE="run_logs/a2tta_pems05_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    echo "[nohup] PID=$!"
    echo "[nohup] tail -f $LOG_FILE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Knobs (env-overridable)
# ---------------------------------------------------------------------------
GPU=${GPU:-1}
# Online-AN seeds 51..55 are present in this repo; default to those so the
# backbone .pkl always resolves. Override with SEEDS="42 43" etc.
SEEDS=${SEEDS:-"51 52 53 54 55"}
BACKBONE_LOGNAME=${BACKBONE_LOGNAME:-"oneline_st_an_pems05"}
BACKBONE_FALLBACK=${BACKBONE_FALLBACK:-"retrain_st_pems05"}
# Methods to run (space-separated). The core component ablation is:
#   backbone     : frozen base, no calibrator
#   calibrator   : warmed-up FiLM calibrator, no online TTA
#   tta_all      : delayed-label TTA on the full pool, no local clone
#   tta_ctx_local: full A2TTA (Ours)
METHODS=${METHODS:-"backbone calibrator tta_all tta_ctx_local"}

# Hyperparameters
ADAPT_LR=${ADAPT_LR:-1e-3}
ADAPT_STEPS=${ADAPT_STEPS:-3}
ADAPT_EVERY_BATCHES=${ADAPT_EVERY_BATCHES:-1}
ADAPT_EVERY_WINDOWS=${ADAPT_EVERY_WINDOWS:-64}
LABEL_DELAY_STEPS=${LABEL_DELAY_STEPS:-12}
BUDGET_FRAC=${BUDGET_FRAC:-0.25}
POOL_SIZE=${POOL_SIZE:-512}
LAMBDA_CONS=${LAMBDA_CONS:-0.05}
LAMBDA_REG=${LAMBDA_REG:-1e-4}
HIDDEN_DIM=${HIDDEN_DIM:-64}
NODE_EMB_DIM=${NODE_EMB_DIM:-16}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
WARMUP_LR=${WARMUP_LR:-1e-3}
EVAL_BATCH=${EVAL_BATCH:-1}
SENSOR_METADATA_ROOT=${SENSOR_METADATA_ROOT:-data/sensor_metadata}
FAST_DEV_RUN=${FAST_DEV_RUN:-0}
CSV_PATH=${CSV_PATH:-"run_logs/a2tta_lite_pems05_results.csv"}

echo "[A2TTA] GPU=$GPU SEEDS='$SEEDS' METHODS='$METHODS'"
echo "[A2TTA] backbone=$BACKBONE_LOGNAME (fallback=$BACKBONE_FALLBACK)"
echo "[A2TTA] adapt_lr=$ADAPT_LR steps=$ADAPT_STEPS budget=$BUDGET_FRAC pool=$POOL_SIZE"
echo "[A2TTA] eval_batch=$EVAL_BATCH cadence=$ADAPT_EVERY_WINDOWS label_delay=$LABEL_DELAY_STEPS"
echo "[A2TTA] λ_cons=$LAMBDA_CONS hidden=$HIDDEN_DIM warmup=$WARMUP_EPOCHS"
echo "[A2TTA] csv=$CSV_PATH fast_dev_run=$FAST_DEV_RUN"

run_one() {
    local METHOD=$1
    local SEED=$2
    local LOGNAME="a2tta_${METHOD}_pems05"
    echo "==================== [A2TTA] method=$METHOD seed=$SEED ===================="
    python a2tta_main.py \
        --conf conf/PEMS05/a2tta_olan_pems05.json \
        --logname "$LOGNAME" \
        --method "$METHOD" \
        --dataset PEMS05 \
        --gpuid "$GPU" \
        --seed "$SEED" \
        --backbone_ckpt_logname "$BACKBONE_LOGNAME" \
        --backbone_ckpt_logname_fallback "$BACKBONE_FALLBACK" \
        --backbone_method TrafficStream \
        --freeze_backbone 1 \
        --adapter_hidden_dim "$HIDDEN_DIM" \
        --node_emb_dim "$NODE_EMB_DIM" \
        --adapt_lr "$ADAPT_LR" \
        --adapt_steps "$ADAPT_STEPS" \
        --adapt_every_batches "$ADAPT_EVERY_BATCHES" \
        --adapt_every_windows "$ADAPT_EVERY_WINDOWS" \
        --label_delay_steps "$LABEL_DELAY_STEPS" \
        --budget_frac "$BUDGET_FRAC" \
        --candidate_pool_size "$POOL_SIZE" \
        --lambda_cons "$LAMBDA_CONS" \
        --lambda_reg "$LAMBDA_REG" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --warmup_lr "$WARMUP_LR" \
        --eval_batch_size "$EVAL_BATCH" \
        --input_normalization train \
        --sensor_id_mode metadata \
        --sensor_metadata_root "$SENSOR_METADATA_ROOT" \
        --csv_path "$CSV_PATH" \
        --fast_dev_run "$FAST_DEV_RUN"
}

for SEED in $SEEDS; do
    for METHOD in $METHODS; do
        run_one "$METHOD" "$SEED"
    done
done

echo "==================== A2TTA ALL DONE ===================="
echo "[A2TTA] Results CSV: $CSV_PATH"
