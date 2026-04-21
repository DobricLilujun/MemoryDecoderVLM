#!/bin/bash
set -e

# ========================================================================
# VLM Memory Decoder Training Pipeline
# ========================================================================
#
# Full pipeline:
#   Step 1: Save VLM embeddings (keys + vals) for the training split
#   Step 2: Build FAISS index from the saved embeddings
#   Step 3: KNN search → save sparse probability distributions
#   Step 4: Prepare text-only dataset for MemDec training
#   Step 5: Train Memory Decoder with KL distillation
#
# Prerequisites:
#   pip install faiss-cpu pyarrow loguru accelerate
#   pip install transformers datasets torch
#
# Usage:
#   bash scripts/vlm_pipeline.sh
# ========================================================================

# ---- Configuration ----------------------------------------------------
VLM_MODEL="Qwen/Qwen2-VL-2B-Instruct"
MEMDEC_MODEL="Qwen/Qwen2-0.5B"
DATASET="jmhessel/newyorker_caption_contest"
DATASET_CONFIG="explanation"
SPLIT="train"
HIDDEN_DIM=1536  # Qwen2-VL-2B hidden size

# Paths
DSTORE_DIR="./vlm_dstore"
MEMDEC_DATA_DIR="./vlm_memdec_data"
CHECKPOINT_DIR="./vlm_memdec_checkpoints"

# Derived paths
DSTORE_PATH="${DSTORE_DIR}/dstore_qwen2_vl_${SPLIT}_${HIDDEN_DIM}.arrow"
METADATA_PATH="${DSTORE_DIR}/metadata_${SPLIT}.json"
INDEX_PATH="${DSTORE_DIR}/${SPLIT}_${HIDDEN_DIM}.index"
VAL_PATH="${DSTORE_DIR}/${SPLIT}_vals.pkl"
KNN_OUTPUT_PATH="${DSTORE_DIR}/knn_qwen2_vl_${SPLIT}_${HIDDEN_DIM}.arrow"

# KNN Configuration
K=1024
KNN_TEMP=16.0
PROBE=32
NCENTROIDS=256    # Small dataset → fewer centroids
CODE_SIZE=32
NUM_KEYS_TO_ADD=1000000

# Training Configuration
BATCH_SIZE_TRAIN=8
GRAD_ACCUM=4
NUM_EPOCHS=30
LR=1e-3
ALPHA=0.5

echo "=========================================="
echo "VLM Memory Decoder Training Pipeline"
echo "=========================================="
echo "VLM:       ${VLM_MODEL}"
echo "MemDec:    ${MEMDEC_MODEL}"
echo "Dataset:   ${DATASET} (${DATASET_CONFIG})"
echo "Split:     ${SPLIT}"
echo "=========================================="
echo ""

# ---- Step 1: Save VLM Embeddings -------------------------------------
echo "[Step 1/5] Saving VLM embeddings..."
echo "  Output: ${DSTORE_PATH}"
echo ""

python vlm_save_embed.py \
    --vlm_model "${VLM_MODEL}" \
    --dataset "${DATASET}" \
    --dataset_config "${DATASET_CONFIG}" \
    --split "${SPLIT}" \
    --output_dir "${DSTORE_DIR}"

echo ""
echo "[Step 1/5] ✓ VLM embeddings saved"
echo ""

# ---- Step 2: Build FAISS Index ----------------------------------------
echo "[Step 2/5] Building FAISS index..."
echo "  Datastore: ${DSTORE_PATH}"
echo "  Centroids: ${NCENTROIDS}"
echo ""

python -m knn_utils.build_index \
    --dstore_path "${DSTORE_PATH}" \
    --num_keys_to_add_at_a_time ${NUM_KEYS_TO_ADD} \
    --ncentroids ${NCENTROIDS} \
    --code_size ${CODE_SIZE} \
    --probe ${PROBE}

echo ""
echo "[Step 2/5] ✓ FAISS index built"
echo ""

# ---- Step 3: KNN Search & Save Distributions --------------------------
echo "[Step 3/5] Computing KNN distributions..."
echo "  K=${K}, Temp=${KNN_TEMP}"
echo "  Output: ${KNN_OUTPUT_PATH}"
echo ""

python -m knn_utils.saveKNNMulti \
    --model_path "${VLM_MODEL}" \
    --dstore_path "${DSTORE_PATH}" \
    --val_path "${VAL_PATH}" \
    --index_path "${INDEX_PATH}" \
    --output_path "${KNN_OUTPUT_PATH}" \
    --k ${K} \
    --knn_temp ${KNN_TEMP} \
    --probe ${PROBE} \
    --batch_size 4096 \
    --ignore_first True \
    --threshold 0

echo ""
echo "[Step 3/5] ✓ KNN distributions saved"
echo ""

# ---- Step 4: Prepare Text-Only MemDec Dataset --------------------------
echo "[Step 4/5] Preparing text-only MemDec dataset..."
echo "  Metadata: ${METADATA_PATH}"
echo "  Output: ${MEMDEC_DATA_DIR}"
echo ""

python vlm_prepare_memdec_data.py \
    --vlm_model "${VLM_MODEL}" \
    --dataset "${DATASET}" \
    --dataset_config "${DATASET_CONFIG}" \
    --split "${SPLIT}" \
    --metadata_path "${METADATA_PATH}" \
    --output_dir "${MEMDEC_DATA_DIR}"

echo ""
echo "[Step 4/5] ✓ MemDec dataset prepared"
echo ""

# ---- Step 5: Train Memory Decoder -------------------------------------
echo "[Step 5/5] Training Memory Decoder..."
echo "  Model: ${MEMDEC_MODEL}"
echo "  KNN: ${KNN_OUTPUT_PATH}"
echo "  Checkpoints: ${CHECKPOINT_DIR}"
echo ""

python train_memdec_vlm.py \
    --model_name_or_path "${MEMDEC_MODEL}" \
    --dataset_name "${MEMDEC_DATA_DIR}" \
    --knn_save_path "${KNN_OUTPUT_PATH}" \
    --output_dir "${CHECKPOINT_DIR}" \
    --per_device_train_batch_size ${BATCH_SIZE_TRAIN} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --lr_scheduler_type linear \
    --num_train_epochs ${NUM_EPOCHS} \
    --alpha ${ALPHA} \
    --seed 42 \
    --checkpointing_steps epoch \
    --logging_steps 1

echo ""
echo "[Step 5/5] ✓ Training complete"
echo ""

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Outputs:"
echo "  - VLM embeddings: ${DSTORE_PATH}"
echo "  - FAISS index:    ${INDEX_PATH}"
echo "  - KNN dists:      ${KNN_OUTPUT_PATH}"
echo "  - MemDec data:    ${MEMDEC_DATA_DIR}"
echo "  - Checkpoints:    ${CHECKPOINT_DIR}"
echo "=========================================="
