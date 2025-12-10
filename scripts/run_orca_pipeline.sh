#!/bin/bash
export HF_HOME=/data/huggingface
export CUDA_VISIBLE_DEVICES=6
export WANDB_MODE=online

# Script to run the ORCA dataset processing pipeline
# Usage: ./run_orca_pipeline.sh

set -e  # Exit on any error

# Default arguments (modify these as needed)
ORCA_DATASET_PATH="${ORCA_DATASET_PATH:-/data/faive_lab/datasets/data_D4}"
ORCA_OUTPUT_PATH="${ORCA_OUTPUT_PATH:-/data/Ctrl-World/datasets}"
SVD_PATH="${SVD_PATH:-stabilityai/stable-video-diffusion-img2vid}"
DATASET_NAME="${DATASET_NAME:-orca_D4}"
DEBUG_FLAG="${DEBUG_FLAG:-}"
MAIN_PROCESS_PORT=12348

echo "=========================================="
echo "ORCA Dataset Processing Pipeline"
echo "=========================================="
echo "ORCA Dataset Path: $ORCA_DATASET_PATH"
echo "ORCA Output Path: $ORCA_OUTPUT_PATH"
echo "SVD Path: $SVD_PATH"
echo "Dataset Name: $DATASET_NAME"
echo "Debug Mode: ${DEBUG_FLAG:-disabled}"
echo "=========================================="
echo ""

# Step 1: Extract latent representations
echo "[Step 1/3] Running extract_latent_orca.py..."
echo "----------------------------------------"
accelerate launch dataset_example/extract_latent_orca.py \
    --orca_dataset_path "$ORCA_DATASET_PATH" \
    --orca_output_path "$ORCA_OUTPUT_PATH/$DATASET_NAME" \
    --svd_path "$SVD_PATH" \
    $DEBUG_FLAG

if [ $? -eq 0 ]; then
    echo "✓ extract_latent_orca.py completed successfully"
else
    echo "✗ extract_latent_orca.py failed"
    exit 1
fi

echo ""

# Step 2: Create meta information
echo "[Step 2/3] Running create_meta_info_orca.py..."
echo "----------------------------------------"
python dataset_meta_info/create_meta_info_orca.py \
    --orca_output_path "$ORCA_OUTPUT_PATH/$DATASET_NAME" \
    --dataset_name "$DATASET_NAME" \
    $DEBUG_FLAG

if [ $? -eq 0 ]; then
    echo "✓ create_meta_info_orca.py completed successfully"
else
    echo "✗ create_meta_info_orca.py failed"
    exit 1
fi

echo ""

# Step 3: Start training with the processed dataset
echo "[Step 3/3] Running train_wm.py..."
echo "----------------------------------------"
accelerate launch --main_process_port $MAIN_PROCESS_PORT scripts/train_wm.py  \
    --dataset_root_path "$ORCA_OUTPUT_PATH" \
    --dataset_meta_info_path "dataset_meta_info" \
    --dataset_names "$DATASET_NAME"

if [ $? -eq 0 ]; then
    echo "✓ train_wm.py completed successfully"
else
    echo "✗ train_wm.py failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Output location: $ORCA_OUTPUT_PATH"
echo "Meta info location: dataset_meta_info/$DATASET_NAME"
echo "Model checkpoints: model_ckpt/$DATASET_NAME"
