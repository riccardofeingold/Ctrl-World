#!/bin/bash
export HF_HOME=/data/huggingface
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online

set -e  # Exit on any error

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a  # Automatically export all variables
    source .env
    set +a  # Disable automatic export
fi

# Default arguments (modify these as needed)
# Dataset path which is converted from HDF5 file to lerobot format (usually in faive_lab directory)
ORCA_DATASET_PATH="${ORCA_DATASET_PATH:-/data/faive_lab/datasets/tests}"
# Where to store the train/val dataset after processing
ORCA_OUTPUT_PATH="${ORCA_OUTPUT_PATH:-/data/Ctrl-World/datasets}"
# Which VAE model to use for latent extraction
SVD_PATH="${SVD_PATH:-stabilityai/stable-video-diffusion-img2vid}"
# Name of the processed dataset
DATASET_NAME="${DATASET_NAME:-test_random_data}"

# Debug flag
DEBUG_FLAG="${DEBUG_FLAG:-}"

# Randomly assign a port for the main process to avoid conflicts
MAIN_PROCESS_PORT=$((10000 + RANDOM % 55535))

# Desired FPS for processing
DESIRED_FPS=10

# WandB tag for the experiment
WANDB_TAG="${DATASET_NAME}_${DESIRED_FPS}Hz_one_view}"

# skip flags
SKIP_EXTRACT_LATENT="${SKIP_EXTRACT_LATENT:-true}"
SKIP_CREATE_META_INFO="${SKIP_CREATE_META_INFO:-true}"
SKIP_TRAINING="${SKIP_TRAINING:-false}"

# Function to send Discord notification
send_discord_notification() {
    local message="$1"
    local status="$2"  # "success" or "failure"
    
    if [ -z "$DISCORD_WEBHOOK_URL" ]; then
        return 0
    fi
    
    # Set color based on status
    local color
    if [ "$status" = "success" ]; then
        color=3066993  # Green
    else
        color=15158332  # Red
    fi
    
    # Get hostname and timestamp
    local hostname=$(hostname)
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create JSON payload
    local json_payload=$(cat <<EOF
{
  "embeds": [{
    "title": "ORCA Pipeline Notification",
    "description": "$message",
    "color": $color,
    "fields": [
      {
        "name": "Dataset",
        "value": "$DATASET_NAME",
        "inline": true
      },
      {
        "name": "Host",
        "value": "$hostname",
        "inline": true
      },
      {
        "name": "Timestamp",
        "value": "$timestamp",
        "inline": false
      }
    ]
  }]
}
EOF
)
    
    # Send notification
    curl -H "Content-Type: application/json" \
         -d "$json_payload" \
         "$DISCORD_WEBHOOK_URL" \
         --silent --output /dev/null
}

echo "=========================================="
echo "ORCA Dataset Processing Pipeline"
echo "=========================================="
echo "ORCA Dataset Path: $ORCA_DATASET_PATH"
echo "ORCA Output Path: $ORCA_OUTPUT_PATH"
echo "SVD Path: $SVD_PATH"
echo "Dataset Name: $DATASET_NAME"
echo "Main Process Port: $MAIN_PROCESS_PORT"
echo "Debug Mode: ${DEBUG_FLAG:-disabled}"
echo "Discord Notifications: ${DISCORD_WEBHOOK_URL:+enabled}"
echo "=========================================="
echo ""

# Send initial notification
send_discord_notification "🚀 Pipeline started for dataset: $DATASET_NAME" "success"

# Step 1: Extract latent representations
if [ "$SKIP_EXTRACT_LATENT" = "true" ]; then
    echo "[Step 1/3] Skipping extract_latent_orca.py as per configuration."
else
    echo "[Step 1/3] Running extract_latent_orca.py..."
    echo "----------------------------------------"
    python dataset_example/extract_latent_orca.py \
        --orca_dataset_path "$ORCA_DATASET_PATH" \
        --orca_output_path "$ORCA_OUTPUT_PATH/$DATASET_NAME" \
        --svd_path "$SVD_PATH" \
        --frame_size "(256,256)" \
        $DEBUG_FLAG

    if [ $? -eq 0 ]; then
        echo "✓ extract_latent_orca.py completed successfully"
        send_discord_notification "✅ Step 1/3 Complete: Latent extraction finished successfully" "success"
    else
        echo "✗ extract_latent_orca.py failed"
        send_discord_notification "❌ Step 1/3 Failed: Latent extraction encountered an error" "failure"
        exit 1
    fi

    echo ""
fi

# Step 2: Create meta information
if [ "$SKIP_CREATE_META_INFO" = "true" ]; then
    echo "[Step 2/3] Skipping create_meta_info_orca.py as per configuration."
else
    echo "[Step 2/3] Running create_meta_info_orca.py..."
    echo "----------------------------------------"
    python dataset_meta_info/create_meta_info_orca.py \
        --orca_output_path "$ORCA_OUTPUT_PATH/$DATASET_NAME" \
        --dataset_name "$DATASET_NAME" \
        $DEBUG_FLAG

    if [ $? -eq 0 ]; then
        echo "✓ create_meta_info_orca.py completed successfully"
        send_discord_notification "✅ Step 2/3 Complete: Meta information created successfully" "success"
    else
        echo "✗ create_meta_info_orca.py failed"
        send_discord_notification "❌ Step 2/3 Failed: Meta information creation encountered an error" "failure"
        exit 1
    fi

    echo ""
fi

# Step 3: Start training with the processed dataset
echo "[Step 3/3] Running train_wm.py..."
echo "----------------------------------------"
if [ "$SKIP_TRAINING" = "true" ]; then
    echo "[Step 3/3] Skipping train_wm.py as per configuration."
    exit 0
fi

accelerate launch --main_process_port $MAIN_PROCESS_PORT scripts/train_wm.py  \
    --dataset_root_path "$ORCA_OUTPUT_PATH" \
    --dataset_meta_info_path "dataset_meta_info" \
    --dataset_names "$DATASET_NAME" \
    --fps $DESIRED_FPS \
    --tag "$WANDB_TAG" 

if [ $? -eq 0 ]; then
    echo "✓ train_wm.py completed successfully"
    send_discord_notification "✅ Step 3/3 Complete: Training finished successfully" "success"
else
    echo "✗ train_wm.py failed"
    send_discord_notification "❌ Step 3/3 Failed: Training encountered an error" "failure"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Output location: $ORCA_OUTPUT_PATH"
echo "Meta info location: dataset_meta_info/$DATASET_NAME"
echo "Model checkpoints: model_ckpt/$DATASET_NAME"

# Send final success notification
send_discord_notification "🎉 Pipeline completed successfully! All 3 steps finished for dataset: $DATASET_NAME" "success"
