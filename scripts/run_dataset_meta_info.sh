#!/bin/bash
export HF_HOME=/data/huggingface
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online

set -e  # Exit on any error

# Define ORCA output path
ORCA_OUTPUT_PATH="${ORCA_OUTPUT_PATH:-/data/Ctrl-World/datasets/2025-12-24T01-20-32}"

# Define list of dataset names to process
DATASET_NAMES=(
    "data_fix_cam_close_mimicgen_hand_mask_10fps"
    "data_fix_cam_close_sine_hand_mask_10fps"
    "data_fix_cam_fixed_ee_object_collisions_open_close_10fps"
    "data_fix_cam_fixed_ee_open_close_10fps"
    "data_fix_cam_fixed_ee_random_10fps"
    "data_fix_cam_fixed_ee_sine_object_collision_high_res_10fps"
    "data_fix_cam_fixed_ee_sine_object_collision_high_res_25fps"
    "data_fix_cam_fixed_ee_sine_object_collision_high_res_50fps"
    "data_fix_cam_fixed_ee_sine_object_collisions_10fps"
    "data_fix_cam_fixed_ee_sine_object_collisions_25fps"
    "data_fix_cam_fixed_ee_sine_object_collisions_50fps"
    "data_fix_cam_fixed_fingers_random_object_locations_10fps"
    "data_fix_cam_view_close_random_motions_10fps"
    "data_sine_fixed_ee_10fps"
)

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a  # Automatically export all variables
    source .env
    set +a  # Disable automatic export
fi

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

# Loop through each dataset
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    echo "========================================"
    echo "Processing dataset: $DATASET_NAME"
    echo "========================================"
    
    echo "----------------------------------------"
    python dataset_meta_info/create_meta_info_orca.py \
        --orca_output_path "$ORCA_OUTPUT_PATH/$DATASET_NAME" \
        --dataset_name "$DATASET_NAME" \
        $DEBUG_FLAG

    if [ $? -eq 0 ]; then
        echo "✓ create_meta_info_orca.py completed successfully for $DATASET_NAME"
        send_discord_notification "✅ Step 2/3 Complete: Meta information created successfully for $DATASET_NAME" "success"
    else
        echo "✗ create_meta_info_orca.py failed for $DATASET_NAME"
        send_discord_notification "❌ Step 2/3 Failed: Meta information creation encountered an error for $DATASET_NAME" "failure"
        exit 1
    fi

    echo ""
done

echo "========================================"
echo "All datasets processed successfully!"
echo "========================================"