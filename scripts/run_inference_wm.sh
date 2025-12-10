export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export HF_HOME=/data/huggingface

accelerate launch --main_process_port 29502 scripts/inference_wm.py