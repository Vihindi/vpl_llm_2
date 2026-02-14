#!/bin/bash
# Training script for VAE Preference Model on Custom Data (Optimized for Separation)
# Run ID: A3 + B2 (KL Weight 0.05, Latent Dim 64)

# Default arguments
MODEL_NAME=${1:-"gpt2"}
DATA_PATH=${2:-"data/custom_vpl_balanced"}
OUTPUT_DIR=${3:-"results/custom_vae_optimized_v1"}

# L4 GPU Optimization
BATCH_SIZE=8
GRAD_ACCUM=2

echo "Starting VAE Training (Optimized Run)..."
echo "Model: $MODEL_NAME"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE (Accum: $GRAD_ACCUM)"
echo "Config: KL=0.05, Latent=64, Hidden=256"

python -m hidden_context.train_llm_vae_preference_model \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --data_subset "all" \
    --other_subsets "custom_personas" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --train_dataset_size 4000 \
    --eval_dataset_size 1000 \
    --learning_rate 3e-6 \
    --weight_decay 0.001 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --fixed_llm_embeddings False \
    --fixed_contexts True \
    --use_last_token_embedding True \
    --remove_unused_columns False \
    --kl_loss_weight 0.05 \
    --latent_dim 64 \
    --hidden_dim 256 \
    --use_annealing True

echo "Training complete."
