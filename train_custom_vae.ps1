# PowerShell Training script for VAE Preference Model on Custom Data
# Optimized for L4 GPU (24GB VRAM)

param (
    [string]$ModelName = "gpt2",
    [string]$DataPath = "data/custom_vpl_balanced",
    [string]$OutputDir = "results/custom_vae_model"
)

# L4 GPU Optimization
$BatchSize = 8
$GradAccum = 2

Write-Host "Starting VAE Training..."
Write-Host "Model: $ModelName"
Write-Host "Data Path: $DataPath"
Write-Host "Output Dir: $OutputDir"
Write-Host "Batch Size: $BatchSize (Accum: $GradAccum)"

python -m hidden_context.train_llm_vae_preference_model `
    --model_name "$ModelName" `
    --data_path "$DataPath" `
    --data_subset "all" `
    --other_subsets "custom_personas" `
    --output_dir "$OutputDir" `
    --per_device_train_batch_size $BatchSize `
    --per_device_eval_batch_size $BatchSize `
    --gradient_accumulation_steps $GradAccum `
    --train_dataset_size 4000 `
    --eval_dataset_size 1000 `
    --learning_rate 3e-6 `
    --weight_decay 0.001 `
    --num_train_epochs 3 `
    --logging_steps 10 `
    --save_strategy "epoch" `
    --evaluation_strategy "epoch" `
    --fixed_llm_embeddings True `
    --use_last_token_embedding True `
    --remove_unused_columns False 

