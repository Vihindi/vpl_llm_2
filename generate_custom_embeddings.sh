#!/bin/bash
# Generate Custom Embeddings for Persona A/B
# Usage: ./generate_custom_embeddings.sh [model_type]
# Example: ./generate_custom_embeddings.sh gpt2

# 1. Parse Arguments (Default to 'gpt2' if not provided)
# This matches the mechanism in 'generate_llm_embeddings_UF_P_2.sh'
MODEL_TYPE=${1:-"gpt2"}

# Dataset Parameters
SUBSET_SIZE=4000
POOL_SIZE=100
CONTEXT_LENGTH=8
NUM_DUPLICATES=2

# Input Paths
PERSONA_A_PATH="data/convoDrift_sample/business_emails/labeled_Main_final_dataset_Persona_A.jsonl"
PERSONA_B_PATH="data/convoDrift_sample/business_emails/labeled_Main_final_dataset_Persona_B.jsonl"

# Output Directory
OUTPUT_DIR="data/custom_vpl_balanced"

echo "Running Custom VPL Dataset Generation..."
echo "Model: $MODEL_TYPE"
echo "Subset Size: $SUBSET_SIZE | Pool Size: $POOL_SIZE"

# 2. Run Generation Script
# The python script automatically handles embedding dimension based on --model_type.
# e.g. gpt2 -> 768, gpt2-medium -> 1024, llama -> 4096.
# Train Split
echo "Processing TRAIN split..."
python -m hidden_context.data_utils.custom_vpl_dataset_gen \
    --data_path "." \
    --persona_a_path "data/grouped_data/Persona_A/train.jsonl" \
    --persona_b_path "data/grouped_data/Persona_B/train.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --with_embeddings True \
    --subset_size $SUBSET_SIZE \
    --pool_size $POOL_SIZE \
    --context_length $CONTEXT_LENGTH \
    --num_duplicates $NUM_DUPLICATES \
    --data_split "train"

# Test Split
echo "Processing TEST split..."
python -m hidden_context.data_utils.custom_vpl_dataset_gen \
    --data_path "." \
    --persona_a_path "data/grouped_data/Persona_A/test.jsonl" \
    --persona_b_path "data/grouped_data/Persona_B/test.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --with_embeddings True \
    --subset_size $SUBSET_SIZE \
    --pool_size $POOL_SIZE \
    --context_length $CONTEXT_LENGTH \
    --num_duplicates 1 \
    --data_split "test"
echo "Done! Output saved to $OUTPUT_DIR"
