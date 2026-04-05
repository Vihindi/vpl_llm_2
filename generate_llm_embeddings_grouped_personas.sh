#!/bin/bash

# Default model type
model_type="llama3"

# Parse command line arguments
model_name_or_path="/content/drive/MyDrive/models/llama-3-8b-instruct"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model_type) model_type="$2"; shift ;;
        -model_path) model_name_or_path="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

other_subsets="grouped_personas"
# Using array for subsets
subsets=("Persona_A" "Persona_B" "Persona_C" "Persona_D" "Persona_E")

echo "Processing subsets: ${subsets[@]}"

survey_size=100

for subset in "${subsets[@]}"
do
    echo "Processing $subset with model $model_type..."
    
    # Train split
    TRAIN_PATH="data/emb_grouped_personas_Without_MMP_/${model_type}/${subset}/train.jsonl"
    echo "[DEBUG] CWD: $(pwd)"
    echo "[DEBUG] Checking train path: ${TRAIN_PATH}"
    echo "[DEBUG] Absolute path: $(pwd)/${TRAIN_PATH}"
    if [ -f "$TRAIN_PATH" ]; then echo "[DEBUG] File EXISTS"; else echo "[DEBUG] File NOT FOUND"; fi
    if [ ! -f "$TRAIN_PATH" ]; then
        python -m hidden_context.data_utils.add_survey_contexts \
            --output_dir "data/emb_grouped_personas_Without_MMP_/" \
            --data_path "data/grouped_data_converted" \
            --data_subset "$subset" \
            --data_split "train" \
            --model_type "$model_type" \
            ${model_name_or_path:+--model_name_or_path "$model_name_or_path"} \
            --other_subsets "$other_subsets" \
            --with_embeddings "True" \
            --survey_size "$survey_size" \
            --num_duplicates 2 \
            --controversial_only=False \
            --random_contexts=True
    else
        echo "Skipping $subset train split as it already exists."
    fi

    Test split
    TEST_PATH="data/emb_grouped_personas_Without_MMP_/${model_type}/${subset}/test.jsonl"
    echo "[DEBUG] Checking test path: ${TEST_PATH}"
    echo "[DEBUG] Absolute path: $(pwd)/${TEST_PATH}"
    if [ -f "$TEST_PATH" ]; then echo "[DEBUG] File EXISTS"; else echo "[DEBUG] File NOT FOUND"; fi
    if [ ! -f "$TEST_PATH" ]; then
        python -m hidden_context.data_utils.add_survey_contexts \
            --output_dir "data/emb_grouped_personas_Without_MMP_/" \
            --data_path "data/grouped_data_converted" \
            --data_subset "$subset" \
            --data_split "test" \
            --model_type "$model_type" \
            ${model_name_or_path:+--model_name_or_path "$model_name_or_path"} \
            --other_subsets "$other_subsets" \
            --with_embeddings "True" \
            --survey_size "$survey_size" \
            --num_duplicates 2 \
            --controversial_only=False \
            --random_contexts=True
    else
        echo "Skipping $subset test split as it already exists."
    fi
done
