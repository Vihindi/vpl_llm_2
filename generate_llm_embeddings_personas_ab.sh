#!/bin/bash

# Default model type
model_type="llama3"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model_type) model_type="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

other_subsets="grouped_personas"
# Only Persona A and Persona B
subsets=("Persona_A" "Persona_B")

echo "Processing subsets: ${subsets[@]}"

survey_size=100

for subset in "${subsets[@]}"
do
    echo "Processing $subset with model $model_type..."

    # Train split
    if [ ! -f "data/emb_grouped_personas_AB_${survey_size}/${model_type}/${subset}/train.jsonl" ]; then
        python -m hidden_context.data_utils.add_survey_contexts \
            --output_dir "data/emb_grouped_personas_AB_${survey_size}/" \
            --data_path "data/grouped_data_converted" \
            --data_subset "$subset" \
            --data_split "train" \
            --model_type "$model_type" \
            --other_subsets "$other_subsets" \
            --with_embeddings "True" \
            --survey_size "$survey_size" \
            --num_duplicates 2 \
            --controversial_only=False \
            --random_contexts=True
    else
        echo "Skipping $subset train split as it already exists."
    fi

    # Test split
    if [ ! -f "data/emb_grouped_personas_AB_${survey_size}/${model_type}/${subset}/test.jsonl" ]; then
        python -m hidden_context.data_utils.add_survey_contexts \
            --output_dir "data/emb_grouped_personas_AB_${survey_size}/" \
            --data_path "data/grouped_data_converted" \
            --data_subset "$subset" \
            --data_split "test" \
            --model_type "$model_type" \
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
