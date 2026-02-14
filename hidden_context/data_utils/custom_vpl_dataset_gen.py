
import argparse
import random
import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from copy import deepcopy
from transformers import HfArgumentParser
from hidden_context.data_utils.data_processing import (
    ScriptArguments as BaseScriptArguments,
    generate_embeddings_with_llm,
)
from dataclasses import dataclass, field
import torch

"""
Custom VPL Dataset Generation
=============================

This script generates a dataset for VPL (Variational Preference Learning) using a custom data source (Persona A/B),
aligning with the structure used for the UltraFeedback dataset in the VPL codebase.

## Data Flow Comparison: UltraFeedback vs. Custom Dataset

### 1. Source Data & Formatting
- **UltraFeedback (`ultrafeedback_augment.py`)**:
    - **Input**: Raw UltraFeedback data + Binarized Preferences.
    - **Formatting**: Constructs `chosen` and `rejected` fields with specific formatting:
      ```
      "chosen": "Human: {prompt}\n\nAssistant: {response_A}"
      "rejected": "Human: {prompt}\n\nAssistant: {response_B}"
      ```
    - **Output**: Saved as `UltraFeedback_single_P_4/train.jsonl`.

- **Custom Dataset (`custom_vpl_dataset_gen.py`)**:
    - **Input**: `labeled_Main_final_dataset_Persona_A.jsonl` (and B).
    - **Formatting**: Applies strictly identical formatting in `format_example`:
      ```
      formatted_chosen = f"Human: {prompt}\n\nAssistant: {chosen_text}"
      formatted_rejected = f"Human: {prompt}\n\nAssistant: {rejected_text}"
      ```
    - **Result**: The text passed to the encoder is structurally identical to UltraFeedback.

### 2. Embedding Generation (`generate_embeddings_with_llm`)
- **UltraFeedback**:
    - The `add_survey_contexts.py` script calls `generate_embeddings_with_llm`.
    - It reads the formatted `chosen`/`rejected` fields.
    - Uses GPT-2/Llama to get the last hidden state of the final token.
    - Adds an `embeddings` column: `{'embedding_chosen': [...], 'embedding_rejected': [...]}`.

- **Custom Dataset**:
    - This script calls `generate_embeddings_with_llm` directly on the formatted dataset.
    - **Process**: Identical. The same function from `data_processing.py` is used.
    - **Output**: Identical `embeddings` column structure.

### 3. Pooling Mechanism (K-N-M)
- **UltraFeedback Logic (`add_survey_contexts.py`)**:
    - **Pool (K)**: Selects K (e.g., 100) "survey options" (controversial pairs) and saves them to `survey_100.jsonl` to ensure consistency across runs.
    - **Context Selection (N)**: For *each* data point, it randomly samples N (e.g., 8) examples *from this fixed pool*.
    - **VAE Input**: The VAE Encoder receives these N contexts as pairs of `(chosen, rejected)`.
      - Input Structure: `[{"embedding_chosen": ..., "embedding_rejected": ...}, ...]` (N times).

- **Custom Implementation**:
    - **Pool (K)**: We select a fixed K=100 pool (`pool_size`) from the balanced subset.
    - **Context Selection (N)**: We sample N=8 (`context_length`) examples from this in-memory pool for each data point.
    - **Augmentation (M)**: We duplicate endpoints M=2 (`num_duplicates`) times, resampling the context for each duplicate.
    - **Alignment**: We ensure Persona A and Persona B use the *exact same* context indices for the same `convo_ID`.
    - **VAE Input**: Identical structure. The `embeddings` for both `chosen` and `rejected` responses in the context pool are provided to the encoder.

"""
@dataclass
class ScriptArguments(BaseScriptArguments):
    persona_a_path: str = field(
        default="data/convoDrift_sample/business_emails/labeled_Main_final_dataset_Persona_A.jsonl",
        metadata={"help": "Path to Persona A dataset"}
    )
    persona_b_path: str = field(
        default="data/convoDrift_sample/business_emails/labeled_Main_final_dataset_Persona_B.jsonl",
        metadata={"help": "Path to Persona B dataset"}
    )
    subset_size: int = field(
        default=4000,
        metadata={"help": "Number of examples to select per persona"}
    )
    pool_size: int = field(
        default=100,
        metadata={"help": "Size of the fixed global context pool (K)"}
    )
    context_length: int = field(
        default=8,
        metadata={"help": "Number of context examples to sample per data point (N)"}
    )
    num_duplicates: int = field(
        default=4,
        metadata={"help": "Number of times to duplicate each example with different context (M)"}
    )
    with_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to generate embeddings (Forces True for this script)"}
    )
    data_split: str = field(
        default="train",
        metadata={"help": "Dataset split name (e.g., train, test)"}
    )

def load_jsonl(file_path):
    data = []
    # Use utf-8-sig to handle BOM if present
    print(f"[INFO] Reading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Skipping invalid JSON at line {i+1}: {e}")
                    print(f"[DEBUG] Line content: {line[:50]}...")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to read file {file_path}: {e}")
        raise
        
    print(f"[INFO] Successfully loaded {len(data)} records from {os.path.basename(file_path)}")
    return data

def select_balanced_subset(data_list, target_size=4000):
    """
    Selects a balanced subset of data based on 'communication_genre'.
    Returns the selected convo_IDs.
    """
    df = pd.DataFrame(data_list)
    genres = df['communication_genre'].unique()
    n_genres = len(genres)
    
    # Calculate samples per genre
    samples_per_genre = target_size // n_genres
    
    selected_ids = []
    
    for genre in genres:
        genre_df = df[df['communication_genre'] == genre]
        # If a genre has fewer samples than needed, take all of them.
        n_samples = min(len(genre_df), samples_per_genre)
        sampled_genre = genre_df.sample(n=n_samples, random_state=42)
        selected_ids.extend(sampled_genre['convo_ID'].tolist())
    
    # If we are short of target_size, fill with random sampling from remaining
    if len(selected_ids) < target_size:
        remaining_df = df[~df['convo_ID'].isin(selected_ids)]
        needed = target_size - len(selected_ids)
        if len(remaining_df) >= needed:
             extra_sample = remaining_df.sample(n=needed, random_state=42)
             selected_ids.extend(extra_sample['convo_ID'].tolist())
        else:
            # Take all remaining if still not enough (unlikely if dataset is large enough)
            selected_ids.extend(remaining_df['convo_ID'].tolist())
        
    return set(selected_ids)

def format_example(row):
    """
    Formats a raw row into the VPL expected format.
    prompt -> prompt
    response_A/B + preference_label -> chosen/rejected
    """
    prompt = row['prompt']
    response_A = row['response_A']
    response_B = row['response_B']
    label = row['preference_label']

    if label == 1:
        chosen_text = response_A
        rejected_text = response_B
    else:
        chosen_text = response_B
        rejected_text = response_A

    formatted_chosen = f"Human: {prompt}\n\nAssistant: {chosen_text}"
    formatted_rejected = f"Human: {prompt}\n\nAssistant: {rejected_text}"

    return {
        'prompt': prompt,
        'chosen': formatted_chosen,
        'rejected': formatted_rejected,
        'responses': [chosen_text, rejected_text],
        'convo_ID': row['convo_ID'],
        'data_subset': row['persona_ID'], # Persona_A or Persona_B
        'Original_label': label,
        'direction': row['direction'],
        'communication_genre': row['communication_genre'],
        'scenario': row['scenario']
    }

def process_single_persona(dataset_path, args, persona_name="Persona"):
    # 1. Load Data
    print(f"\n[INFO] Processing {persona_name} from {dataset_path}")
    data = load_jsonl(dataset_path)
    
    # 2. Select Balanced Subset
    print(f"[INFO] Selecting balanced subset of size {args.subset_size} for {persona_name}...")
    selected_convo_ids = select_balanced_subset(data, target_size=args.subset_size)
    print(f"[INFO] Selected {len(selected_convo_ids)} conversation IDs.")

    # 3. Filter dataset
    subset = [row for row in data if row['convo_ID'] in selected_convo_ids]
    
    # Sort for consistency
    subset.sort(key=lambda x: x['convo_ID'])

    # Format data
    formatted_data = [format_example(row) for row in subset]
    ds = Dataset.from_list(formatted_data)
    print("Here is the dataset",ds[0],ds[1])
    # 4. Generate Embeddings
    if args.with_embeddings:
        # IMPORTANT: Set synthetic_dataset=True so generate_embeddings_with_llm uses the passed dataset
        args.synthetic_dataset = True
        
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")
        
        print(f"\n[INFO] Generating embeddings for {persona_name}...")
        print(f"[DEBUG] Sample Input: {ds[0]['chosen'][:100]}...") 
        ds = generate_embeddings_with_llm(args, ds)
        print(f"[INFO] Completed embeddings. Rows: {len(ds)}")

    # 5. Select Pool
    print(f"\n[INFO] Selecting global context pool (Size K={args.pool_size}) for {persona_name}...")
    pool_indices = np.random.choice(range(len(ds)), args.pool_size, replace=False)
    print(f"[DEBUG] Selected {len(pool_indices)} unique pool indices.")
    
    # Extract pool data for lookup
    def extract_context_dict(dataset, indices):
        context_pool = {}
        for idx in indices:
            row = dataset[int(idx)]
            ctx = {
                'original_id': row['convo_ID'],
                'chosen': row['chosen'],
                'rejected': row['rejected']
            }
            if 'embeddings' in row and row['embeddings'] is not None:
                ctx['embedding_chosen'] = row['embeddings']['embedding_chosen']
                ctx['embedding_rejected'] = row['embeddings']['embedding_rejected']
            context_pool[int(idx)] = ctx
        return context_pool

    print(f"[INFO] Extracting pool contexts...")
    pool_dict = extract_context_dict(ds, pool_indices)
    
    # 6. Augment Data
    print(f"\n[INFO] Generating context plans (Duplicates M={args.num_duplicates}, Context N={args.context_length})...")
    
    context_plans = []
    for m in range(args.num_duplicates):
        choices_for_pass = []
        for _ in range(len(ds)):
            chosen_keys = np.random.choice(list(pool_dict.keys()), args.context_length, replace=False)
            choices_for_pass.append(chosen_keys)
        context_plans.append(choices_for_pass)

    def apply_augmentation(dataset, pool_dict, plans):
        augmented_datasets = []
        for idx, plan in enumerate(plans):
            contexts_embeddings_column = [] # Renamed for clarity
            contexts_text_column = []       # New column for raw text

            for i in range(len(dataset)):
                chosen_keys = plan[i]
                row_contexts = [pool_dict[k] for k in chosen_keys]
                
                # Append to embeddings list
                contexts_embeddings_column.append(row_contexts)
                
                # Append to text list (stripped of embeddings to save space/memory if needed, 
                # but for now we can just use the same dict or a subset)
                # The VAE script expects 'chosen' and 'rejected' keys in the context dicts.
                # text_contexts = []
                # for ctx in row_contexts:
                #     text_contexts.append({
                #         "chosen": ctx["chosen"],
                #         "rejected": ctx["rejected"]
                #     })
                # contexts_text_column.append(text_contexts)
                contexts_text_column.append(row_contexts)

            # Clone dataset and add columns
            new_ds = dataset.add_column("contexts_embeddings", contexts_embeddings_column)
            # Add the 'contexts' column required when fixed_llm_embeddings=False
            new_ds = new_ds.add_column("contexts", contexts_text_column)
            
            # Add context_length column if needed
            new_ds = new_ds.add_column("context_length", [len(c) for c in contexts_embeddings_column])
            
            augmented_datasets.append(new_ds)
        
        return concatenate_datasets(augmented_datasets)

    print(f"\n[INFO] Applying augmentation to {persona_name}...")
    final_ds = apply_augmentation(ds, pool_dict, context_plans)
    print(f"[INFO] {persona_name} Final Size: {len(final_ds)}")
    
    return final_ds

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Set seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Script Args: {script_args}")
    
    # Process Persona A
    ds_a = process_single_persona(script_args.persona_a_path, script_args, "Persona A")
    
    # Process Persona B
    ds_b = process_single_persona(script_args.persona_b_path, script_args, "Persona B")
    
    # Save Output
    # Mimic HH-RLHF structure: output_dir/{model_type}/grouped_data/Persona_{A,B}/{split}.jsonl
    base_output_dir = os.path.join(script_args.output_dir, script_args.model_type, "grouped_data")
    
    # Save Persona A
    dir_a = os.path.join(base_output_dir, "Persona_A")
    os.makedirs(dir_a, exist_ok=True)
    path_a = os.path.join(dir_a, f"{script_args.data_split}.jsonl")
    ds_a.to_json(path_a)
    print(f"Saved Persona A ({script_args.data_split}) to {path_a}")
    
    # Save Persona B
    dir_b = os.path.join(base_output_dir, "Persona_B")
    os.makedirs(dir_b, exist_ok=True)
    path_b = os.path.join(dir_b, f"{script_args.data_split}.jsonl")
    ds_b.to_json(path_b)
    print(f"Saved Persona B ({script_args.data_split}) to {path_b}")
