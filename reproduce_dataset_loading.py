
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Literal
from datasets import load_dataset, concatenate_datasets, Value
from transformers import HfArgumentParser

# Mock classes to avoid full dependency chain
@dataclass
class ScriptArguments:
    data_path: str = field(default="data/custom_vpl_balanced")
    data_subset: str = field(default="both")
    other_subsets: str = field(default="custom_personas")
    fixed_llm_embeddings: bool = field(default=True)
    fixed_contexts: bool = field(default=True)

# Copying get_hh_rlhf_dataset logic roughly
def get_hh_rlhf_dataset(data_path, other_subsets, split="train"):
    datasets = []
    if other_subsets == 'custom_personas':
        target_filename = "train.jsonl" if split == "train" else "test.jsonl"
        
        def load_custom_file(persona_name, subset_name):
            base_grouped_path = os.path.join(data_path, "grouped_data")
            if not os.path.exists(base_grouped_path):
                 print(f"Path {base_grouped_path} does not exist, falling back to data/grouped_data")
                 base_grouped_path = os.path.join("data", "grouped_data")
            
            file_path = os.path.join(base_grouped_path, persona_name, target_filename)
            print(f"Looking for file: {file_path}")
            
            if os.path.exists(file_path):
                print(f"Loading custom dataset ({split}): {file_path}")
                ds = load_dataset("json", data_files=file_path, split="train") 
                return ds
            else:
                print(f"Warning: Custom dataset file not found: {file_path}")
                return None

        ds_a = load_custom_file("Persona_A", "Persona_A")
        if ds_a: datasets.append(ds_a)
        
        return concatenate_datasets(datasets)
    return None

def main():
    print("reproducing dataset loading...")
    # Simulate arguments from shell script
    args = ScriptArguments()
    print(f"Args: {args}")

    ds = get_hh_rlhf_dataset(args.data_path, args.other_subsets, split="train")
    
    if ds:
        print(f"Dataset loaded. Size: {len(ds)}")
        print(f"Columns: {ds.column_names}")
        
        print("Checking for embeddings column...")
        if "embeddings" in ds.column_names:
             print("SUCCESS: 'embeddings' column found.")
        else:
             print("FAILURE: 'embeddings' column NOT found.")
             
        # Simulate accessing it like the preprocessor
        if args.fixed_llm_embeddings:
            print("Simulating fixed_llm_embeddings access...")
            try:
                # Iterate a bit
                for i, item in enumerate(ds):
                    if i > 2: break
                    _ = item["embeddings"] # strict access
                print("Access successful (unexpected).")
            except KeyError as e:
                print(f"Caught expected KeyError: {e}")
            except Exception as e:
                print(f"Caught unexpected exception: {e}")
    else:
        print("Dataset failed to load.")

if __name__ == "__main__":
    main()
