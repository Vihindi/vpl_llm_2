import json
import os
import random
import shutil
from collections import defaultdict

def load_jsonl(file_path):
    data = []
    print(f"[INFO] Reading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Skipping invalid JSON line: {e}")
                    continue
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return []
    print(f"[INFO] Loaded {len(data)} records.")
    return data

def save_jsonl(data, file_path):
    print(f"[INFO] Saving {len(data)} records to {file_path}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def process_persona_data(input_path, output_dir, seed=42):
    random.seed(seed)
    data = load_jsonl(input_path)
    
    # Filter
    filtered_data = [
        d for d in data 
        if d.get('communication_genre') not in ['business email', 'casual email']
    ]
    print(f"[INFO] Filtered down to {len(filtered_data)} records (removed business/casual emails).")
    
    # Group by genre
    genre_groups = defaultdict(list)
    for d in filtered_data:
        genre = d.get('communication_genre')
        if genre:
            genre_groups[genre].append(d)
            
    train_data = []
    test_data = []
    
    print("[INFO] Splitting data by genre (80% Train, 20% Test)...")
    for genre, items in genre_groups.items():
        # Shuffle
        random.shuffle(items)
        
        # Calculate split index
        n_total = len(items)
        n_test = int(n_total * 0.2)
        n_train = n_total - n_test
        
        # Split
        test_items = items[:n_test]
        train_items = items[n_test:]
        
        test_data.extend(test_items)
        train_data.extend(train_items)
        
        print(f"  > Genre '{genre}': Total={n_total}, Train={len(train_items)}, Test={len(test_items)}")

    # Save
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    
    save_jsonl(train_data, train_path)
    save_jsonl(test_data, test_path)

def main():
    base_dir = "data/convoDrift_sample"
    output_base = "data/grouped_data"
    
    # Process all personas
    personas = ["A", "B", "C", "D", "E"]
    
    for p in personas:
        input_file = os.path.join(base_dir, f"business_emails/labeled_Main_final_dataset_Persona_{p}.jsonl")
        output_dir = os.path.join(output_base, f"Persona_{p}")
        
        print(f"\n[INFO] Processing Persona {p}...")
        process_persona_data(input_file, output_dir)

if __name__ == "__main__":
    main()
