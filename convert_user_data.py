import json
import os
import glob
from tqdm import tqdm

def convert_line(line):
    """Converts a single line from user format to VPL/HH-RLHF format.

    Args:
        line (str): A JSON string representing a single line of user data.

    Returns:
        dict: The converted data in VPL format, or None if the label is invalid.
    """
    data = json.loads(line)
    
    prompt = data.get('prompt', '')
    resp_a = data.get('response_A', '')
    resp_b = data.get('response_B', '')
    label = data.get('preference_label')
    
    if label == 1:
        chosen_resp = resp_a
        rejected_resp = resp_b
    elif label == 2:
        chosen_resp = resp_b
        rejected_resp = resp_a
    else:
        return None

    chosen_full = f"Human: {prompt}\n\nAssistant: {chosen_resp}"
    rejected_full = f"Human: {prompt}\n\nAssistant: {rejected_resp}"
    
    responses = [rejected_resp, chosen_resp]
    new_label = 1
    
    new_entry = {
        "chosen": chosen_full,
        "rejected": rejected_full,
        "Index": data.get("convo_ID", "0"),
        "data_subset": data.get("persona_ID", "default"),
        "prompt": prompt,
        "responses": responses,
        "original_label": label,
        "label": new_label,
        "objective": "helpful",
        "controversial": False,
        "communication_genre": data.get("communication_genre", "unknown"),
        "survey_options": True
    }
    
    return new_entry

def main():
    """Main execution entry point to process and convert data across personas."""
    base_dir = "data/grouped_data"
    output_base_dir = "data/grouped_data_converted"
    
    personas = ["Persona_A", "Persona_B", "Persona_C", "Persona_D", "Persona_E"]
    splits = ["train", "test"]
    
    print(f"Converting data from {base_dir} to {output_base_dir}...")
    
    for persona in personas:
        for split in splits:
            input_path = os.path.join(base_dir, persona, f"{split}.jsonl")
            output_dir = os.path.join(output_base_dir, persona)
            output_path = os.path.join(output_dir, f"{split}.jsonl")
            skipped_path = os.path.join(output_dir, f"skipped_{split}.jsonl")
            
            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                continue
                
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Processing {persona}/{split}...")
            
            with open(input_path, 'r', encoding='utf-8') as f_in, \
                 open(output_path, 'w', encoding='utf-8') as f_out, \
                 open(skipped_path, 'w', encoding='utf-8') as f_skipped:
                
                lines = f_in.readlines()
                skipped_count = 0
                for line in tqdm(lines, desc=f"{persona} {split}"):
                    if not line.strip():
                        continue
                    try:
                        new_entry = convert_line(line)
                        if new_entry:
                            f_out.write(json.dumps(new_entry) + "\n")
                        else:
                            f_skipped.write(line)
                            skipped_count += 1
                    except Exception as e:
                        print(f"Error processing line: {e}")
                        f_skipped.write(line)
                        skipped_count += 1
                
                if skipped_count > 0:
                    print(f"Skipped {skipped_count} entries. Saved to {skipped_path}")
                else:
                    f_skipped.close()
                    try:
                        os.remove(skipped_path)
                    except:
                        pass

    print("Conversion complete.")

if __name__ == "__main__":
    main()
