import subprocess
import sys
import os

def run_generation():
    """Generates embedding data for missing subsets by executing data_utils script.
    
    Validates permutations over multiple personas and splits, triggering subprocess
    executions for each valid scenario.
    """
    python_executable = sys.executable
    script_path = "hidden_context.data_utils.add_survey_contexts"
    base_data_path = "data/grouped_data_converted"
    base_output_dir = "data/emb_grouped_personas_100/"
    model_type = "llama3"
    other_subsets = "grouped_personas"
    survey_size = "100"
    
    personas = ["Persona_A", "Persona_B", "Persona_C", "Persona_D", "Persona_E"]
    splits = ["train", "test"]

    print(f"Starting embedding generation with model: {model_type}")
    print(f"Python executable: {python_executable}")

    for persona in personas:
        print(f"\nProcessing {persona}...")
        for split in splits:
            print(f"  - Split: {split}")
            
            cmd = [
                python_executable, "-m", script_path,
                "--output_dir", base_output_dir,
                "--data_path", base_data_path,
                "--data_subset", persona,
                "--data_split", split,
                "--model_type", model_type,
                "--other_subsets", other_subsets,
                "--with_embeddings", "True",
                "--survey_size", survey_size,
                "--num_duplicates", "1"
            ]
            
            env = os.environ.copy()
            env["WANDB_MODE"] = "online"
            try:
                subprocess.run(cmd, env=env, check=True)
                print(f"    -> Success for {persona} {split}")
            except subprocess.CalledProcessError as e:
                print(f"    -> Error processing {persona} {split}: {e}")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    run_generation()
