# Adaptive Latent Persona Modelling Framework for LLMs

This repository contains the codebase for training, evaluating, and simulating Adaptive Latent Persona Modelling models using Large Language Models (LLMs). The core project focuses on learning dynamic, multi-persona user preferences and accurately modeling potential preference drift.

## Dataset

This framework utilizes the **convoDrift** dataset for modeling preference drift. The dataset is available at [https://github.com/Vihindi/_CONVODRIFT_.git](https://github.com/Vihindi/_CONVODRIFT_.git)

## Environment Setup

Follow these steps to set up the codebase.

1. **Navigate to the Project Directory**
   Ensure you are in the root directory (`vpl_llm_2`).
   ```bash
   cd path/to/vpl_llm_2
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   # Activate on Windows:
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   Install the required Python modules via `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

---

## Codebase Overview & Architecture

### 1. Data Preparation
- **`convert_user_data.py`**: Converts raw user-provided data fragments into proper HH-RLHF JSONL formats, standardizing prompt, chosen, and rejected mappings.
- **`hidden_context/data_utils/add_survey_contexts.py`**: Pre-engineers datasets to inject contextual histories and invokes LLMs to generate high-dimensional embeddings for subsequent use.
- **`run_generation.py`**: A batch utility wrapper that automates the execution of embedding generation commands efficiently across multiple personas and data splits.

### 2. Training the Model
- **`hidden_context/train_llm_vae_preference_model.py`**: The primary executable pipeline for training the VAE mechanism on underlying reward preferences. Employs Parameter-Efficient Fine-Tuning (PEFT, specifically LoRA) to train smoothly without excessive VRAM overheads.
- **`hidden_context/train_llm_preference_model.py`**: Houses the fallback structural definitions and utilities utilized strictly by baseline (non-VAE) sequence reward models.

### 3. Emulating Personas
- **`create_multi_persona_simulation.py`**: Constructs intricate temporal simulations modeling "user drift." It pieces paired test cases successively starting from a reliable context segment ("Seed"), migrating through chaotic mixed boundaries ("Drift evaluation"), and lastly checking settling performance ("Final evaluation").
- **`vae_session_inference.py`**: Contains the critical `VAESessionInference` class. It manages evaluating live reward scenarios continuously keeping track of a sliding window memory format, recalculating the inferred target user state (`z_anchor`).

### 4. Continuous Evaluation
- **`evaluate_reward_scores.py`**: Calculates Tier-1 (Per-Persona Accuracy) and Tier-2 (Cross-Persona Latent Accuracy) metrics, ensuring that the latent distribution correctly encodes varying persona traits.
- **`evaluate_adaptation_velocity.py`**: Generates tracking visualizations by comparing varying momentum variables over a temporal grid, determining precisely how fast the model adapts and restores accuracy subsequent to an unrecognized paradigm drift.

---

## Basic Worflow Execution

1. **Format Base Datasets**  
   Pre-format JSON configurations:
   ```bash
   python convert_user_data.py
   ```

2. **Generate Embeddings**  
   Pre-compute contexts required for dynamic VAE loading:
   ```bash
   python run_generation.py
   ```

3. **Train the Preference Model**  
   Configure training inputs mapping local weights tracking inside the internal arguments structure:
   ```bash
   python -m hidden_context.train_llm_vae_preference_model \
       --model_name <base_llm_name> \
       --bf16 True 
   # View the `ScriptArguments` dataclass inside the script for all parameters.
   ```

4. **Construct Sequential Evaluator Simulations**  
   Map transitions between test personas (e.g. `Persona_A` transitioning to `Persona_B`):
   ```bash
   python create_multi_persona_simulation.py --all
   ```

5. **Examine Inference Adjustments**  
   Trace adaptation velocities against baseline performances:
   ```bash
   python evaluate_adaptation_velocity.py --vae_model_path <path_to_saved_model>
   ```
