import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, cast

import numpy as np
import torch
import random
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainingArguments,
    TrainerCallback,
)
from transformers.utils import PaddingStrategy
from .vae_utils import VAETrainer, VAEModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .train_llm_preference_model import (
    get_step_decay_lr_lambda,
    get_cosine_decay_lr_lambda,
    RewardModelType,
    DataSubset,
    get_hh_rlhf_dataset,
    concatenate_datasets
)
from datasets import disable_progress_bar
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()
disable_progress_bar()


@dataclass
class ScriptArguments:
    local_rank: int = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this "
                    "if the model that you want to train doesn't fit on a single GPU."
        },
    )
    disable_tqdm: bool = field(
        default=False,
        metadata={"help": "Whether to disable tqdm progress bars"},
    )
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=3e-6)
    weight_decay: float = field(default=0.001)
    model_name: str = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. "
                    "E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    data_path: str = field(
        default="Anthropic/hh-rlhf",
    )
    data_subset: str = field(
        default="both",
        metadata={
            "help": "Which subset of the data to use. You can choose between 'both', "
                    "'helpful', or 'harmless'."
        },
    )
    reward_model_type: str = field(
        default="base",
        metadata={
            "help": "The type of reward model to use. You can choose between "
                    "'base', 'mean_and_variance', or 'categorical'."
        },
    )
    num_atoms: int = field(
        default=10,
        metadata={
            "help": "The number of atoms to use for the categorical reward model."
        },
    )
    entropy_coeff: float = field(
        default=0.1,
        metadata={"help": "The entropy coefficient for the categorical reward model."},
    )
    variance_penalty: float = field(
        default=0.0,
        metadata={
            "help": "The variance penalty for the mean and variance reward model."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default "
                    "for your model",
        },
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to "
                    "sacrifice a little precision and have a supported GPU."
        },
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to "
                    "sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_dataset_size: int = field(
        default=0,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_dataset_size: int = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    # Added to support user overrides
    output_dir: str = field(default=None, metadata={"help": "Override output directory"})
    logging_steps: int = field(default=100, metadata={"help": "Logging steps"})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy"})
    evaluation_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Remove unused columns"})
    max_length: int = field(default=1024)
    eval_first_step: bool = field(
        default=True,
        metadata={"help": "Whether to run eval after the first step"},
    )
    log_dir: str = field(default="data/reward_models/hh_rlhf")
    kl_loss_weight: float = field(default=0.0, metadata={"help": "weight for KLD loss"})
    
    # [VPL Paper Hyperparameters]
    # latent_dim: Size of the user latent vector 'z'.
    # hidden_dim: Size of internal layers in encoders/decoders.
    # encoder_embed_dim / decoder_embed_dim: Dimensions of the LLM representations used as input.
    latent_dim: int = field(default=512, metadata={"help": "dimension of latent user vector"})    # todo: 64
    hidden_dim: int = field(default=512, metadata={"help": "dimension of hidden layer in vae"})    # todo: 256
    encoder_embed_dim: int = field(default=1024, metadata={"help": "dimension of LLM embeddings for encoder"})
    decoder_embed_dim: int = field(default=1024, metadata={"help": "dimension of LLM embeddings for decoder"})
    use_annealing: bool = field(default=True, metadata={"help": "Whether to use annealing for learning rate"})
    
    # fixed_contexts: If True, uses pre-computed embeddings for the context history C.
    # This acts as the input to the VAE Encoder (SequenceEncoder).
    fixed_contexts: bool = field(
        default=False,
        metadata={"help": "whether to use pre-calculated embeddings for contexts (encoder inputs)"}
    )
    # fixed_llm_embeddings: If True, uses pre-computed embeddings for the target query/response.
    # This acts as the input to the VAE Decoder (Reward Model).
    fixed_llm_embeddings: bool = field(
        default=False,
        metadata={"help": "whether to use pre-calculated embeddings for decoder inputs"}
    )
    seed: int = field(default=0)
    controversial_only: bool = field(
        default=False,
        metadata={"help": "whether to only include controversial data"}
    )
    up_sampling: bool = field(
        default=False,
        metadata={"help": "whether to upsample controversial data during training phase"}
    )
    one_user: str = field(
        default=None,
        metadata={"help": "whether to only train and evaluate on one single user"}
    )
    other_subsets: str = field(
        default=None,
        metadata={"help": "specify the group of subsets if not using helpful/harmless. You can choose between"
                          "ultra_feedback, pos_neg, set, single."},
    )
    use_last_token_embedding: bool = field(
        default=False,
        metadata={"help": "whether to use the last token embedding of last layer as LLM embeddings"}
    )

class HHRLHFPreprocessor(object):
    """
    [VPL Data Processing]
    Responsible for structuring the inputs into:
    1. Context C (user history): A list of previous interactions (chosen/rejected pairs).
    2. Target (current interaction): The current chosen/rejected pair to be evaluated.
    
    This splits the raw dataset examples into the 'contexts' used by the Encoder 
    and the 'targets' used by the Decoder.
    """
    def __init__(self, args, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.args = args
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        """
        Example Input:
            examples: Dict of lists from Dataset
            {
                "chosen": ["Text A1...", "Text A2..."],
                "rejected": ["Text B1...", "Text B2..."],
                "contexts": [
                    [{"chosen": "Hist1_A", "rejected": "Hist1_B"}, ...],  # User 1 history
                    [{"chosen": "Hist2_A", "rejected": "Hist2_B"}, ...]   # User 2 history
                ]
            }
        
        Example Output:
            new_examples: Dict of lists
            {
                "input_ids_chosen": [[101, ...], ...],  # Encoded Target Chosen
                "contexts_tokens": [
                   [ {"input_ids_chosen": ..., "input_ids_rejected": ...}, ... ], # Encoded History
                   ...
                ]
            }
        """
        if self.args.fixed_llm_embeddings:
            new_examples: dict = {
                "embedding_chosen": [],
                "embedding_rejected": [],
                "contexts_embeddings": [],
                "max_lengths": []
            }
            for embeddings, contexts in zip(
                    examples["embeddings"], examples["contexts_embeddings"]
            ):
                new_examples["embedding_chosen"].append(embeddings["embedding_chosen"])
                new_examples["embedding_rejected"].append(embeddings["embedding_rejected"])
                contexts_embeddings = [{"embedding_chosen": context["embedding_chosen"],
                                        "embedding_rejected": context["embedding_rejected"]}
                                       for context in contexts]
                new_examples["contexts_embeddings"].append(contexts_embeddings)
                new_examples["max_lengths"].append(0)
            new_examples["user_type"] = examples["data_subset"]
            return new_examples

        new_examples: dict = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "max_lengths": []
        }
        if self.args.fixed_contexts:
            new_examples["contexts_embeddings"] = []
        else:
            new_examples["contexts_tokens"] = []
        for chosen, rejected, contexts, user_type in zip(
                examples["chosen"], examples["rejected"], examples["contexts"], examples["data_subset"]
        ):
            max_length = 0
            tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
            tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(
                tokenized_chosen["attention_mask"]
            )
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_rejected["attention_mask"]
            )
            max_length = max(max_length, len(tokenized_chosen["input_ids"]))
            max_length = max(max_length, len(tokenized_rejected["input_ids"]))

            if self.args.fixed_contexts:
                # DEBUG: Handle missing embeddings gracefully with logging
                try:
                    contexts_embeddings = [{"embedding_chosen": context["embedding_chosen"],
                                            "embedding_rejected": context["embedding_rejected"]}
                                           for context in contexts]
                    new_examples["contexts_embeddings"].append(contexts_embeddings)
                except KeyError as e:
                    print(f"\\n[ERROR] KeyError accessing context embeddings: {e}")
                    if len(contexts) > 0:
                        print(f"[DEBUG] Context keys present: {contexts[0].keys()}")
                        if 'embeddings' in contexts[0]:
                            print(f"[DEBUG] 'embeddings' content: {contexts[0]['embeddings']}")
                        else:
                            # Fallback if embeddings are nested differently or missing
                            pass
                    raise e
            else:
                tokenized_context = []
                # Tokenize the contexts.
                for context in contexts:
                    chosen, rejected = context["chosen"], context["rejected"]
                    tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
                    tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)
                    tokenized_context.append(
                        {
                            "input_ids_chosen": tokenized_chosen["input_ids"],
                            "attention_mask_chosen": tokenized_chosen["attention_mask"],
                            "input_ids_rejected": tokenized_rejected["input_ids"],
                            "attention_mask_rejected": tokenized_rejected["attention_mask"],
                        }
                    )
                    max_length = max(max_length, len(tokenized_chosen["input_ids"]))
                    max_length = max(max_length, len(tokenized_rejected["input_ids"]))
                new_examples["contexts_tokens"].append(tokenized_context)
            new_examples["max_lengths"].append(max_length)
        new_examples["user_type"] = examples["data_subset"]
        return new_examples


trainer_classes: Dict[RewardModelType, Type[VAETrainer]] = {
    "vae": VAETrainer,
}


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    """
    [VPL Batching]
    Handles the complexity of variable-length user contexts.
    
    Since each user context C has a different number of past interactions, this collator:
    1. Flattens all context interactions from a batch of users into a single list.
    2. Creates 'seq_start_end' indices so the SequenceEncoder knows which interactions belong to which user.
       Format: [(start_idx_1, end_idx_1), (start_idx_2, end_idx_2), ...]
    """
    args: ScriptArguments
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Example Input:
            features: List of samples (Batch Size)
            [
                { "input_ids_chosen": ..., "contexts_tokens": [C1, C2] },
                { "input_ids_chosen": ..., "contexts_tokens": [C3, C4, C5] }
            ]
        
        Example Output:
            batch: Dict of Tensors
            {
                "input_ids_chosen": (B, L_max),
                "contexts_input_ids_chosen": (Sum_Contexts, L_max),  # Flattened history!
                "seq_start_end": Tensor([[0, 2], [2, 5]])  # Validation of flattening
            }
        """
        if self.args.other_subsets is None:
            user_mapping = {
                "helpful": 0,
                "harmless": 1,
            }
        else:   # TODO: set subsets here
            if self.args.other_subsets == 'ultra_feedback':
                subsets = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
            elif self.args.other_subsets == 'single' or self.args.other_subsets == '84':
                subsets = ['8', '4', '2', '1']
            elif self.args.other_subsets == 'custom_personas':
                subsets = ['Persona_A', 'Persona_B']
            else:
                print("SEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE THEEEEEEEE SUBSET IS NULLLLLLLLLLLL")
                subsets = []
            user_mapping = {subset: idx for idx, subset in enumerate(subsets)}
        if self.args.fixed_llm_embeddings:
            batch_size = len(features)
            embeddings_chosen = []
            embeddings_rejected = []
            contexts_embeddings_chosen = []
            contexts_embeddings_rejected = []
            contexts_lengths = [0]
            for feature in features:
                embeddings_chosen.append(
                    feature["embedding_chosen"]
                )
                embeddings_rejected.append(
                    feature["embedding_rejected"]
                )
                contexts_embeddings_chosen.extend(
                    [
                        context["embedding_chosen"] for context in feature["contexts_embeddings"]
                    ]
                )
                contexts_embeddings_rejected.extend(
                    [
                        context["embedding_rejected"] for context in feature["contexts_embeddings"]
                    ]
                )
                contexts_lengths.append(len(feature["contexts_embeddings"]))
            contexts_lengths = torch.cumsum(torch.tensor(contexts_lengths), dim=0)
            seq_start_end = torch.stack(
                [contexts_lengths[:-1], contexts_lengths[1:]], dim=1
            )
            user_type = [user_mapping[feature["user_type"]] for feature in features]
            assert len(seq_start_end) == batch_size
            return {
                "embeddings_chosen": embeddings_chosen,
                "embeddings_rejected": embeddings_rejected,
                "contexts_embeddings_chosen": contexts_embeddings_chosen,
                "contexts_embeddings_rejected": contexts_embeddings_rejected,
                "seq_start_end": seq_start_end,
                "return_loss": True,
                "user_type": user_type,
            }
        if self.args.fixed_contexts:
            batch_size = len(features)
            features_chosen = []
            features_rejected = []
            contexts_embeddings_chosen = []
            contexts_embeddings_rejected = []
            contexts_lengths = [0]
            for feature in features:
                features_chosen.append(
                    {
                        "input_ids": feature["input_ids_chosen"],
                        "attention_mask": feature["attention_mask_chosen"],
                    }
                )
                features_rejected.append(
                    {
                        "input_ids": feature["input_ids_rejected"],
                        "attention_mask": feature["attention_mask_rejected"],
                    }
                )
                # Creating a flattened list of contexts.
                contexts_embeddings_chosen.extend(
                    [
                        context["embedding_chosen"] for context in feature["contexts_embeddings"]
                    ]
                )
                contexts_embeddings_rejected.extend(
                    [
                        context["embedding_rejected"] for context in feature["contexts_embeddings"]
                    ]
                )
                # Keep track of the start and end of each sequence.
                contexts_lengths.append(len(feature["contexts_embeddings"]))

            batch = self.tokenizer.pad(
                features_chosen + features_rejected,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            input_ids = batch["input_ids"].view(
                2, batch_size, batch["input_ids"].shape[-1]
            )
            attention_mask = batch["attention_mask"].view(
                2, batch_size, batch["attention_mask"].shape[-1]
            )

            context_lengths = torch.cumsum(torch.tensor(contexts_lengths), dim=0)
            seq_start_end = torch.stack(
                [context_lengths[:-1], context_lengths[1:]], dim=1
            )
            user_type = [user_mapping[feature["user_type"]] for feature in features]
            assert len(seq_start_end) == batch_size

            return {
                "input_ids_chosen": input_ids[0],
                "attention_mask_chosen": attention_mask[0],
                "input_ids_rejected": input_ids[1],
                "attention_mask_rejected": attention_mask[1],
                "contexts_embeddings_chosen": contexts_embeddings_chosen,
                "contexts_embeddings_rejected": contexts_embeddings_rejected,
                "seq_start_end": seq_start_end,
                "return_loss": True,
                "user_type": user_type,
            }

        batch_size = len(features)
        features_chosen = []
        features_rejected = []
        contexts_features_chosen = []
        contexts_features_rejected = []
        contexts_lengths = [0]
        for feature in features:
            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )

            # Creating a flattened list of contexts.
            contexts_features_chosen.extend(
                [
                    {
                        "input_ids": context["input_ids_chosen"],
                        "attention_mask": context["attention_mask_chosen"],
                    }
                    for context in feature["contexts_tokens"]
                ]
            )
            contexts_features_rejected.extend(
                [
                    {
                        "input_ids": context["input_ids_rejected"],
                        "attention_mask": context["attention_mask_rejected"],
                    }
                    for context in feature["contexts_tokens"]
                ]
            )
            # Keep track of the start and end of each sequence.
            contexts_lengths.append(len(feature["contexts_tokens"]))

        batch = self.tokenizer.pad(
            features_chosen + features_rejected + contexts_features_chosen + contexts_features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        input_ids = batch["input_ids"][:2 * batch_size].view(
            2, batch_size, batch["input_ids"].shape[-1]
        )
        attention_mask = batch["attention_mask"][:2 * batch_size].view(
            2, batch_size, batch["attention_mask"].shape[-1]
        )

        contexts_lengths = torch.cumsum(torch.tensor(contexts_lengths), dim=0)
        seq_start_end = torch.stack(
            [contexts_lengths[:-1], contexts_lengths[1:]], dim=1
        )
        user_type = [user_mapping[feature["user_type"]] for feature in features]
        assert len(seq_start_end) == batch_size
        context_ids = batch["input_ids"][2 * batch_size:].view(
            2, contexts_lengths[-1], batch["input_ids"].shape[-1]
        )
        context_attention_mask = batch["attention_mask"][2 * batch_size:].view(
            2, contexts_lengths[-1], batch["attention_mask"].shape[-1]
        )

        return {
            "input_ids_chosen": input_ids[0],
            "attention_mask_chosen": attention_mask[0],
            "input_ids_rejected": input_ids[1],
            "attention_mask_rejected": attention_mask[1],
            "contexts_input_ids_chosen": context_ids[0],
            "contexts_attention_mask_chosen": context_attention_mask[0],
            "contexts_input_ids_rejected": context_ids[1],
            "contexts_attention_mask_rejected": context_attention_mask[1],
            "seq_start_end": seq_start_end,
            "return_loss": True,
            "user_type": user_type,
        }


def up_sample_controversial(dataset, seed):
    cont = dataset.filter(lambda example: example['controversial'] == True)
    up_sampled_dataset = concatenate_datasets([cont] * 4 + [dataset])
    up_sampled_dataset = up_sampled_dataset.shuffle(seed=seed)
    return up_sampled_dataset


def customized_optimizer(model, lr):
    """
    [VPL Optimization Strategy]
    Often beneficial to align the Encoders and Decoders at different rates.
    """
    encoder_params = [p for p in model.parameters() if p not in model.decoder.parameters()]
    decoder_params = [p for p in model.parameters() if p in model.decoder.parameters()]
    grouped_parameters = [
        {'params': encoder_params, 'lr': lr},
        {'params': decoder_params, 'lr': lr / 10},
    ]
    return


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    seed = script_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.set_default_dtype(torch.bfloat16 if script_args.bf16 else torch.float32)

    if script_args.use_last_token_embedding:
        if script_args.model_name == 'gpt2':
            script_args.decoder_embed_dim = 768
            script_args.encoder_embed_dim = 768
        if script_args.model_name == 'meta-llama/Llama-2-7b-hf':
            script_args.decoder_embed_dim = 4096
            script_args.encoder_embed_dim = 4096
        if script_args.model_name == 'meta-llama/Meta-Llama-3-8B-Instruct':
            script_args.decoder_embed_dim = 4096
            script_args.encoder_embed_dim = 4096
            script_args.use_last_token_embedding = False

    data_subset = cast(DataSubset, script_args.data_subset)
    train_dataset = get_hh_rlhf_dataset(
        data_subset,
        "train",
        script_args.train_dataset_size,
        data_path=script_args.data_path,
        other_subsets=script_args.other_subsets
    )
    eval_dataset = get_hh_rlhf_dataset(
        data_subset,
        "test",
        script_args.eval_dataset_size,
        data_path=script_args.data_path,
        other_subsets=script_args.other_subsets
    )
    print(len(train_dataset), len(eval_dataset))
    if script_args.controversial_only:
        train_dataset = train_dataset.filter(lambda example: example['controversial'] == True)
        eval_dataset = eval_dataset.filter(lambda example: example['controversial'] == True)
    elif script_args.up_sampling:
        train_dataset = up_sample_controversial(train_dataset, seed)

    if script_args.one_user:
        train_dataset = train_dataset.filter(lambda example: example['data_subset'] == script_args.one_user)
        eval_dataset = eval_dataset.filter(lambda example: example['data_subset'] == script_args.one_user)
    reward_model_type = cast(RewardModelType, script_args.reward_model_type)

    # Define the training args. Needs to be done before the model is loaded if you
    # are using deepspeed.
    model_name_split = script_args.model_name.split("/")[-1]
    output_name = (
        f"{script_args.log_dir}/{data_subset}/"
        f"{reward_model_type}_{model_name_split}"
        f"__{script_args.train_dataset_size}_{script_args.learning_rate}"
        f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
    )
    output_name += f"_{script_args.kl_loss_weight}_{script_args.latent_dim}_{script_args.decoder_embed_dim}_seed{script_args.seed}"

    trainer_kwargs: Dict[str, Any] = {}
    if script_args.lr_scheduler_type == "step":
        lr_scheduler_type = "constant"
        trainer_kwargs["lr_lambda"] = get_step_decay_lr_lambda
    elif script_args.lr_scheduler_type == "cosine":
        lr_scheduler_type = "constant"
        trainer_kwargs["lr_lambda"] = get_cosine_decay_lr_lambda
    else:
        lr_scheduler_type = script_args.lr_scheduler_type

    output_name = script_args.output_dir if script_args.output_dir else output_name

    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        eval_strategy=script_args.evaluation_strategy, # Updated
        eval_steps=0.05 if script_args.evaluation_strategy == "steps" else None,
        save_strategy=script_args.save_strategy, # Updated
        save_steps=10000 if script_args.save_strategy == "steps" else None,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=script_args.remove_unused_columns, # Updated
        label_names=[],
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        logging_strategy="steps",
        logging_steps=script_args.logging_steps, # Updated
        optim=script_args.optim,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
        run_name=output_name.split("/")[-1],
        disable_tqdm=script_args.disable_tqdm,
    )
    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True, add_eos_token=False)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=128,
        lora_alpha=256,
        lora_dropout=0.1,
    )

    torch.set_anomaly_enabled(True)

    trainer_classes = {
        "base": VAETrainer,
        "mean_and_variance": VAETrainer,
        "categorical": VAETrainer,
    }

    trainer_class = trainer_classes[reward_model_type]
    decoder_embed_dim = script_args.decoder_embed_dim
    encoder_embed_dim = script_args.encoder_embed_dim

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=decoder_embed_dim, torch_dtype=torch.bfloat16
    )
    # We multiply the final linear layer's weights by 0.01 because this seems to
    # significantly stabilize training and lead to better optimization of the loss.
    model.score.weight.data *= 0.01
    if not script_args.fixed_contexts:
        contexts_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name, num_labels=encoder_embed_dim, torch_dtype=torch.bfloat16
        )
        contexts_model.score.weight.data *= 0.01
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if not script_args.fixed_contexts:
        contexts_model = get_peft_model(contexts_model, peft_config)
        contexts_model.print_trainable_parameters()
        contexts_model.config.pad_token_id = tokenizer.pad_token_id
        contexts_model.config.use_cache = not script_args.gradient_checkpointing
    else:
        contexts_model = None

    # Need to do this for GPT2 and Llama because they don't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    model.config.use_cache = not script_args.gradient_checkpointing
    num_proc = 24  # Can adjust to be higher if you have more processors.
    original_columns = train_dataset.column_names

    train_dataset = train_dataset.map(
        HHRLHFPreprocessor(script_args, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    train_dataset = train_dataset.filter(
        lambda x: x["max_lengths"] <= script_args.max_length
    )

    eval_dataset = eval_dataset.map(
        HHRLHFPreprocessor(script_args, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x["max_lengths"] <= script_args.max_length
    )

    # Train the model.
    latent_dim = script_args.latent_dim
    hidden_dim = script_args.hidden_dim
    vae_model = VAEModel(encoder_embed_dim, decoder_embed_dim, hidden_dim, latent_dim, model, contexts_model,
                         fixed_contexts=script_args.fixed_contexts,
                         fixed_llm_embeddings=script_args.fixed_llm_embeddings,)

    trainer = trainer_class(
        model=vae_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=trainer_class.compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            args=script_args,
            tokenizer=tokenizer,
            max_length=script_args.max_length,
            pad_to_multiple_of=64,
        ),
        kl_loss_weight=script_args.kl_loss_weight,
        use_annealing=script_args.use_annealing,
        **trainer_kwargs,
    )

    def eval_pair_embeddings(model, eval_dataloader):
        # We need to access the inner model if wrapped
        if hasattr(model, "module"):
            inner_model = model.module
        else:
            inner_model = model
            
        print("Visualizing Pair Embeddings...")
        inner_model.eval()
        pair_embeddings = []
        labels = []
        
        # Limit the number of batches to avoid OOM or slow eval
        max_batches = 100
        
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= max_batches: break
                
                # Check if fixed_contexts or not to fetch embeddings
                if script_args.fixed_contexts:
                     # For fixed_contexts=True, collator outputs contexts_embeddings_chosen/rejected
                     
                     # Get device reliably
                     device = next(inner_model.parameters()).device

                     if "contexts_embeddings_chosen" in batch:
                         c_c_raw = batch["contexts_embeddings_chosen"]
                         c_r_raw = batch["contexts_embeddings_rejected"]
                         
                         # Manual tensor conversion if they are lists
                         if isinstance(c_c_raw, list):
                             c_c = torch.tensor(c_c_raw).to(device).float()
                         else:
                             c_c = c_c_raw.to(device).float()

                         if isinstance(c_r_raw, list):
                             c_r = torch.tensor(c_r_raw).to(device).float()
                         else:
                             c_r = c_r_raw.to(device).float()
                     else:
                         print("Warning: contexts_embeddings not found in batch for pair eval.")
                         continue
                else:
                     # For fixed_contexts=False, we need to run context encoder first?
                     # The current VAE architecture might be complex here.
                     # Let's assume fixed_contexts=True for this visualization as per user config.
                     print("Skipping pair eval for non-fixed contexts (not implemented yet).")
                     return

                # Pass through Pair Encoder
                # pair_emb shape: (Sum_Contexts, Latent_Dim)
                # We need to access the VAE model inside
                if hasattr(inner_model, "vae_model"): 
                     vae = inner_model.vae_model
                elif isinstance(inner_model, VAEModel):
                     vae = inner_model
                elif hasattr(inner_model, "peft_config"): # PeftModel
                     # PeftModel wraps the base model
                     if hasattr(inner_model.base_model.model, "vae_model"):
                        vae = inner_model.base_model.model.vae_model
                     elif isinstance(inner_model.base_model.model, VAEModel):
                        vae = inner_model.base_model.model
                     else:
                        print("Could not find VAE model inside PEFT wrapper")
                        return
                else:
                     print(f"Could not find VAE model. Type: {type(inner_model)}")
                     return
                
                try:
                    pair_emb = vae.pair_encoder(c_c, c_r)
                    pair_embeddings.append(pair_emb.cpu().numpy())
                except Exception as e:
                    print(f"Error in pair encoder forward: {e}")
                    continue
                
                # Get labels for coloring
                batch_user_types = batch["user_type"]
                seq_start_end = batch["seq_start_end"]
                
                # batch['user_type'] is a list or tensor of user IDs for the *batch*
                # seq_start_end maps batch indices to context ranges
                # We need to replicate the user_type for each context in that range
                
                if isinstance(batch_user_types, torch.Tensor):
                    batch_user_types = batch_user_types.cpu().numpy()
                
                for j, (start, end) in enumerate(seq_start_end):
                    num_contexts = end - start
                    u_type = batch_user_types[j]
                    labels.extend([u_type] * num_contexts.item())

        print(f"Collected {len(pair_embeddings)} pair embeddings for visualization.")
        if len(pair_embeddings) == 0:
            print("No pair embeddings collected.")
            return

        pair_embeddings = np.concatenate(pair_embeddings, axis=0)
        labels = np.array(labels)

        # Run t-SNE
        if len(pair_embeddings) > 20:
            try:
                tsne = TSNE(n_components=2, random_state=42)
                z_2d = tsne.fit_transform(pair_embeddings)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
                legend1 = ax.legend(*scatter.legend_elements(), title="Personas")
                ax.add_artist(legend1)
                plt.title("t-SNE of Pair Embeddings (Encoder Output)")
                
                # Save and log
                plt.savefig("eval_pair_embeddings.png")
                wandb.log({"eval_pair_embeddings": wandb.Image("eval_pair_embeddings.png")})
                print("Logged eval_pair_embeddings to WandB")
                plt.close()
            except Exception as e:
                print(f"t-SNE visualization failed: {e}")

    class PairEmbeddingCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
            if state.global_step % args.logging_steps == 0 or state.global_step == 0:
                 eval_pair_embeddings(model, eval_dataloader)

    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                control.should_evaluate = True


    trainer.add_callback(EvaluateFirstStepCallback())
    trainer.add_callback(PairEmbeddingCallback())

    trainer.train(script_args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")

    model.save_pretrained(output_name + "_peft_last_checkpoint")
    output_name += "_peft_last_checkpoint"
    os.makedirs(output_name, exist_ok=True)

    output_name = os.path.join(output_name, "model.pt")
    vae_model.save_model(output_name)
