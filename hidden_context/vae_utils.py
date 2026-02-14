import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import Trainer, EvalPrediction
import wandb
from transformers.optimization import get_cosine_schedule_with_warmup


class PairEncoder(nn.Module):
    """
    Model to encode pairs of accepted and rejected responses
    
    [VPL Paper Concept]:
    This component processes a single interaction from the User's Context set C.
    An interaction consists of a prompt x, a chosen response y_w, and a rejected response y_l.
    
    In the paper, this corresponds to the initial processing of context elements before aggregation.
    It maps the raw embeddings of the chosen and rejected responses (concatenated) into a 
    hidden representation.
    """

    def __init__(self, embed_dim, hidden_dim, output_dim):
        super(PairEncoder, self).__init__()

        self._model = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, e_c, e_r):
        """
        Example Input:
            e_c: (Batch, Embed_Dim) e.g. (32, 1024) - Embedding of Chosen response
            e_r: (Batch, Embed_Dim) e.g. (32, 1024) - Embedding of Rejected response
        
        Example Output:
            output: (Batch, Hidden_Dim) e.g. (32, 512) - Encoded pair representation
        """
        # dtype = next(self._model.parameters()).dtype
        # e_c = e_c.to(dtype)
        # e_r = e_r.to(dtype)

        # diff = e_c - e_r
        # prod = e_c * e_r
        # x = torch.cat([e_c, e_r, diff, prod], dim=1)
        # return self._model(x)

        x = torch.cat([e_c, e_r], dim=1)
        return self._model(x)



class SequenceEncoder(nn.Module):
    """
    Model to encode sequence of responses
    
    [VPL Paper Concept]:
    This is the "Encoder" part of the Variational Autoencoder (VAE).
    It takes the sequence of encoded interactions (from PairEncoder) constituting the user's context C.
    
    Role: Approximate the posterior distribution q_phi(z | C).
    
    Mechanism:
    - Uses an attention mechanism (Self-Attention) to aggregate the variable-length sequence of 
      interactions into a fixed-size representation.
    - Outputs 'mean' (mu) and 'log_var' (log sigma^2) which parametrize the Gaussian distribution 
      of the user's latent preference vector z.
    """

    def __init__(self, input_dim, latent_dim):
        super(SequenceEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.linear = nn.Identity()
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)
        self.w_v = nn.Linear(input_dim, input_dim)
        self.mean_layer = nn.Linear(input_dim, latent_dim)
        self.log_var_layer = nn.Linear(input_dim, latent_dim)
        self.layer_norm = nn.Identity()     # nn.LayerNorm(latent_dim)

    def forward(
        self, sequences, seq_start_end
    ):  # (C_1+C_2+...+C_n, D), [(0, C_1), (C_1, C_1+C_2), ..., (C_1+...+C_n-1, C_1+...+C_n)]
        """
        Example Input:
            sequences: (Total_Context_Items, Input_Dim) e.g. (128, 512)
                       - Aggregated interactions from ALL users in the batch.
            seq_start_end: (Batch, 2) e.g. [(0, 4), (4, 10), ...]
                           - Start and End indices for each user's context in 'sequences'.
        
        Example Output:
            mean: (Batch, Latent_Dim) e.g. (32, 64)
            log_var: (Batch, Latent_Dim) e.g. (32, 64)
            
        Sample Data Point (Batch Size = 2):
        -----------------------------------
        User A has 2 interactions (pairs) in history: [Pair_A1, Pair_A2]
        User B has 1 interaction (pair) in history:   [Pair_B1]
        
        'sequences' input (Concatenated Contexts):
        - Shape: (3, Input_Dim)
        - Content: [Pair_A1_Embedding, 
                    Pair_A2_Embedding, 
                    Pair_B1_Embedding]
        
        'seq_start_end' input (Indices):
        - Shape: (2, 2)
        - Content: [
             [0, 2],  # User A: sequences[0:2] -> {Pair_A1, Pair_A2}
             [2, 3]   # User B: sequences[2:3] -> {Pair_B1}
          ]
        """
        outputs = []
        for _, (start, end) in enumerate(seq_start_end):
            context = sequences[start:end]  # C_i x D
            q = self.w_q(context)
            k = self.w_k(context)
            attention_scores = torch.matmul(
                q, k.transpose(0, 1)
            )
            attention_scores = attention_scores / (context.shape[-1] ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)  # C_i x C_i
            weighted_values = torch.matmul(attention_weights, self.w_v(context))  # C_i x D
            output = torch.mean(weighted_values, dim=0)  # D
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)  # n x D

        mean = self.layer_norm(self.mean_layer(outputs))
        log_var = self.layer_norm(self.log_var_layer(outputs))
        print("Here is the mean: ", mean)
        print("Here is the log_var: ", log_var)
        return mean, log_var


class Decoder(nn.Module):
    """
    [VPL Paper Concept]:
    This is the "Decoder" or the "Reward Model" r_psi(x, y, z).
    
    Role: Predict the reward for a query-response pair (x, y), CONDITIONED on the latent user vector z.
    
    Inputs:
    - xc/xr: Embeddings of the query-response pair (target being evaluated).
    - z: The sampled latent user preference vector.
    
    This enables the personalization: the same input (x, y) can receive different rewards 
    depending on the user's z.
    
    """

    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xc, xr, z):
        """
        Example Input:
            xc: (Batch, Embed_Dim) e.g. (32, 1024) - Target Chosen Embedding
            xr: (Batch, Embed_Dim) e.g. (32, 1024) - Target Rejected Embedding
            z: (Batch, Latent_Dim) e.g. (32, 64) - Latent User Vector
        
        Example Output:
            rc: (Batch, 1) - Reward score for Chosen
            rr: (Batch, 1) - Reward score for Rejected
        """
        print("Here is the  z: ", z)
        xc = torch.cat([xc, z], dim=1)
        xr = torch.cat([xr, z], dim=1)
        rc = self._model(xc)
        rr = self._model(xr)
        return rc, rr


class VAEModel(nn.Module):
    """
    [VPL Paper Concept]:
    The sequence VPL model architecture implementing the flow:
    Context C -> Encoder -> z ~ q(z|C) -> Decoder -> Reward r(x,y,z)
    """
    def __init__(self, encoder_embed_dim, decoder_embed_dim, hidden_dim, latent_dim, llm_encoder, llm_contexts_encoder,
                 fixed_contexts=False, fixed_llm_embeddings=False, use_causal_lm=False, use_attention_layer=False,
                 use_transformer=False, concat_chosen_rejected=False):
        super(VAEModel, self).__init__()
        self.llm_encoder = llm_encoder
        self.llm_contexts_encoder = llm_contexts_encoder
        self.pair_encoder = PairEncoder(encoder_embed_dim, hidden_dim, latent_dim)
        self.sequence_encoder = SequenceEncoder(latent_dim, latent_dim)
        self.decoder = Decoder(decoder_embed_dim + latent_dim, hidden_dim)

        self.latent_dim = latent_dim
        self.fixed_contexts = fixed_contexts
        self.fixed_llm_embeddings = fixed_llm_embeddings
        self.use_causal_lm = use_causal_lm
        self.use_attention_layer = use_attention_layer
        self.use_transformer = use_transformer
        self.concat_chosen_rejected = concat_chosen_rejected

        self.saved_embeddings = torch.Tensor(4, latent_dim)
        self.saved_embeddings.uniform_(-1, 1)

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std).to(mean.device)  # sampling epsilon
        z = mean + std * epsilon                         # reparameterization trick
        z = F.normalize(z, p=2, dim=-1) * math.sqrt(z.shape[-1])
        return z

    """
    [VPL Paper Concept - Inference/Forward Pass]
    1. Context Encoding: chosen/rejected pairs from context history are encoded (PairEncoder).
    2. Latent Aggregation: SequenceEncoder computes mu and log_var.
    3. Sampling: z is sampled using the Reparameterization Trick (during training).
       z = mu + sigma * epsilon
    4. Reward Prediction: The Decoder predicts rewards for the TARGET pair using z.
    """
    def encode_pair(self, e_c, e_r):
        return self.pair_encoder(e_c, e_r)

    def encode_sequence(self, sequences, seq_start_end):
        return self.sequence_encoder(sequences, seq_start_end)

    def decode(self, e_c, e_r, z):
        return self.decoder(e_c, e_r, z)

    def forward(
        self,
        target_chosen,
        target_rejected,
        context_chosen,
        context_rejected,
        seq_start_end,
        user_type,
        ground_truth_user_vector=False,
        **kwargs,
    ):
        """
        Example Input:
            target_chosen: (Batch, Embed_Dim) - Current interaction chosen response
            target_rejected: (Batch, Embed_Dim) - Current interaction rejected response
            context_chosen: (Total_Context_Len, Embed_Dim) - History chosen responses
            context_rejected: (Total_Context_Len, Embed_Dim) - History rejected responses
            seq_start_end: (Batch, 2) - Indices map for context
            user_type: (Batch,) - Optional user IDs (not used in standard training)
        
        Example Output:
            rc, rr: (Batch, 1) - Rewards
            mean, log_var, z: (Batch, Latent_Dim) - VAE internals
        """
        pair_embed = self.encode_pair(context_chosen, context_rejected)
        mean, log_var = self.encode_sequence(pair_embed, seq_start_end)
        mean = torch.clamp(mean, -1, 1)

        _log_var = torch.clamp(log_var, -1, 1)
        if ground_truth_user_vector:
            z = torch.zeros_like(mean)
            self.saved_embeddings = self.saved_embeddings.to(mean.device)
            for idx in range(user_type.shape[0]):
                z[idx] = self.saved_embeddings[int(user_type[idx])]
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * _log_var))

        if not self.training and not ground_truth_user_vector:
            z = mean
        rc, rr = self.decode(target_chosen, target_rejected, z)

        return rc, rr, mean, _log_var, z

    def save_model(self, path):
        torch.save(self, path)


class VAETrainer(Trainer):
    """
    [VPL Paper Concept]:
    Handles the optimization of the Evidence Lower Bound (ELBO).
    
    Objective: maximize E_{z~q}[log p(y_w > y_l | x, z)] - beta * KL(q(z|C) || p(z))
    
    This translates to minimizing:
    Loss = Ranking Loss (reconstruction) + KL Divergence Loss
    """
    def __init__(
        self, *args, lr_lambda=None, kl_loss_weight=None, use_annealing=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda
        self.kl_loss_weight = kl_loss_weight
        self.use_annealing = use_annealing
        self.annealer = Annealer(
            total_steps=1e4, shape="cosine", baseline=0.1, cyclical=True    # todo: change total_step here
        )

    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        """
        Example Input:
            rewards_chosen: (Batch, 1) e.g. [[0.8], [1.2]]
            rewards_rejected: (Batch, 1) e.g. [[0.1], [1.5]]
        
        Example Output:
            loss: (Batch, 1) - Per-sample ranking loss
        """
        return -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

    def loss(self, rewards_chosen, rewards_rejected):
        return torch.mean(self.per_sample_loss(rewards_chosen, rewards_rejected))

    def compute_loss(self, wrapped_model, inputs, return_outputs=False, **kwargs):
        if isinstance(wrapped_model, VAEModel):
            model = wrapped_model  # .module
        else:
            model = wrapped_model.module
        device = model.llm_encoder.device
        batch_size = inputs["seq_start_end"].shape[0]
        if model.fixed_llm_embeddings:
            embeddings_chosen = torch.tensor(inputs["embeddings_chosen"]).to(device).bfloat16()
            embeddings_rejected = torch.tensor(inputs["embeddings_rejected"]).to(device).bfloat16()
        else:
            embeddings = model.llm_encoder(
                input_ids=torch.concatenate(
                    [
                        inputs["input_ids_chosen"],
                        inputs["input_ids_rejected"],
                    ],
                    dim=0,
                ),
                attention_mask=torch.concatenate(
                    [
                        inputs["attention_mask_chosen"],
                        inputs["attention_mask_rejected"],
                    ],
                    dim=0,
                ),
            )[0]
            embeddings_chosen = embeddings[:batch_size]
            embeddings_rejected = embeddings[batch_size:]

        if model.fixed_contexts:
            contexts_embeddings_chosen = torch.tensor(inputs["contexts_embeddings_chosen"]).to(device).bfloat16()
            contexts_embeddings_rejected = torch.tensor(inputs["contexts_embeddings_rejected"]).to(device).bfloat16()
        else:
            input_ids_chosen = inputs["contexts_input_ids_chosen"]
            attention_mask_chosen = inputs["contexts_attention_mask_chosen"]
            token_length_chosen = torch.eq(input_ids_chosen,
                                           model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1
            input_ids_rejected = inputs["contexts_input_ids_rejected"]
            attention_mask_rejected = inputs["contexts_attention_mask_rejected"]
            token_length_rejected = torch.eq(input_ids_rejected,
                                             model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1

            with torch.no_grad():
                last_hidden_state_chosen = model.llm_contexts_encoder(
                    input_ids=input_ids_chosen,
                    attention_mask=attention_mask_chosen,
                    output_hidden_states=True
                ).hidden_states[-1]

                weights_for_non_padding_chosen = attention_mask_chosen * torch.arange(
                    start=1, end=last_hidden_state_chosen.shape[1] + 1
                ).unsqueeze(0).to(attention_mask_chosen.device).float()
                sum_embeddings = torch.sum(last_hidden_state_chosen * weights_for_non_padding_chosen.unsqueeze(-1),
                                           dim=1)
                num_of_none_padding_tokens_chosen = torch.sum(weights_for_non_padding_chosen, dim=-1).unsqueeze(-1)
                contexts_embeddings_chosen = sum_embeddings / num_of_none_padding_tokens_chosen
                last_hidden_state_rejected = model.llm_contexts_encoder(
                    input_ids=input_ids_rejected,
                    attention_mask=attention_mask_rejected,
                    output_hidden_states=True
                ).hidden_states[-1]

                weights_for_non_padding_rejected = attention_mask_rejected * torch.arange(
                    start=1, end=last_hidden_state_rejected.shape[1] + 1
                ).unsqueeze(0).to(attention_mask_rejected.device).float()
                sum_embeddings = torch.sum(last_hidden_state_rejected * weights_for_non_padding_rejected.unsqueeze(-1),
                                           dim=1)
                num_of_none_padding_tokens_rejected = torch.sum(weights_for_non_padding_rejected, dim=-1).unsqueeze(-1)
                contexts_embeddings_rejected = sum_embeddings / num_of_none_padding_tokens_rejected
        seq_start_end = inputs["seq_start_end"]
        user_type = torch.tensor(inputs["user_type"]).to(device).bfloat16()
        rewards_chosen, rewards_rejected, mean, log_var, z = model(
            embeddings_chosen,
            embeddings_rejected,
            contexts_embeddings_chosen,
            contexts_embeddings_rejected,
            seq_start_end,
            user_type,
            ground_truth_user_vector=False,  # todo: set to True for debug usage
            # mask_chosen=inputs["attention_mask_chosen"],
            # mask_rejected=inputs["attention_mask_rejected"],
        )

        """
        [VPL Paper Concept - Loss Calculation]
        
        reproduction_loss: The Ranking Loss.
        - Corresponds to maximizing the likelihood of the preferred response given z.
        - -log(sigmoid(reward_chosen - reward_rejected))
        
        kld: The KL Divergence term.
        - Regularizes the learned posterior q(z|C) to be close to the prior p(z) (Standard Normal).
        - Prevents overfitting to specific contexts and enforces a smooth latent space.
        """
        reproduction_loss = self.loss(rewards_chosen, rewards_rejected)
        if self.kl_loss_weight == 0:
            loss = reproduction_loss
            accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
            if not return_outputs:
                self.log(
                    {
                        "train_recon_loss": reproduction_loss.mean().item(),
                        "train_accuracy": accuracy.mean().item(),
                        "rewards_chosen": rewards_chosen.mean().item(),
                        "rewards_rejected": rewards_rejected.mean().item(),
                        "embeddings_chosen": embeddings_chosen.mean().item(),
                        "embeddings_rejected": embeddings_rejected.mean().item(),
                        "mean": mean.mean().item(),
                        "log_var": log_var.mean().item()
                    }
                )
        else:
            kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
            if self.use_annealing:
                kld = self.annealer(kld)
                self.annealer.step()
            kld = self.kl_loss_weight * kld
            loss = reproduction_loss + kld
            accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
            if not return_outputs:
                self.log(
                    {
                        "train_recon_loss": reproduction_loss.mean().item(),
                        "train_kld": kld.mean().item(),
                        "train_accuracy": accuracy.mean().item(),
                        "rewards_chosen": rewards_chosen.mean().item(),
                        "rewards_rejected": rewards_rejected.mean().item(),
                        "embeddings_chosen": embeddings_chosen.mean().item(),
                        "embeddings_rejected": embeddings_rejected.mean().item(),
                        "mean": mean.mean().item(),
                        "log_var": log_var.mean().item()
                    }
                )
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
                "mean": mean,
                "log_var": log_var,
                "z": z,
                "user_type": user_type,
            }
        return loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.03 * num_training_steps),
            num_training_steps=num_training_steps
        )
        self.lr_scheduler = scheduler
        return scheduler

    @classmethod
    def compute_metrics(cls, eval_prediction: EvalPrediction):
        rewards_chosen, rewards_rejected, mean, log_var, z, user_type = (
            eval_prediction.predictions
        )
        rewards_chosen = torch.from_numpy(rewards_chosen)
        rewards_rejected = torch.from_numpy(rewards_rejected)
        mean = torch.from_numpy(mean)
        log_var = torch.from_numpy(log_var)
        z = torch.from_numpy(z)
        loss = cls.per_sample_loss(rewards_chosen, rewards_rejected)
        kld = -torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        accuracy = torch.mean((loss < np.log(2)).float())

        def plot_latent(latent):
            from sklearn.manifold import TSNE
            z_embedding = TSNE(n_components=2, init='random', perplexity=20, learning_rate="auto").fit_transform(latent.numpy())
            import matplotlib.pyplot as plt
            colors = [f"C{int(i)}" for i in user_type]
            plt.scatter(z_embedding[:, 0], z_embedding[:, 1], c=colors)
            im = wandb.Image(plt)
            plt.close()
            return im
        im1 = plot_latent(mean)
        im2 = plot_latent(z)
        
        # Log images to wandb directly to avoid JSON serialization error in Trainer state saving
        if wandb.run is not None:
            wandb.log({"eval_mean_embeddings": im1, "eval_z_embeddings": im2})

        return {
            "loss": loss.mean().item(),
            "accuracy": accuracy.item(),
            "kld": kld.mean().item(),
        }


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return
