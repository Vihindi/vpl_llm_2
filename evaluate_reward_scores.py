"""
evaluate_reward_scores.py
=========================
Tier 1 — Per-persona reward accuracy:
    For each of the 5 personas, load their test split (data/simulation/test_persona_X.jsonl),
    build a latent z using initialization context pairs from that file, then check whether
    rc > rr for every test pair. Reports accuracy and mean margin.

Tier 2 — Cross-persona discriminability:
    For each test pair belonging to Persona X, score the CHOSEN embedding against all 5
    z-vectors (z_A … z_E). The correct z should yield the highest rc. Reports Top-1 accuracy
    and mean rank of the correct z.

Run
---
    python evaluate_reward_scores.py \\
        --vae_model_path  data/reward_models/.../model.pt \\
        --sim_dir         data/simulation \\
        --n_context       8 \\
        --n_test          0 \\
        --output_csv      results/reward_eval.csv \\
        --vae_dev         cuda:0

Arguments
---------
    --vae_model_path  : Path to saved VAEModel .pt file (required)
    --sim_dir         : Directory containing test_persona_A.jsonl … test_persona_E.jsonl
                        Default: data/simulation
    --n_context       : Number of context pairs used from each file to warm-start z.
                        These are drawn from the FIRST n_context rows (which are the
                        oldest, most reliable preference history). Default: 8
    --n_test          : Max test pairs to evaluate per persona (0 = all). Default: 0 (all)
    --output_csv      : Path to save per-pair results as CSV. Default: results/reward_eval.csv
    --vae_dev         : Device for the VAE model. Default: cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

# ── Allow project-root imports ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vae_session_inference import VAESessionInference


PERSONAS = ["A", "B","C","D","E"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    """Loads a JSONL file and returns a list of parsed dictionaries.

    Args:
        path (str): The filename/path to load.

    Returns:
        List[dict]: A list of dictionary objects from the JSONL.
    """
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def to_tensor(emb_list, dtype=torch.float32) -> torch.Tensor:
    """Convert flat list → [1, D] tensor."""
    return torch.tensor(emb_list, dtype=dtype).unsqueeze(0)


def get_rc_rr(
    vae_model,
    h_chosen: torch.Tensor,
    h_rejected: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
    model_type: str = "vae",
) -> tuple[float, float]:
    """Score a pair of embeddings against a given z using the decoder MLP.
    
    Returns (rc, rr)
    """
    dtype = next(vae_model.parameters()).dtype
    hc = h_chosen.to(device=device, dtype=dtype)
    hr = h_rejected.to(device=device, dtype=dtype)
    
    with torch.no_grad():
        if model_type == "vae":
            z_ = z.to(device=device, dtype=dtype)
            rc, rr = vae_model.decode(hc, hr, z_)
            return rc.item(), rr.item()
        else:
            # Baseline scoring
            import torch.nn.functional as F
            logits_c = vae_model(hc)[0]
            logits_r = vae_model(hr)[0]
            if model_type == "categorical":
                num_atoms = logits_c.shape[-1]
                atom_values = torch.linspace(0, 1, num_atoms, device=logits_c.device, dtype=dtype)
                rc = (F.softmax(logits_c, dim=-1) * atom_values).sum(dim=-1).item()
                rr = (F.softmax(logits_r, dim=-1) * atom_values).sum(dim=-1).item()
            elif model_type == "mean_and_variance":
                rc = logits_c[:, 0].item()
                rr = logits_r[:, 0].item()
            else:
                rc = logits_c.item()
                rr = logits_r.item()
            return rc, rr


def build_z_from_contexts(
    vae_model,
    contexts: list,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a latent z from the pre-computed context pairs already embedded
    inside a single data row (row["contexts"]).

    This matches training exactly: the model sees a target pair together with
    its associated context pairs (not arbitrary rows from the file).

    Parameters
    ----------
    vae_model : loaded VAEModel (frozen, eval mode)
    contexts  : list of dicts, each with keys embedding_chosen / embedding_rejected
    device    : torch device

    Returns
    -------
    z : [1, latent_dim] tensor on `device`
    """
    dtype = next(vae_model.parameters()).dtype

    chosen_embs   = torch.stack(
        [torch.tensor(c["embedding_chosen"],   dtype=dtype) for c in contexts]
    ).to(device)   # [N, encoder_dim]
    rejected_embs = torch.stack(
        [torch.tensor(c["embedding_rejected"], dtype=dtype) for c in contexts]
    ).to(device)   # [N, encoder_dim]

    with torch.no_grad():
        # PairEncoder: [N, encoder_dim] x2 → [N, latent_dim]
        pair_hidden = vae_model.encode_pair(chosen_embs, rejected_embs)

        # SequenceEncoder (attention pooling): [N, latent_dim] → mu [1, latent_dim]
        seq_start_end = [(0, len(contexts))]
        mu, _log_var  = vae_model.encode_sequence(pair_hidden, seq_start_end)
        mu = torch.clamp(mu, -1, 1)   # match train-time clamping

    return mu   # [1, latent_dim]  — eval mode uses mean directly (no sampling)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Per-persona reward accuracy
# ─────────────────────────────────────────────────────────────────────────────

def run_tier1(
    vae_model,
    test_rows_per_persona: Dict[str, List[dict]],
    device: torch.device,
    model_type: str = "vae",
) -> Dict[str, dict]:
    """
    For each persona, score their test pairs using the z built from EACH ROW'S
    OWN contexts (row["contexts"]).  This matches training: the model was always
    given a specific set of context pairs alongside each target pair.

    Measures whether rc > rr (chosen beats rejected).
    """
    print("\n" + "=" * 65)
    print("TIER 1 — Per-Persona Reward Accuracy  (rc > rr ?)")
    print(f"{'Persona':<10} {'N':>5} {'Correct':>8} {'Accuracy':>10} "
          f"{'Mean(rc-rr)':>12} {'Std(rc-rr)':>11}")
    print("-" * 65)

    results = {}
    for persona in PERSONAS:
        rows = test_rows_per_persona[persona]
        if not rows: continue
        margins = []
        correct = 0

        for row in rows:
            if hasattr(vae_model, "latent_dim"):
                contexts = row.get("contexts", [])
                if not contexts: continue
                z = build_z_from_contexts(vae_model, contexts, device)
            else:
                z = None # Baseline ignores z

            h_c = to_tensor(row["embeddings"]["embedding_chosen"]).to(device)
            h_r = to_tensor(row["embeddings"]["embedding_rejected"]).to(device)
            rc, rr = get_rc_rr(vae_model, h_c, h_r, z, device, model_type=model_type)
            margin = rc - rr
            margins.append(margin)
            if rc > rr:
                correct += 1

        n   = len(rows)
        acc = correct / n if n > 0 else 0.0
        mean_m = float(np.mean(margins)) if margins else float("nan")
        std_m  = float(np.std(margins))  if margins else float("nan")

        results[persona] = {
            "n": n,
            "correct": correct,
            "accuracy": acc,
            "mean_margin": mean_m,
            "std_margin": std_m,
        }
        print(f"Persona {persona:<4} {n:>5} {correct:>8} {acc:>10.1%} "
              f"{mean_m:>12.4f} {std_m:>11.4f}")

    print("-" * 65)
    overall_acc = np.mean([r["accuracy"]    for r in results.values()])
    overall_mg  = np.mean([r["mean_margin"] for r in results.values()])
    print(f"{'Mean':<10} {'':>5} {'':>8} {overall_acc:>10.1%} {overall_mg:>12.4f}")
    return results


# Tier 2 — Cross-persona discriminability

def run_tier2(
    vae_model,
    test_rows_per_persona: Dict[str, List[dict]],
    device: torch.device,
    model_type: str = "vae",
) -> Dict[str, dict]:
    """
    For each test pair from Persona X, build z from that row's own contexts,
    then ALSO build z-vectors from a representative set of contexts for each
    OTHER persona, and see which z gives the highest rc for the CHOSEN embedding.

    Since we only have per-persona test files (not cross-persona contexts inside
    a single row), we build a GLOBAL representative z per persona by averaging
    the per-row z vectors across the first 20 rows of each persona's test file.
    This gives a stable persona-level z for cross-persona comparison.

    The correct z (the one matching the true persona) should yield the highest
    (rc - rr) margin, NOT just the highest raw rc score.  Using the margin is
    critical when the dataset spans multiple domains (business emails, tweets,
    LinkedIn posts, etc.) because absolute rc values vary by domain.  The margin
    cancels the domain-level offset and isolates the pure preference signal.
    """
    # ── Build a global representative z per persona (average of per-row z's) ──
    print("\n[Tier 2] Building representative z per persona (avg of per-row z's)...")
    z_repr: Dict[str, torch.Tensor] = {}
    is_vae = hasattr(vae_model, "latent_dim")

    for persona in PERSONAS:
        if not is_vae:
            z_repr[persona] = None
            continue
            
        rows = test_rows_per_persona[persona]
        sample = rows[:min(20, len(rows))]
        z_list = []
        for row in sample:
            ctx = row.get("contexts", [])
            if not ctx:
                continue
            z = build_z_from_contexts(vae_model, ctx, device)  # [1, latent_dim]
            z_list.append(z)
        if z_list:
            z_repr[persona] = torch.mean(torch.stack(z_list, dim=0).squeeze(1), dim=0, keepdim=True)
        else:
            z_repr[persona] = torch.zeros(1, vae_model.latent_dim, device=device)
        norm = z_repr[persona].norm().item()
        print(f"  z_repr_{persona} norm = {norm:.4f}")

    n_p = len(PERSONAS)
    random_top1_pct  = f"{100/n_p:.1f}%"
    random_mean_rank = f"{(n_p + 1) / 2:.2f}"
    print("\n" + "=" * 65)
    print("TIER 2 — Cross-Persona Discriminability  (which z gives best rc-rr margin?)")
    print(f"{'Persona':<10} {'N':>5} {'Top-1':>8} {'Top-1 Acc':>10} {'Mean Rank':>10}")
    print(f"{'':10} {'':5} {'':8} {f'(random={random_top1_pct})':>10} {f'(random={random_mean_rank})':>10}")
    print("-" * 65)

    results       = {}
    all_pair_rows = []

    for persona in PERSONAS:
        rows = test_rows_per_persona[persona]
        top1_count = 0
        ranks      = []

        for row in rows:
            h_c = to_tensor(row["embeddings"]["embedding_chosen"]).to(device)
            h_r = to_tensor(row["embeddings"]["embedding_rejected"]).to(device)

            # Compute (rc - rr) margin for each persona z.
            # Margin cancels domain-level absolute score bias:
            #   a tweet embedding scores differently to a business email
            #   regardless of persona z — the margin isolates preference only.
            margins = {}
            for p in PERSONAS:
                rc_p, rr_p = get_rc_rr(vae_model, h_c, h_r, z_repr[p], device, model_type=model_type)
                margins[p] = rc_p - rr_p
            # Rank by margin (higher = better preference discrimination)
            ranked = sorted(PERSONAS, key=lambda p: margins[p], reverse=True)
            rank_of_correct = ranked.index(persona) + 1
            ranks.append(rank_of_correct)
            if rank_of_correct == 1:
                top1_count += 1

            all_pair_rows.append({
                "true_persona": persona,
                "index": row.get("Index", ""),
                **{f"margin_{p}": margins[p] for p in PERSONAS},
                "rank_correct": rank_of_correct,
                "top1": rank_of_correct == 1,
            })

        n        = len(rows)
        top1_acc = top1_count / n if n > 0 else 0.0
        mean_r   = float(np.mean(ranks))
        rank_dist = {r: ranks.count(r) for r in range(1, n_p + 1)}

        results[persona] = {
            "n": n,
            "top1_count": top1_count,
            "top1_accuracy": top1_acc,
            "mean_rank": mean_r,
            "rank_distribution": rank_dist,
        }
        print(f"Persona {persona:<4} {n:>5} {top1_count:>8} {top1_acc:>10.1%} {mean_r:>10.2f}")

    print("-" * 65)
    mean_top1 = np.mean([r["top1_accuracy"] for r in results.values()])
    mean_rank = np.mean([r["mean_rank"]     for r in results.values()])
    print(f"{'Mean':<10} {'':>5} {'':>8} {mean_top1:>10.1%} {mean_rank:>10.2f}")
    print(f"{'Random baseline':<10} {'':>5} {'':>8} {random_top1_pct:>10} {random_mean_rank:>10}")

    print("\nRank distribution (how often was correct z ranked 1st/2nd/…):")
    header = f"{'Persona':<10}" + "".join(f"  Rank{r}" for r in range(1, n_p + 1))
    print(header)
    print("-" * (10 + 8 * n_p))
    for persona in PERSONAS:
        rd = results[persona]["rank_distribution"]
        row_str = f"Persona {persona:<4}" + "".join(
            f"  {rd.get(r, 0):5d}" for r in range(1, n_p + 1)
        )
        print(row_str)

    return results, all_pair_rows


# CSV export

def save_csv(all_pair_rows: list, output_csv: str):
    """Saves per-pair evaluation results to a CSV file.

    Args:
        all_pair_rows (list): List of dictionaries containing the results.
        output_csv (str): Output path for the CSV output.
    """
    if not all_pair_rows:
        return
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_pair_rows)
    print(f"\n[Eval] Per-pair Tier 2 results saved to {output_csv}")



# Argument parsing

def get_args():
    """Parses command-line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed arguments containing configuration options.
    """
    p = argparse.ArgumentParser(
        description="Tier 1+2: Reward accuracy and cross-persona discriminability"
    )
    p.add_argument("--vae_model_path", default=None,
                   help="Path to the saved VAEModel .pt file")
    p.add_argument("--baseline_model_path", default=None,
                   help="Path to saved Baseline .bin model folder")
    p.add_argument("--model_type", type=str, default="vae",
                   choices=["vae", "base", "mean_and_variance", "categorical"],
                   help="Type of reward model being evaluated")
    p.add_argument("--baseline_embed_dim", type=int, default=4096,
                   help="Embedding dimensions for SimpleRewardModel weights")
    p.add_argument("--sim_dir",        default="data/simulation",
                   help="Directory with test_Persona_{A-E}.jsonl files")
    p.add_argument("--n_context",      type=int, default=8,
                   help="[Tier 2] Number of rows to avg for representative persona z (default: 8)")
    p.add_argument("--n_test",         type=int, default=0,
                   help="Max test pairs per persona (0 = all, default: 0)")
    p.add_argument("--output_csv",     default="results/reward_eval_tier2.csv",
                   help="Path to save per-pair Tier 2 CSV results")
    p.add_argument("--vae_dev",        default="cuda:0",
                   help="Device for VAE (default: cuda:0)")
    return p.parse_args()



def main():
    """Main evaluation script for Tier 1 and Tier 2 accuracy tests."""
    args   = get_args()
    device = torch.device(args.vae_dev)
    sim_dir = Path(args.sim_dir)

    # Load raw data 
    print("[Setup] Loading persona test files …")
    all_rows: Dict[str, List[dict]] = {}
    for persona in PERSONAS:
        fpath = sim_dir / f"test_Persona_{persona}.jsonl"
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found — skipping Persona {persona}")
            all_rows[persona] = []
            continue
        rows = load_jsonl(str(fpath))
        all_rows[persona] = rows
        print(f"  Persona {persona}: {len(rows)} rows loaded from {fpath.name}")

    # ── Load VAE model (once, shared across tiers) ────────────────────────────
    if args.model_type == "vae":
        print(f"\n[Setup] Loading VAE model from {args.vae_model_path} …")
        vae_model = torch.load(args.vae_model_path, map_location=device, weights_only=False)
        vae_model.eval()
        for param in vae_model.parameters():
            param.requires_grad = False
    else:
        print(f"\n[Setup] Loading Baseline {args.model_type} model from {args.baseline_model_path} …")
        from hidden_context.train_llm_preference_model import SimpleRewardModel
        if args.model_type == "base": num_labels = 1
        elif args.model_type == "mean_and_variance": num_labels = 2
        else: num_labels = 10
        
        vae_model = SimpleRewardModel(embed_dim=args.baseline_embed_dim, num_labels=num_labels)
        import os
        bin_path = os.path.join(args.baseline_model_path, "pytorch_model.bin")
        vae_model.load_state_dict(torch.load(bin_path, map_location=device))
        vae_model.to(device).bfloat16()
        vae_model.eval()

    # ── All rows are test rows — z is built per-row from row["contexts"] ──────
    test_rows_per_persona: Dict[str, List[dict]] = {}
    for persona in PERSONAS:
        rows = all_rows[persona]
        test_rows = rows if args.n_test == 0 else rows[:args.n_test]
        test_rows_per_persona[persona] = test_rows
        print(f"  Persona {persona}: {len(test_rows)} test pairs")

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    tier1_results = run_tier1(vae_model, test_rows_per_persona, device, model_type=args.model_type)

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    tier2_results, all_pair_rows = run_tier2(
        vae_model, test_rows_per_persona, device, model_type=args.model_type
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_csv(all_pair_rows, args.output_csv)

    # ── Final interpretation hints ────────────────────────────────────────────
    n_p = len(PERSONAS)
    rand_t1 = 100 / n_p
    rand_rank = (n_p + 1) / 2
    good_t1  = rand_t1 + 20
    print("\n" + "=" * 65)
    print("INTERPRETATION GUIDE")
    print(f"  Tier 1 accuracy > 65%         -> reward model is correctly ranking pairs")
    print(f"  Tier 1 accuracy ~ 50%         -> reward model not learning (check training)")
    print(f"  Tier 2 Top-1 > {good_t1:.0f}% (random={rand_t1:.0f}%) -> z is genuinely persona-discriminative")
    print(f"  Tier 2 Top-1 ~ {rand_t1:.0f}%         -> z is being ignored by the decoder (bad)")
    print(f"  Tier 2 Mean rank < {rand_rank-1:.1f}         -> z clusters are well-separated")
    print("=" * 65)


if __name__ == "__main__":
    main()
