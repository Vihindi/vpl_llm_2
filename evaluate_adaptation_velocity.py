"""
evaluate_adaptation_velocity.py
================================
Tracks Rolling Accuracy speed during drift across different momentum coefficients.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from itertools import combinations

import torch
import numpy as np
import matplotlib

PERSONAS = ["A", "B", "C", "D", "E"]
PAIRS    = list(combinations(PERSONAS, 2))
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vae_session_inference import VAESessionInference


def load_jsonl(path: str):
    """Loads a JSONL file and returns a list of dictionaries.

    Args:
        path (str): The path to the JSONL file to load.

    Returns:
        list: A list of dictionaries representing the JSON lines.
    """
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def to_tensor(emb_list) -> torch.Tensor:
    """Converts a standard Python list of embeddings to a PyTorch tensor.

    Args:
        emb_list (list): A list of numerical values.

    Returns:
        torch.Tensor: The converted tensor with an added batch dimension.
    """
    return torch.tensor(emb_list, dtype=torch.float32).unsqueeze(0)


def main():
    """Main execution block handles argument parsing, runs simulations across 
    different momentum values, computes sliding-window accuracy, and outputs the graphs.
    """
    parser = argparse.ArgumentParser(description="Evaluate adaptation speed and momentum values.")
    parser.add_argument("--vae_model_path", required=True)
    parser.add_argument("--sim_dir", default="data/simulation/multi_drift")
    parser.add_argument("--sim_path", default=None)
    parser.add_argument("--output_dir", default="results/adaptation")
    parser.add_argument("--vae_dev", default="cuda:0")
    parser.add_argument("--window", type=int, default=5, help="Rolling window size for Accuracy smoothing")
    args = parser.parse_args()

    # ── 1. Load VAE model ─────────────────────────────────────────────────────
    print(f"\n[Velocity] Loading VAE model: {args.vae_model_path}")
    device = torch.device(args.vae_dev)
    vae_model = torch.load(args.vae_model_path, map_location=device, weights_only=False)
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False

    #  2. Load Simulations 
    if args.sim_path:
        sims_to_run = [Path(args.sim_path)]
    else:
        sim_dir = Path(args.sim_dir)
        sims_to_run = []
        for x, y in PAIRS:
            p = sim_dir / f"sim_{x}_to_{y}.jsonl"
            if p.exists():
                sims_to_run.append(p)

    sim_data_list = []
    print(f"[Velocity] Loading {len(sims_to_run)} simulation files...")
    for p in sims_to_run:
        rows = load_jsonl(p)
        sim_data_list.append({
            "name": p.stem,
            "seed": [r for r in rows if r.get("section") == "seed"],
            "eval": [r for r in rows if r.get("section") == "eval"],
            "final_eval": [r for r in rows if r.get("section") == "final_eval"]
        })
    if not sim_data_list:
        raise FileNotFoundError(f"No simulation files found. Checked {args.sim_path or args.sim_dir}")

    #  3. Processing 
    momentums = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    W = args.window
    results = {}
    momentum_curves = {}

    print("\n[Velocity] Running sweeps over momentums …")

    for m in momentums:
        momentum_curves[m] = []
        rec_steps_80 = []
        rec_steps_70 = []
        unrecovered_count = 0
        final_drift_accs = []
        overall_drift_accs = []
        overall_stable_accs = []
        final_eval_accs = []

        print(f"  Momentum {m} …")

        for sim in sim_data_list:
            seed_rows = sim["seed"]
            eval_rows = sim["eval"]
            final_rows = sim["final_eval"]

            chosen_embs = torch.tensor([r["embeddings"]["embedding_chosen"] for r in seed_rows], dtype=torch.float32)
            rejected_embs = torch.tensor([r["embeddings"]["embedding_rejected"] for r in seed_rows], dtype=torch.float32)

            session = VAESessionInference(model_obj=vae_model, momentum=m, device=args.vae_dev)
            session.initialize_with_pairs(chosen_embs, rejected_embs)

            drift_correct_flags = []
            stable_correct_flags = []

            for row in eval_rows:
                tc = to_tensor(row["embeddings"]["embedding_chosen"])
                tr = to_tensor(row["embeddings"]["embedding_rejected"])
                is_drift = row.get("drift", False)

                ctx_c = ctx_r = None
                if is_drift and row.get("contexts"):
                    ctx = row["contexts"][0]
                    ctx_c = to_tensor(ctx["embedding_chosen"])
                    ctx_r = to_tensor(ctx["embedding_rejected"])

                rc, rr = session.score(
                    tc, tr, drift=is_drift, 
                    new_context_chosen_emb=ctx_c, 
                    new_context_rejected_emb=ctx_r
                )
                # Ensure scalar float
                rc = float(rc) if hasattr(rc, "item") else rc
                rr = float(rr) if hasattr(rr, "item") else rr
                
                pred = 1.0 if rc > rr else 0.0
                if is_drift:
                    drift_correct_flags.append(pred)
                else:
                    stable_correct_flags.append(pred)

            if len(drift_correct_flags) >= W:
                rolling_drift_acc = np.convolve(drift_correct_flags, np.ones(W)/W, mode='valid')
                momentum_curves[m].append(rolling_drift_acc)

                over_80 = np.where(rolling_drift_acc >= 0.80)[0]
                over_70 = np.where(rolling_drift_acc >= 0.70)[0]

                if len(over_80) > 0:
                    rec_steps_80.append(W + over_80[0])
                else:
                    unrecovered_count += 1

                if len(over_70) > 0:
                    rec_steps_70.append(W + over_70[0])

                final_drift_accs.append(rolling_drift_acc[-1])

            if len(drift_correct_flags) > 0:
                overall_drift_accs.append(np.mean(drift_correct_flags))

            if final_rows:
                f_flags = []
                for row in final_rows:
                    tc = to_tensor(row["embeddings"]["embedding_chosen"])
                    tr = to_tensor(row["embeddings"]["embedding_rejected"])
                    rc, rr = session.score(tc, tr, drift=False)
                    rc = float(rc) if hasattr(rc, "item") else rc
                    rr = float(rr) if hasattr(rr, "item") else rr
                    f_flags.append(1.0 if rc > rr else 0.0)
                if f_flags:
                    final_eval_accs.append(np.mean(f_flags))

        results[m] = {
            "rec_80_mean": np.mean(rec_steps_80) if rec_steps_80 else float('nan'),
            "rec_80_std": np.std(rec_steps_80) if rec_steps_80 else float('nan'),
            "rec_70_mean": np.mean(rec_steps_70) if rec_steps_70 else float('nan'),
            "rec_70_std": np.std(rec_steps_70) if rec_steps_70 else float('nan'),
            "unrecovered": unrecovered_count,
            "final_drift_acc": np.mean(final_drift_accs) if final_drift_accs else 0.0,
            "drift_acc": np.mean(overall_drift_accs) if overall_drift_accs else 0.0,
            "stable_acc": np.mean(overall_stable_accs) if overall_stable_accs else 0.0,
            "final_eval_acc": np.mean(final_eval_accs) if final_eval_accs else 0.0
        }

    # ── 4. Plotting ──────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    for m in momentums:
        curves = momentum_curves[m]
        if curves:
            min_len = min(len(c) for c in curves)
            curves_arr = np.array([c[:min_len] for c in curves])
            mean_c = np.mean(curves_arr, axis=0)
            std_c = np.std(curves_arr, axis=0)
            steps = np.arange(W, min_len + W)
            
            line, = plt.plot(steps, mean_c, label=f'Momentum {m}', alpha=0.9)
            plt.fill_between(steps, mean_c - std_c, mean_c + std_c, color=line.get_color(), alpha=0.1)

    plt.axhline(0.80, color='red', linestyle='--', label='80% Threshold', alpha=0.6)
    plt.axhline(0.70, color='orange', linestyle='--', label='70% Threshold', alpha=0.6)
    plt.xlabel('Drift Interaction Step Index')
    plt.ylabel(f'Rolling Drift Accuracy (Window={W})')
    plt.title("Drift Adaptation Velocity (Drift Rows Only)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    sim_name = Path(args.sim_path).stem if args.sim_path else "All_Simulations"
    plt.suptitle(f'Adaptation Velocity & Momentum Sweep - {sim_name}')
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_img = os.path.join(args.output_dir, f"{sim_name}_velocity.png")
    plt.savefig(out_img, dpi=120)
    plt.close()

    # ── 5. Print Summary Table ───────────────────────────────────────────────
    print("\n" + "="*120)
    print(f"{'Momentum':<10} | {'Rec@0.8 (Mean±Std)':<20} | {'Rec@0.7 (Mean±Std)':<20} | {'Unrecovered':<12} | {'Final Drift Acc':<15} | {'Drift Acc':<10} | {'Final-Eval Acc':<15}")
    print("-"*120)
    for m in momentums:
        r = results[m]
        r80 = f"{r['rec_80_mean']:.1f} ± {r['rec_80_std']:.1f}" if not np.isnan(r['rec_80_mean']) else "N/A"
        r70 = f"{r['rec_70_mean']:.1f} ± {r['rec_70_std']:.1f}" if not np.isnan(r['rec_70_mean']) else "N/A"
        print(f"{m:<10} | {r80:<20} | {r70:<20} | {r['unrecovered']:<12} | {r['final_drift_acc']:.3f}        | {r['drift_acc']:.3f}     | {r['final_eval_acc']:.3f}")
    print("="*120)

if __name__ == "__main__":
    main()
