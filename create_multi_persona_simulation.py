"""
create_multi_persona_simulation.py
===================================
Builds a simulation JSONL file for a persona-drift scenario from X → Y.


Structure of output (45 rows by default):
  Section 1 (seed)       : N_SEED rows  — all Persona X  (no drift key)
  Section 2 (eval)       : N_DRIFT drift=True (Persona Y) + N_STABLE drift=False (Persona X), shuffled
  Section 3 (final_eval) : N_FINAL rows — all Persona Y  (no drift key)

    # Single pair (A→B)
    python create_multi_persona_simulation.py \\
        --persona_x A --persona_y B \\
        --sim_dir   data/simulation \\
        --out_dir   data/simulation/multi_drift

    # All 10 transitions at once
    python create_multi_persona_simulation.py --all \\
        --sim_dir   data/simulation \\
        --out_dir   data/simulation/multi_drift

Requires
--------
    data/simulation/test_persona_{A,B,C,D,E}.jsonl
    (each containing 100 rows with embeddings + contexts)
"""

import argparse
import json
import random
from itertools import combinations
from pathlib import Path

PERSONAS  = ["A", "B","C","D","E"]
SEED      = 42

#  Section size defaults 
N_SEED   = 10   # Section 1: Persona X seed (initialise z_anchor)
N_DRIFT  = 20   # Section 2: Persona Y rows with drift=True
N_STABLE = 5    # Section 2: Persona X rows with drift=False
N_FINAL  = 10   # Section 3: Persona Y final eval


def load_jsonl(path: str):
    """Loads a JSONL file and returns a list of dictionaries.

    Args:
        path (str): The path to the JSONL file to load.

    Returns:
        list: A list of dictionaries representing the JSON lines.
    """
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_simulation(
    rows_x: list,
    rows_y: list,
    seed: int,
    n_seed: int   = N_SEED,
    n_drift: int  = N_DRIFT,
    n_stable: int = N_STABLE,
    n_final: int  = N_FINAL,
) -> list:
    """Samples and labels rows from persona X and persona Y to build a single
    45-row drift simulation in the simulation2.jsonl format.

    Args:
        rows_x (list): Configured rows from the SOURCE persona (before drift).
        rows_y (list): Configured rows from the TARGET persona (after drift).
        seed (int): Random seed for reproducibility.
        n_seed (int): Number of seed iterations. Defaults to N_SEED.
        n_drift (int): Number of drift iterations. Defaults to N_DRIFT.
        n_stable (int): Number of stable iterations. Defaults to N_STABLE.
        n_final (int): Number of final eval iterations. Defaults to N_FINAL.

    Returns:
        list: List of dictionaries, ready to be written as JSONL.
    """
    rng = random.Random(seed)

    need_x = n_seed + n_stable
    need_y = n_drift + n_final

    if len(rows_x) < need_x:
        raise ValueError(
            f"Persona X has only {len(rows_x)} rows but needs {need_x} "
            f"({n_seed} seed + {n_stable} stable eval). "
            "Lower --n_seed / --n_stable or use a larger test file."
        )
    if len(rows_y) < need_y:
        raise ValueError(
            f"Persona Y has only {len(rows_y)} rows but needs {need_y} "
            f"({n_drift} drift + {n_final} final eval)."
        )

    # ── Sample unique indices ──────────────────────────────────────────────────
    x_indices = rng.sample(range(len(rows_x)), need_x)
    y_indices = rng.sample(range(len(rows_y)), need_y)

    seed_rows   = [rows_x[i] for i in x_indices[:n_seed]]
    stable_rows = [rows_x[i] for i in x_indices[n_seed:]]
    drift_rows  = [rows_y[i] for i in y_indices[:n_drift]]
    final_rows  = [rows_y[i] for i in y_indices[n_drift:]]

    # ── Deep-copy & label (avoid mutating the loaded dicts) ───────────────────
    import copy

    def tag(row, section, drift=None):
        """Tags a row with the section and drift indicator."""
        r = copy.deepcopy(row)
        r["section"] = section
        r.pop("drift", None)
        if drift is not None:
            r["drift"] = drift
        return r

    labeled_seed   = [tag(r, "seed")                        for r in seed_rows]
    labeled_drift  = [tag(r, "eval",       drift=True)      for r in drift_rows]
    labeled_stable = [tag(r, "eval",       drift=False)     for r in stable_rows]
    labeled_final  = [tag(r, "final_eval")                  for r in final_rows]

    eval_pool = labeled_drift + labeled_stable
    rng.shuffle(eval_pool)

    all_rows = labeled_seed + eval_pool + labeled_final
    return all_rows


def save_jsonl(rows: list, path: Path):
    """Saves a list of dictionaries to a JSONL file.

    Args:
        rows (list): The list of dictionaries to save.
        path (Path): The pathlib.Path pointing to the target file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_and_save(
    persona_x: str,
    persona_y: str,
    sim_dir: Path,
    out_dir: Path,
    n_seed: int,
    n_drift: int,
    n_stable: int,
    n_final: int,
    seed: int,
):
    """Builds one X->Y simulation and writes it to disk.

    Args:
        persona_x (str): Source persona identifier.
        persona_y (str): Target persona identifier.
        sim_dir (Path): Base directory containing source test files.
        out_dir (Path): Output directory.
        n_seed (int): Number of seed interactions.
        n_drift (int): Number of drifted interactions.
        n_stable (int): Number of stable interactions.
        n_final (int): Number of final evaluations.
        seed (int): Random seed.
        
    Returns:
        Path: Path to the generated simulation file.
    """
    file_x = sim_dir / f"test_Persona_{persona_x}.jsonl"
    file_y = sim_dir / f"test_Persona_{persona_y}.jsonl"
    for f in (file_x, file_y):
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")

    rows_x = load_jsonl(str(file_x))
    rows_y = load_jsonl(str(file_y))

    rows = build_simulation(
        rows_x, rows_y, seed=seed,
        n_seed=n_seed, n_drift=n_drift, n_stable=n_stable, n_final=n_final,
    )

    out_path = out_dir / f"sim_{persona_x}_to_{persona_y}.jsonl"
    save_jsonl(rows, out_path)

    total = len(rows)
    drift_count  = sum(1 for r in rows if r.get("section") == "eval" and r.get("drift"))
    stable_count = sum(1 for r in rows if r.get("section") == "eval" and not r.get("drift"))
    print(
        f"  {persona_x}->{persona_y}: {total} rows written -> {out_path.name}  "
        f"[seed={n_seed}, drift={drift_count}, stable={stable_count}, final={n_final}]"
    )
    return out_path


def get_args():
    """Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    p = argparse.ArgumentParser(
        description="Build drift-simulation JSONL files for all persona pairs"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all",           action="store_true",
                   help="Build simulations for all 10 ordered persona pairs")
    g.add_argument("--persona_x",     type=str, choices=PERSONAS,
                   help="Source persona (before drift)")
    p.add_argument("--persona_y",     type=str, choices=PERSONAS,
                   help="Target persona (after drift)  [required unless --all]")
    p.add_argument("--sim_dir",       default="data/simulation",
                   help="Dir containing test_persona_X.jsonl files")
    p.add_argument("--out_dir",       default="data/simulation/multi_drift",
                   help="Output directory for simulation files")
    p.add_argument("--n_seed",        type=int, default=N_SEED)
    p.add_argument("--n_drift",       type=int, default=N_DRIFT)
    p.add_argument("--n_stable",      type=int, default=N_STABLE)
    p.add_argument("--n_final",       type=int, default=N_FINAL)
    p.add_argument("--seed",          type=int, default=SEED)
    return p.parse_args()


def main():
    """Main execution point for multi-persona drift simulation generation."""
    args    = get_args()
    sim_dir = Path(args.sim_dir)
    out_dir = Path(args.out_dir)

    if args.all:
        # All ordered pairs (X→Y where X ≠ Y, 5P2 = 20, or unique combos = 10)
        pairs = list(combinations(PERSONAS, 2))   # 10 unordered pairs
        print(f"Building {len(pairs)} drift simulations in {out_dir} ...")
        for x, y in pairs:
            build_and_save(x, y, sim_dir, out_dir,
                           args.n_seed, args.n_drift, args.n_stable, args.n_final,
                           args.seed)
    else:
        if args.persona_y is None:
            raise ValueError("--persona_y is required when not using --all")
        if args.persona_x == args.persona_y:
            raise ValueError("--persona_x and --persona_y must be different")
        print(f"Building simulation {args.persona_x}->{args.persona_y} in {out_dir} ...")
        build_and_save(args.persona_x, args.persona_y, sim_dir, out_dir,
                       args.n_seed, args.n_drift, args.n_stable, args.n_final,
                       args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
