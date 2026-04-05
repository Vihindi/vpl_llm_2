"""
VAE Session Inference with Sliding Window z-Anchor

    session = VAESessionInference(model_path="path/to/model.pt", max_context_size=8, momentum=0.9)

    # ── Warm start: pre-load N random pairs before any scoring ──
    session.initialize_with_pairs(
        chosen_embs=tensor_of_shape_N_x_D,    # [N, encoder_embed_dim]
        rejected_embs=tensor_of_shape_N_x_D,  # [N, encoder_embed_dim]
    )

    # Score a pair without updating the anchor
    rc, rr = session.score(target_chosen_emb, target_rejected_emb, drift=False)

    # New preference pair arrives → remove oldest hidden vector, recompute z_anchor
    rc, rr = session.score(
        target_chosen_emb, target_rejected_emb,
        drift=True,
        new_context_chosen_emb=ctx_chosen_emb,
        new_context_rejected_emb=ctx_rejected_emb,
    )

    # Reset the session entirely (e.g. new user)
    session.reset()
"""

import torch
import torch.nn.functional as F
from collections import deque


class VAESessionInference:
    """
    Inference wrapper around a frozen VAEModel.

    Maintains:
      - A sliding window (deque) of pair-encoder hidden vectors across turns.
      - A momentum-smoothed z_anchor representing the current user persona estimate.

    When drift=True, a new (chosen, rejected) context pair is provided, its hidden
    vector is computed via the PairEncoder, appended to the window (oldest removed
    if at capacity), and z_anchor is updated via the SequenceEncoder + momentum.

    When drift=False, the existing z_anchor is used directly for scoring.
    """

    def __init__(
        self,
        model_path: str = None,
        max_context_size: int = 8,
        momentum: float = 0.9,
        device: str = None,
        model_obj = None,
    ):
        """
        Args:
            model_path:       Path to the saved VAEModel (.pt file saved with torch.save(self, path)).
            max_context_size: Maximum number of pair-encoder vectors in the sliding window.
                              When a new vector arrives and the window is full, the oldest is dropped.
            momentum:         Momentum coefficient m for z_anchor update.
                              z_anchor = m * z_anchor + (1 - m) * z_new
            device:           'cuda', 'cpu', or None (auto-detect).
            model_obj:        Pre-loaded model object. If provided, model_path is ignored.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.momentum = momentum
        self.max_context_size = max_context_size

        if model_obj is not None:
            self.model = model_obj
        else:
            if model_path is None:
                raise ValueError("Must provide either model_path or model_obj")
            # Load the frozen model
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.latent_dim = self.model.latent_dim
        # Detect model dtype (bfloat16 if trained with bf16=True, else float32)
        self.dtype = next(self.model.parameters()).dtype
        print(f"Model dtype: {self.dtype}")

        # Sliding window: each element is a hidden vector from the PairEncoder
        # shape of each element: [latent_dim]
        self._hidden_window: deque = deque(maxlen=max_context_size)

        # The running persona estimate (momentum-smoothed z) — matches model dtype
        self._z_anchor: torch.Tensor = torch.zeros(1, self.latent_dim, device=self.device, dtype=self.dtype)


    def score(
        self,
        target_chosen_emb: torch.Tensor,
        target_rejected_emb: torch.Tensor,
        drift: bool = False,
        new_context_chosen_emb: torch.Tensor = None,
        new_context_rejected_emb: torch.Tensor = None,
    ):
        """
        Score a (chosen, rejected) target pair using the current z_anchor.

        Args:
            target_chosen_emb:      Embedding of the target chosen response.  Shape: [1, decoder_embed_dim]
            target_rejected_emb:    Embedding of the target rejected response. Shape: [1, decoder_embed_dim]
            drift:                  If True, update z_anchor using the new context pair before scoring.
            new_context_chosen_emb: Embedding of new context chosen (required if drift=True). Shape: [1, encoder_embed_dim]
            new_context_rejected_emb: Embedding of new context rejected (required if drift=True). Shape: [1, encoder_embed_dim]

        Returns:
            rc (float): Reward score for chosen response.
            rr (float): Reward score for rejected response.
        """
        if drift:
            if new_context_chosen_emb is None or new_context_rejected_emb is None:
                raise ValueError("drift=True requires both new_context_chosen_emb and new_context_rejected_emb.")
            self._update_anchor(new_context_chosen_emb, new_context_rejected_emb)

        # Cast to model dtype (e.g. bfloat16) and move to device
        target_chosen_emb   = target_chosen_emb.to(device=self.device, dtype=self.dtype)
        target_rejected_emb = target_rejected_emb.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            rc, rr = self.model.decode(target_chosen_emb, target_rejected_emb, self._z_anchor)

        return rc.item(), rr.item()

    def initialize_with_pairs(
        self,
        chosen_embs: torch.Tensor,
        rejected_embs: torch.Tensor,
    ):
        """
        Warm-start the session by pre-loading N preference pairs.

        Each pair is passed through the PairEncoder to produce a hidden vector.
        All N hidden vectors fill the sliding window (if N > max_context_size,
        only the last max_context_size are kept — oldest dropped first).
        The window is then passed through the SequenceEncoder to compute
        the initial z_anchor.

        Args:
            chosen_embs:   Tensor of shape [N, encoder_embed_dim]  — chosen embeddings
            rejected_embs: Tensor of shape [N, encoder_embed_dim]  — rejected embeddings
        """
        if chosen_embs.shape[0] != rejected_embs.shape[0]:
            raise ValueError("chosen_embs and rejected_embs must have the same number of rows (N).")

        self._hidden_window.clear()
        self._z_anchor = torch.zeros(1, self.latent_dim, device=self.device, dtype=self.dtype)

        chosen_embs  = chosen_embs.to(device=self.device, dtype=self.dtype)
        rejected_embs = rejected_embs.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            # Encode all N pairs in one batched call → [N, latent_dim]
            all_hidden = self.model.encode_pair(chosen_embs, rejected_embs)

        # Add each hidden vector one by one so the deque enforces maxlen properly
        for i in range(all_hidden.shape[0]):
            self._hidden_window.append(all_hidden[i].cpu())

        # Compute initial z_anchor from the full window (no momentum — clean cold start)
        with torch.no_grad():
            stacked = torch.stack(list(self._hidden_window), dim=0).to(self.device)
            seq_start_end = [(0, len(self._hidden_window))]
            mu, _log_var = self.model.encode_sequence(stacked, seq_start_end)
            mu = torch.clamp(mu, -1, 1)
            self._z_anchor = mu   # set directly, no momentum blending at init

        print(
            f"[VAESession] Initialized with {len(self._hidden_window)} pairs | "
            f"window={len(self._hidden_window)}/{self.max_context_size} | "
            f"z_anchor_norm={self._z_anchor.norm().item():.4f}"
        )

    def reset(self):
        """Reset the session: clear the sliding window and z_anchor."""
        self._hidden_window.clear()
        self._z_anchor = torch.zeros(1, self.latent_dim, device=self.device, dtype=self.dtype)
        print("[VAESession] Session reset.")


    @property
    def z_anchor(self) -> torch.Tensor:
        """Returns the current z_anchor (the smoothed user persona vector)."""
        return self._z_anchor.clone()

    @property
    def context_size(self) -> int:
        """Returns the number of hidden vectors currently in the sliding window."""
        return len(self._hidden_window)


    def _update_anchor(
        self,
        new_context_chosen_emb: torch.Tensor,
        new_context_rejected_emb: torch.Tensor,
    ):
        """Updates the session's z_anchor with a new preference pair.

        Args:
            new_context_chosen_emb (torch.Tensor): Chosen embedding of the new context pair.
            new_context_rejected_emb (torch.Tensor): Rejected embedding of the new context pair.
        """
        # Cast to model dtype (e.g. bfloat16) and move to device
        new_context_chosen_emb   = new_context_chosen_emb.to(device=self.device, dtype=self.dtype)
        new_context_rejected_emb = new_context_rejected_emb.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            # Step 1: PairEncoder → hidden vector for this new context pair
            # encode_pair expects [N, embed_dim] inputs
            h_new = self.model.encode_pair(new_context_chosen_emb, new_context_rejected_emb)  # [1, latent_dim]
            h_new = h_new.squeeze(0)  # [latent_dim]

        # Step 2: Append to sliding window (deque enforces maxlen automatically, drops oldest)
        self._hidden_window.append(h_new.cpu())

        with torch.no_grad():
            # Step 3: Stack all hidden vectors in the window → [N, latent_dim]
            stacked = torch.stack(list(self._hidden_window), dim=0).to(self.device)  # [N, latent_dim]

            # Step 4: SequenceEncoder (attention pooling) → mu, log_var
            # seq_start_end: one group spanning all N vectors (single user)
            seq_start_end = [(0, len(self._hidden_window))]
            mu, _log_var = self.model.encode_sequence(stacked, seq_start_end)  # [1, latent_dim]
            mu = torch.clamp(mu, -1, 1)  # match train-time clamping

            # Step 5: Momentum update of z_anchor
            # z_anchor = m * z_anchor + (1 - m) * mu
            self._z_anchor = self.momentum * self._z_anchor + (1 - self.momentum) * mu

        print(
            f"[VAESession] z_anchor updated | "
            f"window_size={len(self._hidden_window)}/{self.max_context_size} | "
            f"z_anchor_norm={self._z_anchor.norm().item():.4f}"
        )


def run_simulation(
    model_path: str,
    sim_data_path: str = "data/simulation/simulation2.jsonl",
    max_context_size: int = 8,
    momentum: float = 0.9,
    plot: bool = True,
    plot_save_path: str = None,
):
    """
    Run the z-anchor simulation on simulation2.jsonl (3-section format).

    Layout of simulation2.jsonl (45 rows):
      Section 1 rows  1-10  : Persona A seed       (section='seed',       no drift key)
      Section 2 rows 11-35  : 20 B (drift=True) + 5 A (drift=False), shuffled (section='eval')
      Section 3 rows 36-45  : Persona B final eval  (section='final_eval', no drift key)

    Two sessions run in parallel:
      - SESSION A (z-anchor): updated only on drift=True eval rows; frozen for final_eval rows
      - SESSION B (baseline): NEVER updated (frozen z throughout)

    Args:
        model_path:       Path to saved VAEModel .pt file.
        sim_data_path:    Path to simulation2.jsonl (default).
        max_context_size: Sliding window capacity for the anchor session.
        momentum:         Momentum coefficient for the anchor session.
        plot:             Whether to show plots.
        plot_save_path:   If given, saves figures to disk instead of showing.

    Returns:
        dict with per-section accuracy, drift info, and raw score lists.
    """
    import json
    import matplotlib
    matplotlib.use("Agg" if plot_save_path else "TkAgg")
    import matplotlib.pyplot as plt

    # ── Load and split by section ─────────────────────────────────────────────
    all_rows = []
    with open(sim_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))

    seed_rows       = [r for r in all_rows if r.get("section") == "seed"]
    eval_rows       = [r for r in all_rows if r.get("section") == "eval"]
    final_eval_rows = [r for r in all_rows if r.get("section") == "final_eval"]

    assert len(seed_rows) > 0,       "No seed rows found (section='seed')"
    assert len(eval_rows) > 0,       "No eval rows found (section='eval')"
    assert len(final_eval_rows) > 0, "No final_eval rows found (section='final_eval')"

    print(f"[Simulation] Loaded {len(all_rows)} rows | "
          f"seed={len(seed_rows)} | eval={len(eval_rows)} | final_eval={len(final_eval_rows)}")

    # ── Build sessions ───────────────────────────────────────────────────────
    session_anchor   = VAESessionInference(model_path, max_context_size, momentum)
    session_baseline = VAESessionInference(model_path, max_context_size, momentum)

    # Warm-start BOTH sessions with the same seed pairs
    chosen_embs   = torch.tensor(
        [r["embeddings"]["embedding_chosen"]   for r in seed_rows], dtype=torch.float32
    )
    rejected_embs = torch.tensor(
        [r["embeddings"]["embedding_rejected"] for r in seed_rows], dtype=torch.float32
    )
    session_anchor.initialize_with_pairs(chosen_embs, rejected_embs)
    session_baseline.initialize_with_pairs(chosen_embs, rejected_embs)

    # ── Evaluation loop ──────────────────────────────────────────────────────
    anchor_correct   = []
    baseline_correct = []
    drift_flags      = []     # True/False per eval turn (section 2 only)
    drift_indices    = []     # turn indices where drift=True
    # reward score tracking (section 2 = eval turns)
    anchor_rc_list   = []
    anchor_rr_list   = []
    base_rc_list     = []
    base_rr_list     = []
    # final eval tracking (section 3)
    final_anchor_correct  = []
    final_base_correct    = []
    final_anchor_rc_list  = []
    final_base_rc_list    = []
    final_anchor_rr_list  = []
    final_base_rr_list    = []

    print("\n" + "=" * 80)
    print("SECTION 2 — Drift-aware eval (anchor updates on drift=True)")
    print(f"{'Turn':<5} {'Drift':<7} {'Subset':<11} "
          f"{'Anchor rc':>10} {'Anchor rr':>10} {'A ok?':>6} "
          f"{'Base rc':>9} {'Base rr':>9} {'B ok?':>6} "
          f"{'w_sz':>5} {'z_norm':>7}")
    print("-" * 80)

    drift_count = 0
    for turn, row in enumerate(eval_rows):
        is_drift = row["drift"]
        subset   = row.get("data_subset", "?")
        drift_flags.append(is_drift)
        if is_drift:
            drift_count += 1
            drift_indices.append(turn)

        # Target embeddings (what we score)
        tc = torch.tensor([row["embeddings"]["embedding_chosen"]],   dtype=torch.float32)
        tr = torch.tensor([row["embeddings"]["embedding_rejected"]], dtype=torch.float32)

        # Context embeddings (used to update anchor when drift=True)
        ctx   = row["contexts"][0]
        ctx_c = torch.tensor([ctx["embedding_chosen"]],   dtype=torch.float32)
        ctx_r = torch.tensor([ctx["embedding_rejected"]], dtype=torch.float32)

        # SESSION A (z-anchor): respect drift flag
        rc_a, rr_a = session_anchor.score(
            tc, tr,
            drift=is_drift,
            new_context_chosen_emb=ctx_c  if is_drift else None,
            new_context_rejected_emb=ctx_r if is_drift else None,
        )
        ok_a = rc_a > rr_a
        anchor_correct.append(ok_a)
        anchor_rc_list.append(rc_a)
        anchor_rr_list.append(rr_a)

        # SESSION B (baseline): NEVER update anchor
        rc_b, rr_b = session_baseline.score(tc, tr, drift=False)
        ok_b = rc_b > rr_b
        baseline_correct.append(ok_b)
        base_rc_list.append(rc_b)
        base_rr_list.append(rr_b)

        print(
            f"{turn+1:<5} {'TRUE' if is_drift else 'false':<7} {subset:<11} "
            f"{rc_a:>10.4f} {rr_a:>10.4f} {'Y' if ok_a else 'N':>6} "
            f"{rc_b:>9.4f} {rr_b:>9.4f} {'Y' if ok_b else 'N':>6} "
            f"{session_anchor.context_size:>3}/{session_anchor.max_context_size} "
            f"{session_anchor.z_anchor.norm().item():>7.4f}"
        )

    # ── Section 3: Final eval (no anchor updates) ────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 3 — Final eval (Persona B, anchor FROZEN, no updates)")
    print(f"{'Turn':<5} {'Subset':<11} "
          f"{'Anchor rc':>10} {'Anchor rr':>10} {'A ok?':>6} "
          f"{'Base rc':>9} {'Base rr':>9} {'B ok?':>6}")
    print("-" * 60)
    for f_turn, row in enumerate(final_eval_rows):
        tc = torch.tensor([row["embeddings"]["embedding_chosen"]],   dtype=torch.float32)
        tr = torch.tensor([row["embeddings"]["embedding_rejected"]], dtype=torch.float32)
        subset = row.get("data_subset", "?")
        # Score with final anchor — no updates for either session
        rc_a, rr_a = session_anchor.score(tc, tr, drift=False)
        rc_b, rr_b = session_baseline.score(tc, tr, drift=False)
        ok_a, ok_b = rc_a > rr_a, rc_b > rr_b
        final_anchor_correct.append(ok_a)
        final_base_correct.append(ok_b)
        final_anchor_rc_list.append(rc_a)
        final_anchor_rr_list.append(rr_a)
        final_base_rc_list.append(rc_b)
        final_base_rr_list.append(rr_b)
        print(f"{f_turn+1:<5} {subset:<11} "
              f"{rc_a:>10.4f} {rr_a:>10.4f} {'Y' if ok_a else 'N':>6} "
              f"{rc_b:>9.4f} {rr_b:>9.4f} {'Y' if ok_b else 'N':>6}")

    # ── Accuracy summary ─────────────────────────────────────────────────────
    n  = len(eval_rows)
    nf = len(final_eval_rows)
    anchor_acc   = sum(anchor_correct)        / n  * 100
    baseline_acc = sum(baseline_correct)      / n  * 100
    f_anchor_acc = sum(final_anchor_correct)  / nf * 100
    f_base_acc   = sum(final_base_correct)    / nf * 100

    # Per-drift breakdown (section 2)
    drift_anchor   = [ok for ok, f in zip(anchor_correct,   drift_flags) if f]
    drift_baseline = [ok for ok, f in zip(baseline_correct, drift_flags) if f]
    nodrift_anchor   = [ok for ok, f in zip(anchor_correct,   drift_flags) if not f]
    nodrift_baseline = [ok for ok, f in zip(baseline_correct, drift_flags) if not f]
    n_drift   = len(drift_anchor)
    n_nodrift = len(nodrift_anchor)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  {'':35s} {'z-anchor':>12}  {'Baseline':>12}")
    print(f"  {'-'*61}")
    print(f"  {'[Sec 2] Overall':35s} {sum(anchor_correct):>5}/{n}={anchor_acc:>5.1f}%  {sum(baseline_correct):>5}/{n}={baseline_acc:>5.1f}%")
    if n_drift > 0:
        a_d = sum(drift_anchor)   / n_drift   * 100
        b_d = sum(drift_baseline) / n_drift   * 100
        print(f"  {'[Sec 2] drift=True  (Persona B)':35s} {sum(drift_anchor):>5}/{n_drift}={a_d:>5.1f}%  {sum(drift_baseline):>5}/{n_drift}={b_d:>5.1f}%")
    if n_nodrift > 0:
        a_nd = sum(nodrift_anchor)   / n_nodrift * 100
        b_nd = sum(nodrift_baseline) / n_nodrift * 100
        print(f"  {'[Sec 2] drift=False (Persona A)':35s} {sum(nodrift_anchor):>5}/{n_nodrift}={a_nd:>5.1f}%  {sum(nodrift_baseline):>5}/{n_nodrift}={b_nd:>5.1f}%")
    print(f"  {'[Sec 3] Final eval  (Persona B)':35s} {sum(final_anchor_correct):>5}/{nf}={f_anchor_acc:>5.1f}%  {sum(final_base_correct):>5}/{nf}={f_base_acc:>5.1f}%")
    print(f"  {'-'*61}")
    print(f"  Sec2 Improvement (overall) : {anchor_acc - baseline_acc:+.1f}%")
    print(f"  Sec3 Improvement (final)   : {f_anchor_acc - f_base_acc:+.1f}%")
    print(f"  Total drift events         : {drift_count}/{n}")


    # ── Plot: cumulative accuracy vs turn, with drift markers ────────────────
    if plot or plot_save_path:
        turns         = list(range(1, n + 1))
        cum_anchor    = [sum(anchor_correct[:i])   / i * 100 for i in turns]
        cum_baseline  = [sum(baseline_correct[:i]) / i * 100 for i in turns]
        cum_drifts    = [sum(drift_flags[:i]) for i in turns]  # x2: drifts seen so far

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("VAE Session Inference — Simulation Results", fontsize=13, fontweight="bold")

        # ── Left: Accuracy vs Turn (with drift markers) ──────────────────────
        ax = axes[0]
        ax.plot(turns, cum_anchor,   label="z-anchor (drift-aware)", color="steelblue",  linewidth=2)
        ax.plot(turns, cum_baseline, label="Baseline (frozen z)",     color="darkorange", linewidth=2, linestyle="--")
        for di in drift_indices:
            ax.axvline(x=di + 1, color="red", alpha=0.25, linewidth=1)
        ax.axvline(x=drift_indices[0] + 1, color="red", alpha=0.25, linewidth=1, label="Drift event")
        ax.set_xlabel("Evaluation turn")
        ax.set_ylabel("Cumulative accuracy (%)")
        ax.set_title("Accuracy vs Turn")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── Right: Accuracy vs Number of Drifts Seen ────────────────────────
        ax2 = axes[1]
        # Gather accuracy at each drift event (x = cumulative drifts so far)
        drift_x, anchor_at_drift, base_at_drift = [], [], []
        cumulative_drifts = 0
        for i, (flag, a_ok, b_ok) in enumerate(zip(drift_flags, anchor_correct, baseline_correct)):
            if flag:
                cumulative_drifts += 1
                drift_x.append(cumulative_drifts)
                anchor_at_drift.append(sum(anchor_correct[:i+1]) / (i+1) * 100)
                base_at_drift.append(sum(baseline_correct[:i+1]) / (i+1) * 100)

        ax2.plot(drift_x, anchor_at_drift,  marker="o", label="z-anchor", color="steelblue",  linewidth=2)
        ax2.plot(drift_x, base_at_drift,    marker="s", label="Baseline",  color="darkorange", linewidth=2, linestyle="--")
        ax2.set_xlabel("Number of drift events seen")
        ax2.set_ylabel("Cumulative accuracy (%)")
        ax2.set_title("Accuracy vs Number of Drifts")
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if plot_save_path:
            plt.savefig(plot_save_path, dpi=150, bbox_inches="tight")
            print(f"[Simulation] Plot saved to {plot_save_path}")
        else:
            plt.show()

        # ── Figure 2: Chosen reward fluctuation across turns ─────────────────
        fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig2.suptitle("Chosen Reward Fluctuation — z-Anchor vs Baseline", fontsize=13, fontweight="bold")

        # Background shading: red for drift=True turns, green for drift=False
        for t_idx, flag in enumerate(drift_flags):
            color = "#ffdddd" if flag else "#ddffdd"
            ax3.axvspan(t_idx + 0.5, t_idx + 1.5, color=color, alpha=0.4)
            ax4.axvspan(t_idx + 0.5, t_idx + 1.5, color=color, alpha=0.4)

        # ── Top subplot: rc and rr for both sessions ──────────────────────────
        ax3.plot(turns, anchor_rc_list, color="steelblue",      linewidth=1.8)
        ax3.plot(turns, anchor_rr_list, color="steelblue",      linewidth=1.2, linestyle=":", alpha=0.7)
        ax3.plot(turns, base_rc_list,   color="darkorange",     linewidth=1.8, linestyle="--")
        ax3.plot(turns, base_rr_list,   color="darkorange",     linewidth=1.2, linestyle=(0,(3,1,1,1)), alpha=0.7)
        ax3.axhline(0, color="black", linewidth=0.8, linestyle=":")
        # Correctness markers on rc line: green circle = correct, red X = wrong
        from matplotlib.lines import Line2D
        for t, rc_a, rc_b, ok_a, ok_b in zip(turns, anchor_rc_list, base_rc_list,
                                               anchor_correct, baseline_correct):
            ax3.scatter(t, rc_a, marker="o" if ok_a else "x",
                        color="green" if ok_a else "red", s=60, zorder=5, linewidths=2)
            ax3.scatter(t, rc_b, marker="o" if ok_b else "x",
                        color="green" if ok_b else "red", s=60, zorder=5, linewidths=2)
        ax3.legend(handles=[
            Line2D([0],[0], color="steelblue",  linewidth=2,   label="Anchor rc (chosen)"),
            Line2D([0],[0], color="steelblue",  linewidth=1.2, linestyle=":",  alpha=0.7, label="Anchor rr (rejected)"),
            Line2D([0],[0], color="darkorange", linewidth=2,   linestyle="--", label="Baseline rc (chosen)"),
            Line2D([0],[0], color="darkorange", linewidth=1.2, linestyle=(0,(3,1,1,1)), alpha=0.7, label="Baseline rr (rejected)"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="green", markersize=8, label="Correct (rc > rr)"),
            Line2D([0],[0], marker="x", color="red", markersize=8, linewidth=2, label="Wrong (rc < rr)"),
        ], loc="upper left", fontsize=8)
        ax3.set_ylabel("Reward score")
        ax3.set_title("Chosen & Rejected Reward per Turn   |   red bg = drift=True,  green bg = drift=False   |   markers = correctness")
        ax3.grid(True, alpha=0.3)

        # ── Bottom subplot: margin (rc - rr) for both sessions ────────────────
        anchor_margin = [rc - rr for rc, rr in zip(anchor_rc_list, anchor_rr_list)]
        base_margin   = [rc - rr for rc, rr in zip(base_rc_list,   base_rr_list)]
        ax4.plot(turns, anchor_margin, color="steelblue",  linewidth=1.8)
        ax4.plot(turns, base_margin,   color="darkorange", linewidth=1.8, linestyle="--")
        ax4.axhline(0, color="black", linewidth=1.2, linestyle="-")
        # Correctness markers on margin line
        for t, m_a, m_b, ok_a, ok_b in zip(turns, anchor_margin, base_margin,
                                             anchor_correct, baseline_correct):
            ax4.scatter(t, m_a, marker="o" if ok_a else "x",
                        color="green" if ok_a else "red", s=60, zorder=5, linewidths=2)
            ax4.scatter(t, m_b, marker="o" if ok_b else "x",
                        color="green" if ok_b else "red", s=60, zorder=5, linewidths=2)
        ax4.legend(handles=[
            Line2D([0],[0], color="steelblue",  linewidth=2, label="Anchor margin"),
            Line2D([0],[0], color="darkorange", linewidth=2, linestyle="--", label="Baseline margin"),
            Line2D([0],[0], color="black",      linewidth=1.5, label="Decision boundary (0)"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="green", markersize=8, label="Correct"),
            Line2D([0],[0], marker="x", color="red",   markersize=8, linewidth=2, label="Wrong"),
        ], loc="upper left")
        ax4.set_xlabel("Evaluation turn")
        ax4.set_ylabel("Margin (rc - rr)")
        ax4.set_title("Reward Margin (rc - rr) per Turn   |   above 0 = correct prediction   |   markers = correctness")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if plot_save_path:
            reward_plot_path = plot_save_path.replace(".png", "_reward_fluctuation.png")
            fig2.savefig(reward_plot_path, dpi=150, bbox_inches="tight")
            print(f"[Simulation] Reward fluctuation plot saved to {reward_plot_path}")
        else:
            plt.show()

    return {
        "anchor_correct"   : anchor_correct,
        "baseline_correct" : baseline_correct,
        "anchor_acc"       : anchor_acc,
        "baseline_acc"     : baseline_acc,
        "turns"            : list(range(1, n + 1)),
        "drift_flags"      : drift_flags,
        "drift_indices"    : drift_indices,
    }


if __name__ == "__main__":
    import os

    MODEL_PATH   = os.getenv("VAE_MODEL_PATH", "path/to/your/model.pt")   # ← set via env or edit here
    SIM_DATA     = os.getenv("SIM_DATA_PATH",  "data/simulation/simulation.jsonl")
    PLOT_SAVE    = os.getenv("PLOT_SAVE_PATH",  None)   # e.g. "simulation_plot.png" in Colab

    run_simulation(
        model_path    = MODEL_PATH,
        sim_data_path = SIM_DATA,
        max_context_size = 8,
        momentum         = 0.9,
        plot             = True,
        plot_save_path   = PLOT_SAVE,
    )
