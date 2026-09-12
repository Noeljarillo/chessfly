# ChessFly — engineering plan

Training the FlyWire *Drosophila melanogaster* connectome (139,255 neurons,
~54.5M synapses) to play chess.

Prior art in this genre: `nftechie/doomfly`, `nftechie/stonkfly`,
`philshiu/Drosophila_brain_model` (Shiu et al. 2024, *Nature*) — the last is the
substrate baseline used here.

---

## 0. Verdict — lead with this

**3600 Elo is unreachable. Not by a tuning margin — by ~3 orders of magnitude in
parameter capacity, and by a second, independent gap that capacity cannot close.**

Two separate walls:

**Wall 1 — capacity.** Fitting a log-linear curve through the three public
policy-only (1-node, no search) anchors:

| Network | Params | Policy-only Elo (approx.) |
|---|---|---|
| Maia-1900 (6×64 ResNet) | 1.2M | ~1900 |
| Leela 20×256 | 22M | ~2575 |
| Leela BT4 (transformer) | ~200M | ~2850 |

`Elo ≈ −694 + 433·log₁₀(N_params)`  (R² > 0.99 on 3 points; extrapolation beyond
200M is **unsound** — the curve certainly flattens. Treat the number below as a
*generous lower bound* on the requirement.)

Solving for 3600 gives **8.2 × 10⁹ parameters**. The connectome offers:

- **2.7M** trainable weights if one weight per *neuron-to-neuron edge* (≥5-synapse threshold)
- **54.5M** trainable weights if one weight per *individual synapse* (still fully connectome-constrained — see §2)

That is **151× short** at best, **3,000× short** at worst. In information terms,
at the empirical ~2 bits/parameter capacity law (Allen-Zhu & Li 2024): 109 Mbit
available vs. 16.5 Gbit required.

**Wall 2 — search.** Stockfish's 3600 is *not* a network property. It is a
~10⁸ nodes/sec alpha-beta search to depth 30+, with a ~23–69M-parameter NNUE as
the leaf evaluator. No policy network of any size currently reaches 3600 at one
node; the best measured 1-node play is ~2850–2900. A feedforward/recurrent
policy, however large, is ~700 Elo short of the target *before* capacity even
enters. A single fly-brain forward pass is, by construction, one node.

**Consequence.** At the realistic ceiling derived in §2 (~1,850 Elo), the
expected score against Stockfish-max is **2.4 × 10⁻⁵** — roughly one
non-loss per 42,000 games. Any plan that promises otherwise is lying.

### What this project is instead

Reframed target, and it is a genuinely good one:

> **Produce the first measured capacity↔competence curve for a biological
> connectome held topologically fixed**, using chess as the complexity dial —
> and ship a fly that plays real, legal, non-trivial chess at **1,500–1,900 Elo**.

The scientific deliverable is the curve and the ablation set (§7), not the Elo.
The Elo is the demo.

### Prior artifact: the blackjack-dealer checkpoint

**Verdict: keep the config, discard the weights. Net liability as an
initialization.**

*Why it's a liability.* A blackjack dealer is a lookup table: ~20 reachable
states (hard/soft totals 4–21) × 2 actions, gated by one rule (hit below 17).
Total task entropy ≈ **7 bits**. Training 2.7M–54.5M parameters to convergence on
a 7-bit target does not leave the network "warmed up" — it drives it into a
low-dimensional attractor with severe rank collapse and a large fraction of
dormant units. This is the loss-of-plasticity failure documented by Dohare et al.
(2024, *Nature*): networks trained to convergence on a simple task measurably lose
the ability to learn a subsequent hard one. Fine-tuning from that state on a task
needing ~10⁶× more entropy is worse than a fresh init.

*Falsifiable test, run in week 1.* Compute per-region effective rank (stable rank
‖W‖_F²/‖W‖₂²) and the dormant-unit fraction (units with <0.1 Hz across 10k random
inputs) of the blackjack checkpoint. **If effective rank < 5% of dimension or
dormant fraction > 30% in the central brain, the weights are confirmed dead** —
discard.

*What is genuinely reusable, and worth ~30–50 GPU-hours:*
- Neuron parameters (τ_m, τ_syn, thresholds, refractory, per-region gain) already
  tuned to a stable non-saturating regime. This is §5 Stage 0, already done.
- The I/O harness: sensory-neuron driver, spike-count decoder plumbing, the
  CSR connectome loader and its index layout.
- Negative results on which surrogate-gradient β values explode.

*If you insist on warm-starting the weights:* shrink-and-perturb (Ash & Adams
2020) at λ=0.3, σ=0.01·std(W). Expect it to match a fresh init, not beat it.
Budget one 20-GPU-hour A/B; do not spend more.

---

## 1. Substrate specification

| Choice | Value |
|---|---|
| Neuron model | Current-based LIF + adaptive threshold (ALIF) on a designated 20% |
| Timestep | dt = 0.5 ms |
| Trial length | T = 96 steps = 48 ms simulated |
| Simulator | **Custom JAX** (`lax.scan` + custom-VJP spike op + Pallas SpMM) |
| Precision | bf16 activations, fp32 master weights |
| Dale's law | Enforced via sign-constrained parametrization |

**Baseline calibration** from Shiu et al. 2024 / `philshiu/Drosophila_brain_model`:
τ_m = 20 ms, V_rest = −52 mV, V_θ = −45 mV, V_reset = −52 mV, refractory 2.2 ms,
τ_syn = 5 ms, unitary synaptic weight 0.275 mV. *(Values as published; treated as
the starting point of the Stage-0 search, not as fixed.)*

**Dale's law parametrization.** `W_ij = s_i · softplus(a_ij)`, where `s_i = ±1` is
fixed by the FlyWire neurotransmitter prediction for the presynaptic neuron
(Eckstein et al. 2024: ~48% ACh excitatory, ~22% GABA, ~22% Glu, ~4% monoamines).
Only `a_ij` is trained, so a neuron can never flip sign. Cornford et al. (2021)
show sign-constrained networks can match unconstrained ones under this
parametrization; I budget a ~75-Elo penalty anyway (§2).

**Adaptation.** 20% of central-brain neurons get an adaptive threshold with
τ_adapt sampled log-uniformly over 20 ms – 2 s (Bellec et al. 2018 ALIF). This is
a *time constant and a gain* — both explicitly permitted — not new state
topology. Rationale: mean shortest path in FlyWire is ~2.8–3.5 hops, so
computational depth must come from time, and pure LIF over 96 steps has a ~20 ms
memory horizon. **Marked as a relaxation if challenged**: it adds one scalar
dynamic variable per neuron.

### Fidelity/trainability trade — three tiers, tier B chosen

| Tier | Model | Trainable? | Verdict |
|---|---|---|---|
| A | Multi-compartment, conductance-based, NT-typed receptor kinetics | ~50× cost, no stable gradients through 96 steps | Rejected — unaffordable |
| **B** | **Current-based LIF/ALIF, Dale-constrained, NT signs from FlyWire, dt=0.5 ms** | **Yes, surrogate-grad BPTT** | **Chosen** |
| C | Rate-based ReLU surrogate, same adjacency | Trivially | Stage-1 shadow only |

Tier B keeps every property the *claim* depends on — real connectome adjacency,
real neurotransmitter signs, discrete spikes, biological time constants — and
drops only the properties that are unobservable at the readout (dendritic
compartments, conductance-based reversal potentials, stochastic release).

Tier C is used, but it is honestly **"a sparse RNN with a fly-shaped mask."** It
is a pretraining device and an ablation control (§7), never the headline result.
Both numbers get reported.

**Why not Brian2/NEST.** Neither differentiates. BPTT requires autodiff through
the full 96-step unroll; Brian2 (C++ standalone) and NEST (MPI) give forward
simulation only, and GeNN/Brian2CUDA give fast forward without gradients. JAX
gives `lax.scan` + `jax.checkpoint` + a custom VJP for the spike nonlinearity.
Brian2 is retained as a **cross-validation oracle**: forward-simulate 1,000
trials in both and require max per-neuron spike-count divergence < 2%.

**Kernel note.** `jax.experimental.sparse.bcoo_dot_general` is 5–20× off what a
hand-written kernel achieves here. Budget two engineer-weeks for a Pallas/Triton
SpMM over a static CSR with a fixed neuropil-blocked permutation (~78 neuropils;
most inter-neuropil blocks are empty, nonzero blocks run 10–20% dense).

---

## 2. Capacity analysis

### 2.1 What is actually trainable

| Quantity | Count | Note |
|---|---|---|
| Neurons | 139,255 | FlyWire FAFB v783, proofread |
| Synapses | ~54.5M | — |
| Neuron-to-neuron edges (≥5 syn) | ~2.7M | — |
| Per-neuron params (τ_m, τ_syn, V_θ, refrac, gain) | ~0.70M | 5 × 139,255 |
| Neuromodulatory gains (per NT × per neuropil) | ~470 | 6 NT × 78 neuropils |

**The single most important design decision in this plan** is edge-weights vs.
per-synapse-weights. Both are strictly connectome-constrained — a synapse that
exists in FlyWire is not an invented connection. Choosing per-synapse weights
buys a **20× capacity increase for free** and costs only ~3× wall-clock (the
sparse kernel is bandwidth-bound on neuron state, not synapse count, until
~10M nnz). **Take it.** Trainable parameter count: **≈ 55.2M**.

### 2.2 Bits

At the empirical ~2 bits/parameter capacity law:

| Configuration | Params | Capacity |
|---|---|---|
| Edge weights only | 3.4M | 6.8 Mbit (850 KB) |
| **Per-synapse weights (chosen)** | **55.2M** | **110 Mbit (13.8 MB)** |
| Required for 3600 (log-linear fit) | 8.2B | 16.5 Gbit (2.1 GB) |

**Shortfall: 151×** (per-synapse) to **2,400×** (edge-only).

### 2.3 Comparison to known strong chess networks

| Network | Params | Strength |
|---|---|---|
| Maia-1100 / 1900 | ~1.2M | 1100 / 1900 (human-matched) |
| ChessFly, edge weights | 3.4M | — |
| Stockfish NNUE HalfKAv2_hm (L1=1024) | ~23M | 3600 *with* 10⁸ nps search |
| **ChessFly, per-synapse (chosen)** | **55.2M** | **target 1,500–1,900** |
| Stockfish NNUE big net (L1=3072) | ~69M | 3600 *with* search |
| Leela BT4 | ~200M | ~2850 at 1 node; ~3600 with 10⁵ nps MCTS |

The comparison that matters: ChessFly sits between the two Stockfish NNUE nets in
raw parameter count and will still be ~1,700 Elo weaker, because (a) NNUE gets
10⁸ nodes/sec of search on top and (b) NNUE's parameters sit in a topology
designed for the task, whereas ChessFly's sit wherever the fly's evolution put
them.

### 2.4 Realistic Elo ceiling

Start from the fit at 55.2M params: **2,656**. Apply four penalties, each an
engineering estimate — **all four are assumptions, not literature results:**

| Penalty | Elo | Reason |
|---|---|---|
| No 8×8 translation equivariance | −300 | Connectome has no convolutional prior; a same-size MLP loses this much on board games |
| Topology fixed, not optimizable | −225 | Cannot allocate width where the task needs it; severe bottlenecks (§7 KC-3) |
| Spike discretization + surrogate gradient | −150 | Typical SNN-vs-ANN gap of 2–6% task accuracy |
| Dale's law sign constraint | −75 | Per Cornford et al., small but nonzero |
| **Net ceiling** | **≈ 1,905** | |

**Stated ceiling: 1,500–1,900 Elo. Point estimate 1,750. Floor if two or more
penalties come in worse than estimated: 1,150.**

Edge-weight-only variant, same arithmetic: **1,100–1,400**, point estimate 1,340.

### 2.5 What is *not* the bottleneck

Worth stating so the analysis isn't mistaken for hand-waving about "the brain is
small":

- **Output channel.** ~1,300 descending neurons × ~2 bits/neuron/trial = 2.6 kbit.
  Selecting among ~35 legal moves needs 5.1 bits. Overhead: 500×. Fine.
- **Input channel.** 768 retinotopic units × ~3 bits = 2.3 kbit, against ~160 bits
  of board state. Fine.
- **Compute.** ~1,050 GPU-hours total (§5). The 24 GB GPU is not the constraint.
- **Depth.** 96 timesteps of recurrence gives effective depth ~96, well past the
  ~20 layers strong chess nets use.

**The bottleneck is weight entropy and fixed topology, and nothing else.** That
is a clean, publishable finding on its own.

---

## 3. I/O encoding

### 3.1 Board → spikes

**Piece placement → retinotopic visual pathway.** Each optic lobe has ~800
medulla columns. Allocate 12 columns per board square per eye: 64 × 12 = 768
columns, ~96% of one lobe. Column *k* of square *s* is driven iff piece type *k*
occupies *s* (6 white + 6 black = 12 planes). This is exactly the standard
768-feature chess bitboard, mapped onto the fly's *actual* retinotopy — the
8×8 board is literally projected onto the eye.

- Encoding: **rate**, Poisson, 80 Hz active / 2 Hz background, sustained across
  all 96 steps (the board is static within one decision, so rate is correct).
- Plus a **synchronous onset burst** at t = 0–2 ms across all active columns.
  This gives the network a phase reference / clock. *Assumption*: this materially
  speeds convergence; ablate it in Stage 1 and report.
- Drive R1–R8 → lamina L1–L5 → medulla; both lobes get the same board (bilateral
  redundancy is a feature — see the mirror test in §6.4).

**Everything else → non-visual pathways.** The antennal lobe has 51 glomeruli;
17 are allocated:

| State | Pathway | Encoding |
|---|---|---|
| Side to move | Ocellar pathway (L-neurons) | Global DC rate offset, 0 or 60 Hz |
| Castling rights (4 bits) | AL glomeruli DM1, DM2, DL5, VA1v | 70 Hz / 2 Hz per bit |
| En passant file (8 + none) | 9 AL glomeruli | 1-hot, 70 Hz |
| Repetition count (0/1/2) | 3 AL glomeruli | thermometer code |
| Halfmove clock (0–100 → 8 bins) | Johnston's organ JO-A/B | rate, 5–75 Hz |
| Fullmove number (log-binned, 4) | JO-C | rate |

Rationale for splitting modalities: the visual and olfactory pathways converge
only in the central brain (LH, MB, SIP/SMP), which forces the network to actually
integrate placement with rights rather than treating them as 785 undifferentiated
input lines. *Assumption* — ablate against a flat "all inputs to medulla" control.

**Timing.** T = 96 steps × 0.5 ms = 48 ms. Biologically motivated: fly
looming-evoked escape runs on a 30–60 ms sensorimotor latency, so 48 ms is one
fly "decision." Readout window is the last 32 steps (t = 32–48 ms), after the
onset transient has propagated the ~3-hop mean path length several times over.

### 3.2 Spikes → move

**Factored readout over descending neurons.** ~1,300 DNs exist; a 1-hot over the
1,968 legal UCI move encodings is impossible. Instead:

- 64 DN groups → **from-square** logits
- 64 DN groups → **to-square** logits
- 5 DN groups → **promotion piece** logits
- 133 groups × 10 DNs each = 1,330 DNs ≈ the full population.

`logit(g) = (spike count of the 10 DNs in g over last 32 steps) / 10`, then
`score(move) = logit_from[f] + logit_to[t] + logit_promo[p]`.

A second, separately-pooled set of ~40 DNs projects a scalar **value head**
(WDL) via rate over the same window.

### 3.3 Illegal moves — and the honest accounting

Three-layer handling:

1. **Training:** mask illegal moves before the softmax (standard AlphaZero
   practice). The mask comes from `python-chess`, externally.
2. **Play:** take the argmax over legal moves.
3. **Reporting:** *always publish the **unmasked legal-move rate*** — the fraction
   of held-out positions where the argmax over all 4,672 move encodings is legal
   without any mask. Target ≥ 95% by M4.

**This is a declared relaxation.** The legality oracle is external and not
learned. Justification: learning move legality from scratch consumes an estimated
30–40% of available weight entropy to encode rules that are perfectly known a
priori, for an estimated −500 Elo. Every strong chess engine does the same thing.
The unmasked rate metric exists so the relaxation cannot be quietly used to
inflate the claim that "the fly knows chess."

---

## 4. Learning rule

**Chosen: surrogate-gradient BPTT**, full unroll over T = 96, with gradient
checkpointing every 12 steps.

Surrogate: triangular / boxcar (Bellec et al. 2018), `∂S/∂v ≈ γ·max(0, 1 − |v − V_θ|/V_θ)`,
γ = 0.3. Fast-sigmoid (SuperSpike, Zenke & Ganguli 2018) as the fallback; Zenke &
Vogels (2021) show results are robust across surrogate shape but sensitive to
scale, so γ is swept in Stage 0 and then frozen.

### Why not the alternatives

| Rule | Why rejected |
|---|---|
| **e-prop** | O(1)-in-time memory, biologically plausible, but its eligibility traces discard the non-local-in-time component of the gradient, costing 10–30% on tasks with real temporal structure. Here T=96 is short enough that full BPTT fits in 24 GB (§5). **Held in reserve**: switch to e-prop only if the memory budget breaks. |
| **Reward-modulated STDP** | Three-factor rules with a scalar global reward have gradient variance scaling with parameter count. At d = 55.2M the sample complexity is ~10⁴–10⁶× BPTT's. Dead on arrival. |
| **Evolution strategies** | ES gradient variance also scales with d. For d = 55.2M you need ~10⁴–10⁵ perturbations per update × ~10⁴ updates ≈ 10⁹ full simulations ≈ 10⁵ GPU-hours. Dead for the weights. **Retained for a small subspace**: the ~470 neuromodulatory gains and a rank-16 per-neuropil weight modulation (~10⁴ params) are tuned by CMA-ES as an outer loop — that is where ES is actually the right tool. |

### Credit assignment

The key design move is **to never require long-horizon credit assignment at all**
in Stages 0–3. One training example = one position → one move. The episode is
48 ms. Game-level credit is delivered by the *target* (Stockfish's evaluation),
not by a return propagated over 80 moves. Only Stage 4 does RL, and it uses
1-step TD against a bootstrapped value head from a 32-simulation search — still no
long horizon.

Within the 48 ms trial, the risks are standard RNN pathologies:

- **Vanishing:** mitigated by ALIF adaptation spanning 20 ms – 2 s, and monitored
  directly (§7 KC-2) as `‖∂L/∂h₀‖ / ‖∂L/∂h_T‖`, required to stay in [10⁻³, 10³].
- **Exploding:** global grad-norm clip at 1.0; spectral radius of the linearized
  Jacobian held at 0.95–1.05 by the Stage-0 homeostatic objective.
- **Dead gradient from silence:** a firing-rate regularizer
  `λ_r · Σ_i (r_i − r_target)²` with r_target = 5 Hz, λ_r = 1e-3, which is also
  what keeps the simulation biologically plausible.

### Rate→spike annealing

Train the Tier-C rate surrogate first, then anneal to spikes over 20k steps
(quantization-clip-floor style, Bu et al. 2022; Rueckauer et al. 2017). This
converts the hardest optimization problem (spiking, from scratch, on a fixed
sparse graph) into two easy ones. *Assumption*: the rate solution transfers with
< 15% loss in top-1 agreement. Measured at M2; if it doesn't transfer, the anneal
schedule is the first thing to lengthen.

---

## 5. Training curriculum

**Throughput model** (4090, bf16, per-synapse weights, B=512, T=96, grad
checkpoint every 12 steps): **≈1,900 positions/s → 6.9M positions/GPU-hour**;
VRAM ≈ 6 GB at B=512, ≈10.5 GB at B=1024. Forward-only is ~3× faster.
H100 SXM ≈ 3.3× the 4090 on this workload (HBM3 bandwidth-bound).

| Stage | What | Data | Batch | Stop criterion | GPU-h (4090-eq) |
|---|---|---|---|---|---|
| **0** | Substrate calibration — no chess | 10M random board patterns | 512 | ≥95% of neurons in 3–8 Hz; silent <15%; saturated (>50 Hz) <5%; Jacobian spectral radius 0.95–1.05 | 20 |
| **1** | Rate-surrogate distillation (Tier C) | 200M positions, Lichess + Leela T80; targets = SF17 @ depth 12, MultiPV 20 | 1024 | held-out top-1 agreement with SF-d12 ≥ 35% | 40 |
| **1b** | ANN→SNN anneal | 40M positions | 512 | spiking top-1 ≥ 0.85 × rate top-1 | 25 |
| **2** | Spiking distillation, full BPTT | same 200M, 2 epochs | 512 | top-1 ≥ 33%; value MSE ≤ 0.09; unmasked legal ≥ 80% | 120 |
| **3** | Tactical puzzles | Lichess puzzle DB (~5M), curriculum 400→2400 rating | 512 | ≥55% on the 1500–1800 band; ≥25% on 2000–2200 | 70 |
| **4** | Self-play RL, 32-sim MCTS targets | 200k self-play games × 15 iterations | 512 | +100 Elo over Stage-3 ckpt over 2,000 games, **or** 3 consecutive iterations gaining <15 Elo | 400 |
| **5** | Endgame patch | Syzygy 5-piece (1 GB), 50M DTZ-labelled positions | 1024 | ≥90% on KRK/KQK/KPK; ≥70% KRP vs KR | 50 |
| — | Eval gauntlets (12 × full) | — | — | — | 25 |
| | | | | **Subtotal** | **750** |
| | | | | **× 1.4 contingency** | **1,050** |

**= 1,050 4090-hours ≈ 320 H100-hours ≈ $640–$1,280 at $2–4/GPU-hour.**

Local 4090 at 20 h/day covers 1,050 h in 53 days of pure compute across a
38-week schedule. **Cloud burst is needed only to compress Stage 4** (self-play
generation parallelizes trivially; 8× H100 for ~2 days). Everything else runs on
the workstation.

### Data sources

- **Lichess open database** — monthly PGN dumps, ~100M games/month. Use 2023–2025
  for training; **hold out everything after 2026-01-01** as a temporal test set.
- **Lichess elite database** (2400+ players), ~9M games, for the Stage-4 opening
  distribution.
- **Leela T80/T82 training data** — positions with policy + WDL targets already
  attached; saves re-running Stockfish on 200M positions.
- **Lichess puzzle database** — ~5M puzzles, themed and rated.
- **Syzygy tablebases**, 5-piece (1 GB). 6-piece (150 GB) is optional; 7-piece
  (18.4 TB) is out of scope.
- **Stockfish 17** locally, depth 12 MultiPV 20, for any position not covered
  above (~3,000 CPU-hours; run on cheap spot CPU, not the GPU).

**Stage 4 is the one to cut if the schedule slips.** Its expected contribution is
+80 to +150 Elo for 38% of the compute budget. Stages 1–3 + 5 deliver ~85% of the
final strength.

---

## 6. Evaluation protocol

### 6.1 Methodology

Elo by **gauntlet against absolutely-rated opponents**, never self-play-relative.
Ratings computed with `ordo` (or BayesElo) anchored to the CCRL scale. Match play
driven by `cutechess-cli`; ChessFly exposes a UCI wrapper.

### 6.2 Opponent pool

| Opponent | Anchor | Purpose |
|---|---|---|
| Stockfish 17 `UCI_LimitStrength`, `UCI_Elo` ∈ {1320, 1500, 1700, 1900, 2100} | SF's own calibration | Primary ladder |
| Maia-1100 / 1500 / 1900 | Human-move-matched | Human-relative calibration |
| Fruit 2.1, Stash 25 | CCRL-rated | Independent absolute anchor |
| **Lichess BOT account** | Real human opponents | The only opponent that isn't a model of a human |
| Stockfish 17, unlimited | 3600 | The headline result, run once, at 20,000 games, to report the true expected score |

### 6.3 Statistics

- **2,000 games per opponent** → ±25 Elo at 95% CI at typical draw rates.
  Full gauntlet = 8 opponents × 2,000 = **16,000 games**.
- **SPRT** (bounds [0, 10] Elo, α = β = 0.05) for checkpoint-vs-checkpoint A/B
  during development — the Fishtest methodology. Typical stop: 15k–40k games.
- **Openings:** 8-ply balanced book (UHO_Lichess_4852_v1.epd), every opening
  played from **both colours as a reverse pair**, so opening bias cancels exactly.
- **Time control:** ChessFly's cost is fixed (one 96-step forward pass, ~0.5 ms on
  the 4090, plus optional search). So the primary control is **fixed nodes** for
  the opponent (1k / 10k / 100k nodes) for reproducibility, with a secondary
  **60+0.6 wall-clock blitz** set for realism. Report both.

### 6.4 Memorization vs. generalization

Four independent probes. Memorization is the most likely way this project
produces a number that is real and meaningless.

1. **Temporal holdout.** Split by *game* and by *date*: all evaluation games
   played after 2026-01-01, none of which can be in training. Report the
   train/test top-1 gap; a gap > 8 points is a memorization flag.
2. **Position-novelty stratification.** Zobrist-hash every eval position against
   the training set. Report Elo and top-1 separately for *seen* vs *unseen*
   positions, and for plies < 12 vs ≥ 12. A > 150-Elo gap between seen and unseen
   means the opening book is doing the work.
3. **Chess960 transfer.** Evaluate on shuffled back ranks. A generalizer loses
   150–250 Elo (it genuinely hasn't seen these structures); a memorizer collapses.
   **Threshold: a drop > 400 Elo means the model memorized opening theory rather
   than learning chess.**
4. **Bilateral mirror test.** Evaluate on left-right mirrored positions (castling
   disabled to keep them legal). The connectome is bilaterally symmetric and both
   optic lobes receive the same board, so a model that learned board *structure*
   should be near-invariant. **Threshold: > 100-Elo asymmetry means the model
   learned lateralized lookup, not spatial reasoning.** This test is only
   available because the substrate is a fly brain — it's free, and it's the
   sharpest probe in the set.

---

## 7. Failure modes and kill criteria

Each is a measured number with a date. Hitting one means stop, not "try harder."

**KC-1 — Capacity wall (checked at M1, week 6).**
Stage-1 *rate surrogate* (Tier C, no spikes, easiest possible version) plateaus at
held-out top-1 agreement **< 30%** with training loss still **> 1.8 nats** after 3
epochs. Since Tier C is a plain sparse RNN, failure here indicts the *topology*,
not the spiking. → The connectome cannot represent a chess policy at all.
**Kill, or relax the topology constraint and report that as the finding.**

**KC-2 — Gradient pathology (checked at M2, week 10).**
Either: (a) `‖∂L/∂h₀‖ / ‖∂L/∂h_T‖` outside [10⁻³, 10³], or (b) per-parameter
gradient SNR (|mean|/std over 100 batches) **< 0.02 for more than 60% of weights**.
→ BPTT is not reaching the recurrent core; the deep neuropils are being trained by
noise. **Kill BPTT, fall back to e-prop for one 100-GPU-hour attempt, then kill.**

**KC-3 — Information bottleneck (checked at M2, week 10).**
Train a linear probe to reconstruct piece placement from the narrowest population
on the visual→central pathway. If probe accuracy **< 85%** (≈ < 40 bits of the
~160-bit board recovered) at any point between medulla and the central complex, the
board physically cannot reach the decision-making regions. → **Kill.** This is the
most likely single point of failure, because the fly's visual pathway is optimized
for wide-field motion, not for high-resolution static pattern.

**KC-4 — Elo stall (checked at M6, week 38).**
**< 1,200 Elo** (CCRL-anchored, 2,000 games, ±25) with all stages complete. →
The realistic ceiling of §2.4 was optimistic by more than the stated floor.
**Ship as a demo, retract the Elo claim, publish the negative capacity result.**

**KC-5 — Motor decode degeneracy (checked at M5, week 30).**
Unmasked legal-move rate **< 40%**. → 1,300 DNs cannot carry a factored move
distribution; the masking relaxation is doing all the work and the "fly plays
chess" claim is not supportable. **Relax to an external legality head and say so
explicitly in every subsequent claim.**

**KC-6 — Connectome-irrelevance control (checked at M4, week 22).**
The adversarial control: train an identical pipeline on a **degree-preserving
rewired** connectome (configuration model, same in/out degree sequence, shuffled
targets). If the rewired network reaches **within 50 Elo** of the real one, the
biology contributes nothing and this is a paper about sparse RNNs. → Not a kill,
but it is the result, and it must be reported first, not buried.

---

## 8. Milestones

Dated from **2026-09-12**. Each has one measurable pass/fail.

| ID | Date | Week | Deliverable | Pass | Fail → |
|---|---|---|---|---|---|
| **M0** | 2026-09-26 | 2 | Substrate runs | 139,255 neurons + 54.5M synapses loaded; 96-step forward at B=512 in <300 ms on the 4090; Brian2 cross-check <2% spike divergence; blackjack-checkpoint rank report filed | Fix kernel |
| **M1** | 2026-10-24 | 6 | Rate surrogate (Tier C) | Held-out SF-d12 top-1 **≥ 35%** | **KC-1** |
| **M2** | 2026-11-21 | 10 | Spiking net after anneal | top-1 **≥ 30%**; unmasked legal **≥ 60%**; gradient ratio in band; probe **≥ 85%** | **KC-2 / KC-3** |
| **M3** | 2026-12-19 | 14 | First rated play | **≥ 1,200 Elo** vs SF@1320 + Maia-1100, 2,000 games, ±25 | Extend Stage 2 |
| **M4** | 2027-02-13 | 22 | Puzzles + rewire control | Unmasked legal **≥ 95%**; **≥ 1,500 Elo**; rewired control ≥50 Elo weaker | **KC-6** |
| **M5** | 2027-04-10 | 30 | Self-play complete | **≥ 1,700 Elo**; Chess960 drop **≤ 350**; mirror asymmetry **≤ 100** | **KC-5** |
| **M6** | 2027-06-05 | 38 | Final | **≥ 1,800 Elo** target / **≥ 1,500** floor; full 16,000-game gauntlet; 20,000-game SF-max run reporting the true expected score; capacity curve published | **KC-4** |
| **M7** | 2027-07-03 | 42 | *Optional, clearly labelled* | ChessFly as leaf eval inside a 6-ply alpha-beta: **2,200–2,500 Elo**. **This is not the fly playing chess** and is reported under a separate name. | — |

### The one-line summary for the funder

We will not beat Stockfish. We will build the first fly brain that plays real
chess at around club strength (~1,750 Elo), and in doing so produce the first
measured capacity↔competence curve for a biologically fixed connectome — plus a
rewiring control that says whether the biology mattered at all.

---

### Assumptions register

Items marked as assumptions above, collected for auditing:

1. 2 bits/parameter capacity law transfers from language models to chess policies.
2. The three-point log-linear Elo↔params fit is valid in the 1M–200M range.
3. All four Elo penalties in §2.4 (−300/−225/−150/−75) are engineering estimates
   with no direct literature support.
4. Rate→spike annealing transfers with <15% top-1 loss.
5. The synchronous onset burst (§3.1) materially speeds convergence.
6. Splitting board state across sensory modalities beats a flat input map.
7. ALIF adaptation counts as "tuning a time constant" rather than adding state.
8. ~1,300 DNs is the correct FlyWire count; ~800 medulla columns per eye;
   neuropil neuron counts throughout are approximate.
