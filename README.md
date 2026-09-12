# ChessFly

Engineering plan for training the FlyWire *Drosophila melanogaster* connectome
(139,255 neurons, ~54.5M synapses) to play chess — plus a playable 3D harness.

**▶ Play it: https://noeljarillo.github.io/chessfly/**

**Verdict up front: 3600 Elo is unreachable.** Short by ~151× in parameter
capacity, and by a second, independent gap (search) that capacity cannot close.
Realistic ceiling: **1,500–1,900 Elo**. Full derivation in [`PLAN.md`](PLAN.md) §2.

| | |
|---|---|
| Trainable params | 55.2M (one weight per synapse, connectome-constrained) |
| Required for 3600 | 8.2 × 10⁹ (log-linear fit over Maia / Leela anchors) |
| Substrate | LIF + ALIF, dt = 0.5 ms, T = 96 steps, custom JAX |
| Learning rule | Surrogate-gradient BPTT |
| Total compute | ~1,050 4090-hours ≈ 320 H100-hours ≈ $640–1,280 |
| Schedule | 38 weeks, M0 → M6, six kill criteria |

## `index.html`

Self-contained demo — no build step, open it directly.

- 3D board (Three.js), click to play against the fly
- Hand-written 0x88 move generator; **perft verified to depth 4** (20 / 400 / 8902 / 197281)
- Brain panel: 39-neuropil LIF surrogate, 96 timesteps, with spike raster,
  neuropil activity graph, and factored descending-neuron move readout
- Anatomical plate over the board: the same 39 neuropils drawn on a frontal
  schematic of the fly brain — optic-lobe stack, mushroom body, central complex,
  GNG — lit by live per-region rate and with spikes riding the tracts
- The fly is an animal, not a cursor: head with compound eyes, banded abdomen,
  beating wings and six jointed legs. It flies in from its perch, closes its
  legs around the piece, carries it, sets it down and leaves
- Deliberation playback is independent of `prefers-reduced-motion`: reduced
  motion steps the trial instead of sweeping it, and never skips it

The opponent ladder is the §8 milestone table: M3 ≈ 1200, M4 ≈ 1500, M6 ≈ 1800.
Stockfish-max is listed and disabled, annotated with its capacity shortfall.

**The demo is the harness, not a result.** Moves come from piece-square tables
plus α–β; the brain panel is untrained and its readout is clamped to the played
move. Both facts are stated on the page.

## Prior art

`philshiu/Drosophila_brain_model` (Shiu et al. 2024, *Nature*) — substrate baseline.
`nftechie/doomfly`, `nftechie/stonkfly` — genre. Both target policies under ~10 bits.
Chess at 3600 is not one of those.
