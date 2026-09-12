# Caissa — plan and progress

AlphaZero-style self-play reinforcement learning for board games, playable in the browser.

This document is the source of truth for where the project is and why it is built the way
it is. It is written to be picked up cold, by a later session or a different agent.

---

## Goals, in priority order

1. **Understand reinforcement learning.** This is the primary goal. The project is a
   vehicle for learning the concepts, so implementations favour clarity over cleverness,
   and every phase is expected to come with an explanation of the RL ideas involved.
2. **A multi-game AlphaZero framework, playable in the browser.** The algorithm must stay
   game-agnostic. Games are added to prove the abstraction holds, not as an afterthought.
3. **An engine strong enough to be a challenge.** The owner plays at roughly 1600–2000,
   so chess is targeted at ~1800–1900 — which means the supervised bootstrap alone will
   not be enough and self-play RL is required, not optional.

## Constraints

**Hosting — GitHub Pages, static only.** No server, so all inference runs on the visitor's
device. No control over HTTP headers, which means no `SharedArrayBuffer` and therefore no
multi-threaded WASM unless a `coi-serviceworker` shim is added later. Hard limit of 100 MB
per file, 1 GB per site, 100 GB/month bandwidth (soft).

**Training — one machine.** Apple M4 Max, 10 performance cores, 32-core GPU, 36 GB unified
memory. No cluster, no distributed self-play.

**Time — weekends.** Phases are sized to be individually completable and independently
useful.

---

## Measured facts

Benchmarked on the M4 Max with a 6×64 residual network at batch 512. **Do not re-derive
these**; re-measure only if the hardware or architecture changes.

| Config | Params | Inference | Training |
|---|---|---|---|
| PyTorch / MPS, conv policy head | **0.63 M** | **138,126 pos/s** | **34,934 pos/s** |
| PyTorch / MPS, dense policy head | 10.16 M | 114,336 pos/s | 30,426 pos/s |
| MLX / GPU, dense policy head | 10.16 M | 82,685 pos/s | 14,195 pos/s |
| PyTorch / CPU, conv policy head | 0.63 M | 4,941 pos/s | 819 pos/s |

Three consequences:

- **PyTorch on MPS beats MLX** here — 1.4× on inference, 2.5× on training. This inverts the
  usual assumption about Apple Silicon, but it was measured on this machine with this
  architecture. PyTorch also has far more reference implementations to learn from, so there
  is no trade-off to make.
- **A convolutional policy head is 16× smaller than a dense one and slightly faster.** Emit
  the policy as feature planes over board squares rather than flattening into a big linear
  layer. For chess this is AlphaZero's 73-plane encoding (64 squares × 73 = 4672). At
  0.63 M parameters a network is ~630 KB quantised to int8, which removes every GitHub
  Pages size and bandwidth concern.
- **The GPU is not the bottleneck — MCTS is.** At 138k evals/s the GPU will sit ~95% idle
  while Python walks the search tree. Expected end-to-end self-play throughput:

  | MCTS implementation | End-to-end | Games/hr @100 sims | Per 40-hr weekend |
  |---|---|---|---|
  | Python, single process | ~5–8 k/s | ~2,500 | ~100 k games |
  | Python, N workers + batched eval server | ~30–50 k/s | ~15,000 | ~600 k games |
  | Rust MCTS | ~80–120 k/s | ~35,000 | ~1.4 M games |

For scale: AlphaZero's chess run used **44 M games at 800 sims/move**, roughly 4.7×10¹²
network evaluations. A weekend here buys well under 1% of that. This is the single most
important number in the project, and it is why chess is bootstrapped from human games
rather than trained from zero.

---

## Core conventions

These are load-bearing. Changing one after training has started means retraining from
scratch, so treat them as fixed unless there is a deliberate decision to revisit.

1. **Canonical perspective.** Every position is described from the point of view of the
   player about to move: their pieces `+1`, the opponent's `-1`. `Game.apply()` negates the
   board on the way out so the flip is automatic. The network therefore learns one function
   rather than one per side, and every self-play position trains both sides at once.

2. **Mover-relative values.** Every value answers *"how good is this for the player to
   move?"* — `+1` win, `-1` loss, `0` draw. `terminal_value()` will therefore normally
   return only `-1` or `0`, never `+1`: a player cannot be on move having already won.
   Confusing this with an absolute "good for player one" convention is the classic way to
   build an agent that trains hard toward losing.

3. **Action masks, not action lists.** `legal_actions()` returns a boolean array as wide as
   the network's policy output. Illegal actions are set to `-inf` before the softmax, so no
   probability mass leaks onto moves that do not exist.

4. **Symmetries are free training data.** `Game.symmetries()` returns equivalent
   (position, policy) pairs. The policy permutation must exactly match the board
   permutation — if it does not, augmentation pairs positions with the wrong move
   distributions and poisons training while everything still appears to run.

---

## Game ladder

Games are ordered so each adds exactly one new difficulty. This is a teaching sequence, not
an arbitrary list.

| Game | New concept it forces | Actions | Status |
|---|---|---|---|
| Connect 4 | Baseline. Solved, so strength can be checked against perfect play. | 7 | done |
| Reversi | **Pass moves** — no legal action does not mean the game is over. 8-fold symmetry. | 65 | |
| Gomoku | Large action space; policy targets become very sparse. | 81–225 | |
| Isola | **Compound actions** (move *and* remove a tile) — action encoding design. | large | |
| Dots & Boxes | **Accumulated score** rather than win/loss — what should the value head predict? | ~60 | |
| Chess | Everything at once, plus a supervised bootstrap. | 4672 | |

Isola is deliberate rehearsal for chess: a compound action space is exactly the problem
AlphaZero's 73-plane encoding solves. Dots & Boxes is the most conceptually interesting,
because a score-based outcome breaks the usual win/loss value target.

---

## Phases

### Phase 0 — Game contract and Connect 4 — **done**

- [x] `uv` project, Python 3.12, package layout under `src/caissa/`
- [x] `Game` protocol (`src/caissa/games/base.py`) — the only interface the algorithm sees
- [x] Connect 4 (`src/caissa/games/connect4.py`)
- [x] Test suite (`tests/test_connect4.py`), 19 tests
- [x] Mutation-tested: deliberately breaking the canonical flip fails 8 tests; breaking the
      policy permutation fails the symmetry test. The suite is known to be load-bearing.

### Phase 1 — Network and MCTS

- [ ] Policy + value residual network, game-configurable (input planes, board shape,
      action space). Convolutional policy head where the action space maps to board
      squares; dense head otherwise (Connect 4's 7 columns do not map to its 42 cells).
- [ ] MCTS with PUCT selection, network priors, value backup
- [ ] Dirichlet noise at the root
- [ ] Temperature-based action selection
- [ ] Tests: search with a perfect oracle network must find forced wins; visit
      distributions must concentrate on good moves as simulation count rises

### Phase 2 — Self-play and training

- [ ] Self-play game generation, storing (position, visit distribution, outcome)
- [ ] Replay buffer
- [ ] Training step: cross-entropy on policy, MSE on value
- [ ] Symmetry augmentation wired in
- [ ] Checkpointing

### Phase 3 — Evaluation and gating

- [ ] Arena: play two checkpoints against each other
- [ ] Elo tracking across generations
- [ ] Comparison against a perfect Connect 4 solver — the external yardstick
- [ ] Gate: only promote a new network if it beats the incumbent by a margin

### Phase 4 — Browser, Connect 4 playable

- [ ] ONNX export
- [ ] `onnxruntime-web` inference in a Web Worker
- [ ] TypeScript game rules, validated against Python-generated test vectors
- [ ] Game-agnostic board UI (Vite + TypeScript)
- [ ] Point `.github/workflows/deploy.yml` at the build output rather than the repo root
- [ ] Deployed and playable on desktop and mobile

### Phase 5 — Reversi

- [ ] Reversi rules, including pass handling
- [ ] 8-fold symmetry
- [ ] Train, and confirm the framework needed no algorithm changes

### Phase 6+ — More games, then chess

- [ ] Further games from the ladder
- [ ] Chess representation (`python-chess`, AZ input planes, 73-plane move encoding)
- [ ] Supervised bootstrap on Lichess games filtered to ~1800–1900
- [ ] Self-play RL starting from the bootstrapped network
- [ ] Rust MCTS (`shakmaty`) for ~10× throughput, shared between training and browser
- [ ] Ship two personalities: the human-like bootstrap and the RL-strengthened network

---

## Decision log

Decisions already argued through. Revisit deliberately, not by accident.

| Decision | Rationale |
|---|---|
| Write our own engine rather than wrapping Stockfish | The engine is the point; learning RL is the goal. |
| AlphaZero-style, but chess bootstrapped from human games rather than zero-start | Zero-start chess on one machine over a few weekends lands around 800–1400 Elo, with real risk of not converging at all. Bootstrapping gives a usable opponent immediately and makes the RL gain measurable against a baseline. |
| Validate on Connect 4 before spending chess compute | Connect 4 is solved, so correctness is checkable against perfect play in an afternoon rather than after weeks of chess training that silently goes nowhere. |
| PyTorch, not MLX | Measured faster on this machine (see above) *and* has far more reference implementations. |
| `python-chess` / `shakmaty` rather than hand-written movegen | Move generation is a tarpit and is not what this project is for. |
| Game rules implemented twice (Python + TypeScript) rather than once in Rust | These games are 50–100 lines each; a Rust toolchain now would tax the actual learning goal. Divergence is caught by shared test vectors generated from Python. Chess sidesteps the issue since both languages have mature libraries. |
| Browser ship before chess | Proves the deployment path early and keeps the project playable throughout. |

---

## Concepts covered

A running glossary, extended as each phase introduces new ideas. Intended as a refresher,
not a substitute for the AlphaZero paper.

### Phase 0

- **Canonical perspective** — presenting every position from the mover's point of view so
  the network learns a single function instead of one per side. Halves what must be
  learned and doubles the usable training signal.
- **Mover-relative value** — the convention that values are always from the perspective of
  the player to move. Sign errors here are the most common silent failure in AlphaZero
  implementations.
- **Action masking** — constraining the policy distribution to legal moves by setting
  illegal logits to `-inf` before the softmax, so network capacity is not spent learning
  what the rules already say.
- **Symmetry augmentation** — exploiting board symmetries to multiply training data. A
  large sample-efficiency win when data is expensive to generate, as self-play data is.

---

## Commands

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```
