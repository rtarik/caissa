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

**Single-position inference is the wrong shape entirely.** Measured with the default
Connect 4 network (352 k parameters):

| | CPU | MPS |
|---|---|---|
| Single position | **1,046/s** | 662/s |
| Batch of 512 | 14,199/s | **281,835/s** |

One position at a time is *faster on CPU* than on the GPU, because the round trip costs
more than the work. But batching on the GPU is 426x faster than single positions on the
GPU. Self-play must therefore run many games concurrently and evaluate their leaves in one
batch; evaluating position by position leaves a factor of several hundred unclaimed.

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

### Phase 1 — Network and MCTS — **done**

- [x] `Evaluator` protocol (`src/caissa/evaluator.py`) separating search from knowledge,
      plus `UniformEvaluator` for knowledge-free testing
- [x] MCTS with PUCT selection, priors, and sign-alternating value backup
      (`src/caissa/mcts.py`)
- [x] Dirichlet noise at the root
- [x] Temperature-based action selection
- [x] Tests (`tests/test_mcts.py`), 17 tests, mutation-verified against seven
      deliberate bugs including both value-sign inversions
- [x] Policy + value residual network (`src/caissa/network.py`), game-configurable and
      with no Connect 4 specifics. Dense policy head for now, since Connect 4's 7 columns
      do not map onto its 42 cells; a convolutional head slots in for chess, where
      4672 = 64 squares x 73 move types and the saving is ~16x in parameters.
- [x] `NetworkEvaluator` wrapping the network behind the `Evaluator` protocol, forcing
      `eval()` mode so search cannot corrupt the batch-norm running statistics
- [x] Tests (`tests/test_network.py`), 14 tests, mutation-verified against five
      deliberate bugs

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

### Phase 1

- **Policy improvement operator** — the central idea of AlphaZero. Search takes the
  network's mediocre priors and, by spending simulations, produces a *better* move
  distribution. Training then compresses that improvement back into the weights, and the
  loop repeats. Everything else is machinery around this one step.
- **PUCT** — the selection rule, `Q(a) + c_puct · P(a) · √N_parent / (1 + N(a))`. The first
  term exploits what search has learned, the second explores what the network recommends
  but the search has not yet examined. The `1 + N(a)` denominator makes the optimism decay
  as a move gets a fair hearing.
- **Visit counts as the training target** — the policy label is the distribution of
  *visits*, not the network's own priors and not the Q values. A move earns visits only by
  repeatedly winning the PUCT argument, so visits aggregate all the evidence, whereas a Q
  value can rest on a single lucky simulation.
- **Sign-alternating backup** — a simulation's result is pushed up the path with its sign
  flipped at every ply, because a result good for one player is bad for their opponent.
  This is the direct consequence of the mover-relative value convention, and inverting it
  produces an agent that trains smoothly toward losing.
- **Dirichlet noise at the root** — random perturbation of the root priors, forcing the
  agent to try moves it currently dislikes. Without it the loop is self-reinforcing: the
  network proposes, search explores only what was proposed, training makes the proposal
  stronger, and a dismissed move is never reconsidered. Applied at the root only, since the
  aim is to vary the games played rather than to corrupt evaluation inside a line.
- **Temperature** — `pi(a) ∝ N(a)^(1/T)`. At `T=1`, play proportionally to visits, keeping
  self-play varied so the network sees a wide spread of positions. At `T→0`, play greedily,
  which is how to compete rather than learn. Usually `T=1` for the opening moves, then 0.
- **Separating search from knowledge** — the `Evaluator` protocol means search can be
  tested with no network at all. This keeps "the search is broken" distinguishable from
  "the network is untrained", which is the single most valuable diagnosis to be able to
  make while building a reinforcement learner.
- **Shared trunk, two heads** — one representation serves both questions the agent asks of
  a position, because "who is threatening what" underlies both "which move is good" and
  "who is winning". Cheaper than learning it twice, and the two tasks regularise each
  other: a shortcut that helps the value head usually hurts the policy head, so it does
  not survive.
- **Residual connections** — each block learns a *correction* to its input rather than a
  replacement, so gradients reach the early layers and depth stops being a liability.
  Verified directly: with the convolutions zeroed, a block must be the identity.
- **Bounded value output** — `tanh` restricts the value head to `[-1, 1]`, exactly the
  range of the thing being predicted. A network that cannot express an impossible value
  does not need to spend capacity learning not to.
- **The batch-norm evaluation trap** — in training mode, batch-norm normalises using the
  current batch's statistics rather than the learned running averages, *and* updates those
  averages as a side effect. Since search calls the network hundreds of times per move, a
  missing `eval()` means playing the game silently corrupts the network. No error, no
  obviously wrong output, just a strength collapse later.
- **Loss choice determines whether gradient exists at all** — `logits.sum().backward()`
  gives exactly zero gradient at the trunk of a batch-normalised network, because batch
  norm makes its output invariant to shifts in its input. The real loss (cross-entropy on
  the policy, squared error on the value) carries signal where a naive one carries none.
- **Batching is not an optimisation, it is the architecture** — on this hardware, batched
  GPU evaluation is 426x faster than single-position GPU evaluation. Self-play has to run
  many games concurrently and evaluate their leaves together, which shapes how Phase 2 is
  written rather than being something to add afterwards.

---

## Commands

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```
