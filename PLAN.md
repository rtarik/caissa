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

### Phase 2a — Self-play and training — **done**

- [x] Self-play generation (`src/caissa/selfplay.py`), storing (position, visit
      distribution, final outcome) with mover-relative value targets
- [x] Replay buffer (`src/caissa/replay.py`) — a sliding window over recent play
- [x] Training step (`src/caissa/train.py`) — cross-entropy on the policy against search's
      full visit distribution, squared error on the value, L2 through the optimiser
- [x] Symmetry augmentation wired in
- [x] The loop and checkpointing (`src/caissa/learn.py`), optimiser state included
- [x] Training entry point (`scripts/train.py`) with validated tactical probes
- [x] Tests, 81 in total, mutation-verified against eight deliberate bugs

### Phase 2b — Parallel self-play — **done**

- [x] Measured where the time actually goes before choosing: self-play is **98% network,
      2% tree** (952 us per simulation, of which 24 are tree). Batching leaves across games
      could therefore only ever reach the tree rate of ~42,000 evals/s — a 40x cap — so
      multiprocessing whole games was taken instead. It lifts both halves and is simpler.
- [x] `torch.set_num_threads(1)` per worker, worth **2.6x on its own**. PyTorch's intra-op
      threading costs more than it saves on one 6x7 position, and unset, ten workers would
      each try to use ten threads on ten cores.
- [x] `ParallelSelfPlay` (`src/caissa/parallel.py`) — a reusable spawn pool, since starting
      a process means importing torch afresh
- [x] Guard against the fork bomb: starting a pool from inside a worker raises with an
      explanation instead of taking the machine down
- [x] Validated against the sequential path — N workers reproduce N sequential runs exactly,
      in order, seed for seed
- [x] Training moved to the GPU, with a CPU mirror for self-play
- [x] Tests, 99 in total, mutation-verified

**Results.** Benchmarked at 30 games, 50 simulations per move:

| | games/s | games/hour |
|---|---|---|
| sequential, 10 torch threads (the original) | 0.82 | 2,950 |
| sequential, 1 torch thread | 2.13 | 7,661 |
| **parallel, 10 workers** | **17.47** | **62,907** |

Peak is at exactly 10 workers — the performance-core count. 12 and 14 are *slower*: the
extra workers land on efficiency cores, and every iteration waits for its slowest worker.

**The bottleneck then moved.** Per iteration at 200 games, self-play fell from 72s to 11s
while training stayed at 72s — 87% of the time, and it had been 5%. Training is batched, so
the GPU is ~39x faster there (59,757 against 1,525 positions/s at batch 512; a 6x7
convolution has too little arithmetic per byte for a CPU, and the backward pass is six times
the forward). Moving it took the iteration from 82s to 14s, with self-play dominating again
at 11.2s against 2.8s — which is the balance worth keeping.

End to end: **27x faster per game than where Phase 2a left off.**

### Phase 2c — Batched leaf evaluation (optional)

Not needed yet, and the ceiling is known: batching removes only the network cost, leaving
~42,000 evals/s of tree walking per process. Revisit only if self-play becomes dominant
again at a scale where 10 processes are not enough.

### Phase 3 — Evaluation and gating — **done**

- [x] Arena (`src/caissa/arena.py`) — colour-reversed pairs from shared random openings,
      so a match measures the players and not the first-move advantage
- [x] Elo with a confidence interval, and a `significant` flag when the interval
      excludes zero
- [x] `games_needed` / `resolvable_elo` — the sample size a claim requires, and the
      smallest difference a sample can resolve
- [x] Champion-challenger gating, off by default (AlphaZero dropped it too)
- [x] Generational checkpoints (`--checkpoint-every`)
- [x] Perfect Connect 4 solver (`src/caissa/solver.py`) — bitboard alpha-beta with a
      transposition table
- [x] `accuracy()` against ground truth, excluding positions where no mistake is available
- [x] `scripts/evaluate.py` (two checkpoints) and `scripts/accuracy.py` (against perfect play)
- [x] Tactical probes retired
- [x] Tests, 146 in total, mutation-verified against fourteen deliberate bugs

**Solver cost.** Exponential in the empty squares, so Python can grade the endgame and not
the opening:

| ply | median solve |
|---|---|
| 22+ | < 1 ms |
| 18 | 14 ms |
| 16 | ~0 ms, 0.4 s worst case |
| 14 | 0.73 s, 8 s worst case |
| 12 | 35 s, 116 s worst case |

Ply 18–22 is the working window: fast, and ~60% of random positions there still have a
wrong answer available. Deeper positions are mostly already decided — at ply 24–30, nine
sampled positions in twelve had no mistake to make.

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

## Results

### Connect 4, first training run (14 iterations, 35 games each, 50 simulations)

352 k parameter network, ~16 minutes on the M4 Max, 490 games and 21 k augmented positions.

**It learns.** Measured head to head, alternating who moves first:

| Matchup | Result |
|---|---|
| Trained vs untrained, policy only (100 games) | **81%** |
| Trained vs uniform evaluator, policy only (100 games) | **83%** |
| Trained vs untrained, 50 simulations (30 games) | **75%** |

**But weakly, and the loss curve said nothing useful about it.** Training loss fell cleanly
from 2.02 to 0.99 across the run while the tactical probes did not improve at all — "win in
one" ended at 0.12, no better than the untrained 0.14. Measured on the network's own
self-play distribution:

| | |
|---|---|
| Value head, sign agreement on decisive positions | 72% (76% within 3 plies of the end) |
| Policy head, agrees with search's top move | 71% |
| Policy cross-entropy on fresh data | 1.41, against a target-entropy floor of 0.41 |

That last row is the most informative: the network is badly *underfitting its own training
targets*, which points at data volume rather than at a defect. 490 games is on the order of
half a percent of what an AlphaZero-style Connect 4 run normally needs.

**The probe was a bad yardstick and should not be trusted.** The probe positions hold 5–7
pieces where self-play positions average 11.8, and the nearest self-play position differs
in 5+ squares — they are off-distribution. They also score the *raw network*, when search
is what actually plays: search with the trained network gets all three probes right. Phase
3's arena replaces this.

Three things to fix, in order: Phase 2b for the data volume, Phase 3 for the measurement,
and only then any tuning. Nothing here suggests an implementation bug — the head-to-head
result rules that out, and every component is mutation-tested.

### Connect 4, second run (40 iterations, 250 games each, 50 simulations)

10,000 games — 20x the first run — in about 14 minutes, which Phase 2b made possible.

| | run 1 (490 games) | run 2 (10,000 games) |
|---|---|---|
| Trained vs untrained, policy only | 81% | **97%** |
| Trained vs uniform evaluator, policy only | 83% | **99%** |
| Trained vs untrained, 50 simulations | 75% | **100%** |
| Policy agrees with search's best move | 71% | **78%** |
| Policy cross-entropy, gap to floor | 1.41 − 0.41 = 1.00 | **0.79 − 0.24 = 0.55** |
| Value sign agreement (decisive positions) | 72% | 69% |
| Value MSE vs predicting zero | 0.80 / 0.83 | 0.75 / 0.73 |

**The diagnosis held.** The gap between the network and its own search targets halved, which
is what "needs more data" was predicted to fix. Playing strength moved much further than the
loss curve suggested it would — 100% against the untrained network with search.

**The value head is the remaining weakness.** It extracts real signal (69% beats chance) but
hedges its magnitudes, so its MSE barely improves on predicting zero everywhere. That is the
rational response to a noisy label, and the label genuinely is noisy: self-play applies
Dirichlet noise at every root and samples moves at temperature 1 for the first eight plies,
so the same position can lead to either outcome. Whether 69% is near the ceiling for this
data or a real shortfall is not answerable without comparing checkpoints — which is Phase 3.

**Gap found: no checkpoint history.** `scripts/train.py` overwrites a single file, so run 1
is gone and cannot be played against run 2. The arena needs generational checkpoints to
measure against, so Phase 3 must add them.

### Connect 4 against perfect play (run 2 checkpoint, 10,000 games)

237 random positions at ply 18–22 where a mistake was available.

| | accuracy |
|---|---|
| random legal move | 28.2% |
| untrained network, no search | 23.4% |
| **trained network, no search** | **59.7%** |
| search alone, uniform evaluator, 50 sims | 86.9% |
| trained network, 50 sims | 86.9% |
| search alone, uniform evaluator, 200 sims | 96.6% |
| trained network, 200 sims | 93.7% |

**The network has learned real chess-like knowledge**: 59.7% against 28.2% for random, with
no search at all, and the untrained network is *worse* than random.

**But its contribution on top of search is not detectable here.** Paired McNemar tests:
50 simulations with and without the network differ on 50 of 237 positions, 25 each way,
p = 1.00. At 200 simulations, 14 favour plain search and 7 the network, p = 0.19. What *is*
significant is more search: 50 against 200 simulations differs 1 to 17, p < 0.001.

**An earlier 124-position run of the same comparison read 83.9% against 78.2%** and looked
like a clear regression caused by the network. It was noise. The paired test on a larger
sample put the two at exactly equal. This happened in the phase built specifically to stop
such conclusions, which is the strongest possible argument for the phase existing.

**Caveat on coverage.** Ply 18–22 is the regime where raw search is strongest, because
terminal positions are within reach of even 50 simulations. A learned value function should
matter most in the opening — which is exactly what this solver cannot reach. The measurement
is honest about the endgame and silent about the rest.

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

### Phase 2

- **Monte Carlo returns, not bootstrapping** — the value target is the *actual final
  result* of the game, applied back to every position in it, rather than `reward + V(s')`.
  Bootstrapping is low-variance but biased, and inherits whatever the network currently
  gets wrong, so those errors can circulate and reinforce. The final outcome is unbiased
  but noisy, since one late blunder relabels every earlier position. AlphaZero takes the
  unbiased target and drowns the variance in volume, because search already supplies the
  lookahead that bootstrapping would otherwise provide.
- **The value target alternates** — the outcome is recorded from the perspective of
  whoever was to move in each position, so the label flips sign along the game. Same
  convention and same failure mode as the search backup.
- **The replay buffer solves two problems** — *correlation*, since successive positions in
  a game differ by one piece and a batch drawn from a contiguous run is one position
  repeated; and *non-stationarity*, which has no supervised-learning equivalent: the data
  is produced by the network being trained, so the distribution moves as the network
  moves. Window size is a real trade-off — too small is unstable and forgetful, too large
  trains on data from noticeably weaker versions of the network.
- **Soft policy targets** — the label is search's whole visit distribution, not its argmax.
  A one-hot label would discard everything search learned about the alternatives,
  including how close the decision was. Consequence to expect: cross-entropy against a
  soft target bottoms out at the *entropy of the target*, not at zero, so a policy loss
  settling near 1.0 for a seven-action game is at the floor, not stalled.
- **Duplicate positions make the value head learn expectations** — the same opening appears
  in many games with different results, so no function can fit them all and it converges
  to their mean. That is the desired behaviour: the value head should predict the expected
  outcome, not memorise individual games.
- **The train/eval coupling bug** — `train_step` leaves the network in training mode, and
  the evaluator holds a *reference* to that same object. Without re-asserting `eval()` on
  every call, the first self-play game after the first gradient step begins corrupting the
  batch-norm running statistics. Neither a test of self-play alone nor of training alone
  can catch it; it is created by their interaction.
- **Validate your yardstick before you trust it** — the first tactical probe written for
  this project had *no* correct answer, because an opponent three-in-a-row with both ends
  open is a double threat. A measurement that has not been checked is not a measurement,
  and in reinforcement learning the yardstick is often the only thing standing between you
  and a confident, wrong conclusion.
- **A falling loss is not evidence of learning** — observed directly in the first training
  run: loss fell from 2.02 to 0.99 while tactical ability did not move at all. The loss
  measures agreement with targets the system generated itself, so it can fall while the
  targets stay weak. Only an outside measurement — head to head against another opponent —
  answered the question, and it took three different diagnostics before the picture was
  clear.
- **Check the yardstick is on-distribution too** — the probes were not merely unvalidated
  in the earlier sense; they held 5–7 pieces where self-play positions average 11.8. A
  network can be genuinely competent on the positions it meets and hopeless on positions
  drawn from nowhere near them.
- **Underfitting its own targets is the diagnostic that pointed at data** — a policy
  cross-entropy of 1.41 against a 0.41 floor says the network cannot yet reproduce what its
  own search found. That is a volume problem, not a correctness problem, and it is what
  separates "needs more games" from "needs debugging".

### Phase 2b

- **Profile before optimising, even when the answer seems obvious** — the assumption here
  was that tree walking cost about half of self-play. It cost 2%. The measurement inverted
  the design decision, and turned up a free 2.6x nobody was looking for.
- **More parallelism is not more throughput** — 10 workers beat 12 and 14, because the
  extra processes run on efficiency cores and an iteration is only as fast as its slowest
  worker. Scaling stops at the count of *fast* cores, not logical ones.
- **Processes, not threads** — the GIL stops threads from running tree search concurrently.
  The price is that everything crossing the boundary is pickled, which is why workers get a
  state dict and rebuild the model rather than receiving a live one.
- **A fast path must be provably identical to the slow one** — an optimisation that
  silently changes the data shows up only as a training run that behaves worse for no
  visible reason. The test asserts N workers reproduce N sequential runs exactly. It failed
  first time because the *reference* was wrong: it used two random generators where the
  worker shares one.
- **The two halves want opposite hardware** — self-play evaluates one position at a time and
  belongs on CPU; training is batched and belongs on the GPU. Keeping the network on the GPU
  with a CPU mirror for play is what lets both have what they want.
- **Fixing the bottleneck moves it** — self-play was 95% of the time, then 13%. Any change
  large enough to be worth making invalidates the measurement that justified it, so measure
  again afterwards rather than assuming the shape held.

### Phase 3

- **A hierarchy of yardsticks, not one number** — training loss is fully circular; beating
  the previous generation is relative and can drift; beating a fixed opponent is honest but
  narrow; agreeing with perfect play is the only absolute. Each level costs more and each
  one can contradict the level below it.
- **Elo assumes transitivity, and self-play breaks it** — a network can beat its predecessor
  while losing to something older. A rising curve built only from consecutive matchups can
  describe an agent going in circles, which is why it needs an anchor.
- **Sample size is the measurement** — near even, one Elo point is worth 0.0014 of a point
  per game against a per-game standard deviation of 0.5, so precision costs quadratically:
  100 Elo takes ~50 games, 10 Elo takes ~4,600. AlphaGo Zero's promotion gate of 400 games
  at 55% is not a round number; 400 games resolves 34 Elo and 55% *is* 35 Elo. The threshold
  was chosen to match the sample.
- **A gate below what its sample resolves promotes on noise** — 40 games resolve ~111 Elo,
  so a 55% threshold on 40 games fires on results that mean nothing, and the agent wanders
  while every report looks healthy.
- **Colour-reversed pairs from a shared opening** — both players face the same position from
  both sides, so an opening that simply favours whoever starts cancels instead of adding
  variance. Essential in a game like Connect 4 that the first player wins outright.
- **A benchmark must only ask questions that can be answered wrongly** — in a lost position
  every move preserves the outcome, so scoring it is a free mark. Measured in enough
  hopeless positions, an agent scores well for doing nothing.
- **Test where the agent does not choose to go** — grading it on positions from its own
  games flatters it, because it never has to answer the questions it is bad at. A fixed
  random sample is the same exam for every generation.
- **I nearly published noise as a finding** — a 124-position comparison said the network
  made search 5.7 points worse. A paired test on 237 positions put them exactly level. The
  discipline this whole phase is about caught an error made while building it.

---

## Commands

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```
