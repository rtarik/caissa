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
3. **An engine strong enough to be a challenge.** The owner plays at roughly 1600–2000 FIDE
   (over the board; Lichess ratings run a few hundred points higher at club level), so chess
   is targeted at ~1800–1900 FIDE. Imitating strong games plus search may get close; self-play
   RL is how it should get past the humans it copied, and is the point of the project anyway.

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

**Sustained load is safe, and did not throttle.** Across every training run macOS recorded
no thermal warning and no performance warning, and throughput per unit of work stayed flat:
+1.9% over the 30-iteration Reversi run, +9.9% over the 40-iteration Connect 4 run (deeper
search trees in longer games, not the chip backing off). A throttling machine gets slower
*per position*; these did not. Apple Silicon firmware reduces clocks well below any harmful
temperature, so full load is a supported operating mode — the real costs are fan noise, a
warm chassis, and other applications feeling sluggish while all performance cores are busy.
`--workers 8` leaves two cores free for roughly 20% less throughput.

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
   board whenever the turn passes, so the flip is automatic. The network therefore learns one
   function rather than one per side, and every self-play position trains both sides at once.

2. **Mover-relative values.** Every value answers *"how good is this for the player to
   move?"* — `+1` win, `-1` loss, `0` draw. Confusing this with an absolute "good for
   player one" convention is the classic way to build an agent that trains hard toward
   losing.

   Whether `+1` ever appears is a property of the game, not of the convention. Connect 4
   ends the instant someone wins, so the game always ends on the loser's turn and
   `terminal_value()` returns only `-1` or `0`. Reversi ends when *neither* side can move,
   and the player to move may hold more discs — it returns `+1` about a third of the time.
   Phase 5 found this by asserting the Connect 4 behaviour as if it were universal.

3. **Action masks, not action lists.** `legal_actions()` returns a boolean array as wide as
   the network's policy output. Illegal actions are set to `-inf` before the softmax, so no
   probability mass leaks onto moves that do not exist.

4. **Symmetries are free training data.** `Game.symmetries()` returns equivalent
   (position, policy) pairs. The policy permutation must exactly match the board
   permutation — if it does not, augmentation pairs positions with the wrong move
   distributions and poisons training while everything still appears to run.

5. **Seats are stated, never counted.** `Game.to_play()` says which player is to move. Search,
   self-play labels and the arena negate a value only when the seat changes between one
   position and the next — never merely because a move was made. For seven phases those were
   the same thing, and Reversi and Isolation were each designed to keep them the same. Dots &
   Boxes, where closing a box earns another move, is the game that separated them: see
   Phase 8.

---

## Game ladder

Games are ordered so each adds exactly one new difficulty. This is a teaching sequence, not
an arbitrary list.

| Game | New concept it forces | Actions | Status |
|---|---|---|---|
| Four in a Row | Baseline. Solved, so strength can be checked against perfect play. | 7 | done |
| Reversi | **Pass moves** — no legal action does not mean the game is over. 8-fold symmetry. | 65 | done |
| Gomoku | Large action space; policy targets become very sparse. | 81 | done |
| Isolation | **Compound actions** (move *and* remove a tile) — action encoding design. | 392 | done |
| Dots & Boxes | **Bonus moves** — closing a box earns another turn, so turns stop alternating. The result is a count. | 60 | done |
| Chess | Everything at once, plus a supervised bootstrap. | 4672 | *in progress* |

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

### Phase 4 — Browser, Connect 4 playable — **done**

- [x] ONNX export (`src/caissa/export.py`, `scripts/export.py`), verified against PyTorch
      on probabilities and best moves before it is allowed to ship
- [x] `onnxruntime-web` in a Web Worker (`web/src/engine/`), single-threaded
- [x] TypeScript rules (`web/src/games/connect4.ts`) validated against 250
      Python-generated vectors — legal moves, terminal values and encodings
- [x] TypeScript MCTS (`web/src/engine/mcts.ts`) validated against Python's **exact visit
      counts** on 40 searches
- [x] Board UI with engine evaluation and visit distribution (`web/src/main.ts`)
- [x] `.github/workflows/deploy.yml` builds `web/`, typechecks, runs the tests and
      publishes `web/dist`
- [x] Verified playing in the browser, desktop and mobile viewports
- [x] Tests: 154 Python, 21 TypeScript, mutation-verified

**Sizes.** 13.3 MB of WebAssembly (3.4 MB gzipped, which is what Pages serves) plus a
1.3 MB model — about 4.6 MB on a cold visit, so roughly 21,000 visits a month inside the
bandwidth allowance, and cached afterwards. onnxruntime-web ships four WASM builds totalling
80 MB; only the plain CPU one is needed, and Vite emits exactly that.

**Speed.** 200 simulations in ~160 ms in the browser. Fast enough that the "Careful" setting
at 600 simulations is still comfortable.

**The header constraint from the very first planning session finally bit.** Multi-threaded
WASM needs `SharedArrayBuffer`, which needs COOP and COEP response headers, which GitHub
Pages will not set. `numThreads` is therefore pinned to 1 explicitly rather than left to
fall back silently — the symptom of the silent version is only an engine that thinks more
slowly than expected.

### Phase 4b — Observed in play

The browser engine blocks threats correctly and reports honest evaluations, but its opening
is weak — it answered 1.d4 (centre) with a move to the edge column and then scored itself at
30%, which is both bad play and an accurate self-assessment.

That lines up exactly with the Phase 3 caveat. The solver could only grade positions from
ply 16 onward, where it scored 86.9% at 50 simulations. The opening is the part that was
never measured, and it is visibly the weakest part of its game.

- [ ] Opening book or deeper search at low ply, or simply more training
- [ ] A faster solver (Rust, or a precomputed opening table) would let the opening be
      measured rather than guessed at

### Phase 5 — Reversi — **done**

- [x] Reversi rules (`src/caissa/games/reversi.py`), passing as an explicit 65th action
- [x] The full dihedral symmetry — eight variants per position, with the pass action
      deliberately held out of the permutation
- [x] Trained, and **the framework needed no algorithm changes**
- [x] TypeScript port with cross-language vectors, and the search validated against
      Python's exact visit counts for Reversi as well as Connect 4
- [x] Both games playable in the browser, chosen from a menu
- [x] Tests: 182 Python, 30 TypeScript, mutation-verified against eight deliberate bugs

**The abstraction held.** The only edit to existing framework code was three lines adding
Reversi to the registry. Search, network, training, replay, parallel self-play, arena and
Elo all ran unchanged on a different board size, a different action space, a different
terminal rule and four times the symmetry.

**Passing is an explicit action, not an implicit skip.** A player with no legal move does
not lose and the game does not end — which breaks the assumption every simple board game
encourages. Making it the 65th action keeps the invariant everything else depends on:
`apply` always switches the player, so the canonical sign flip stays exactly one per move.
The implicit version would sometimes leave the same player to move.

**One termination rule, not two.** The game ends on two consecutive passes. A full board
reaches that the same way, via two forced passes, so "the board is full" and "nobody can
move" cannot disagree, because only the second exists.

### Phase 6 — Gomoku — **done**

- [x] Gomoku rules on 9x9 (`src/caissa/games/gomoku.py`), five in a row, free-style
- [x] Eight symmetries, every action a square so nothing is held out of the permutation
- [x] TypeScript port, rules and search both validated against Python
- [x] Tests, 203 Python and 40 TypeScript, mutation-verified against seven deliberate bugs
- [x] Trained and exported to the browser

Played on 9x9 rather than the traditional 15x15: identical rules, the same difficulty for
the framework, and a fraction of the compute. 225 actions with games running past 200 plies
would cost several times as much for no new lesson.

**What it adds is scale.** Connect 4 offers seven actions and Reversi about ten at a time;
Gomoku offers every empty intersection — 81 at the start, still 50 by mid-game. The policy
target becomes *sparse*, which is harder to learn than it sounds, since a network can drive
the loss a long way down by predicting "almost nothing, almost everywhere". And search gets
thinner: 200 simulations over 50 moves is four apiece against nearly thirty when there are
seven, so the same budget buys much less certainty. That is the real reason large action
spaces are hard.

### Phase 6b — The web app as a whole

- [x] A game menu covering the full ladder, with untrained games shown but disabled
- [x] Availability read from `models/index.json`, written by the export script from the
      files that actually exist — so the menu cannot offer a game whose network is missing
- [x] Two-column layout on desktop, stacked on mobile; the menu scrolls rather than wraps
- [x] Per-game board views: Connect 4 columns, Reversi squares with legal-move hints,
      Gomoku stones on an open board with the winning five highlighted
- [x] Visit counts drawn per game: bars where the actions are few, and a board-shaped
      heatmap where they are squares — eighty-one slivers say nothing, but the same
      numbers laid over the board show exactly where the search looked

### Phase 6c — Why the first Gomoku engine was weak — **fixed**

Diagnosed after the owner, an experienced Gomoku player, beat it easily and reported that it
did not block an open three. It does not, and the reason is not a coding bug.

**What the engine actually did**, on constructed threat positions:

| threat | engine's reply |
|---|---|
| opponent has an open **four** | blocks, at every budget |
| opponent has an open **three** | **misses, even at 3,000 simulations** |
| engine has an open four | wins immediately |

At 3,000 simulations it *visits both* blocking squares and plays elsewhere anyway, because
it values them identically to everything else: blocking scored Q = −0.85, a random move
Q = −0.83. The network had learned "opponent has three in a row → I am losing" and not
"…and this square fixes it".

That is a **self-play bootstrap failure**, not a defect. In the training games neither side
could punish or defend, so "open three → loss" was statistically true *regardless of the
reply*, and the value head learned exactly that. Search could not correct it either: with a
knowledge-free evaluator, 20,000 simulations return a value of exactly **0.00 for every
move**, because in an 81-wide tree no rollout ever reaches a terminal position.

Three causes, all fixed:

- [x] **The branching factor was too high for any search to work.** Play is now restricted to
      empty points within one square of an existing stone — 79 legal moves in the opening
      becomes 13, converging back to the full board by the midgame. This is deliberate
      domain knowledge and a departure from "zero"; it is how practical Gomoku engines have
      always worked, and without it the loop cannot start at this compute budget.
- [x] **The simulation budget was too small.** 50 → 250 per move.
- [x] **The Dirichlet alpha was not scaled to the action space.** It is now derived per
      position as `10 / legal moves` — the rule AlphaZero used to pick 0.3 for chess and
      0.03 for Go. A fixed 1.0 put only 22.9% of the noise on the top five actions against
      AlphaZero's 60–64%, smearing a quarter of the root prior across moves that were mostly
      bad.

Two performance faults surfaced while fixing the above, both in the new rules code:

- [x] `legal_actions` looped over every stone in Python, so the game got slower as it went
      on. Replaced with a vectorised dilation of the occupied mask: 22.7 µs → 13.7 µs.
- [x] `apply` validated a move by building the *whole* legal mask, so expanding a node with
      thirty children did that work thirty times. Replaced with a single-square check, with
      a test asserting the two agree on every square of 60 positions.

Together: **147 s per training iteration → 40 s.**

**Still open: the arena runs sequentially.** A 100-game evaluation at 250 simulations takes
about ten minutes in the main process while eight self-play workers sit idle — so on the
retraining run, evaluation cost roughly three times what training did. `play_match` should
use the same worker pool as self-play. Worth fixing before chess, where the per-move cost is
far higher again.

- [x] Parallelise `arena.play_match` across the self-play pool. `ParallelArena` extends
      `ParallelSelfPlay`, so one pool serves both jobs. Pairs are split rather than games,
      so every worker plays whole colour-reversed pairs and the variance reduction survives
      the split — verified identical to a sequential match, seed for seed. A 100-game
      evaluation at 400 simulations went from ~23 minutes to under three.
- [x] `--resume` on the training script, so an interrupted run continues from its
      checkpoint instead of starting over (the checkpoint existed; nothing could read it)

**Carry into Isola and chess.** Simulations per move must be sized against the branching
factor, not carried over from the previous game — 50 simulations over 73 legal moves visits
two of them, and the "improved" policy target is then a sharpened copy of the prior.

**At fixed wall-clock, trading games for simulations is probably the larger win.** 3,600
games at 50 simulations and 450 games at 400 are the same total compute, but only the second
produces training targets that search actually improved. The premise of the whole algorithm
is that search quality creates the learning signal.

### Phase 7 — Isola — **done**

- [x] Isola rules on 7x7 (`src/caissa/games/isola.py`) — step to an adjacent square, then
      destroy any square still standing; a player with no legal action loses
- [x] **Compound actions**, encoded as a product: `direction * 49 + square`, giving 392.
      Chess does the same thing with 73 move types x 64 squares for its 4672.
- [x] **Convolutional policy head** (`NetworkConfig.policy_head="conv"`) — one plane per
      action type, read off at each square. Isola drops from 966k parameters to 386k, and
      this is the head chess will need.
- [x] Symmetries that transform *both halves* of a compound action
- [x] TypeScript port with rules and search validated against Python
- [x] A two-phase board: step, then choose the demolition, with the origin shown as a ghost
- [x] Tests, 238 Python and 50 TypeScript, mutation-verified against seven deliberate bugs
- [x] Trained and exported — 30 iterations of 90 games at 350 simulations. Its results were
      never written down before the run's log was lost, so this phase has no Results entry;
      the iteration-30 checkpoint is what ships.

**Why a product and not two half-moves.** Splitting the turn into two plies — one for the
step, one for the demolition — would be simpler to encode and would break the invariant
everything else depends on: that `apply` hands the position to the *other* player, so the
canonical flip happens exactly once per move. The product keeps one turn as one action.
(Phase 8 later retired that invariant, because Dots & Boxes could not be bent to fit it. The
product encoding stays, but for a different reason: it keeps each turn a single decision for
the search.)

**The compound action shows up again in the symmetries**, and nowhere else so far. Under a
board rotation the destroyed square moves like any square *and* the direction rotates with
it. Transforming one and not the other still produces a legal action on a plausible board,
which is why it needed a commuting-diagram test over 500 position/action pairs rather than
an inspection.

**On the branching factor.** Isola opens with 235 legal actions — worse than Gomoku's 79 —
and at 400 simulations a knowledge-free search reaches no terminal position from the opening,
exactly as Gomoku did. But the board *shrinks by one square every turn*, so the game forces
itself toward tractable positions: by ply 32 there are 30 legal actions, search concentrates
96.7% of its visits on one move, and the root value reads +0.96. There is a built-in
curriculum, and the endgame signal can propagate backward. Gomoku had no such escape, which
is why it needed the neighbourhood restriction and Isola does not.

### Phase 8 — Dots & Boxes — **done**

- [x] Board confirmed before building: **5x5 boxes, 6x6 dots**, the standard. With 25 boxes a
      draw is impossible. (Sizes are quoted both ways; "5x5" meaning dots would have been a
      4x4-box game.)
- [x] **The framework learns about turns.** `Game.to_play()` added to the contract and to every
      game. Search nodes record whether the seat changed on the way in; selection, backup,
      self-play value labels and arena credit all flip only when it did.
- [x] Proved **bit-identical** for the four alternating games: their rules and search vectors
      regenerate exactly from the seat-aware code, visit counts and root values included
- [x] Rules on bitmasks (`src/caissa/games/dotsandboxes.py`): an expansion costs 0.04 ms
      against 0.55 ms for a network evaluation — cheaper than Gomoku's rules at a third of the
      branching
- [x] An 11x11 lattice encoding with constant orientation planes, and symmetries that restore
      those planes after a quarter-turn swaps horizontal lines for vertical ones
- [x] Search checked against **brute-force exact play** on endgames with bonus moves
- [x] TypeScript port, rules and search validated against Python, with search vectors drawn
      from endgames so they actually reach finished games
- [x] Lattice board, running score, and bonus-move status in the page
- [x] Fixed a latent page bug: a forced pass was detected as "the only legal move is the last
      action index", so in Four in a Row with only column 7 open the page played it for you and
      announced a pass. Games now declare `passAction`; only Reversi has one.
- [x] Tests: 271 Python and 68 TypeScript; mutations caught 15/15 in Python and 10/10 in
      TypeScript
- [x] Trained and exported — 40 iterations of 100 games at 300 simulations, interrupted after
      iteration 15 and resumed. The iteration-40 checkpoint shipped first (gen55 has since
      replaced it — Phase 8b), and the page was checked against it: handed a box on three
      sides, the engine takes it and says it moves again.
- [x] `scripts/evaluate.py --workers` plays a match across the pool, so a 400-game match is
      minutes rather than most of an hour

**Why the framework had to change.** Reversi and Isolation each kept turns alternating by
design — an explicit pass action, and a compound action fusing two half-moves — because
everything downstream assumed it. Neither trick works here: a player can close box after box
in one turn, so a "whole turn" action would be unboundedly long. Handled naively, each bonus
move is scored as the opponent's, and the agent learns that completing a box is *bad for it*.
Training would run perfectly while it learned to give boxes away.

**Sizing, measured rather than estimated.** 2,380 games an hour at 300 simulations on 8
workers, exactly 60 plies a game, flat across iterations. A 40 x 100-game run projects to 1.7 h
of self-play, 0.3 h of evaluation and under 0.1 h of training — about 2.1 h. The estimate made
before building was 2¼–2½ h.

### Phase 8b — Observed in play

Handed a box in the opening, the gen40 engine usually declined it: it took 31% at the page's
default strength and 36% at *Careful*. A declined box is still there for the person to take
back, so the cost is mostly that beginner blunders go unpunished. See the Results entries.

**Why the existing exploration could not reach these positions.** Root noise and the
temperature window both vary the agent's choice *among moves search already rates*. Handing
over a box is never among them, so no amount of either produces a gift. What has to change is
where games start. A random opening is the simplest way, and needs no knowledge of the game —
but it only helps if it reaches the positions in question: with 1–16 random plies, 26% of
openings leave a box on offer; with 1–20, 38%.

- [x] **Measured first:** `scripts/gifts.py` hands each network a box after 2–14 plies of
      sensible play and counts how often it takes it. gen40: 30% on the policy alone, 40.5%
      with 200 simulations. It reproduces the collapse from gen15 (55.5%) to gen20 (13%) that
      the one-off probe found.
- [x] **Random openings in self-play** (`--random-openings`, `--random-opening-plies`): a share
      of games starts from a position reached by random moves, and only the positions after
      them are recorded — nobody searched the random moves, and the result that follows would
      credit or blame the players for choices they never made. Game-agnostic: it reuses the
      arena's random opening.
- [x] `--min-buffer` holds training back until the replay buffer has refilled, so a resumed run
      does not spend its first iterations memorising. (Saving the buffer instead would add
      about 1.3 GB to every checkpoint at this size.)
- [x] Tests: 276 Python. Six deliberate bugs in the new code, all caught — one of them, a worker
      that silently drops the opening settings, only by the new parallel-equivalence case.
- [x] Fine-tuned from gen40 with half the games opening randomly (1–20 plies), 15 iterations.
      gen55 ships: +186 Elo over gen40 in 400 games, 90% of traps with search, and its policy
      alone takes 50% of handed-over boxes, up from 30% — though search no longer adds to that.
- [ ] A score head: the network also predicts the final margin, and search adds a small share
      of that prediction to the value, so that when the chances of winning are equal the agent
      prefers the bigger win. The fine-tuning showed the blind spot is an objective problem, not
      a data problem — games here are won by a median of 14 boxes, so an agent rewarded only
      for winning is right not to care about one. KataGo does this; Reversi's disc count would
      use it too. Deferred.
- [ ] Count traps in the agent's own games, to test whether rarity is why the policy never
      learned to decline
- [ ] A direct gen30 vs gen15 match, to settle whether the buffer reset slowed iterations 16–30

### Phase 9 — Chess — *in progress*

Chess is built and trained in stages. Every stage ends with something checked and, from 9.2
on, a network the owner can play in the browser before deciding on the next stage. Downloads
and training runs are sized so that any stage can stop and resume; nothing assumes one long
session.

**The shape of the plan, and why.** Zero-start self-play is out of reach on one machine —
AlphaZero's chess run was ~4.7×10¹² network evaluations (*Measured facts*). So the network
first **imitates** strong human games, supervised, the way AlphaGo started; self-play then
improves on what it copied. The first network plays like the strong club players it learned
from; the second should be stronger than they are. A network pitched at the owner's own level
is a separate, optional personality (9.4b).

| Stage | Produces | Ends with |
|---|---|---|
| 9.0 Plan | this section | the owner's go-ahead |
| 9.1 Rules and encoding | `Chess` behind the `Game` protocol | tests; the cost of a search step, measured |
| 9.2 Browser | a chess page | the owner plays an *untrained* network |
| 9.3 Data, a month at a time | filtered human games as compact shards | statistics for each month |
| 9.4 Imitation, stage by stage | the imitation network | the owner plays it after every stage |
| 9.4b Owner's level, optional | a human-like network pitched at the owner | the owner plays it |
| 9.5 Yardsticks | puzzles and a Stockfish ladder | a rating on an outside scale |
| 9.6 Self-play, stage by stage | the stronger network | the owner plays it after every stage |
| 9.7 Ship | both personalities on the site | |

9.1–9.3 are the setup; 9.4 and 9.6 are the training.

**9.1 — Rules and encoding (Python)** — done

- [x] `src/caissa/games/chess.py` on python-chess 1.11. Check, mate, stalemate, castling, en
      passant, promotion, insufficient material, the fifty-move rule and repetition are all
      delegated, not rewritten (decision log). python-chess is GPL-3.0: fine here; the site
      uses chess.js, so nothing GPL reaches the browser. A state is a board without its move
      stack, plus the position keys since the last irreversible move — all that repetition
      needs, and never more than a hundred long.
- [x] **Canonical perspective**: for Black the board is mirrored (ranks reversed, colours
      swapped), so the network always sees the mover playing up the board, and Black's
      castling is the same action as White's. Tested on thousands of positions and their
      colour-mirrored twins: identical planes, identical legal actions.
- [x] **Input**: 19 planes — 12 for the pieces (the mover's first), 4 castling rights, en
      passant (marked only when the capture is actually possible), the fifty-move counter and
      a repeated-position flag. No history planes yet; they are the first thing to try if
      move prediction stalls.
- [x] **Actions**: 73 move types × 64 origin squares = 4,672, type-major as the conv head emits
      them. **1,858 stay on the board** — 1,456 queen-like, 336 knight, 66 underpromotions —
      the count Leela Chess Zero's policy map also arrives at, which checks the table's
      geometry against an outside number.
- [x] **No symmetry augmentation**: `symmetries()` is the identity alone
- [x] Draws applied automatically, as engines do: threefold repetition, the fifty-move rule at
      the hundredth half-move (a mate delivered on it still wins), stalemate, insufficient
      material
- [x] Tests: 26 for chess, plus chess joining the turn-alternation test. 18 deliberate bugs in
      the translation and in the search change below, all caught.
- [x] **Measured a search step, then built children lazily** (`scripts/searchcost.py`; table
      below). The search now builds a child's position on its first visit.
- [x] `scripts/testvectors.py` records each game's case count, so regenerating with its defaults
      reproduces the committed vectors — the no-change check below first "failed" on exactly
      that

**A search step, measured** — one CPU thread, 6×64 network, as a self-play worker runs:

| per simulation | chess, children built up front | chess, built on first visit | Dots & Boxes, on first visit |
|---|---|---|---|
| network | 623 µs | 644 µs | 824 µs |
| rules and tree | 311 µs (31%) | **73 µs (10%)** | 28 µs (3%) |
| simulations a second, per worker | 996 | **1,337** | 1,190 |

Each simulation *recorded* 31.4 chess children and *visited* 0.95, so building every child at
expansion was work thrown away 97 times in 100. Building on first visit is the same search in
the same order: the Python↔TypeScript search vectors of all five earlier games regenerate bit
for bit. The network is now nine tenths of a chess search step, so the next factor comes from
batching evaluations across games (Phase 2c), not from faster rules.

**9.2 — Browser** — done

- [x] chess.js 1.4.0 (BSD licence) for the rules in TypeScript, with the encoding and the move
      table ported and checked against Python on 300 positions and 40 searches, as for every
      other game. Chess vectors list legal moves by index — a 4,672-wide mask per position
      would have made the file ~6 MB — and half their searches start where a move mates at
      once, so 16 of 40 reach a decided game. The first set reached none.
- [x] **chess.js's private generator, contained.** Its public `moves({verbose: true})` builds
      every move's notation and the position before and after it: 819 µs a call, more than a
      network evaluation, and it made the web tests take 68 s. The private `_moves` underneath
      returns plain moves. It is reached in one adapter, chess.js is pinned to exactly 1.4.0,
      and a test checks it lists the same moves as the public API. The tests now take 3 s.
- [x] The TypeScript search builds children on first visit as well; every game's search vectors
      still match Python exactly
- [x] The board: our own SVG pieces (`web/src/ui/pieces.ts`) on slate squares. Click a piece,
      then its square; legal targets, the last move, check and the selection are highlighted;
      coordinates; a promotion picker; turned to face Black when the owner plays Black; the
      moves in standard notation. White and Black name the seats.
- [x] **Instinct** — the network alone, no search — joins the strength levels for every game:
      how Maia plays human-like chess, and the plainest view of what a network has learned
- [x] Playable now, against an untrained network labelled as such: 200 simulations take about
      0.3 s in the browser
- [x] Third-party notices for chess.js and ONNX Runtime Web, linked from the footer. Both licences
      require the notice to travel with the code, and minifying strips it from the bundle.
- [x] **The deploy caught what the laptop did not.** The chess legal-move test replayed every
      case from the opening, so its work grew with the square of a game's length: 2.2 s on the
      M4 Max, over the 5 s limit on the deploy runner. The harness now continues each case from
      the one before, replaying each game once (0.2 s), and move lookups compare chess.js's own
      squares rather than building names. The suite runs in 1.5 s.
- [x] A picker for which exported stage to play: built in 9.4, once there was a second network

**9.3 — Data, a month at a time**

Lichess publishes every rated game, month by month, as zstd-compressed PGN under CC0. Read
from the server:

| Month | Games | Download |
|---|---|---|
| 2016-01 | 4.8 M | 0.87 GB |
| 2017-01 | 10.7 M | 1.90 GB |
| 2018-01 | 17.9 M | 5.47 GB |
| 2019-01 | 33.9 M | 10.1 GB |
| 2022-01 | 102 M | 33.2 GB |
| 2026-08 | 91.9 M | 30.2 GB |

Games from 2017 on carry clock times in the move text, which is part of why later months grow
faster than their game counts.

- [x] `scripts/lichess.py <month>` downloads resumably, checks Lichess's published checksum and
      converts across 8 worker processes. Games are split out of the zstd stream and judged on
      their headers alone; only survivors have their moves parsed and replayed, through `Chess`
      itself. Filter: **both players 2200+** (decision log), blitz or slower, finished on the
      board — no time forfeits, whose result a clock decided — no bots, at least 20 plies.
- [x] Stored as what happened (`caissa.data.chess`): 44 bytes a position — the board as nibbles,
      side to move, rights, repetition count, the move played and its game — and encoded at
      training time by a vectorised twin of `Chess.encode` that the tests hold to it exactly
- [x] 1% held out for validation **by game**, chosen from the game's ID so the same games are
      held out on every run: positions from one game are near-duplicates, and letting them
      straddle the split would measure memory rather than skill
- [x] Tests: 25. Of 22 deliberate bugs, 21 were caught; the other is an equivalent mutant —
      relabelling a lookup table's columns in both places that use them changes nothing
- [x] **January 2020 converted** (below). The raw file is kept rather than deleted, so 9.4b's
      owner-level band can be filtered from it without a second 13.75 GB download.
- [x] `.gitignore`'s `data/` — meant for the downloads — also matched `src/caissa/data/` and hid
      the new package from git; a pattern without a leading slash matches at any depth. Now
      `/data/`.

**January 2020, as converted:**

| | |
|---|---|
| read | 46,800,709 games — Lichess's published count to the game — 98.8 GB of text in 225 s |
| kept | **530,632 games (1.1%)**, **39.6 M positions**, 74.6 plies a game |
| rejected | rating 22.2 M, bullet 17.5 M, not finished on the board 6.6 M, too short 5,733, bots 2,185 |
| results | White 48.4%, drawn 8.6%, Black 43.1% |
| players | median 2320, 90th percentile 2503, highest 2974 |
| held out | 5,210 games, 391 k positions |
| on disk | 1.78 GB of positions, 10 MB of games |
| checked | no illegal move in a 2,000-position spot check; 19,904 random stored positions encode exactly as `Chess.encode` does |

Two things the numbers say. **At 2200+, "blitz or slower" means blitz**: 98% of what is kept,
because strong Lichess players hardly play the slower controls. And **the first 2% of the month
predicted the whole**: a 2 GB sample, converted in 7 s before the download had finished,
projected 570 k games; the month gave 531 k. Maia trained on 12 M games per 100-point band; one
month here is about a twentieth of that, which is why months are added a stage at a time.

**9.4 — Imitation, stage by stage**

- [x] `scripts/imitate.py --months <list> --stage <n>`: behaviour cloning on stored months. The
      policy target is the move played — `imitation_loss`, tested to be exactly the AlphaZero
      loss with a one-hot target — and the value target the game's result for the mover. AdamW,
      batches of 1,024, a short warm-up then a cosine; about 28,000 positions a second, the batch
      encoder keeping the GPU fed.
- [x] Network: 6 residual blocks × 64 channels with the conv policy head, 563 k parameters —
      Maia's size
- [x] After each measurement, on 20,000 positions from held-out games: move-prediction accuracy
      (overall, opening, middlegame, endgame), held-out policy and value loss, and the training
      losses beside them
- [x] **Value overfitting, watched rather than guessed at**: every position of a game shares one
      result, so the value head could learn to recognise games instead of judging positions —
      AlphaGo trained value on one position per game for this reason. Across the first pass,
      held-out and training value loss agreed to three decimal places (0.713 each at the end).
      One pass shows each position once; the risk returns with repeated passes, and the same two
      numbers will show it.
- [x] A picker on the page for which stage to play: `export.py --name --label` gives each stage
      its own file, `index.json` lists them all, and the page offers the newest by default
- [x] **Stage 1: January 2020, one pass, 22 minutes** — 49.3% of held-out moves predicted
      (Results). For reference, Maia's 6×64 networks, with move history, predict about half.
- [ ] Further stages: add months while held-out accuracy still rises — it was still rising, if
      slowly, at the end of stage 1

**9.4b — A human-like network at the owner's level (optional)**

- [ ] The same pipeline with a different filter: a Lichess band matched to the owner's FIDE
      1600–2000 — probably somewhere around 2000–2200 on Lichess, to be checked against the
      owner's own games before any month is filtered for it. Played from its raw policy, the
      way Maia plays: fewer simulations make an engine weaker, not more human.

**9.5 — Yardsticks: a rating on an outside scale**

Every Elo so far has been relative — one version of the agent against another. Chess is the
first game with calibrated opponents to measure against.

- [ ] Lichess puzzles (304 MB, 6.1 M rated puzzles, CC0): solve rate by puzzle rating, for the
      raw policy and with search — a tactics rating
- [ ] A Stockfish ladder: matches against Stockfish capped with `UCI_LimitStrength`/`UCI_Elo` at
      several levels, driven through python-chess. Those levels are calibrated to computer
      rating lists, not Lichess, so read the result as a range. Needs `brew install stockfish`.
- [ ] Optional: Syzygy endgame tablebases (3–5 pieces, about 1 GB) for exact endgame grading,
      the chess counterpart of `endgames.py`

**9.6 — Self-play, stage by stage**

- [x] **The machinery (9.6a)**, all of it tested against deliberate breakage:
  - [x] The replay buffer saved with each checkpoint (`--save-buffer`) — stages restart the
        process by design, and Phase 8 measured what an empty buffer costs. Stored compactly:
        half-precision planes and only the actions search visited, 2.7 KB a sample against 42 KB
        written out in full, so a 120,000-position chess window is 325 MB rather than 5 GB.
  - [x] A maximum game length (`--max-plies`, drawn at the cap) and resignation
        (`--resign-below`, two moves running below the threshold), both labelling honestly: the
        cap is a draw for everyone, a resignation a loss for the side that resigned.
  - [x] Fine-tuning settings: `--learning-rate`, which now overrides the rate inside a loaded
        optimiser state — resuming from the tail of the imitation run's cosine schedule would
        otherwise have trained at 5e-5 — and `--temperature-moves`, 30 for chess as in AlphaZero.
  - [x] The held-out human exam lifted out of `scripts/imitate.py` into `caissa.data.heldout`,
        so every stage sits the same exam and it can be tested (it could not be, inside a script)
- [x] Throughput, measured before committing to a run: 100 games at 200 simulations on 10
      workers in 222 s — **1,620 games an hour**, 93 plies a game. Better than the 9.1 estimate.
      The lever is still batching evaluations across games (Phase 2c).
- [x] **Stage 1, first attempt: self-play made it worse, and the reason is worth the section
      below.** Five iterations of plain AlphaZero fine-tuning (150 games an iteration, 200
      simulations, 200 gradient steps, lr 2e-4) cost **-228 Elo** against the network it started
      from. See *Chess, self-play stage 1* in the results.
- [x] **Rehearsal** (`--human`, `--human-share`): a fixed set of human positions mixed into
      every batch, the fix the diagnosis pointed at. Half of each batch by default for chess.
- [x] Stage 1, second attempt, with rehearsal: -104 Elo at 200 simulations, **-3 at 50**. The
      collapse became a search-efficiency loss. Neither attempt produced a network worth
      shipping, and both produced a measurement worth keeping.
- [x] Tools this needed, all now in `scripts/`: `humanmoves.py` (held-out exam),
      `improvement.py` (does search still beat the raw policy), `resignations.py` (what
      resignation costs), and per-player `--simulations` / `--c-puct` in `evaluate.py`, which is
      what made the diagnosis possible at all
- [ ] **Open: what to try next.** The value head is the binding constraint — 65% sign accuracy
      on held-out positions, and search is worth 541 Elo when it is hedged enough not to be
      believed. Candidates, in the order the evidence supports:
      1. more imitation data (another month or two) so the value head is *accurate* before
         self-play makes it confident;
      2. policy-only self-play — freeze the value head, needs a `--value-weight` flag;
      3. more simulations a move (800 rather than 200), which is 4x the compute for a policy
         target that improves on the prior by 2.3 points instead of 1.0;
      4. accept Imitation 1 as the chess engine and spend the time on Phase 9.5's yardsticks
         instead, so any future stage is measured on an outside scale.
- [x] **How often resignation is wrong — measured** (`scripts/resignations.py`, 60 games at 200
      simulations from the imitation network, played out in full with the rule replayed over the
      record). At AlphaZero's -0.9 threshold: **83% of games resign, 100% of them genuinely
      lost, 47% of all plies saved** — self-play throughput roughly doubles for no wrong labels.
      -0.85 mislabels 2% of games and -0.80 mislabels 4%, so the threshold stays at -0.9.

      | threshold | resigned | was lost | was drawn | was won | plies saved |
      |---|---|---|---|---|---|
      | -0.99 | 68% | 100% | 0% | 0% | 30% |
      | -0.95 | 82% | 100% | 0% | 0% | 42% |
      | **-0.90** | **83%** | **100%** | **0%** | **0%** | **47%** |
      | -0.85 | 87% | 98% | 2% | 0% | 51% |
      | -0.80 | 90% | 96% | 2% | 2% | 54% |

- [x] **A bug the first version of this hid.** Resignation counted *plies* below the threshold,
      not a player's own moves. Values are the mover's own, so in a game where turns alternate
      the rule asked both players to despair on consecutive plies — which a consistent evaluator
      never does, so resignation would simply never have fired, silently. The test that passed
      used a stub handing the same dismal value to both sides, the one case where the two
      readings agree: it pinned the implementation instead of the intent. The rule is now one
      function, `selfplay.despairing`, shared by self-play and the audit so they cannot drift.
- [ ] Don't forget the humans: keep a share of human positions in the buffer, and play every
      stage against the imitation network as well as its predecessor
- [ ] After each stage: arena results, the human exam (`scripts/humanmoves.py`) for forgetting,
      the yardsticks, then the owner plays it

**9.7 — Ship**

- [ ] Personalities: the imitation network from its raw policy (a strong club player's style),
      the self-play network with search (the strongest), and the owner-level network if 9.4b
      was built
- [ ] Budget: a network of a few MB, and a move in a second or two on an ordinary laptop

### Phase 10 — The site, for people who just want to play — **done**

The engine work stops here (9.6's open list stands). What was left was a site that read like a
lab notebook: six text tabs, an analysis panel of simulation counts and visit heatmaps on
screen by default, and every game wearing Four in a Row's red-and-yellow-on-blue.

- [x] **A gallery, then a game.** `#/` lists the six games as cards with their own icon and a
      line on how to play; `#/play/<game>` is a focused board with a back link. The URL hash
      does the routing, so the back button, a bookmark and a reload all behave.
- [x] **The technical half, folded away.** The play screen shows the board, whose turn it is
      and the result. A quiet *Engine details* toggle brings back the evaluation bar, the visit
      heatmap and the per-game notes, and remembers the choice. Everything about *how* it plays
      moved to a *How it works* page, linked once, quietly, from the header.
- [x] **A palette per game**, driven entirely by the variables the board rules already read:
      red and yellow on blue stay Four in a Row's alone, Reversi gets black and white on green
      felt, Gomoku a wooden board with a drawn grid, Isolation teal and coral on slate, Dots &
      Boxes pencil on paper, chess its existing set.
- [x] **Icons** drawn for each game, the chess one borrowed from the board's own knight rather
      than drawn twice. Isolation's pieces are the board's pawn too: they were circles, which
      say "counter" when the game is about a piece that walks.
- [x] **Second pass, from the owner playing it.** The gallery's headline was larger than the
      cards it introduced and said what the *How it works* page already says, so it is now a
      small "Choose a game" and the copy moved. The footer sits at the bottom of the page and
      carries a footer's worth of text; the line about nothing leaving the device belongs on
      the page that explains the device. The selects are drawn rather than left to the platform
      (still native controls - a hand-built dropdown is worse on a phone and worse with a
      keyboard). Dots & Boxes turned out wrong on a dark page: cream paper was the brightest
      thing on screen by a wide margin, so it is chalk on slate in the dark and keeps the paper
      in the light. Isolation's teal-and-coral became brass and pewter, and the step hints are
      a ring rather than a tint - mixing a brass piece colour into the floor turned the tiles
      olive.
- [x] **Third pass.** Cards became tiles on anything wider than a phone: six chips in a row
      read as a file listing on a laptop, and the games are what the page is for (rows are
      kept on a phone, where tiles would mean scrolling past two games to see the rest).
      Dashes came out of the copy people read. The repository link was wrong, which is now a
      test rather than a thing to remember.
- [x] **The *How it works* page became the guide.** It walks the method in order: what the two
      halves are and how they disagree, the self-play loop step by step, how it can fail
      (Phase 9.6 in a paragraph), then a card per game with its board, its action count, how it
      learned and a link to its rules. A section on what makes chess different - 530,000
      Lichess games where both players were 2200+, because learning chess from nothing is a
      compute budget rather than a method - and on Gomoku's training restriction being a
      training aid rather than a rule. Every claim links to the file that implements it, and
      `web/test/site.test.ts` fails if one of those files is renamed or moved.
- [x] **Difficulty renamed and measured.** Beginner / Casual / Strong / Master, and **Master is
      now the default** - it takes under a second a move in every game (898 ms for chess at 600
      simulations, 990 ms for Dots & Boxes), so there was nothing to protect anyone from.
      `scripts/levels.py` measures what each rung is worth per game; the table ships with the
      site and is in the results below.

**Two bugs the site had, both found by playing it rather than by reading it.**

- [x] **Isolation could not destroy the square you came from** - the one square players reach
      for first. Neither rules implementation was wrong: chess had arrived with a global
      `.piece` CSS rule carrying `pointer-events: none`, Isola had been using `piece` as a cell
      class since long before, and the collision made that one button dead. Nothing in either
      language could see it; the button was there, correctly classed, and did nothing. The
      chess pieces are now `.chess-piece`, and `web/test/styles.test.ts` fails if any unscoped
      rule claims a class the boards use for cells, or turns off pointing at one.
- [x] **Gomoku refused most of the board** - the first stone had to be the centre point, and
      later stones had to touch an existing one. That was the training restriction
      (`NEIGHBOURHOOD`, added because self-play from a knowledge-free network cannot otherwise
      reach a finished game) leaking into the game people play. It is now a constructor flag,
      off by default and available for retraining. Safe because the trained network *learned*
      the restriction rather than relying on it: with the mask lifted it puts 0.0-0.5% of its
      prior on the points the mask used to hide, and picks the same move in 59 of 60 positions.
      The Gomoku vectors needed `SEARCH_WIN_IN_ONE` afterwards - free-style, a knowledge-free
      search reaches a result in none of forty searches, which is the restriction's own
      docstring coming true in the test file.


### Phase 11 — A play screen, and chess you can study — *in progress*

Agreed with the owner after a round of sketches. Two halves: a play screen built around the
game rather than around a settings form, and the chess features that make it worth playing
seriously - a rating to measure yourself against, games you can take elsewhere, positions you
can set up, openings that vary.

| Step | What | Scope |
|---|---|---|
| 11.1 | **Done.** The new play screen: larger type, player cards above and below the board, a new-game sheet with level cards and segmented choices in place of the settings form | all games |
| 11.2 | **Done.** Move list, stepping back and forth through the game, undo - with stale engine answers discarded | all games |
| 11.3 | **Done.** PGN export and a history of games kept in the browser | chess |
| 11.4 | **Done.** Board editor and FEN; the engine accepts a starting position | chess |
| 11.5 | Varied openings from a book of our own 2200+ games, and opening names | chess |
| 11.6 | **First measurement done** (results below). To do after 11.5: measure again with varied openings, and for *both* chess networks - the untrained one needs `--name chess` and will likely sit below Stockfish's 1320 floor, which the fit reports as a bound. A rough rating for each level against Stockfish at known strengths, shown on the level cards | chess |

Decisions taken in the discussion:

- **A sheet, not a setup page.** New game opens over the board; the board is never more than one
  click away.
- **Undo and browsing are for every game**, since every game is a list of moves. PGN, the editor
  and openings are chess only.
- **Stockfish is the only rating reference.** A rough estimate is the goal; Stockfish's
  `UCI_Elo` is calibrated to computer rating lists rather than to people, which is said on the
  page rather than hidden. (Maia would anchor to human ratings more directly, and was dropped as
  more setup than the precision is worth.)
- **Variety, not a practice mode.** The engine should stop playing the same line every game;
  choosing an opening to rehearse is not wanted. Opening names still come from Lichess's CC0
  openings list, for the move list and the PGN headers.
- **History lives in the browser**, with PGN export so a game can be analysed elsewhere. No
  import.

**11.1, as built.** The sheet opens when you enter a game, and the engine loads behind it, so
the wait for a network to arrive is spent choosing an opponent rather than staring at a board.
Its choices are native radio buttons drawn as cards and segments, so the keyboard and screen
readers work without any widget code. Everything behind the sheet is `inert` while it is open,
and the engine does not make the first move of a game the player has not started yet. Whose
turn it is moved from a sentence under the board onto the card of the side to move; the status
line keeps what needs a sentence - the result, a prompt, a pass. The board sizes itself to the
window's height as well as its width, because sized by width alone it pushed your own card below
the fold on a laptop. Two bugs found on the way, both by looking: the king on your card was black
when you played White (the piece colours were defined inside the board's scope only), and the
stylesheet guard from Phase 10 caught hidden radio inputs switched off with `pointer-events`.

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

### Reversi, first training run (30 iterations, 100 games each, 50 simulations)

499 k parameter network, ~12 minutes, 3,000 games and 1.5 M augmented positions (the 8-fold
symmetry doing its work: 100 games produce 50,000 training samples).

**The in-training evaluations said progress had stopped. They were wrong.**

| | reported | sample resolves |
|---|---|---|
| gen10 vs gen5, 40 games | 97.5%, +636 Elo | ~111 Elo |
| gen15 vs gen10, 40 games | 80%, +241 Elo | ~111 Elo |
| gen20 vs gen15, 40 games | 77.5%, +215 Elo | ~111 Elo |
| gen25 vs gen20, 40 games | 46.2%, −26 Elo, **not significant** | ~111 Elo |
| gen30 vs gen25, 40 games | 60%, +70 Elo, **not significant** | ~111 Elo |
| **gen30 vs gen20, 200 games** | **82%, +263 Elo [+207, +335]** | **~48 Elo** |

Two consecutive 40-game matches showed nothing, one of them slightly negative, and the
obvious reading was that training had plateaued after iteration 20. A 200-game match across
the same ten iterations says **+263 Elo, decisively**.

Nothing plateaued. A 40-game match cannot resolve better than ~111 Elo, and the per-step
gains had dropped to roughly 60–70 — real, and invisible to the instrument. Early gains were
large enough to clear the bar; later ones were not, which makes an underpowered evaluation
look exactly like a plateau at precisely the point where it stops being able to see.

The default evaluation sample is now 100 games rather than 40, and the help text says what
that resolves. `resolvable_elo` was written in Phase 3 specifically to prevent this mistake,
and it was still made — because the number was printed and not acted on.

### Gomoku, first training run (30 iterations, 120 games each, 50 simulations)

593 k parameter network on 9x9, ~11 minutes, 3,600 games. Evaluated every 5 iterations over
100 games — the sample size raised after Phase 5, and it earned its keep immediately.

| | result |
|---|---|
| gen5 vs gen0 | 76.0%, **+200 Elo** [+128, +293] |
| gen10 vs gen5 | 63.0%, **+92 Elo** [+24, +168] |
| gen15 vs gen10 | 70.0%, **+147 Elo** [+78, +230] |
| gen20 vs gen15 | 62.0%, **+85 Elo** [+17, +160] |
| gen25 vs gen20 | 60.0%, **+70 Elo** [+2, +144] |
| gen30 vs gen25 | 57.0%, +49 Elo [−19, +121], not significant |

Five consecutive significant gains and no plateau. **Four of those six results would have
been invisible at the old 40-game default**, which cannot resolve better than 111 Elo — the
same failure that made Reversi look finished when it was not.

**Games got shorter as it improved**: 35 plies at iteration 1, 16 by iteration 30. The
opposite of Connect 4 and Reversi, where stronger play lengthened games. A stronger Gomoku
player builds an unanswerable threat faster, so improvement shows up as games ending sooner.

**The sparse-target difficulty is visible in the loss.** Policy loss settled at 1.37 against
Connect 4's 0.57, on an action space twelve times larger. Value loss, by contrast, fell to
0.07 — far lower than either earlier game, because Gomoku positions are decisive and
readable once a threat is on the board.

### Gomoku, second run — after the fixes (40 iterations, 110 games, 250 simulations)

3,300 games in about 90 minutes, of which roughly two thirds was *evaluation* rather than
training — see the open item on the sequential arena.

| | first run (broken) | second run |
|---|---|---|
| gen10 vs gen5 | +92 Elo | **+315** |
| gen15 vs gen10 | +147 | **+512** |
| gen20 vs gen15 | +85 | +182 |
| gen25 vs gen20 | +70 | **+346** |
| gen30 vs gen25 | +49, not significant | **+308** |
| gen35 vs gen30 | — | **+147** |
| gen40 vs gen35 | — | **+78** |
| final value loss | 0.07 | **0.135** |

**All eight evaluations significant**, and still gaining at gen40.

**Two indicators that the fix is real, beyond the Elo.**

*Games got longer, not shorter.* 25 plies at iteration 1, dipping to 16 by iteration 10 as
the attacker improved first, then rising to **62 by iteration 40** as defence caught up. On
an 81-point board that means most of it fills. The first, broken run went monotonically the
other way — 35 plies down to 16 — because a stronger attacker was beating a defenceless
opponent faster. Length rising is what mutual competence looks like.

*Draws appeared.* The first run produced none at all. In the second, gen40's evaluation match
was **+39 =44 -17** — nearly half the games drawn, because both networks now block each other
to a full board.

**And the reported failure is gone.** On the position the owner described:

| | before | after |
|---|---|---|
| prior on the blocking squares | 0.09 | **0.789 + 0.209** |
| value | −0.85 ("lost") | −0.20 ("worse, but playable") |
| blocks at 50 / 3000 sims | ✗ ✗ | **✓ ✓** |

It also now values an opponent's *open four* at exactly −1.00 and blocks anyway — which is
correct play: an open four cannot be stopped, and the block is the best try.

### Dots & Boxes, first training run (40 iterations, 100 games each, 300 simulations)

4,000 games in 2.2 hours of compute, against the 2.1 sized in advance: 170 s an iteration, all
but 15 s of it self-play, and under three minutes per evaluation. The run was interrupted
after iteration 15 and resumed from the gen15 checkpoint.

**Arena**, each checkpoint against the one five iterations earlier, 100 games:

| | score | Elo, 95% interval |
|---|---|---|
| gen5 vs untrained | +82 -18 | **+263** [+186, +374] |
| gen10 vs gen5 | +81 -19 | **+252** [+175, +358] |
| gen15 vs gen10 | +77 -23 | **+210** [+137, +305] |
| gen20 vs gen15 | +58 -42 | +56 [-12, +129], not significant |
| gen25 vs gen20 | +54 -46 | +28 [-41, +99], not significant |
| gen30 vs gen25 | +58 -42 | +56 [-12, +129], not significant |
| gen35 vs gen30 | +58 -42 | +56 [-12, +129], not significant |
| gen40 vs gen35 | +64 -36 | **+100** [+32, +177] |

**Chained steps overstated the progress.** The five matches after gen15 add up to +296 Elo.
Played directly over 400 games, gen40 against gen15 is +291 -109 (72.8%): **+171** [+134,
+211]; gen40 against gen30 is +264 -136 (66.0%): **+115** [+80, +153], where the chain says
+156. The chain carries about ±150 Elo of compounded uncertainty, so the gaps may be pure
chance — and the fine-tuning run below erred the other way, chaining to +84 where a direct
match found +186. That is what the log's "assumes transitivity" is warning about. Even the fastest stretch, +115 over the last ten
iterations, is under 60 per five — below the ~69 Elo a 100-game match can resolve, which is
why four of those five matches looked flat. The run was still improving when it stopped.

**Against exact play.** `scripts/endgames.py` solves endgames 6–12 lines from the end by brute
force and scores how often the chosen move keeps a won position won. Every network sits the
same exam — 200 endgames and 50 traps, seed 0 — so rows compare directly, but 50 traps carry
roughly ±7 points of noise. Network entries are *policy alone / with 300 simulations*.

| | all endgames | traps |
|---|---|---|
| random play | 23.6% | 11.7% |
| greedy: take any box | 91.3% | 0.0% |
| gen3 (the sizing run) | 29.5% / 61.0% | 2.0% / 16.0% |
| gen5 | 82.0% / 94.5% | 4.0% / 40.0% |
| gen10 | 91.0% / 95.5% | 0.0% / 42.0% |
| gen15 | 92.0% / 97.0% | 0.0% / 54.0% |
| gen20 | 93.0% / 96.5% | 2.0% / 62.0% |
| gen25 | 92.0% / 98.0% | 2.0% / 64.0% |
| gen30 | 93.0% / 98.5% | 0.0% / 76.0% |
| gen35 | 94.5% / 99.5% | 0.0% / 84.0% |
| gen40 | 94.0% / 98.5% | 2.0% / 88.0% |

**The raw network stayed a greedy player; search learned to decline the box.** From gen10 on,
the policy's first choice scores 91–94.5% on all endgames against greedy's 91.3%, and grabs
the box in all but at most one of the 50 traps, right through gen40. The same network with 300
simulations declines it more and more often: 40% at gen5, 54% at gen15, 76% at gen30, 88% at
gen40. This is the failure this game was put on the ladder to show — greedy play beats weak
play, so self-play can settle into it — and it was search that got out of it.

**The value head learned double-dealing; the policy never did.** A one-off probe of the same 50
traps:

| | gen5 | gen10 | gen15 | gen20 | gen25 | gen30 | gen35 | gen40 |
|---|---|---|---|---|---|---|---|---|
| policy's prior on captures | 0.72 | 0.85 | 0.95 | 0.96 | 0.94 | 0.94 | 0.96 | 0.95 |
| policy's prior on winning moves | 0.08 | 0.05 | 0.03 | 0.03 | 0.06 | 0.06 | 0.04 | 0.05 |
| value head, one move deep, prefers a winning move | 16% | 18% | 30% | 40% | 40% | 70% | 54% | 78% |
| search at 300 simulations declines | 40% | 42% | 54% | 62% | 64% | 76% | 84% | 88% |

The policy grew *more* sure about capturing, never less, while the value head went from
seeing the trap one time in six to four times in five. Search needs only a few simulations to
try a move holding 5% of the prior; after that the value head decides.

The policy was not short of a correct target, either. At gen40, search at 300 simulations puts
80% of its visits on winning moves in these traps (gen15: 47%), and visit counts are exactly
what the policy trains toward. It still puts 5% of its prior there. The likeliest reason is
frequency: every ordinary capture in self-play teaches "take the box" at full strength, and a
position like these is rare enough to be outvoted. Counting traps in the agent's own games
would settle it; that has not been done.

**A checkpoint is not the whole training state.** It holds the network and the optimiser, not
the replay buffer. Iteration 15 drew its 1,000 x 256 training samples from 480,000 positions,
ten iterations of games; iteration 16 drew the same number from the 48,000 of a single
iteration, so each was seen about five times instead of half a time. Loss fell from 2.93 to
2.42 (value 0.63 to 0.50) in one iteration. That is memorising 100 games, not learning: the
loss is measured on the very samples being fitted. As the buffer refilled the loss climbed
back — 2.42, 2.36, 2.40, 2.46, 2.49 over iterations 16–20 — and it is full again by iteration
25. Did the reset cost strength? The endgame exam never dipped. The arena is ambiguous: gen40 is
+171 over gen15 but +115 over gen30, which would leave only about +56 for iterations 16–30 —
if Elo chained, which it does not reliably do. But the reset did leave a mark where neither
instrument looks. How often the network takes a box handed to it in the opening fell from 48%
at gen15 to 9% at gen20 — see below — and never fully came back. The likely mechanism is the
very thing a buffer is for: its older games, from weaker networks, are where boxes get handed
over, and a buffer holding only the newest games has none to learn from. (The resumed process
also restarts the log's cumulative Elo from zero; that total is not in the checkpoint either.)

**The gen40 engine often declines a box handed to it.** Take a few plies of sensible play,
then one blunder — a line that leaves a box on three sides. Taking that box is right: declining
only hands the same box to the opponent. Over 200 such positions:

| | gen5 | gen10 | gen15 | gen20 | gen25 | gen30 | gen35 | gen40 |
|---|---|---|---|---|---|---|---|---|
| policy alone takes the box | 26% | 42% | 48% | 9% | 14% | 27% | 22% | 24% |
| search at 200 simulations takes it | 32% | 26% | 31% | 9% | 17% | 33% | 25% | 31% |

At the page's *Careful* setting, 600 simulations, gen40 takes 36%; gifts in chaotic
random-play openings give the same picture. It is not uniform — in the position that first
showed it, a corner box after three plies, gen40 takes the box with 95% of its visits, where
gen25 valued taking it at −0.52 against +0.00 for declining. At gen10 and gen15 search took
*fewer* boxes than the policy alone, which only happens when the value head steers away from
the capture. A competent network almost never hands over a box early, so self-play never
corrects this. One plausible reading, not tested: in the endgames the agent does see, whoever
captures and must move again is often about to open a chain for the opponent, and the value
head carries that lesson into openings where it does not apply.

Value loss also *rose* over iterations 3–15, from 0.57 to 0.63, while every arena match showed
large gains — in self-play the loss is measured against a target that moves with the agent.

### Dots & Boxes, fine-tuning with random openings (gen40 → gen55)

Resumed from gen40 with half the self-play games opening with 1–20 random plies. The first
three iterations only refilled the buffer — random openings record fewer positions a game,
so it took three rather than the two planned — and iterations 44–55 trained.

| | gifts taken: policy alone | gifts taken: 200 sims | endgames / traps, 300 sims | arena, 100 games |
|---|---|---|---|---|
| gen40 | 30.0% | 40.5% | 98.5% / 88% | — |
| gen45 | 39.0% | 57.0% | 99.0% / 80% | +21 vs gen40 |
| gen50 | 45.5% | 51.5% | 99.5% / 86% | +63 vs gen45 |
| gen55 | 50.5% | 49.5% | 99.0% / **90%** | 0 vs gen50 |

At gen55 the policy alone also declines the box in 10% of traps, where every earlier
generation managed 0–4%.

Played directly over 400 games, **gen55 beats gen40 +298 -102 (74.5%): +186 Elo** [+149, +228].
The three 100-game steps added up to only +84 — the chain erred low this time, where on the
first run it erred high. Opening half the games randomly cost normal play nothing measurable;
whether it helped cannot be separated from twelve more iterations of training, which near the
end of the first run were worth about +115 per ten. **gen55 ships.**

**What the value head thinks one early box is worth**, over the same 200 gift positions: the
position after taking the box, against the same lines with that box the opponent's instead.

| | after taking | box given away | difference | taking valued higher |
|---|---|---|---|---|
| gen40 | +0.075 | +0.014 | +0.062 | 57% |
| gen45 | +0.164 | +0.023 | +0.142 | 62% |
| gen50 | +0.030 | −0.093 | +0.123 | 56% |
| gen55 | +0.043 | +0.118 | −0.075 | 48% |

**The value head is not misreading the board; it is indifferent, and nearly rightly so.** In
16 gen55 self-play games at 100 simulations the final margins were 1, 1, 3, 5, 7, 9, 13, 13,
15, 15, 15, 17, 17, 17, 17 and 21 boxes — median 14. Between players of this strength the game
is decided by who controls the long chains, usually by double digits, and an early box changes
the result only in the rare close game. A value target that records only who won therefore
prices one box at close to nothing, and search, choosing between moves its value head rates
equal, follows the prior. That is what the first table shows: the policy's habit of taking
boxes grew from 30% to 50%, and search stopped adding anything to it. Random openings supplied
the positions; they cannot supply a reason to care. The agent does what it was asked —
maximise the chance of winning — and a person reads the result as a blunder because a person
also counts the score.

### Chess ratings against Stockfish (Phase 11.6)

Each level against Stockfish 19 with `UCI_LimitStrength` at 1320, 1500, ... 2500: 24 games a
step, colours alternating, Stockfish on 0.1 s a move, 672 games in all. One rating per level
fitted to all of its results by maximum likelihood (`caissa/rating.py`), 95% likelihood-ratio
ranges.

| Level | Rating | 95% range | Shown as |
|---|---|---|---|
| Beginner (no search) | 1469 | 1386-1548 | about 1450 |
| Casual (40 simulations) | 1767 | 1692-1843 | about 1750 |
| Strong (200) | 2180 | 2104-2258 | about 2200 |
| Master (600) | 2288 | 2209-2369 | about 2300 |

**Checked against the obvious objection.** Stockfish's limiter was calibrated with more thinking
time than 0.1 s, and a rushed Stockfish might play below its label and flatter us. Master was
replayed against 2300 at five times the time: 52% against 58% before, a fitted 2314 against
2288. The same number, inside its range.

**The finding: self-play Elo stretches gaps.** Measured against each other (Phase 10's ladder),
Master beat Strong by 374 Elo. Measured against Stockfish, the two are about 110 apart. The same
network at two depths shares every misjudgement, so the deeper search knows exactly where its
shallower twin will go wrong and aims for it; an outside opponent does not make those tailored
mistakes. A rating is a statement about a pool of players, and a pool made of one family's
members inflates the distances inside it - one more reason every strength claim in this project
is made against something outside the training loop.

**11.2, as built.** A move panel for every game: chess in standard notation, the others by
column, square or the two dots a line joins, all in absolute board coordinates so a move keeps
its name whichever side you sat on. Clicking a move, the arrow keys or Home/End step the board
through the game without changing it; the board is locked and outlined while you look back, and
a line under it says so with a way back. Undo takes back your last move and the reply to it,
walking past forced passes (undoing a pass would only have the page pass again).

Two pieces of this were about time rather than layout. **Every engine request is numbered** and
only the answer to the latest is played: undo while the engine thinks, and its answer for the
abandoned position arrives and is dropped - checked by undoing mid-search and watching the
game stay put. The same numbering closed an older hole: New game pressed during a search used to
let the old game's answer land in the new one. And **positions are cached along the whole game**,
extended by a move or cut back by an undo from the longest shared prefix, so stepping back one
move never replays a chess game from the start. The list's layout groups a side's consecutive
moves into one turn, so a Dots & Boxes chain stays in one cell and the two columns still mean
something; a mutation that the tests could not catch turned out to guard an impossible case
(turns alternate by construction), and the guard was removed rather than kept untestable.

**11.3, as built.** Every chess game is saved as it is played, once it has a move of yours in
it - so a reload loses nothing and an unfinished game can still be exported. Stored as the
engine's own move numbers plus what the game was (sides, level, network, the rating shown),
not as PGN: the record is the fact, and PGN is one way of writing it out, regenerated on
demand. Saving on every move is safe because a save *replaces* the game's earlier save rather
than adding one. PGN is written by chess.js, which already knows the format's rules - tag order,
numbering, the SetUp and FEN tags a set-up position needs for 11.4 - and every export is read
back by chess.js's own PGN reader in the tests and compared move for move. Why a game ended is
worked out with the same conditions, in the same order, as the rules' `terminalValue`, so the
file cannot call a game drawn by repetition that the board thought was still going.

Resign arrived with it, so a lost game can be finished rather than abandoned. The page it all
lands on, `#/games/chess`, lists the games with how each ended, their moves, and copy /
download / delete, plus the whole history as one file; clearing it takes two clicks.

Found by playing it: undo back past your *only* move emptied the game, there was then no
record to write, and the history kept "resigned" for a game rewound to its first position. A
game with nothing of yours left in it now takes its record with it. And a Copy button tested by
a script reported failure, correctly - browsers only let a page write the clipboard during a
real click, which is what a real click then confirmed.

**11.4, as built.** The owner asked the right question first: why would the engine need
teaching to play a new position? It does not. The network judges whatever position it is
shown and search works from anywhere; nothing was retrained. What changed was plumbing: the
browser engine is told the moves, not the board, and replayed them from a starting position
that was hard-coded. It is now told where the game began (`positionFrom`, an optional method on
the game interface, so the worker still never mentions chess). The real caveat is statistical:
a network that learned from strong human games has seen little of a position with three
queens, and its instincts there are worth less - search carries it.

The editor itself: a palette, click to place and click again to remove, who moves, castling
rights that follow the pieces (a right with no rook to castle with cannot be switched on), and a
FEN box that takes a pasted position without ever being rewritten under the cursor. chess.js
refuses a missing or doubled king and pawns on the edge ranks; the editor adds what it lets
through - the side not to move standing in check, nine pawns, and a game already over. The new-game
sheet shows a set-up position as a small board, and it stays for the next game until changed.

**The bug that the plumbing change exposed.** The first game from a set-up position froze the
moment the engine replied. The chess view kept its own replay of the game, for the last-move
highlight, and that replay began at the usual first position: the engine's reply as Black -
Ke8-f7 - decoded as a white king's move from e1, which is illegal there, and rendering threw. The
page already keeps every position of the game, so the view now receives the position before the
last move instead of replaying anything. The regression test renders exactly that game. A
second finding came from mutation testing: a test that only checked the engine's start position
was "not finished" passed just as well when the engine ignored it, since the usual start is not
finished either; it now checks it is the position asked for.

**A bug the owner spotted within minutes.** The first version keyed ratings by *game*, so
switching the chess engine to the untrained network kept showing "about 2300" - a rating measured
on a different network entirely. Ratings are now keyed by network (`ratings.json` holds one entry
per network, and `scripts/stockfish.py --name` files a measurement under the name the site loads
it by); an unmeasured network shows no rating rather than someone else's.

**On FIDE.** Stockfish's scale comes from computer rating lists, not from people, so these are
"roughly FIDE" at best; the site says so. At club level engine lists and FIDE are commonly taken
to be in the same neighbourhood, which is the whole of the claim.

### The four difficulty levels, measured (Phase 10)

Each level plays the one below it, 60 games a rung (24 for chess), the same network on both
sides, `scripts/levels.py`. The only difference is how many positions it searches: 0, 40, 200,
600. Read them within a game - Elo does not transfer between games, and none of these numbers
mean anything on a human scale.

| Game | Casual over Beginner | Strong over Casual | Master over Strong | Beginner to Master |
|---|---|---|---|---|
| Four in a Row | +176 | +199 | +114 | **~488** |
| Reversi | +325 | +352 | +290 | **~967** |
| Gomoku | +168 | +154 | +58 | **~381** |
| Isolation | +147 | +108 | +223 | **~478** |
| Dots & Boxes | +260 | +176 | +134 | **~569** |
| Chess | +470 | clean sweep | +374 | unmeasurable |

What the spread says: **search is worth wildly different amounts in different games.** Reversi
gains most - nearly a thousand Elo end to end - because a disc flip a few moves ahead is
invisible to a network and obvious to a search. Gomoku gains least, and its last rung is worth
only 58 Elo: with eighty-one points to consider, tripling the budget from 200 to 600 barely
deepens anything. Chess's middle rung was a 24-0 sweep, so it has a floor and no ceiling.

That is the same lesson Phase 9.6 met from the other side - search was worth +541 Elo to the
chess network, more than any training difference in the project - and it is the argument for
spending the next effort on the search rather than the network.

### Chess, self-play stage 1 — first attempt: a collapse, diagnosed

Five iterations of self-play from the imitation network, as AlphaZero would do it: 150 games an
iteration at 200 simulations, 200 gradient steps of batch 256 at lr 2e-4, resignation at -0.9,
30 plies of temperature, no human data in the loop. Every loss inside the run fell — total
2.07 → 1.92, policy 1.57 → 1.52, value 0.50 → 0.40 — and the agent got much weaker.

| measurement | imitation 1 | gen 5 |
|---|---|---|
| **arena, 200 simulations a move** | — | **21.2%, -228 Elo** [-367, -133] |
| arena, no search (raw policy) | — | 47.2%, -19 Elo (not significant) |
| held-out human move accuracy | 49.8% | 47.3% |
| held-out value loss | 0.718 | 0.862 |
| mean \|v\| on human positions | 0.295 | 0.519 |
| positions called decided (\|v\| > 0.9) | 5.9% | 17.0% |
| search agrees with strong humans | 48.5% (+1.0 over its own policy) | 42.5% (**-3.3**) |

Read in order, those rows tell the whole story:

1. **The policy head was fine.** Without search the two networks are indistinguishable. Whatever
   went wrong is not "the network forgot how to play".
2. **The value head lost its calibration.** Its confidence nearly doubled and the share of
   positions it called decided tripled, while its error on held-out human positions rose 20%.
   It did not become better or worse at *ranking* positions so much as certain about them.
3. **Search is steered by the value head, so search broke.** For the imitation network, 200
   simulations agree with strong humans slightly *more often* than its own policy does: search
   improves on the priors, which is the assumption the whole method rests on. For gen 5, search
   agrees *less* often than its own policy. Search had stopped being a policy improvement
   operator and become a policy *degradation* operator.
4. **The arena measures search against search**, so it reported the full -228 Elo while the
   networks themselves were level.

Why the value head went: every position of a self-play game carries that game's single result,
so a window of 750 games is 750 labels however many positions it holds — and 1,000 gradient
steps were taken on them. This is AlphaGo's value-overfitting problem arriving from the
self-play side rather than the supervised side. AlphaZero does not meet it because its window
is 500,000 games; at 750 there is nothing to stop the head fitting the noise, and an
overconfident value head is worse than an uncertain one, because search believes it.

The fix is **rehearsal**: keep well-labelled human positions in every batch while the network
learns from its own games, so the value head keeps answering to 39 M outcomes rather than to a
few hundred. The alternative — far more games per gradient step — is the same fix by volume,
and this machine does not have the volume.

**The loss curves said none of this.** They fell throughout, because a policy loss measured
against a search that has itself drifted falls when the network agrees with a worse teacher.
Nothing inside the training loop can catch this; only measurements from outside it can —
the arena, the held-out exam, and the comparison between the network's own policy and what
search does with it.

### Chess, self-play stage 1 — second attempt: rehearsal, and a deeper reason

The same five iterations with half of every batch drawn from human positions, and 100 gradient
steps an iteration instead of 200. Rehearsal did what it was meant to:

| measurement | imitation 1 | gen 5, no rehearsal | gen 5, rehearsal |
|---|---|---|---|
| held-out human accuracy | 49.8% | 47.3% | **48.5%** |
| held-out value loss | 0.718 | 0.862 | **0.744** |
| mean \|v\| on human positions | 0.295 | 0.519 | **0.349** |
| positions called decided | 5.9% | 17.0% | **7.2%** |
| policy entropy over legal moves | 1.54 | — | 1.58 |
| arena, no search | — | 47.2% (ns) | 49.2% (ns) |

The value head kept its calibration, the policy never moved, and the two networks are
indistinguishable when they play from the policy head alone. And yet, with search, gen 5 was
still clearly weaker — until the same match was played at a different depth:

| simulations a move | gen 5 (rehearsal) vs imitation 1 |
|---|---|
| **50** | 49.5%, **-3 Elo** [-49, +42] — indistinguishable |
| **200** | 35.5%, **-104 Elo** [-151, -60] |

Two hundred games each, the same pair of networks, opposite answers. That is the whole finding:

**The value head did not get worse at ranking positions — it got more confident, and PUCT reads
confidence as authority.** On self-play positions both heads call the winner equally often
(75.6%), gen 5's squared error is even slightly lower, and its mean \|v\| is 0.309 → 0.392. At
50 simulations the priors decide most selections and the two networks play alike; at 200 there
are enough visits for Q to take over, and the extra search is spent exploiting a signal no more
accurate than before. More search made the imitation network stronger and this one *relatively*
weaker.

The imitation value head is heavily hedged - mean \|v\| 0.295 where the true mean is 0.876 - and
that hedging is load-bearing. It keeps a 65%-accurate value head from overruling a 50%-accurate
policy that cost 39 M positions to train. Self-play's first effect is to remove the hedge.

**The obvious knob does not fix it.** If search leans too hard on the values, lean it back onto
the priors — that is what `c_puct` is for. It made things worse:

| match, 200 games at 200 simulations | result |
|---|---|
| gen 5 at c_puct 3 vs imitation 1 at 1.5 | 28.7%, -158 Elo (worse than the -104 at 1.5) |
| imitation 1 at c_puct 3 vs itself at 1.5 | 36.2%, **-98 Elo** |

`c_puct` is not a trust dial, it is an exchange rate between what search has learned and what
the network expected, and raising it buys exploration with depth. At a fixed budget of 200
simulations both networks would rather commit. (It is also inert when every Q is equal —
scaling the only non-zero term cannot reorder it — which is why the search tests pin it in a
position where the rules have answered something.)

**What self-play actually cost was the ability to use search.** Each network played against
*itself* at a shallower depth:

| network | 200 simulations vs 50 |
|---|---|
| imitation 1 | 95.8%, **+541 Elo** |
| gen 5 (rehearsal) | 93.5%, **+463 Elo** |

Search is worth about 540 Elo to the imitation network — far more than any training difference
measured anywhere in this project. Self-play left the moves alone and took about 80 Elo off
*that*: the four numbers agree to within their intervals (-3 at 50 simulations, minus 541, plus
463, gives -81 against a measured -104). The loss is not in what the network plays, it is in
what search can still do with it.

So the useful way to state the result: **five iterations of self-play cost this agent roughly a
sixth of what search was worth to it, and left everything else untouched.**

### Chess, imitation stage 1 (January 2020: 39.2 M positions, one pass)

22 minutes on the M4 Max. Accuracy is how often the network's first choice among the legal
moves is the move the 2200+ player made, on 20,000 positions from games held out whole:

| positions seen | accuracy | opening | middlegame | endgame | held-out policy loss | value loss, training / held out |
|---|---|---|---|---|---|---|
| none (untrained) | 6.5% | | | | | |
| 0.4 M (a 1% trial) | 32.6% | 45% | 27% | 30% | 2.63 | |
| 2.0 M | 38.0% | 48% | 34% | 36% | 2.22 | 0.795 / 0.758 |
| 9.8 M | 44.7% | 53% | 41% | 44% | 1.82 | 0.738 / 0.734 |
| 19.6 M | 47.2% | 54% | 44% | 46% | 1.68 | 0.726 / 0.725 |
| 29.4 M | 48.9% | 55% | 46% | 48% | 1.61 | 0.717 / 0.717 |
| **39.2 M** | **49.3%** | **56%** | **46%** | **49%** | **1.59** | **0.713 / 0.713** |

**Half of strong players' moves, from one month.** Maia's networks of the same size predict
about half of human moves, with move history and 12 M games per rating band; this one reaches
49% from the current position alone and 530 k games. Two-thirds of the final figure came from
the first 1% of the data — the common patterns are learned almost at once, and every further
point costs more.

**The opening is the most predictable and the middlegame the least**: well-trodden theory,
against positions nobody has quite seen before.

**The value head judges modestly**: from a single position it names the winner of a decisive
game 66% of the time. Blitz results are noisy, and a position forty moves from the end says
only so much about how it ends.

**Against the untrained network**, 100 games each: **+90 =10 −0 with no search, +100 =0 −0 with
100 simulations**. The draws came only without search. Most likely because human games almost
never reach positions as lopsided as beating a random mover: the copied policy knows how strong
players play, not how to finish off a helpless opponent, and can shuffle into a repetition or a
stalemate. A little search removes the problem entirely.

**It plays like its teachers.** After 1.e4 its search at 200 simulations splits c5 42%, e5 23%,
e6 10% — the Sicilian, the open game and the French, in the order strong players choose them.

### Names, and two deliberate deviations from the published games

**Display names differ from internal keys for two games.** The page says *Four in a Row* and
*Isolation*; the keys stay `connect4` and `isola`, so every model file, checkpoint and
registry lookup is untouched. "Connect 4" is a live Hasbro trademark from 1974 and "Isola"
is Ravensburger's. Reversi, Gomoku and Dots & Boxes need no such care — those are already
the unencumbered names, and it is *Othello* that is the trademark, which is why nearly all
software says Reversi.

**Isolation is played on 7x7 here; the published board is 6x8.** Chosen before checking,
and kept deliberately once checked. Switching would mean a full retrain and, less obviously,
*half the augmentation*: a non-square rectangle has only four symmetries — identity, two
mirrors and a half-turn — where a square has eight. The 7x7 network is trained and verified
(95% on finding an immediate win, 100% on avoiding an immediate loss), so the cost was not
worth paying for a variant of the same game.

**Two further departures from the published Isolation rules**, both consequences of the
above: the board is 49 squares rather than 48, and the two starting squares are ordinary
tiles rather than the permanent platforms the Ravensburger board uses for its 46 tiles in 48
squares. Under the real rules each player always has an indestructible home square, which
changes the endgame.

**Gomoku is 9x9 where the standard is 15x15**, recorded earlier for the same reason: five in
a row spans most of a 9x9 board, which crowds out the double-threat construction that makes
the full-size game interesting.

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
| Chess in stages, each ending in a network the owner plays | The owner's own games are part of the evaluation, and no stage has to fit in one sitting. The price: every stage restarts training, so the chess replay buffer must survive checkpoints. |
| Chess games stored as what happened, not as the network sees them | Board, move played and result, encoded at training time. Changing the input planes or the move encoding then never means re-downloading or re-filtering a month. |
| Chess main line imitates strong games (both players 2200+), not the owner's level | The value head learns who went on to win, and at club level that is often decided by a blunder long after the position; stronger players' results track positions better. A stronger prior also wastes fewer simulations, and every bit of strength imitation provides is self-play compute not spent. The price is data — a month holds a fifth as many such games — and human-likeness at the owner's level, which becomes its own optional network. Filtered from the official CC0 dumps rather than the Lichess Elite Database, which states no licence. |
| Self-play positions stored compactly, and the window saved with the checkpoint | Half-precision planes and only the actions search visited: 2.7 KB a chess sample against 42 KB, because a 4,672-wide policy is about thirty visited moves and 4,640 zeros. That is what makes a window affordable to keep in memory *and* to write out, and a written-out window is what lets a stage resume without spending its first iterations training on a single iteration's correlated games. |
| Resignation measured offline rather than by playing a share of games out | AlphaZero played ~10% of resigned games to the end to count how often resignation was wrong. Doing that here means a second outcome per game travelling back through the worker pool, complicating every stage and every game for one number. A separate script that replays the condition with resignation off answers the same question, on demand, for any network. |
| The held-out human exam lives in the library, not in the imitation script | More than one stage has to sit it - self-play is graded on it for forgetting - and it has to be identical across stages to be comparable. A measurement that decides things earns tests, and code inside `scripts/` cannot be imported by them. |
| Self-play from an imitated network rehearses human positions (half of every batch) | Measured, not assumed: without it, five iterations cost -228 Elo by wrecking the value head's calibration (results). A self-play window of a few hundred games is a few hundred value labels, and AlphaZero's answer - 500,000 games in the window - is not available on one laptop. Rehearsal buys the same protection with the 39 M human outcomes already on disk. It is a departure from AlphaZero, and the reason is a hardware budget, not a disagreement. |
| Strength is judged with search *and* without it | The two disagreed by 209 Elo on the same pair of networks, which is what localised the fault to the value head in one match rather than a day of guessing. `--simulations 0` plays straight from the policy head. |

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

### Phase 4

- **A port is only trustworthy if it is proved identical** — MCTS with a uniform evaluator
  and no root noise is fully deterministic, so the TypeScript search can be checked against
  Python's exact visit counts rather than "looks about right". The algorithm has two
  separate sign flips, a specific tie-breaking order and a root expanded before the loop;
  none of that survives being reimplemented from memory.
- **Rules duplicated across languages need a mechanical check** — 250 generated vectors
  covering legal moves, terminal values and encodings. The encodings matter most: a board
  encoded differently in the browser feeds the network inputs it was never trained on,
  while still producing legal, plausible-looking moves.
- **Choose the quantity before the tolerance** — comparing raw logits rejected a perfectly
  good export, because scaled-up weights put them in the hundreds of thousands where a
  difference of 22 is ordinary floating point. What the system consumes is the masked
  softmax, and on that the agreement was exact. The wrong quantity makes any tolerance
  meaningless.
- **Watch for a comparison that cannot fail** — `tanh` saturation made the value check pass
  by being blind, and a verifier comparing PyTorch to itself passed every test in the file.
  Both now have tests that fail when the check stops checking.
- **The canonical board is not the display board** — the engine's representation flips sign
  every ply so the network always sees itself as +1, which is exactly what a UI must not
  do. The view is derived from the move list instead, so the two cannot be confused.
- **Send the move list, not the position** — the worker replays from scratch. Sending a
  board would mean two copies of the game state that have to agree, and the bug where they
  stop agreeing is silent.

### Phase 5

- **An abstraction is only proved by the second implementation** — a contract validated
  against one game is a guess. Reversi differs in board size, action space, terminal rule
  and symmetry count, and required three lines of registration. That is the evidence the
  Phase 0 design was right; nothing before it was.
- **No legal move does not mean the game is over** — the assumption Connect 4 quietly
  encourages, and the one Reversi breaks. A framework that equated an empty move list with
  a finished game would have scored a forced pass as a result.
- **A convention's visible consequence can be game-specific** — Phase 0 recorded that
  `terminal_value` "normally returns only -1 or 0, never +1", which is true of Connect 4
  and *false* of Reversi: it ends when neither side can move, and the player to move may
  hold more discs, so +1 appears about a third of the time. The convention never changed;
  the observation had been generalised from a single sample. The TypeScript port asserted
  it for both games and Reversi rejected it.
- **Symmetry has to know what is not a square** — the pass action means the same thing
  however the board is turned, so it is held out of the permutation. Rotating it with the
  squares would pair every augmented position with a policy whose last entry belongs to a
  different action, and with eight variants per position that corrupts data four times
  faster than Connect 4's two ever could.
- **A clean sweep is a bound, not a measurement** — 40-0 makes the sample variance zero, and
  a normal approximation collapses the interval to a point, printing a clamped +3600 Elo as
  though it had been measured. The rule of three gives the honest version: zero losses in n
  games puts the 95% bound on losing at 3/n, so 40-0 means ">+436 Elo" and 400-0 means
  ">+849". Found by reading the training log rather than by a failing test.
### Phase 6

### Phase 7

- **A compound action is a product, not a concatenation** — one turn stays one action, so
  `apply` still hands the position to the other player and the canonical flip still happens
  exactly once. Splitting the turn into two plies would be easier to encode and would break
  the invariant every other component relied on — until Phase 8 retired it.
- **Every transform must act on every component** — a board symmetry rotates the destroyed
  square *and* relabels the direction. Doing one produces a legal action on a plausible
  board and silently wrong training data, multiplied eight times over by augmentation.
- **A convolutional policy head needs the action space to be a product of board squares** —
  Isola's 392 is 8 x 49 and chess's 4672 is 73 x 64, so both qualify; Connect 4's seven
  columns are not its forty-two cells, so it does not. Where it applies it is less than half
  the parameters and a better fit, because the action keeps its place on the board instead
  of being scattered through a matrix.
- **A shrinking board is its own curriculum** — Isola's branching is worse than Gomoku's at
  the opening, but every turn removes a square, so the game walks itself into positions
  search can actually solve. Whether a high branching factor is fatal depends on whether the
  game ever gets easier on its own.
- **A failing test can be the rules teaching you something** — a position where a player can
  step but has nothing to destroy seemed obviously possible and is not: the square just
  vacated is always standing and never occupied, so a legal step always has a legal
  demolition. The game therefore ends on immobility alone.

- **A metric can move the right way for the wrong reason** — Phase 6 recorded that Gomoku
  games got shorter as the network improved, and explained it as a stronger winner finishing
  sooner. That was true and it was also the signature of a broken run: the "winner" was
  beating an opponent that could not defend. After the fix games got *longer*, from 25 plies
  to 62, and nearly half of them drew. Direction alone means nothing without asking which
  side of the game is producing it.
- **Restricting the action space can be the difference between learning and not** — with 79
  legal moves, a knowledge-free search of 20,000 simulations reaches no terminal position and
  returns a value of exactly zero everywhere. Cutting the branching factor to 13 is domain
  knowledge and a departure from "zero", but AlphaZero had roughly a million times this
  budget. The honest move is to take the shortcut and write down that you took it.
- **Improvement does not always look the same** — Connect 4 and Reversi games got *longer*
  as the networks improved, because a stronger loser survives further. Gomoku games got
  shorter, from 35 plies to 16, because a stronger winner finishes sooner. Neither
  direction is a warning sign on its own; what matters is knowing which one the game should
  produce before reading anything into it.
- **A large action space thins the search, not just the policy** — 200 simulations over 50
  legal moves is four apiece, against nearly thirty when there are seven. That, more than
  the sparse policy target, is why big action spaces are hard: the same budget buys much
  less certainty per move.
- **Search must be wide enough to visit the moves it is choosing between** — the policy
  improvement operator only improves anything if the search examines alternatives. At 50
  simulations over 73 legal moves it visits two, so the "improved" target is a sharpened
  copy of the prior and the loop learns almost nothing beyond the game result. Simulation
  budget has to scale with the branching factor, not be carried over from the previous game.
- **Exploration noise has a shape, and the shape depends on the action count** — Dirichlet
  noise should be *concentrated*, so the agent tries a few specific alternatives properly.
  At a fixed alpha it becomes a uniform smear as the action space grows, diluting the prior
  instead of probing it. It is one constant, it was right for Connect 4, and it was wrong by
  a factor of eight for Gomoku.
- **Raising the evaluation sample paid for itself immediately** — four of Gomoku's six
  progress measurements were between 49 and 92 Elo, all invisible to a 40-game match. The
  Phase 5 fix turned a run that would have looked like it stalled after gen5 into one with
  five consecutive significant gains.

- **An underpowered measurement looks exactly like a plateau** — and it starts looking like
  one at precisely the moment it loses the ability to see. Two 40-game matches showed no
  progress across ten Reversi iterations; a 200-game match across the same span showed
  +263 Elo. The gains had fallen to roughly 60-70 Elo per step and a 40-game sample cannot
  resolve better than 111. Before concluding that learning has stopped, check whether the
  instrument could still have detected it.

### Phase 8

- **Whose turn it is is a fact the game has to state** — canonical perspective deliberately
  hides it, and for seven phases counting moves gave the right answer by coincidence. With
  bonus moves, counting hands the bonus to the wrong player, labels a player's own capture as
  a gift to their opponent, and credits the result to whoever happened to move last on an even
  count. All three fail silently.
- **Some games cannot be encoded around a framework's assumptions** — Reversi's pass and
  Isolation's compound action each bent a game to fit "every move passes the turn". A chain of
  captures has no bound, so this game could not be bent and the assumption had to go instead.
  Recognising which situation you are in is most of the design work.
- **A generalisation should be provably a no-op where it was not needed** — the seat-aware
  search regenerates the four existing games' test vectors bit for bit. That is a far stronger
  regression check than "the old tests still pass".
- **A test vector that cannot fail proves nothing** — the first Dots & Boxes search vectors came
  from early positions, where a knowledge-free search values everything at exactly zero. Zero
  looks the same whichever way it is flipped, so they would have passed a port with the sign
  rule inverted. Moved to endgames with larger budgets, 18 of 40 reach finished games, every one
  through a bonus move.
- **One lucky case hides a whole class of bug** — every game here is exactly sixty moves, so an
  arena crediting results by counting moves agrees with the correct rule whenever a game ends
  with seat 0 to move. The first arena test used a single game that did. Mutation testing caught
  it, and the test now requires games that end on both seats.
- **On the hot path, the rules are not a detail** — `apply` runs once for every child of every
  expanded node, sixty at a time in this opening, while the network runs once per expansion.
  Bitmasks put the rules under a tenth of the network's cost.
- **Declare, don't infer** — the page decided a player must pass when their only legal move was
  the last action index. That describes Reversi's pass, and also Four in a Row's seventh column.
  Anything the code needs to know about a game belongs in the game's contract.
- **A checkpoint is not the whole training state** — it held the network and the optimiser but
  not the replay buffer, so the resumed run trained its first iteration on one iteration's
  games instead of ten. The loss dropped by a sixth at once: memorising looks like progress on
  a loss measured over the samples being fitted. Anything a run needs in order to continue
  unchanged belongs in the checkpoint, or has to be rebuilt before training resumes.
- **The value head and the policy head can learn different things** — on double-dealing traps
  the policy grew *more* sure that taking the box is right while the value head learned that it
  loses. Search joins them: the prior decides what gets looked at, the value decides what wins.
  Scored without search, this network looks as if it never learned the game's central idea.
- **An exact yardstick sees what a match between two versions of the agent cannot** — four of
  the five arena matches after gen15 were not significant while the endgame score climbed every
  generation. The arena measures strength in the engines' own games, at a resolution set by
  its sample size; the yardstick measures one skill against the truth.
- **Elo gains do not chain** — the five step results after gen15 summed to +296 where one direct
  400-game match found +171; three steps of the fine-tuning run summed to +84 where the direct
  match found +186. Wrong in both directions, so not a bias to correct for: each small step
  carries its own ±70 Elo of noise, and adding steps adds their noise. Measure the span directly.
- **An agent is only tested where its own play goes** — competent networks almost never hand
  each other a box early, so this one never learned what to do when a person does, and no
  arena match, self-play loss or endgame exam could show it. What the replay buffer holds
  decides what the network can still handle: gift-taking collapsed when the resumed run's
  buffer lost its older, weaker games. Positions the agent does not generate itself have to be
  supplied on purpose — randomised openings, or opponents other than itself.
- **Exploring moves is not exploring positions** — root noise and temperature make the agent
  try moves it half-likes, which is how it finds better ones. They cannot take it anywhere that
  only a move it never considers leads to; changing where games start can. And what was not
  searched must not be trained on: a random move has no search policy to imitate, and the
  result that follows it reflects a choice nobody made.
- **The objective decides what counts as a mistake** — an agent rewarded only for winning is
  indifferent to the margin, so when games are decided by a median of 14 boxes it has no
  reason to take a free one, and it doesn't. Better data could not change that; only a
  different objective could. KataGo adds a score term to its search for exactly this reason:
  when the chances of winning are equal, prefer the bigger win.

### Phase 9

- **Delegate the rules, own the translation** — python-chess decides what is legal. What can
  still go wrong is ours: mirroring the board for Black, and a 4,672-entry table naming every
  move. That is where the tests and the eighteen deliberate bugs went, and an outside number —
  Leela's 1,858 on-board moves — checked the table's geometry independently.
- **Most of a search tree is never visited** — expansion records every legal move of a leaf, and
  the next simulation visits one of them. Building positions only on first visit cut the rules
  from 31% of a chess search step to 10%, with the tree and every result unchanged.
- **Prove that a speed-up changed nothing** — a change meant to leave every result as it was
  should be shown to: five games' search vectors regenerated bit for bit. The first attempt
  "failed" because the committed files had been made with case counts nobody had written down
  — an unrecorded setting is a result nobody can reproduce.
- **A test that cannot fail, again** — the first chess search vectors reached a decided game in
  none of 40 searches, so every root value was zero, which agrees with a port whatever it does
  with signs: the Phase 8 lesson in a new game. Starting half the searches where a move mates
  at once made 16 of them decisive.
- **A convenient API can cost more than the work** — chess.js's full move list spent 0.8 ms per
  position on notation the search never reads, more than the network evaluation itself.
  Measure what a library does per call before building a hot loop on it.
- **Reject on the cheapest evidence first** — 99% of a month's games are ruled out by their
  headers, and only the 1.1% that survive pay for parsing and replaying their moves. That
  ordering, not faster parsing, is why 99 GB of games convert in under four minutes.
- **Imitation has its own data hygiene** — hold out whole games, not positions, or validation
  measures how well the network remembers games it has half seen; and drop results a clock
  decided, or the value head learns that being ahead loses.
- **Behaviour cloning copies choices, not reasons** — the network predicts half of what strong
  players play, yet without search it drew ten games against a random mover: the teachers never
  showed it a position that lopsided. Imitation is only as good as the positions its data
  covers — the Dots & Boxes gift lesson from the other side — and search, then self-play, are
  what reach beyond them.
- **The learning curve is steep, then long** — about three-fifths of the whole gain, 6.5% to
  49.3%, came from the first 1% of the data. Most of what can be imitated is common; each
  further point of accuracy comes from rarer and harder decisions.
- **Resignation is a trade between compute and labels** — cutting a lost game short is free
  speed only while the side resigning really was lost; every false resignation labels a whole
  game a loss for someone who was not losing. That makes it the one self-play setting that can
  corrupt the training signal, so the threshold is a measured quantity, not a preference:
  -0.9 saves 47% of all plies here and was never wrong in 60 games.
- **Who a counter belongs to** — resignation first counted plies below the threshold rather
  than a player's own moves. Since values are always the mover's own, that asked both players
  to despair at once: it could not fire in a game where turns alternate, and nothing said so.
  Any rule that spans moves has to name whose moves it spans — the Phase 8 lesson about seats,
  met again in a new place.
- **A stub can be too helpful** — the test for that rule used an evaluator that told both sides
  they were losing, which no real evaluator does, and that is exactly the case where the broken
  reading and the correct one agree. A test built from an impossible situation pins the
  implementation rather than the intent.
- **Storage shapes what is affordable to keep** — a chess policy target is thirty visited moves
  and 4,640 zeros; keeping only the visits, in half precision, is 2.7 KB a sample against 42 KB,
  and that alone is the difference between a replay window that fits in memory and one that
  does not. Compression is not an optimisation here, it is what makes the window possible.
- **Resuming carries more than weights** — an optimiser state carries the learning rate it was
  saved with. Resuming self-play from the tail of an imitation run's cosine schedule would have
  trained at 5e-5 while the run reported healthy losses, because every loss falls when nothing
  moves.
- **Search is a policy improvement operator — until it isn't** — AlphaZero trains the policy
  towards search's visit counts because those visits are better than the priors that produced
  them. That is an assumption about the *value head*, which is what search follows, and it can
  fail: a network whose value head has become overconfident searches *worse* than it plays. Then
  every iteration teaches the network to imitate a worse teacher, and nothing inside the loop
  objects. Measured here: search agreed with strong humans +1.0 points more often than the raw
  policy before, and -3.3 points less after.
- **Confidence is not accuracy, and search believes confidence** — the collapsed value head's
  error on human positions rose only 20%, but the share of positions it called decided tripled.
  PUCT follows values, so a head that is *certain and wrong* prunes the right move away, while a
  head that is *uncertain and wrong* lets the priors and the visit counts carry the search.
- **The effective sample size for a value head is games, not positions** — every position in a
  game carries that game's single result. A 30,000-position window of 750 games holds 750 value
  labels. AlphaGo hit this training on whole human games and answered with one position per
  game; AlphaZero avoids it with a window of 500,000 games. On a laptop, rehearsal is the
  affordable answer.
- **Rehearsal** — keeping examples of the old task in every batch while a network learns a new
  one, the standard defence against catastrophic forgetting. Here: half of each batch drawn
  from human games, so self-play can move the network without being the only thing that does.
- **Falling losses prove nothing about strength** — the collapsed run's every loss fell
  monotonically while it lost 228 Elo. A policy loss measured against a drifting search falls
  when the network agrees with a worse teacher. Only measurements from outside the loop - the
  arena, a held-out exam, a comparison of the network with and without search - can see it.
- **A rating is a statement about a pool** — Elo differences only mean something among the
  players they were measured against. Anchoring a ladder takes opponents with known ratings (here,
  Stockfish at fixed strengths) and a maximum-likelihood fit across all of them at once. And a pool
  of one engine's own variants inflates its gaps: Master over Strong was +374 against itself and
  about +110 against Stockfish, because a deeper search exploits precisely the blind spots its
  shallower twin shares with it.
- **Measure with search and without it** — the same two networks were 209 Elo apart depending
  on whether search was switched on. That gap is a diagnosis: equal without search and far apart
  with it means the fault is in the value head, not the policy.
- **Measure at more than one depth, too** — the same pair of networks was 104 Elo apart at 200
  simulations and level at 50. A single depth would have reported a straightforward regression;
  two depths said what it actually was, a loss of search *efficiency* rather than of skill.
- **c_puct is an exchange rate, not a trust dial** — it prices what search has learned against
  what the network expected. When every Q is equal it does nothing at all, since scaling the
  only non-zero term cannot reorder it; when results disagree with priors, raising it buys
  breadth by spending depth. At a fixed simulation budget, both chess networks measured here
  preferred to commit: c_puct 1.5 beat 3.0 by ~98 Elo.
- **Search is worth more than training, at this scale** — going from 50 to 200 simulations is
  +541 Elo for the imitation network. Every training difference measured in this project is
  smaller than that, which sets the priority: protect what search can do before chasing what
  the network knows.

---

## Commands

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```
