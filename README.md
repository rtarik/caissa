# Caissa

AlphaZero-style self-play reinforcement learning for board games, playable in the browser.

The agent knows nothing about any particular game. Everything it needs sits behind the
`Game` protocol in `src/caissa/games/base.py`, so the same algorithm learns Four in a Row,
Reversi or chess without changes to the search or training code.

## Status

Phases 0 through 8 complete: Four in a Row, Reversi, Gomoku, Isolation and Dots & Boxes are
playable in the browser against networks learned entirely from self-play, with search running
in a Web Worker and nothing leaving the device. Reversi, Gomoku and Isolation each needed only
a few lines of change to existing framework code. Dots & Boxes deliberately changed the
framework itself: it is the first game where a player can move twice in a row, so whose turn
it is now has to be stated by the game rather than counted.

Chess (Phase 9) learns from human games first and is playable; its first self-play stage made
the engine *weaker*, which turned out to be worth more than a stronger one would have been —
the diagnosis is in PLAN.md under *Chess, self-play stage 1*. Phase 10 rebuilt the site around
playing rather than around the project: a gallery of games, a palette and an icon for each, and
the engine's internals behind one quiet toggle.
See [PLAN.md](PLAN.md) for the full roadmap, decision log and progress checklist.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

Train Four in a Row (key `connect4`):

```bash
.venv/bin/python scripts/train.py --iterations 40 --games 250 --workers 10
```

Checkpoints are written after every iteration, so an interrupted run resumes with
`--resume models/<game>-latest.pt`.

Self-play runs across worker processes on the CPU; the gradient steps run on the GPU. Any
script that starts the pool must guard its entry point with `if __name__ == "__main__":`.

Measure a checkpoint against perfect play, or against another checkpoint:

```bash
.venv/bin/python scripts/accuracy.py models/connect4-latest.pt --positions 200
```

```bash
.venv/bin/python scripts/evaluate.py models/connect4-gen0030.pt models/connect4-latest.pt --games 400 --workers 8
```

Dots & Boxes endgames are small enough to solve exactly, which grades a checkpoint on the
double-dealing traps that greedy play falls into:

```bash
.venv/bin/python scripts/endgames.py models/dotsandboxes-latest.pt
```

Chess learns first from human games. This downloads a month of Lichess's public games
(about 14 GB, checked against the published checksum) and keeps those between players rated
2200+, as training positions in `data/chess/<month>/` — a few minutes for a month:

```bash
.venv/bin/python scripts/lichess.py 2020-01
```

Then train an imitation stage on the converted months, and export it beside the earlier ones —
the page offers every exported stage:

```bash
.venv/bin/python scripts/imitate.py --months 2020-01 --stage 1
```

```bash
.venv/bin/python scripts/export.py models/chess-imitation1.pt --name chess-imitation1 --label "Imitation 1"
```

Self-play continues from there, at a lower learning rate, carrying its replay window from
checkpoint to checkpoint, and rehearsing human positions in half of every batch — without which
the value head loses its calibration in a few iterations and takes search down with it (PLAN.md,
*Chess, self-play stage 1*):

```bash
.venv/bin/python scripts/train.py --game chess --resume models/chess-imitation1.pt --iterations 21 --games 150 --simulations 200 --train-steps 100 --blocks 6 --channels 64 --policy-head conv --temperature-moves 30 --max-plies 200 --resign-below -0.9 --buffer 120000 --min-buffer 20000 --learning-rate 2e-4 --human data/chess --human-share 0.5 --save-buffer --workers 10
```

Three scripts watch a self-play stage from outside its own loop, where its losses cannot
flatter it. `humanmoves.py` grades checkpoints on held-out human moves, `improvement.py` asks
whether search still improves on the network's own policy, and `resignations.py` measures how
often resigning would have thrown a game away:

```bash
.venv/bin/python scripts/humanmoves.py models/chess-imitation1.pt models/chess-gen0020.pt
```

```bash
.venv/bin/python scripts/improvement.py models/chess-gen0020.pt --simulations 200 800
```

```bash
.venv/bin/python scripts/resignations.py models/chess-gen0020.pt --games 60
```

`scripts/levels.py` measures what the site's four difficulty levels are worth in each game, by
playing each level against the one below it. It writes `web/public/levels.json`, which the
*How it works* page reads:

```bash
.venv/bin/python scripts/levels.py --games 60
```

`scripts/gifts.py` checks the other end of the game: how often a checkpoint takes a box handed
to it early, a position its own self-play almost never produces.

```bash
.venv/bin/python scripts/gifts.py models/dotsandboxes-gen0040.pt models/dotsandboxes-latest.pt
```

## The web app

Export a checkpoint, regenerate the cross-language test vectors, then run it:

```bash
.venv/bin/python scripts/export.py models/connect4-latest.pt && .venv/bin/python scripts/testvectors.py
```

```bash
cd web && npm install && npm test && npm run dev
```

`web/` is a Vite + TypeScript app. The rules and the search exist in TypeScript as well as
Python, and both are checked against vectors generated from Python — the TypeScript search
must reproduce Python's visit counts exactly.
