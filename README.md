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
it is now has to be stated by the game rather than counted. Chess is under way, in stages
(Phase 9).
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
