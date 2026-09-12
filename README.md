# Caissa

AlphaZero-style self-play reinforcement learning for board games, playable in the browser.

The agent knows nothing about any particular game. Everything it needs sits behind the
`Game` protocol in `src/caissa/games/base.py`, so the same algorithm learns Connect 4,
Reversi or chess without changes to the search or training code.

## Status

Phases 0 through 3 complete: the `Game` contract and Connect 4, MCTS with PUCT, the
policy/value network, the self-play and training loop, parallel self-play (27x faster per
game), and evaluation — an arena with Elo and confidence intervals, plus a perfect solver
to grade against. Next is Phase 4, the browser.
See [PLAN.md](PLAN.md) for the full roadmap, decision log and progress checklist.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

Train Connect 4:

```bash
.venv/bin/python scripts/train.py --iterations 40 --games 250 --workers 10
```

Self-play runs across worker processes on the CPU; the gradient steps run on the GPU. Any
script that starts the pool must guard its entry point with `if __name__ == "__main__":`.

Measure a checkpoint against perfect play, or against another checkpoint:

```bash
.venv/bin/python scripts/accuracy.py models/connect4-latest.pt --positions 200
```
