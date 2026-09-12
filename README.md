# Caissa

AlphaZero-style self-play reinforcement learning for board games, playable in the browser.

The agent knows nothing about any particular game. Everything it needs sits behind the
`Game` protocol in `src/caissa/games/base.py`, so the same algorithm learns Connect 4,
Reversi or chess without changes to the search or training code.

## Status

Phases 0, 1 and 2a complete: the `Game` contract and Connect 4, MCTS with PUCT, the
policy/value network, and the self-play and training loop. Next is Phase 2b, parallel
self-play, then Phase 3, evaluation and gating.
See [PLAN.md](PLAN.md) for the full roadmap, decision log and progress checklist.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

Train Connect 4:

```bash
.venv/bin/python scripts/train.py --iterations 14 --games 35
```
