# Caissa

AlphaZero-style self-play reinforcement learning for board games, playable in the browser.

The agent knows nothing about any particular game. Everything it needs sits behind the
`Game` protocol in `src/caissa/games/base.py`, so the same algorithm learns Connect 4,
Reversi or chess without changes to the search or training code.

## Status

Phases 0 and 1 complete: the `Game` contract and Connect 4, MCTS with PUCT, and the
policy/value network. Next is Phase 2, the self-play and training loop.
See [PLAN.md](PLAN.md) for the full roadmap, decision log and progress checklist.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```
