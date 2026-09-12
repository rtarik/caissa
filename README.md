# Caissa

AlphaZero-style self-play reinforcement learning for board games, playable in the browser.

The agent knows nothing about any particular game. Everything it needs sits behind the
`Game` protocol in `src/caissa/games/base.py`, so the same algorithm learns Connect 4,
Reversi or chess without changes to the search or training code.

## Status

| Phase | | |
|---|---|---|
| 0 | Game contract + Connect 4 | done |
| 1 | Network + MCTS | next |
| 2 | Self-play + training loop | |
| 3 | Evaluation, arena, gating | |
| 4 | Browser shell, Connect 4 playable | |
| 5 | Reversi | |
| 6+ | More games, chess bootstrap, self-play RL | |

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```
