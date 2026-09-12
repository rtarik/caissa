# Working on Caissa

**Read [PLAN.md](PLAN.md) first.** It holds the goals, constraints, measured benchmarks,
the phase checklist, and a decision log explaining why things are built the way they are.
Keep it current: tick checkboxes as work lands, add to the decision log when something is
settled, and extend the concepts glossary as each phase introduces new ideas.

## The point of this project

The owner is learning reinforcement learning. That is the **primary** goal — ahead of
shipping, ahead of strength. Two consequences:

- **Explain the concepts as you go.** After each step, describe what was built and, above
  all, the reinforcement-learning ideas behind it — what the mechanism does, why it is
  there, and what breaks without it. Do not just report that code was written.
- **Work in small, reviewable steps.** Favour clarity over cleverness. A clear
  implementation that teaches the idea beats a fast one that obscures it.

## Conventions that must not be broken

Detailed in PLAN.md under *Core conventions*. In short: canonical perspective (board always
seen from the mover's side, `apply()` flips the sign), mover-relative values (`+1` win for
the player to move, so `terminal_value()` normally returns only `-1` or `0`), action masks
rather than action lists, and symmetry policy permutations that exactly match their board
permutations. Changing any of these after training starts means retraining from scratch.

The algorithm is **game-agnostic**. Search and training code must never import a concrete
game — only the `Game` protocol in `src/caissa/games/base.py`.

## Testing

Tests must be load-bearing. After writing a suite, deliberately break the code it covers
and confirm the tests fail. A test that passes whether or not the code is correct is worse
than no test, because it buys false confidence.

## Git

**Never run `git add`, `git commit` or `git push`.** Build and verify the work, then hand
over a single copy-pasteable command in one ```bash fence:

`git add -A && git commit -m "<one short line>" && git push`

Detail belongs in the chat reply, not the commit message.

## Commands

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```
