"""A fixed exam of held-out human positions, and how a network is graded on it.

The exam lives here rather than inside the imitation script because more than one
stage has to sit it. Self-play trains on the agent's own games, which are worse
than the human ones it was cloned from, and the risk is quiet: the network drifts
towards its own play and forgets moves it used to find, while every self-play loss
keeps falling because the thing it is measured against fell with it. Grading each
generation on the same held-out human moves is what catches that - so the exam has
to be identical across stages, which means one definition, seeded, in one place.

Two measurements, matching the two heads:

* **move-prediction accuracy** - how often the network's first choice *among the
  legal moves* is the move the human played, overall and by phase of the game.
  The legality mask matters: an unmasked argmax grades a network on moves it
  would never be allowed to play, which is not the question.
* **value loss on games held out whole** - beside the training value loss, a gap
  says the value head is recognising games rather than judging positions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from caissa.data.chess import board_of, encode, load_months, values
from caissa.games.chess import action_of
from caissa.network import PolicyValueNet


#: (name, first ply, last ply + 1) for the accuracy breakdown.
PHASES = (("opening", 0, 20), ("middlegame", 20, 60), ("endgame", 60, 100_000))


class Validation:
    """A fixed sample of held-out positions, measured the same way every time."""

    def __init__(self, positions: np.ndarray, games: np.ndarray, count: int,
                 rng: np.random.Generator):
        held_out = np.flatnonzero(games["validation"][positions["game"]].astype(bool))
        rows = np.sort(rng.choice(held_out, size=min(count, len(held_out)), replace=False))
        sample = positions[rows]
        planes, self.actions = encode(sample)
        self.planes = torch.from_numpy(planes)
        self.values = values(sample, games)
        self.plies = sample["ply"].astype(np.int64)
        # The legal moves of each position, as action indices, so the network's
        # first choice can be taken among legal moves - as in play.
        legal = []
        for stored in sample:
            board = board_of(stored)
            legal.append([action_of(board, move) for move in board.legal_moves])
        self.offsets = np.cumsum([0] + [len(moves) for moves in legal])
        self.legal = np.concatenate([np.asarray(moves, dtype=np.int64) for moves in legal])

    @torch.no_grad()
    def measure(self, net: PolicyValueNet, device: torch.device, batch: int = 2048) -> dict:
        net.eval()
        correct = np.zeros(len(self.actions), dtype=bool)
        predicted_value = np.zeros(len(self.actions), dtype=np.float32)
        policy_loss = 0.0
        for start in range(0, len(self.actions), batch):
            end = min(start + batch, len(self.actions))
            logits, value = net(self.planes[start:end].to(device))
            logits = logits.float().cpu()
            actions = torch.from_numpy(self.actions[start:end])
            policy_loss += F.cross_entropy(logits, actions, reduction="sum").item()
            predicted_value[start:end] = value.float().cpu().numpy()

            legal = self.legal[self.offsets[start]:self.offsets[end]]
            rows = np.repeat(np.arange(end - start), np.diff(self.offsets[start:end + 1]))
            masked = torch.full_like(logits, -torch.inf)
            masked[rows, legal] = logits[rows, legal]
            correct[start:end] = masked.argmax(dim=1).numpy() == self.actions[start:end]

        decisive = self.values != 0
        result = {
            "accuracy": float(correct.mean()),
            **{f"accuracy_{name}": float(correct[(self.plies >= low) & (self.plies < high)].mean())
               for name, low, high in PHASES},
            "policy_loss": policy_loss / len(self.actions),
            "value_loss": float(np.mean((predicted_value - self.values) ** 2)),
            # How often the value head at least picks the right winner.
            "value_sign": float(np.mean(np.sign(predicted_value[decisive])
                                        == self.values[decisive])),
        }
        net.train()
        return result
