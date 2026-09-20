"""Imitation: teach the chess network the moves strong humans played.

    python scripts/imitate.py --months 2020-01 --stage 1

Behaviour cloning, the supervised stage AlphaGo began with. Every stored
position is a training example: its policy target is the move the human played,
its value target how the game ended for the player to move.

Progress is measured on games held out whole (see caissa.data.lichess) every
twentieth of the run:

* **move-prediction accuracy**: how often the network's first choice among the
  legal moves is the move actually played - overall, and for the opening (the
  first 20 plies), the middlegame and the endgame (from ply 60);
* **held-out value loss beside the training value loss**: a gap that opens up
  means the value head is recognising games rather than judging positions -
  the overfitting AlphaGo met when it trained value on whole games.

Writes models/chess-imitation<stage>.pt, which export.py and evaluate.py read like
any other checkpoint, with the measurements inside it and beside it as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from caissa.data.chess import encode, values
from caissa.data.heldout import PHASES, Validation, load_months
from caissa.games.chess import ACTIONS, Chess
from caissa.network import NetworkConfig, PolicyValueNet, best_device
from caissa.train import imitation_step


def learning_rate(step: int, steps: int, peak: float) -> float:
    """A short linear warm-up, then a cosine down to a twentieth of the peak."""
    warmup = max(200, steps // 50)
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


def save(path: Path, net: PolicyValueNet, optimizer, config: NetworkConfig, stage: int,
         months: list[str], log: list[dict]) -> None:
    torch.save({
        "iteration": stage,
        "game": "chess",
        "network": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "config": {"network": asdict(config)},
        "imitation": {"months": months, "log": log},
    }, path)
    path.with_suffix(".json").write_text(json.dumps({"months": months, "log": log}, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", required=True, help="comma-separated, e.g. 2020-01,2020-02")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/chess"))
    parser.add_argument("--resume", type=Path, default=None,
                        help="continue from an earlier stage's checkpoint")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--validation", type=int, default=20_000)
    parser.add_argument("--measurements", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early: to time a run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    months = args.months.split(",")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    positions, games = load_months(args.data, months)
    training = np.flatnonzero(~games["validation"][positions["game"]].astype(bool))
    validation = Validation(positions, games, args.validation, rng)

    config = NetworkConfig(blocks=args.blocks, channels=args.channels, policy_head="conv")
    net = PolicyValueNet.for_game(Chess(), config)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = NetworkConfig(**checkpoint["config"]["network"])
        net = PolicyValueNet.for_game(Chess(), config)
        net.load_state_dict(checkpoint["network"])
    device = best_device()
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps = int(args.epochs * len(training) / args.batch)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    every = max(1, steps // args.measurements)
    path = args.out / f"chess-imitation{args.stage}.pt"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(training):,} training positions from {len(games):,} games ({', '.join(months)}); "
          f"{len(validation.actions):,} held-out positions measured")
    print(f"{net.parameter_count():,} parameters on {device}; {steps:,} steps of {args.batch}, "
          f"measured every {every:,}\n", flush=True)

    log = [{"step": 0, "positions": 0, **validation.measure(net, device)}]
    print(f"untrained: accuracy {log[0]['accuracy']:.1%}", flush=True)

    order = rng.permutation(training)
    cursor = 0
    started = time.perf_counter()
    window = np.zeros(3)
    for step in range(1, steps + 1):
        if cursor + args.batch > len(order):
            order, cursor = rng.permutation(training), 0
        batch = positions[order[cursor:cursor + args.batch]]
        cursor += args.batch
        planes, actions = encode(batch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps, args.lr)
        losses = imitation_step(net, optimizer, (
            torch.from_numpy(planes).to(device),
            torch.from_numpy(actions).to(device),
            torch.from_numpy(values(batch, games)).to(device),
        ), args.value_weight)
        window += (losses.policy, losses.value, 1)

        if step % every == 0 or step == steps:
            elapsed = time.perf_counter() - started
            held_out = validation.measure(net, device)
            entry = {"step": step, "positions": step * args.batch,
                     "train_policy": window[0] / window[2], "train_value": window[1] / window[2],
                     **held_out, "seconds": round(elapsed)}
            log.append(entry)
            window[:] = 0
            rate = step * args.batch / elapsed
            print(f"{step / steps:4.0%}  {entry['positions'] / 1e6:5.1f} M  "
                  f"train policy {entry['train_policy']:.3f} value {entry['train_value']:.3f} | "
                  f"held out: accuracy {held_out['accuracy']:.1%} (opening "
                  f"{held_out['accuracy_opening']:.0%}, middle {held_out['accuracy_middlegame']:.0%}, "
                  f"end {held_out['accuracy_endgame']:.0%})  policy {held_out['policy_loss']:.3f} "
                  f"value {held_out['value_loss']:.3f} sign {held_out['value_sign']:.0%} | "
                  f"{rate:,.0f} pos/s, {(steps - step) * args.batch / rate / 60:.0f} min left",
                  flush=True)
            save(path, net, optimizer, config, args.stage, months, log)

    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
