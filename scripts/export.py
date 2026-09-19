"""Export a checkpoint for the browser.

    python scripts/export.py models/connect4-latest.pt

Writes the ONNX graph and a manifest into the web app's public directory, then
verifies the exported graph reproduces PyTorch's decisions. A graph that loads
and returns plausible numbers - but not the same ones - gives a browser opponent
quietly weaker than the one that was benchmarked, and nothing about it looks
wrong from either side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from caissa.export import export_onnx, verify_onnx, write_index, write_manifest
from caissa.games import GAMES
from caissa.network import NetworkConfig, PolicyValueNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("web/public/models"))
    parser.add_argument("--simulations", type=int, default=200,
                        help="default search budget the web app should use")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--name", default=None,
                        help="file name, without extension; defaults to the game's. Give each "
                             "chess stage its own, so the page can offer them side by side")
    parser.add_argument("--label", default=None, help="how the page names this network")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    game = GAMES[checkpoint["game"]]()
    net = PolicyValueNet.for_game(game, NetworkConfig(**checkpoint["config"]["network"]))
    net.load_state_dict(checkpoint["network"])

    name = args.name or game.name
    labelled = {"label": args.label} if args.label else {}
    model = export_onnx(net, args.out / f"{name}.onnx", game)
    write_manifest(args.out / f"{name}.json", game, net,
                   generation=checkpoint["iteration"], simulations=args.simulations, **labelled)

    index = write_index(args.out)
    difference = verify_onnx(net, model, game, positions=128)
    print(f"{model}  ({model.stat().st_size / 1024:.0f} KB, "
          f"{net.parameter_count():,} parameters, generation {checkpoint['iteration']})")
    print(f"  max softmax difference {difference['policy']:.2e}, "
          f"value {difference['value']:.2e}, "
          f"best-move agreement {difference['agreement']:.1%}")

    print(f"  {index} lists "
          f"{len(json.loads(index.read_text())['models'])} trained model(s)")

    if difference["policy"] > args.tolerance or difference["agreement"] < 1.0:
        raise SystemExit("export does not reproduce PyTorch; refusing to ship it")


if __name__ == "__main__":
    main()
