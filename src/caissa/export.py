"""Getting a trained network out of PyTorch and into the browser.

ONNX is an interchange format: a frozen graph of operators plus weights, which
`onnxruntime-web` can execute in WebAssembly with no Python anywhere. The export
has to be checked rather than assumed. A graph that loads and produces
plausible-looking numbers, but not the *same* numbers, gives a browser opponent
that is subtly weaker than the one that was measured - and nothing about it looks
wrong from either side.

So :func:`verify_onnx` re-runs both and compares. It is the same discipline as
the parallel self-play equivalence test: a second implementation of something
already working must be proved identical, not assumed to be.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from caissa.network import PolicyValueNet

#: Old enough to be supported everywhere onnxruntime-web runs, new enough to
#: cover the operators a residual network uses.
OPSET = 17


def export_onnx(net: PolicyValueNet, path: str | Path, game) -> Path:
    """Write ``net`` to an ONNX file, with a dynamic batch dimension.

    The batch axis is left dynamic so the browser can evaluate several search
    leaves in one call later without re-exporting. Everything else is fixed: the
    board shape and action count are properties of the game, not of the caller.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    net = net.eval().cpu()
    example = torch.zeros(1, game.input_planes, *game.board_shape)

    torch.onnx.export(
        net,
        example,
        str(path),
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={"board": {0: "batch"},
                      "policy": {0: "batch"},
                      "value": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    return path


def write_manifest(path: str | Path, game, net: PolicyValueNet, **extra) -> Path:
    """Describe the model so the browser does not have to hard-code its shape.

    The web code reads this instead of carrying a duplicate of the board size and
    action count. Two copies of a constant is one copy too many, and the one that
    goes stale is always the one you are not looking at.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "game": game.name,
        "inputPlanes": game.input_planes,
        "boardShape": list(game.board_shape),
        "actionSize": game.action_size,
        "parameters": net.parameter_count(),
        **extra,
    }, indent=2) + "\n")
    return path


def write_index(directory: str | Path) -> Path:
    """List the exported models, so the web app can show what actually exists.

    The page offers every game on the ladder, but only some have been trained at
    any given moment. Deriving that list from the directory rather than hard-coding
    it in TypeScript means the menu cannot claim a game that is not there, and
    cannot hide one that is.
    """
    directory = Path(directory)
    entries = []
    for manifest in sorted(directory.glob("*.json")):
        if manifest.name == "index.json":
            continue
        data = json.loads(manifest.read_text())
        model = directory / f"{data['game']}.onnx"
        if not model.exists():
            continue
        entries.append({
            "game": data["game"],
            "generation": data.get("generation"),
            "parameters": data.get("parameters"),
        })

    path = directory / "index.json"
    path.write_text(json.dumps({"models": entries}, indent=2) + "\n")
    return path


def verify_onnx(net: PolicyValueNet, path: str | Path, game, positions: int = 64,
                seed: int = 0) -> dict[str, float]:
    """Compare the exported graph against PyTorch on random positions.

    Comparison is on **probabilities and decisions**, not raw logits, because
    that is what the rest of the system consumes: search receives a masked
    softmax and a value, and never sees a logit. Logits are unbounded, so an
    absolute tolerance on them means nothing - a well-trained network can emit
    values in the thousands, where a difference of 20 is ordinary floating-point
    accumulation and changes no decision whatsoever. Choosing the tolerance
    before choosing the quantity is how a working export gets rejected, or a
    broken one accepted.

    Returns the worst disagreement in softmax probability and in value, plus the
    fraction of positions where the two agree on the best move.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(positions):
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 20))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        boards.append(game.encode(state))
    batch = np.stack(boards).astype(np.float32)

    net = net.eval().cpu()
    with torch.no_grad():
        torch_policy, torch_value = net(torch.from_numpy(batch))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_policy, onnx_value = session.run(None, {"board": batch})

    torch_probabilities = torch.softmax(torch_policy, dim=1).numpy()
    onnx_probabilities = torch.softmax(torch.from_numpy(onnx_policy), dim=1).numpy()

    return {
        "policy": float(np.abs(torch_probabilities - onnx_probabilities).max()),
        "value": float(np.abs(torch_value.numpy() - onnx_value.reshape(-1)).max()),
        "agreement": float(
            (torch_policy.numpy().argmax(1) == onnx_policy.argmax(1)).mean()
        ),
    }
