"""Tests for the ONNX export.

The point of every test here is that the exported graph computes the *same*
function as the network it came from. A browser opponent that is quietly a
different network from the one that was benchmarked would look fine from both
sides and be wrong in neither an obvious nor a detectable way.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from caissa.export import export_onnx, verify_onnx, write_manifest
from caissa.games.connect4 import Connect4
from caissa.network import NetworkConfig, PolicyValueNet

ort = pytest.importorskip("onnxruntime")


@pytest.fixture
def game() -> Connect4:
    return Connect4()


@pytest.fixture
def net(game) -> PolicyValueNet:
    torch.manual_seed(0)
    return PolicyValueNet.for_game(game, NetworkConfig(blocks=2, channels=16))


def test_export_writes_a_loadable_graph(game, net, tmp_path):
    path = export_onnx(net, tmp_path / "net.onnx", game)
    assert path.exists() and path.stat().st_size > 0
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["board"]
    assert [o.name for o in session.get_outputs()] == ["policy", "value"]


def test_exported_graph_matches_pytorch(game, net, tmp_path):
    """The check that makes the rest of Phase 4 trustworthy."""
    path = export_onnx(net, tmp_path / "net.onnx", game)
    difference = verify_onnx(net, path, game, positions=64)
    assert difference["policy"] < 1e-5, difference
    assert difference["value"] < 1e-5, difference
    assert difference["agreement"] == 1.0, difference


def test_weights_away_from_their_initial_scale_survive_too(game, net, tmp_path):
    """Untrained weights are small and forgiving; trained ones need not be.

    Moved well off the initial scale, but not so far that the policy head
    saturates - at which point ``tanh`` pins every value to +/-1 and the value
    comparison passes by being blind rather than by being right.
    """
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.3)

    path = export_onnx(net, tmp_path / "net.onnx", game)
    difference = verify_onnx(net, path, game, positions=64)
    assert difference["policy"] < 1e-4, difference
    assert difference["value"] < 1e-4, difference
    assert difference["agreement"] == 1.0, difference


def test_the_value_comparison_is_not_saturated(game, net, tmp_path):
    """Guards the test above: a saturated tanh would make its value check vacuous.

    With every output pinned to +/-1, torch and ONNX agree perfectly no matter
    what the trunk did. The comparison has to happen where the function is still
    sensitive to its input.
    """
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.3)

    rng = np.random.default_rng(0)
    states = []
    for _ in range(32):
        state = game.initial_state()
        for _ in range(int(rng.integers(1, 12))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        states.append(state)

    net.eval()
    with torch.no_grad():
        _, value = net(torch.from_numpy(np.stack([game.encode(s) for s in states])))
    assert (value.abs() < 0.9999).any(), "tanh saturated; the value check proves nothing"


def test_batch_dimension_is_dynamic(game, net, tmp_path):
    """Exported at batch 1, but usable at any batch - so the browser can batch
    search leaves later without a re-export."""
    path = export_onnx(net, tmp_path / "net.onnx", game)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for size in (1, 3, 32):
        board = np.zeros((size, game.input_planes, *game.board_shape), dtype=np.float32)
        policy, value = session.run(None, {"board": board})
        assert policy.shape == (size, game.action_size)
        assert value.reshape(-1).shape == (size,)


def test_export_leaves_the_network_in_eval_mode(game, net, tmp_path):
    """Exporting in training mode would bake batch statistics into the graph."""
    net.train()
    export_onnx(net, tmp_path / "net.onnx", game)
    assert not net.training


def test_manifest_describes_the_model(game, net, tmp_path):
    path = write_manifest(tmp_path / "model.json", game, net, generation=7)
    manifest = json.loads(path.read_text())

    assert manifest["game"] == "connect4"
    assert manifest["boardShape"] == [6, 7]
    assert manifest["actionSize"] == 7
    assert manifest["inputPlanes"] == 2
    assert manifest["parameters"] == net.parameter_count()
    assert manifest["generation"] == 7


def test_verification_detects_a_mismatched_graph(game, net, tmp_path):
    """The detector must be shown to detect.

    Without this, ``verify_onnx`` comparing PyTorch against *itself* would pass
    every test in this file while checking nothing at all - the export could be
    arbitrarily broken and the verification would still report zero difference.
    A check that cannot fail is not a check.
    """
    torch.manual_seed(99)
    other = PolicyValueNet.for_game(game, NetworkConfig(blocks=2, channels=16))
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.5)

    path = export_onnx(other, tmp_path / "other.onnx", game)
    difference = verify_onnx(net, path, game, positions=32)

    assert difference["policy"] > 1e-3, "a different network was reported as identical"
    assert difference["agreement"] < 1.0
