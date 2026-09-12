"""The policy and value network.

One network answers two questions about a position: *which moves look worth
considering* (the policy head) and *who is winning* (the value head). They sit on
top of a shared trunk, because both questions depend on the same understanding -
what is attacked, what is connected, whose threats arrive first. Learning that
understanding once and using it twice is cheaper than learning it twice, and the
two tasks regularise each other: a trunk that lets the value head cheat with some
brittle shortcut will usually hurt the policy head, so the shortcut does not
survive.

The network is deliberately not very good on its own. It has one forward pass in
which to form an opinion, whereas search gets hundreds of evaluations. Its job is
not to be right; it is to be right *often enough to point search in a useful
direction*, and to absorb what search discovers so that next time it points
better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NetworkConfig:
    #: Residual blocks in the trunk. Depth buys tactical sight; AlphaZero used
    #: 19-20 for chess and Go. Small games need far less.
    blocks: int = 4
    #: Trunk width. Together with ``blocks`` this is the main capacity dial.
    channels: int = 64
    #: Width of the 1x1 projection at the start of each head. Heads are kept
    #: narrow so almost all parameters live in the shared trunk.
    policy_channels: int = 32
    value_channels: int = 8
    #: Hidden units in the value head's MLP.
    value_hidden: int = 128


class ResidualBlock(nn.Module):
    """Two convolutions with a skip connection around them.

    The skip is what makes depth usable. Each block only has to learn a
    *correction* to its input rather than a whole new representation, so
    gradients reach the early layers intact and adding blocks does not make
    training harder. Without it, a deep stack tends to be worse than a shallow
    one - the failure that residual networks were invented to fix.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """Shared trunk, one policy head and one value head."""

    def __init__(
        self,
        input_planes: int,
        board_shape: tuple[int, int],
        action_size: int,
        config: NetworkConfig | None = None,
    ):
        super().__init__()
        self.config = config = config or NetworkConfig()
        self.input_planes = input_planes
        self.board_shape = board_shape
        self.action_size = action_size

        height, width = board_shape
        squares = height * width
        channels = config.channels

        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(config.blocks))
        )

        # Policy head. This one emits raw logits and nothing else - masking to
        # legal moves happens in the evaluator, because legality is a rule of the
        # game rather than something the network should have to learn.
        #
        # The head is dense because Connect 4's seven actions are *columns*, and
        # do not correspond to its forty-two cells. Where an action space does
        # map onto board squares - chess's 4672 being 64 squares x 73 move types
        # - a convolutional head emitting those planes directly is roughly
        # sixteen times smaller and slightly faster. See PLAN.md.
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, config.policy_channels, 1, bias=False),
            nn.BatchNorm2d(config.policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(config.policy_channels * squares, action_size),
        )

        # Value head. The tanh is not decoration: it bounds the output to
        # [-1, 1], which is exactly the range of the thing being predicted, since
        # game outcomes are -1, 0 or +1 under the mover-relative convention. A
        # network that cannot express an impossible value cannot waste capacity
        # learning not to.
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, config.value_channels, 1, bias=False),
            nn.BatchNorm2d(config.value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(config.value_channels * squares, config.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

    @classmethod
    def for_game(cls, game, config: NetworkConfig | None = None) -> PolicyValueNet:
        return cls(game.input_planes, game.board_shape, game.action_size, config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits, value)`` for a batch of encoded positions.

        ``value`` has shape ``(batch,)``, not ``(batch, 1)``, so callers do not
        have to remember to squeeze it.
        """
        features = self.trunk(self.stem(x))
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def best_device() -> torch.device:
    """Apple GPU when available, otherwise CPU.

    **Not** the right choice for single-position inference, which is what
    :class:`NetworkEvaluator` does today: moving one small tensor to the GPU and
    back costs more than the computation saves. Measured on an M4 Max with the
    default Connect 4 network::

        single position    cpu  1,046/s     mps    662/s
        batch of 512       cpu 14,199/s     mps 281,835/s

    So the evaluator defaults to CPU, and this function is for the batched work
    in training and parallel self-play. The 426x gap between single-position and
    batched evaluation on the GPU is the reason Phase 2 must run many games at
    once and evaluate their leaves together, rather than one position at a time.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NetworkEvaluator:
    """Adapts a :class:`PolicyValueNet` to the search's evaluator interface.

    This is the join between the two halves of AlphaZero. Search asks for priors
    and a value; the network supplies them; nothing else about the search
    changes. Swapping this for :class:`~caissa.evaluator.UniformEvaluator` turns
    the agent back into pure search, which is how the two are told apart when
    something goes wrong.
    """

    def __init__(self, net: PolicyValueNet, device: torch.device | None = None):
        # Default to wherever the network already lives. `Module.to()` moves
        # parameters *in place*, so a default of CPU would silently drag a
        # training network off the GPU the moment anything evaluated a position
        # with it - detaching it from its optimiser's state in the process.
        # Moving someone else's model is not this class's business.
        self.device = device or next(net.parameters()).device
        self.net = net.to(self.device) if device is not None else net
        # Evaluation mode, always. In training mode the batch-normalisation
        # layers normalise using the statistics of whatever batch they are handed
        # - here a single position - instead of the running averages learned
        # during training, and they *update* those running averages as a side
        # effect. Both are wrong: the evaluations become noise, and the act of
        # searching quietly corrupts the network. This single line is one of the
        # most common bugs in AlphaZero implementations.
        self.net.eval()

    @torch.no_grad()
    def evaluate(self, game, state) -> tuple[np.ndarray, float]:
        # Asserted on every call, not just at construction. The evaluator holds a
        # *reference* to the network, and the training step puts that same object
        # back into training mode. Without this line, the first self-play game
        # after the first gradient step would begin quietly corrupting the
        # batch-norm statistics - a bug that only appears once self-play and
        # training share a network, which is to say, in the real loop and never
        # in a unit test of either half alone.
        self.net.eval()
        encoded = torch.from_numpy(game.encode(state)).unsqueeze(0).to(self.device)
        logits, value = self.net(encoded)

        logits = logits[0].float().cpu().numpy()
        legal = game.legal_actions(state)

        # Mask, then softmax. Illegal moves are sent to -inf so they receive
        # exactly zero probability rather than merely a small one; search must
        # never be able to reach them at all.
        logits = np.where(legal, logits, -np.inf)
        logits -= logits.max()  # stability: keeps exp() from overflowing
        priors = np.exp(logits)
        priors /= priors.sum()

        return priors.astype(np.float32), float(value.item())
