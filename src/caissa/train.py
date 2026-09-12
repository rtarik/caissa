"""The training step: turning search's discoveries back into network weights.

The loss has two terms and they are doing quite different jobs.

The **value** term is ordinary regression: predict how the game ended.

The **policy** term is cross-entropy against search's full visit distribution,
and the word *full* is the important one. The target is not "search chose move
four"; it is "search spent 46% of its visits on move four, 31% on move two, and
the rest scattered". A one-hot label would throw away everything search learned
about the alternatives, including how close the decision was. Training on the
whole distribution transfers the search's *uncertainty* along with its
preference, which is what lets the network reproduce search's judgement rather
than just its conclusions.

One consequence worth expecting: cross-entropy against a soft target does not
bottom out at zero. Its minimum is the entropy of the target itself, so a policy
loss that settles around 1.0 for a seven-action game is not stalled - it is close
to the floor. Watching it for a fall to zero will only cause confusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from caissa.network import PolicyValueNet


@dataclass
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 2e-3
    #: L2 penalty, applied through the optimiser. AlphaZero used 1e-4.
    weight_decay: float = 1e-4


@dataclass
class Losses:
    total: float
    policy: float
    value: float


def alphazero_loss(policy_logits: torch.Tensor, value: torch.Tensor,
                   target_policy: torch.Tensor, target_value: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(total, policy, value)`` losses for one batch.

    Illegal moves carry zero probability in the target, so they contribute
    nothing directly; they still sit in the softmax denominator, which gently
    pushes their logits down. That is harmless and mildly useful, and it is what
    AlphaZero does.
    """
    policy_loss = -(target_policy * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_value)
    # Equal weighting. The two terms are on comparable scales here; a value term
    # that dominates produces a network that judges positions well but proposes
    # moves badly, which search then has to compensate for at every node.
    return policy_loss + value_loss, policy_loss, value_loss


def make_optimizer(net: PolicyValueNet, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def train_step(net: PolicyValueNet, optimizer: torch.optim.Optimizer,
               batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Losses:
    """One gradient step. Leaves the network in training mode."""
    positions, target_policy, target_value = batch

    net.train()
    optimizer.zero_grad(set_to_none=True)
    policy_logits, value = net(positions)
    total, policy_loss, value_loss = alphazero_loss(
        policy_logits, value, target_policy, target_value
    )
    total.backward()
    optimizer.step()

    return Losses(total.item(), policy_loss.item(), value_loss.item())
