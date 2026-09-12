"""Monte Carlo tree search, AlphaZero style.

The idea in one paragraph: the network has an opinion about which moves are good
(its priors) and how positions turn out (its value). Those opinions are cheap but
mediocre. Search spends compute to improve on them - it explores the moves the
network likes, discovers where that optimism was unfounded, and ends up
distributing its visits more sensibly than the raw priors did. The improved
distribution is then used both to *play* and, crucially, as the **training
target** that drags the network toward what search discovered.

That last point is the whole algorithm. Search is a *policy improvement
operator*: given a policy, it produces a better one. Training turns the better
one back into network weights, and the loop repeats. Everything in this file
exists to make that one step work.

A single simulation has four phases:

1. **Select** - walk down from the root, at each node taking the move that
   maximises PUCT, until reaching a node that has not been expanded yet.
2. **Expand** - ask the evaluator for that leaf's priors and value, and create a
   child for every legal move.
3. **Back up** - push the value back along the path taken, flipping its sign at
   every ply.
4. Repeat, so the tree deepens where it matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from caissa.evaluator import Evaluator


@dataclass
class MCTSConfig:
    #: Network evaluations per move. The main strength/time dial.
    simulations: int = 200

    #: Exploration weight in PUCT. Low values trust the priors and search
    #: narrowly; high values spread visits out. Around 1-2 is typical for small
    #: games. Worth tuning per game, but not worth agonising over.
    c_puct: float = 1.5

    #: Dirichlet noise mixed into the root priors, to force exploration of moves
    #: the network currently dislikes. ``alpha`` scales roughly as
    #: ``10 / average number of legal moves`` - AlphaZero used 0.3 for chess
    #: (~35 moves) and 0.03 for Go (~250). Connect 4 has 7, hence 1.0.
    dirichlet_alpha: float = 1.0
    #: How much of the root prior is replaced by noise. AlphaZero used 0.25.
    dirichlet_epsilon: float = 0.25


class Node:
    """One position in the search tree.

    Statistics are stored per node rather than per edge, which is equivalent and
    reads more easily. ``value_sum`` accumulates outcomes **from the perspective
    of the player to move at this node**, following the project convention.
    """

    __slots__ = ("prior", "visit_count", "value_sum", "children", "state")

    def __init__(self, prior: float, state=None):
        #: The evaluator's probability for the move that led here.
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}
        self.state = state

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        """Mean outcome over all simulations through this node, for its mover."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, game, evaluator: Evaluator, config: MCTSConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.game = game
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------ search

    def search(self, state, add_noise: bool = True) -> Node:
        """Build a search tree from ``state`` and return its root.

        ``add_noise`` should be True while generating self-play data and False
        when actually trying to play well - see :meth:`_add_dirichlet_noise`.
        """
        if self.game.terminal_value(state) is not None:
            raise ValueError("cannot search from a finished game")

        root = Node(prior=0.0, state=state)

        # Expand the root first, so there is something to add noise to.
        self._backup([root], self._evaluate(root))
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.config.simulations - 1):
            path = self._select(root)
            self._backup(path, self._evaluate(path[-1]))

        return root

    def _select(self, root: Node) -> list[Node]:
        """Descend by PUCT until reaching an unexpanded or terminal node."""
        path = [root]
        node = root
        while node.expanded:
            node = self._best_child(node)
            path.append(node)
        return path

    def _best_child(self, node: Node) -> Node:
        """Pick the child maximising the PUCT score.

        PUCT balances two competing pressures::

            score(a) = Q(a)  +  c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
                       \\____/    \\_______________________________________/
                     exploit                    explore

        **Q(a)** is what we have actually learned by searching: the mean result
        of simulations through that child. Note the minus sign below. A child's
        statistics are recorded from *its own* mover's perspective, and that is
        the opponent of the player choosing here, so a result that is good for
        them is bad for us. Dropping this negation produces an agent that
        confidently walks into losing lines, and nothing else about the run will
        look wrong.

        **The second term** is optimism. It is large when the evaluator liked the
        move (``P``) and shrinks as the move accumulates visits (``1 + N``), so
        promising-but-unexplored moves get pulled in, and the pull decays once
        they have been given a fair hearing. The ``sqrt(N_parent)`` factor keeps
        exploration alive as the node is visited more, rather than letting an
        early lucky result monopolise the search.

        Unvisited children have ``Q = 0``, which in a mover-relative convention
        means "a draw" - neutral, neither attractive nor repellent. They are
        therefore selected on prior alone, which is exactly what the network's
        opinion is for.
        """
        sqrt_parent_visits = math.sqrt(node.visit_count)

        best_score = -math.inf
        best_child = None
        for child in node.children.values():
            exploit = -child.value() if child.visit_count else 0.0
            explore = (
                self.config.c_puct
                * child.prior
                * sqrt_parent_visits
                / (1 + child.visit_count)
            )
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None, "expanded node with no children"
        return best_child

    def _evaluate(self, node: Node) -> float:
        """Return the value of ``node``, expanding it if the game continues.

        A finished position needs no network: the rules already give the exact
        answer, and it would be strictly worse to use an estimate. Terminal nodes
        are therefore never expanded, which also stops the descent in
        :meth:`_select`.
        """
        terminal = self.game.terminal_value(node.state)
        if terminal is not None:
            return terminal

        priors, value = self.evaluator.evaluate(self.game, node.state)
        for action in np.flatnonzero(self.game.legal_actions(node.state)):
            action = int(action)
            node.children[action] = Node(
                prior=float(priors[action]),
                state=self.game.apply(node.state, action),
            )
        return value

    def _backup(self, path: list[Node], value: float) -> None:
        """Record ``value`` along the path from leaf to root, alternating sign.

        ``value`` arrives from the leaf's mover's point of view. Its parent is
        the opponent, so the same outcome is worth the negation to them, and so
        on alternately up the tree. This is the same convention as
        ``terminal_value`` and is the direct reason every node's statistics can
        be read without tracking whose turn it is.
        """
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _add_dirichlet_noise(self, root: Node) -> None:
        """Mix random noise into the root priors, to force exploration.

        Without this the algorithm is self-reinforcing in the worst way: the
        network proposes, the search explores what was proposed, the resulting
        games train the network to propose the same things more strongly. A move
        the network dismisses early is never played, never appears in training
        data, and so is never reconsidered - the agent locks into a narrow
        repertoire and stops improving, while every metric it reports about
        itself continues to look healthy.

        Noise at the **root only** is what breaks that loop. Deeper nodes are
        left alone because the goal is to diversify the *games actually played*,
        not to degrade the search's judgement inside a line.
        """
        actions = list(root.children)
        noise = self.rng.dirichlet([self.config.dirichlet_alpha] * len(actions))
        epsilon = self.config.dirichlet_epsilon
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * sample

    # ------------------------------------------------------------------ output

    def policy(self, root: Node, temperature: float = 1.0) -> np.ndarray:
        """Turn the search tree into a move distribution over ``action_size``.

        **Visit counts are the output, not Q values and not the network's own
        priors.** This is easy to skim past and is the core of AlphaZero. A move
        accumulates visits only by repeatedly winning the PUCT argument against
        its siblings, so the visit distribution aggregates every simulation's
        worth of evidence. A raw Q value, by contrast, might rest on a single
        lucky rollout: attractive, but unsupported.

        The resulting distribution is strictly better than the priors the
        network supplied - that is what the search bought - which is why it
        serves as the training target. The network is being taught to predict,
        in one forward pass, where search *would have* spent its visits.

        ``temperature`` controls how that distribution is sharpened::

            pi(a) proportional to N(a) ** (1 / temperature)

        At 1.0 moves are played in proportion to their visits, which keeps
        self-play games varied so the network sees a wide spread of positions.
        Approaching 0 it becomes greedy - always the most-visited move - which is
        how to play when the aim is to win rather than to learn. Typical practice
        is temperature 1 for the opening handful of moves and then effectively 0.
        """
        counts = np.zeros(self.game.action_size, dtype=np.float64)
        for action, child in root.children.items():
            counts[action] = child.visit_count

        if temperature == 0:
            policy = np.zeros_like(counts)
            policy[int(counts.argmax())] = 1.0
            return policy

        # Scaling by the maximum first keeps this from overflowing at small
        # temperatures, where raising raw visit counts to a large power would
        # otherwise run out of float range.
        weights = (counts / counts.max()) ** (1.0 / temperature)
        return weights / weights.sum()

    def run(self, state, temperature: float = 1.0, add_noise: bool = True
            ) -> tuple[np.ndarray, float]:
        """Search ``state`` and return ``(policy, root value)``.

        The root value is the search's own estimate of the position for the
        player to move - a better estimate than the network's, for the same
        reason the visit distribution is better than the priors.
        """
        root = self.search(state, add_noise=add_noise)
        return self.policy(root, temperature), root.value()
