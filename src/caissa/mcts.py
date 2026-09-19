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
2. **Expand** - ask the evaluator for that leaf's priors and value, and record a
   child for every legal move. A child's position is only built the first time a
   simulation goes there, because most never get a visit: in chess a simulation
   records ~31 children and visits one.
3. **Back up** - push the value back along the path taken, flipping its sign
   wherever the turn passed between one position and the next.
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
    #: the network currently dislikes.
    #:
    #: ``None`` derives it from the position, as ``10 / legal moves`` - the rule
    #: AlphaZero used to pick 0.3 for chess (~35 moves) and 0.03 for Go (~250).
    #: The scaling is not optional: Dirichlet noise is *concentrated* for small
    #: alpha and *near-uniform* for large, and the point of it is to probe a few
    #: specific alternatives properly rather than dilute the prior across
    #: everything. A fixed 1.0 is right for Connect 4's seven moves and wrong by
    #: a factor of eight for Gomoku's eighty, where it smears a quarter of the
    #: root prior evenly over moves that are mostly bad.
    dirichlet_alpha: float | None = None
    #: How much of the root prior is replaced by noise. AlphaZero used 0.25.
    dirichlet_epsilon: float = 0.25


class Node:
    """One position in the search tree.

    Statistics are stored per node rather than per edge, which is equivalent and
    reads more easily. ``value_sum`` accumulates outcomes **from the perspective
    of the player to move at this node**, following the project convention.
    """

    __slots__ = ("prior", "visit_count", "value_sum", "children", "state", "flip", "action")

    def __init__(self, prior: float, state=None, flip: bool = True, action: int | None = None):
        #: The evaluator's probability for the move that led here.
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}
        #: The position, or ``None`` until the search first comes this way.
        self.state = state
        #: The move from the parent that leads here, to build ``state`` from.
        self.action = action
        #: Whether the player to move here is a *different* player from the one
        #: to move in the parent. True for every move of a game whose turns
        #: alternate; False after a Dots & Boxes line that closed a box and earned
        #: a bonus move. Every sign flip in the search is conditional on this,
        #: because a value only changes sign when it changes hands. Known once
        #: ``state`` is built - and never read before, since an unvisited child
        #: has no value to flip.
        self.flip = flip

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
            child = self._best_child(node)
            if child.state is None:
                self._build(node, child)
            node = child
            path.append(node)
        return path

    def _build(self, parent: Node, child: Node) -> None:
        """Create a child's position on its first visit, and note whether the turn passed.

        Expansion only records a prior per move. Building every child up front
        costs a call to the rules per legal move, and in chess that was 31% of a
        whole search step - spent on positions that, 97 times in 100, were
        never visited. Building on demand does the same search, in the same
        order, with the same result; it just skips the work nobody looks at.
        """
        child.state = self.game.apply(parent.state, child.action)
        child.flip = self.game.to_play(child.state) != self.game.to_play(parent.state)

    def _best_child(self, node: Node) -> Node:
        """Pick the child maximising the PUCT score.

        PUCT balances two competing pressures::

            score(a) = Q(a)  +  c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
                       \\____/    \\_______________________________________/
                     exploit                    explore

        **Q(a)** is what we have actually learned by searching: the mean result
        of simulations through that child. Note the conditional negation below.
        A child's statistics are recorded from *its own* mover's perspective.
        Usually that is the opponent of the player choosing here, so a result
        that is good for them is bad for us and has to be negated. Dropping that
        negation produces an agent that confidently walks into losing lines, and
        nothing else about the run will look wrong.

        But not always. After a Dots & Boxes line that closes a box, the child's
        mover is the *same* player, and negating their result would teach the
        search that earning a bonus move is a disaster.

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
            if child.visit_count == 0:
                exploit = 0.0
            elif child.flip:
                exploit = -child.value()  # the opponent's result, so ours is its negation
            else:
                exploit = child.value()   # a bonus move: still our own result
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
            # A prior and a move; the position waits for a visit (see _build).
            node.children[action] = Node(prior=float(priors[action]), action=action)
        return value

    def _backup(self, path: list[Node], value: float) -> None:
        """Record ``value`` along the path from leaf to root.

        ``value`` arrives from the leaf's mover's point of view. Each step up the
        tree it is negated if - and only if - the turn passed on the way down,
        because the same outcome is worth its negation to an opponent and exactly
        the same to the player themselves. In a strictly alternating game that is
        every step, which is the familiar alternation. In Dots & Boxes a run of
        box-closing lines is a run of steps where the sign must *not* change.
        """
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            if node.flip:
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
        # Derived from this position's branching factor unless pinned, so a game
        # with a large action space is not handed near-uniform noise.
        alpha = self.config.dirichlet_alpha
        if alpha is None:
            alpha = max(0.03, 10.0 / len(actions))
        noise = self.rng.dirichlet([alpha] * len(actions))
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
