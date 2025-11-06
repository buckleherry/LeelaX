# leelax/mcts/puct.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import chess
import torch

from leelax.env.encode import state_to_tensor
from leelax.env.policy_index import move_to_index, legal_policy_mask, index_to_move


@dataclass
class MCTSNode:
    board: chess.Board
    parent: Optional["MCTSNode"] = None
    parent_action: Optional[int] = None  # policy index that led here
    # statistics
    n_visits: int = 0
    value_sum: float = 0.0
    # action -> child node
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    # priors (from network policy), shape (4672,)
    priors: Optional[np.ndarray] = None
    # legal mask cached
    legal_mask: Optional[np.ndarray] = None
    # terminal?
    is_expanded: bool = False

    def value(self) -> float:
        if self.n_visits == 0:
            return 0.0
        return self.value_sum / self.n_visits

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def add_child(self, action_idx: int, child: "MCTSNode") -> None:
        self.children[action_idx] = child


class PUCT:
    """PUCT-based MCTS for AlphaZero-style chess.

    Usage:
        mcts = PUCT(network_fn)
        policy, action_idx = mcts.run(root_board)
    """

    def __init__(
        self,
        network_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        n_simulations: int = 64,
        cpuct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            network_fn: callable that takes [1, 24, 8, 8] tensor and returns (policy_logits[1,4672], value[1,1])
            n_simulations: how many tree walks per move
            cpuct: exploration constant
            dirichlet_alpha: root noise alpha
            dirichlet_eps: mixing factor for root noise
        """
        self.network_fn = network_fn
        self.n_simulations = n_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, board: chess.Board, add_dirichlet: bool = True) -> Tuple[np.ndarray, int]:
        """Run MCTS from given board, return visit-based policy and chosen action index."""
        root = MCTSNode(board=board.copy())
        self._expand(root)
        if add_dirichlet:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_simulations):
            node, path = self._select(root)
            value = self._evaluate(node)
            self._backup(path, value)

        # build final policy from root visits
        policy = np.zeros_like(root.priors, dtype=np.float32)
        for action_idx, child in root.children.items():
            policy[action_idx] = child.n_visits
        if policy.sum() > 0:
            policy = policy / policy.sum()

        # pick action with highest visit count
        best_action = int(np.argmax(policy))
        return policy, best_action

    # ------------------------------------------------------------------
    # core steps
    # ------------------------------------------------------------------
    def _select(self, root: MCTSNode):
        """Select leaf by greedy PUCT."""
        node = root
        path = [node]

        while not node.is_leaf():
            # pick action with max UCB
            best_score = -1e9
            best_action = None
            best_child = None

            sqrt_sum = np.sqrt(node.n_visits + 1e-8)
            for action_idx, child in node.children.items():
                # Q
                q = child.value()
                # U
                p = node.priors[action_idx]
                u = self.cpuct * p * sqrt_sum / (1 + child.n_visits)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_action = action_idx
                    best_child = child

            assert best_child is not None
            node = best_child
            path.append(node)

        return node, path

    def _expand(self, node: MCTSNode) -> None:
        """Expand node using network: set priors and create children placeholders."""
        board = node.board
        # evaluate network
        state = state_to_tensor(board).unsqueeze(0).to(self.device)  # [1,24,8,8]
        with torch.no_grad():
            policy_logits, value = self.network_fn(state)
        policy_logits = policy_logits.detach().cpu()
        value = value.item()

        legal_mask = legal_policy_mask(board)  # (4672,)
        # masked softmax in numpy
        logit_np = policy_logits.numpy().reshape(-1)
        # mask illegal to -inf
        logit_np[legal_mask == 0] = -1e9
        # softmax
        max_logit = np.max(logit_np)
        exp = np.exp(logit_np - max_logit)
        exp[legal_mask == 0] = 0.0
        sum_exp = exp.sum()
        if sum_exp > 0:
            priors = exp / sum_exp
        else:
            priors = legal_mask / legal_mask.sum()  # fallback uniform over legal

        node.priors = priors
        node.legal_mask = legal_mask
        node.is_expanded = True

        # create children lazily on demand (in select we only follow existing)
        # but we can also pre-create: for each legal move create a child placeholder
        for mv in board.legal_moves:
            a_idx = move_to_index(mv)
            child_board = board.copy()
            child_board.push(mv)
            child_node = MCTSNode(board=child_board, parent=node, parent_action=a_idx)
            node.add_child(a_idx, child_node)

        # store network value on node (will be used in backup)
        node.value_from_net = value  # type: ignore[attr-defined]

    def _evaluate(self, node: MCTSNode) -> float:
        board = node.board
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0
            elif result == "0-1":
                return -1.0
            else:
                return 0.0

        # if node was just selected but never expanded (can happen below root), expand now
        if not node.is_expanded:
            self._expand(node)

        return node.value_from_net  # type: ignore[attr-defined]


    def _backup(self, path, value: float) -> None:
        """Propagate value back up the path, alternating signs because players alternate."""
        # value is from the perspective of the node at the end of the path
        for i, node in enumerate(reversed(path)):
            node.n_visits += 1
            # alternate perspective:
            # last node (leaf) gets +v,
            # parent sees -v, parent of parent sees +v, ...
            sign = 1.0 if i % 2 == 0 else -1.0
            node.value_sum += sign * value

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Add exploration noise to the root priors."""
        legal = root.legal_mask
        assert legal is not None
        legal_indices = np.where(legal == 1)[0]
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_indices))

        priors = root.priors.copy()
        priors[legal_indices] = (
            (1 - self.dirichlet_eps) * priors[legal_indices]
            + self.dirichlet_eps * noise
        )
        root.priors = priors

