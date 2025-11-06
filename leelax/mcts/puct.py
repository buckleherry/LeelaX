from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Callable

import numpy as np
import chess
import torch

from leelax.env.encode import state_to_tensor
from leelax.env.policy_index import move_to_index, index_to_move, legal_policy_mask


@dataclass
class MCTSNode:
    board: chess.Board
    parent: Optional["MCTSNode"] = None
    parent_action: Optional[int] = None  # canonical action index
    n_visits: int = 0
    value_sum: float = 0.0
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    priors: Optional[np.ndarray] = None
    legal_mask: Optional[np.ndarray] = None
    is_expanded: bool = False

    def value(self) -> float:
        return 0.0 if self.n_visits == 0 else self.value_sum / self.n_visits

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def add_child(self, action_idx: int, child: "MCTSNode") -> None:
        self.children[action_idx] = child


class PUCT:
    def __init__(
        self,
        network_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        n_simulations: int = 64,
        cpuct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.network_fn = network_fn
        self.n_simulations = n_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device

    def run(self, board: chess.Board, add_dirichlet: bool = True) -> Tuple[np.ndarray, int]:
        root = MCTSNode(board=board.copy())
        self._expand(root)
        if add_dirichlet:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_simulations):
            node, path = self._select(root)
            value = self._evaluate(node)
            self._backup(path, value)

        # build visit-based policy
        policy = np.zeros_like(root.priors, dtype=np.float32)
        for a_idx, child in root.children.items():
            policy[a_idx] = child.n_visits
        if policy.sum() > 0:
            policy = policy / policy.sum()

        best_action = int(np.argmax(policy))
        return policy, best_action

    def _select(self, root: MCTSNode):
        node = root
        path = [node]

        while not node.is_leaf():
            best_score = -1e9
            best_child = None
            best_action = None

            sqrt_visits = np.sqrt(node.n_visits + 1e-8)
            for a_idx, child in node.children.items():
                q = child.value()
                p = node.priors[a_idx]
                u = self.cpuct * p * sqrt_visits / (1 + child.n_visits)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_child = child
                    best_action = a_idx

            node = best_child  # type: ignore[assignment]
            path.append(node)

        return node, path

    def _expand(self, node: MCTSNode) -> None:
        board = node.board
        state = state_to_tensor(board, canonical=True).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.network_fn(state)
        policy_logits = policy_logits.detach().cpu()
        value = value.item()

        legal_mask = legal_policy_mask(board)
        logits = policy_logits.numpy().reshape(-1)
        logits[legal_mask == 0] = -1e9

        max_logit = np.max(logits)
        exp = np.exp(logits - max_logit)
        exp[legal_mask == 0] = 0.0
        s = exp.sum()
        priors = exp / s if s > 0 else legal_mask / legal_mask.sum()

        node.priors = priors
        node.legal_mask = legal_mask
        node.is_expanded = True
        node.value_from_net = value  # type: ignore[attr-defined]

        # create children for all mappable legal moves
        for mv in board.legal_moves:
            try:
                a_idx = move_to_index(board, mv)
            except KeyError:
                continue
            child_board = board.copy()
            child_board.push(mv)
            child_node = MCTSNode(board=child_board, parent=node, parent_action=a_idx)
            node.add_child(a_idx, child_node)

    def _evaluate(self, node: MCTSNode) -> float:
        board = node.board
        if board.is_game_over():
            res = board.result()
            if res == "1-0":
                return 1.0
            elif res == "0-1":
                return -1.0
            else:
                return 0.0

        if not node.is_expanded:
            self._expand(node)
        return node.value_from_net  # type: ignore[attr-defined]

    def _backup(self, path, value: float) -> None:
        # alternate perspective
        for i, node in enumerate(reversed(path)):
            node.n_visits += 1
            sign = 1.0 if i % 2 == 0 else -1.0
            node.value_sum += sign * value

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
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

