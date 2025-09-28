from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class VIStructure(ABC):
    """
    The VIStructure class encapsulates the structure of a variational inference model.
    It is used to define the latent variables and their relationships in a tree-structured or graph-structured
    variational inference setting.

    Attributes:
        adj_matrix (torch.Tensor): A binary (0/1) matrix of shape (N, N) representing edge connections.
        edge_list (List[Tuple[int, int]]): List of (parent, child) tuples defining the tree.
        gamma (Dict[Tuple[int, int], torch.Tensor]): Dictionary mapping each edge (parent, child)
            to its learned correlation vector (shape: [latent_dim]). Values should be in (-1, 1).
        parent_map (Dict[int, Optional[int]]): Mapping from node index to its parent (None for root).
        ordering (List[int]): List of node indices in a topological (ancestral) order.
    """

    @abstractmethod
    def get_ordering(self) -> List[int]:
        pass

    @abstractmethod
    def get_parent(self, node: int) -> Optional[int]:
        pass

    @abstractmethod
    def get_path(self, node: int) -> List[int]:
        pass

    @abstractmethod
    def compute_effective_correlation(self, start: int, end: int) -> torch.Tensor:
        pass
