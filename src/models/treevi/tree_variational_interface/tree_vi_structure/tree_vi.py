from typing import Dict, List, Optional, Tuple
import torch
from src.models.treevi.tree_variational_interface.vi_structure import VIStructure

from typing import Dict, List, Optional, Tuple
import torch
from src.models.treevi.tree_variational_interface.vi_structure import VIStructure


class TreeStructure(VIStructure):
    """
    Optimized TreeStructure for tree-structured variational inference.
    It stores the adjacency matrix, edge list, and learned edge correlation parameters.
    It builds a parent map and a topological ordering for ancestral sampling.
    This class allows efficient computation of effective correlations between nodes in the tree.
    The `parent` is stored as a LongTensor([N]) where parent[i] = -1 indicates a root node,
    and `children` is a list of lists where children[i] contains all direct children of node i.
    The `eff_root` is a list of tensors ([D]) representing the prefix product of gamma values
    from the root to each node, allowing for efficient O(1) computation of effective correlations.
    """

    def __init__(
        self,
        adj_matrix: torch.Tensor,
        edge_list: List[Tuple[int, int]],
        gamma_dict: Dict[Tuple[int, int], torch.Tensor],
    ) -> None:
        """
        Args:
            adj_matrix:   tensor [N, N], matrix of neighbours (only for saving topology).
            edge_list:    list of tuples (parent, child).
            gamma_dict:   dictionary { (parent, child) → Tensor([D]) }, with values: (-1,1).
        """
        self.adj_matrix = adj_matrix
        self.edge_list = edge_list
        self.gamma = gamma_dict

        num_nodes = self.adj_matrix.size(0)
        device = self.adj_matrix.device

        parent = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        for p, c in self.edge_list:
            parent[c] = p
        self.parent = parent

        self.children: List[List[int]] = [[] for _ in range(num_nodes)]
        for p, c in self.edge_list:
            self.children[p].append(c)

        self.roots = (self.parent == -1).nonzero(as_tuple=False).view(-1).tolist()
        if len(self.roots) == 0:
            self.roots = [0]

        self.ordering: List[int] = []
        visited = [False] * num_nodes

        def _dfs(u: int):
            visited[u] = True
            self.ordering.append(u)
            for v in self.children[u]:
                if not visited[v]:
                    _dfs(v)

        for r in self.roots:
            if not visited[r]:
                _dfs(r)

        if len(self.gamma) > 0:
            example_gamma = next(iter(self.gamma.values()))
            latent_dim = example_gamma.shape[0]
        else:
            latent_dim = 1

        self.eff_root = torch.zeros((num_nodes, latent_dim), device=device)

        for r in self.roots:
            self.eff_root[r] = torch.ones(latent_dim, device=device)

        for i in self.ordering:
            p = self.parent[i].item()
            if p == -1:
                continue
            g_pi = self.gamma.get((p, i), None)
            if g_pi is None:
                raise ValueError(f"Lack of gamma parameter for edge: {(p, i)}!")
            self.eff_root[i] = self.eff_root[p] * g_pi

    def get_ordering(self) -> List[int]:
        return self.ordering

    def get_parent(self, node: int) -> Optional[int]:
        p = self.parent[node].item()
        return None if p < 0 else p

    def get_path(self, node: int) -> List[int]:
        path: List[int] = []
        current = node
        while current is not None:
            path.append(current)
            p = self.parent[current].item()
            current = None if p < 0 else p
        path.reverse()
        return path

    def compute_effective_correlation(self, start: int, end: int) -> torch.Tensor:
        """
        In this version quiet fast counting O(D):
            eff_root[end] / eff_root[start].
        If start == end, returns ones([D])
        """
        if start == end:
            return torch.ones_like(self.eff_root[start])

        eff_start = self.eff_root[start]
        eff_end = self.eff_root[end]
        return eff_end / (
            eff_start + 1e-10
        )  # +1e-10 in case if eff_start was 0 (in correct tree, it should not happen)


class OrginalTreeStructure(VIStructure):
    """
    The OrginalTreeStructure class encapsulates the tree over latent variables for
    tree-structured variational inference. It stores the adjacency matrix (if any),
    the edge list, the learned edge correlation parameters, and builds a parent map
    and a topological ordering for ancestral sampling.

    Attributes:
        adj_matrix (torch.Tensor): A binary (0/1) matrix of shape (N, N) representing edge connections.
        edge_list (List[Tuple[int, int]]): List of (parent, child) tuples defining the tree.
        gamma (Dict[Tuple[int, int], torch.Tensor]): Dictionary mapping each edge (parent, child)
            to its learned correlation vector (shape: [latent_dim]). Values should be in (-1, 1).
        parent_map (Dict[int, Optional[int]]): Mapping from node index to its parent (None for root).
        ordering (List[int]): List of node indices in a topological (ancestral) order.
    """

    def __init__(
        self,
        adj_matrix: torch.Tensor,
        edge_list: List[Tuple[int, int]],
        gamma_dict: Dict[Tuple[int, int], torch.Tensor],
    ) -> None:
        self.adj_matrix = adj_matrix  # assumed binary, dtype=torch.int or float
        self.edge_list = edge_list
        self.gamma = (
            gamma_dict  # mapping: (parent, child) -> torch.Tensor of shape [latent_dim]
        )

        self.parent_map = {}  # child -> parent
        # Build parent_map from edge_list
        for parent, child in self.edge_list:
            self.parent_map[child] = parent
        # Determine all nodes (assume nodes are from 0 to N-1)
        num_nodes = self.adj_matrix.size(0)
        self.all_nodes = list(range(num_nodes))
        # Identify roots: nodes that are not in parent_map keys (i.e., have no parent)
        self.roots = [node for node in self.all_nodes if node not in self.parent_map]
        if len(self.roots) == 0:
            # If all nodes appear as children, choose node 0 as root
            self.roots = [0]
        # Build ordering using DFS from each root
        self.ordering = []
        visited = [False] * num_nodes
        for root in self.roots:
            self._dfs(root, visited)

    def _dfs(self, node: int, visited: List[bool]) -> None:
        """Private helper: Perform DFS to compute ancestral ordering."""
        visited[node] = True
        self.ordering.append(node)
        # Find children: any edge (node, child) in self.edge_list
        children = [child for (parent, child) in self.edge_list if parent == node]
        for child in children:
            if not visited[child]:
                self._dfs(child, visited)

    def get_ordering(self) -> List[int]:
        return self.ordering

    def get_parent(self, node: int) -> Optional[int]:
        return self.parent_map.get(node, None)

    def get_path(self, node: int) -> List[int]:
        path = []
        current = node
        while current is not None:
            path.insert(0, current)
            current = self.get_parent(current)
        return path

    def compute_effective_correlation(self, start: int, end: int) -> torch.Tensor:
        path = self.get_path(end)
        # Find the index of 'start' in the path.
        if start not in path:
            raise ValueError(f"Node {start} is not in the path to {end}.")
        start_index = path.index(start)
        # The effective gamma is the product of gamma for each edge along the subpath from start to end.
        effective_gamma = None
        # Loop over indices from start_index to len(path)-1 (each adjacent pair forms an edge)
        for i in range(start_index, len(path) - 1):
            edge = (path[i], path[i + 1])
            gamma_val = self.gamma.get(edge, None)
            if gamma_val is None:
                raise ValueError(
                    f"Gamma parameter for edge {edge} not found in TreeStructure."
                )
            if effective_gamma is None:
                effective_gamma = gamma_val
            else:
                effective_gamma = (
                    effective_gamma * gamma_val
                )  # elementwise multiplication
        if effective_gamma is None:
            # If there are no edges from start to end (start equals end), return ones.
            # Infer latent dimension from any gamma if available, else default to 1.
            latent_dim = list(self.gamma.values())[0].shape[0] if self.gamma else 1
            effective_gamma = torch.ones(
                latent_dim, device=next(iter(self.gamma.values())).device
            )  # assume same device
        return effective_gamma
