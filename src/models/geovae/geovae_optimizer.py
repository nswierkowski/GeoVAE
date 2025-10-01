import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from typing import List, Tuple, Dict, Optional


class GeoVAEOptimizer(nn.Module):
    """
    GNN-based tree optimizer:
    - Takes current_edge_index [2, E_inst] and mu-nodes [B, D].
    - Returns new_edge_index [2, 2*(B-1)] (MST base on predicted scores) and new_gamma_dict {(i,j)->Tensor([D])}.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        config: dict = {},
        device: str = "cuda",
    ):
        super(GeoVAEOptimizer, self).__init__()
        self.device = device
        self.D = latent_dim
        self.hidden_dim = hidden_dim

        self.gcn1 = GCNConv(latent_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.score_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        self.gamma_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

    def _kruskal_mst(self, scores: np.ndarray, num_nodes: int) -> List[Tuple[int, int]]:
        """
        Standard Kruskal algorithm to get MST from dense scores (num_nodes×num_nodes).
        Return list of tuples (i,j), where i < j. The length of this list is (num_nodes-1).
        """
        all_edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                all_edges.append((i, j, float(scores[i, j])))
        all_edges.sort(key=lambda t: t[2], reverse=True)

        parent_uf = list(range(num_nodes))
        rank_uf = [0] * num_nodes

        def find(u):
            while parent_uf[u] != u:
                parent_uf[u] = parent_uf[parent_uf[u]]
                u = parent_uf[u]
            return u

        def union(u, v):
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            if rank_uf[ru] < rank_uf[rv]:
                parent_uf[ru] = rv
            elif rank_uf[ru] > rank_uf[rv]:
                parent_uf[rv] = ru
            else:
                parent_uf[rv] = ru
                rank_uf[ru] += 1
            return True

        mst_edges: List[Tuple[int, int]] = []
        for i, j, sc in all_edges:
            if union(i, j):
                mst_edges.append((i, j))
            if len(mst_edges) == num_nodes - 1:
                break
        return mst_edges

    def forward(
        self,
        current_edge_index: torch.LongTensor,
        mu: torch.Tensor,
        gamma_dict: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
    ) -> Tuple[torch.LongTensor, Dict[Tuple[int, int], torch.Tensor]]:
        """
        1) Pass mu: [B,D] nodes though 2 layers of GCNConv: current_edge_index -> h:[B, hidden_dim].
        2) For each pair i < j count s_{ij} = score_mlp([h_i || h_j]) and build dense scores [B, B].
        3) Generate MST (Tuple list i < j):  scores → mst_edges.
        4) Build new_edge_index bidirectional from mst_edges.
        5) For each (i,j) in mst_edges forecast gamma_{ij} = gamma_mlp([h_i ∥ h_j]).

        Return:
        - (new_edge_index, new_gamma_dict)
        """
        B = mu.size(0)
        device = self.device

        h = F.relu(self.gcn1(mu, current_edge_index))
        h = F.relu(self.gcn2(h, current_edge_index))

        scores = torch.zeros((B, B), device=device)
        for i in range(B):
            for j in range(i + 1, B):
                hi_hj = torch.cat([h[i], h[j]], dim=0)
                s_ij = self.score_mlp(hi_hj)
                scores[i, j] = s_ij
                scores[j, i] = s_ij

        scores_np = scores.detach().cpu().numpy()
        mst_edges = self._kruskal_mst(scores_np, B)

        src_new = []
        dst_new = []
        for i, j in mst_edges:
            src_new += [i, j]
            dst_new += [j, i]
        if len(src_new) == 0:
            new_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            new_edge_index = torch.tensor(
                [src_new, dst_new], dtype=torch.long, device=device
            )

        new_gamma_dict: Dict[Tuple[int, int], torch.Tensor] = {}
        for i, j in mst_edges:
            hi_hj = torch.cat([h[i], h[j]], dim=0)
            gamma_ij = self.gamma_mlp(hi_hj)
            new_gamma_dict[(i, j)] = gamma_ij
            new_gamma_dict[(j, i)] = gamma_ij.clone()

        return new_edge_index, new_gamma_dict
