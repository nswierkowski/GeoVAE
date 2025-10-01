from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from src.models.treevi.TreeStructure import TreeStructure

class TreeOptimizer:
    """
    TreeOptimizer learns an optimized tree structure by solving the constrained optimization problem:
        minimize_A ℓ(A)  subject to  h(A) = 0,
    where ℓ(A) is a loss function (e.g., negative ELBO) provided via a callback,
    and the acyclicity constraint is enforced via
        h(A) = trace( A * exp(A^2) ).

    The augmented Lagrangian is defined as:
        Lᵨ(A, α) = ℓ(A) + (ρ/2) * [h(A)]² + α * h(A)
    The optimizer uses a gradient-based method (LBFGS) to update A and performs a dual ascent
    update on the dual variable α.

    Attributes:
        penalty_parameter (float): The penalty parameter ρ for the augmented Lagrangian.
        dual_variable (float): The dual variable α (initialized to dual_init).
        threshold (float): The threshold ω to binarize the learned matrix A.
        max_iterations (int): Maximum number of outer iterations.
        tolerance (float): Convergence tolerance for h(A).
        lbfgs_lr (float): Learning rate for the LBFGS optimizer.
        lbfgs_max_iter (int): Maximum number of iterations for LBFGS per outer iteration.
        loss_callback (Optional[Callable[[torch.Tensor], torch.Tensor]]):
            Callback function that receives the current A (torch.Tensor) and returns the scalar loss ℓ(A).
        device (torch.device): Computation device.
    """

    def __init__(
        self,
        config: dict,
        loss_callback: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        tree_cfg = config.get("tree_optimizer", {})
        self.penalty_parameter: float = float(tree_cfg.get("penalty_parameter", 1.0))
        self.dual_variable: float = float(tree_cfg.get("dual_init", 0.0))
        self.threshold: float = float(tree_cfg.get("threshold", 0.3))
        self.max_iterations: int = int(tree_cfg.get("max_iterations", 100))
        self.tolerance: float = float(tree_cfg.get("tolerance", 1e-8))
        self.lbfgs_lr: float = float(tree_cfg.get("lbfgs_lr", 0.01))
        self.lbfgs_max_iter: int = int(tree_cfg.get("lbfgs_max_iter", 20))
        self.loss_callback = loss_callback  # This callback computes ℓ(A)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _h_function(self, A: torch.Tensor) -> torch.Tensor:
        return torch.trace(A * torch.exp(A.pow(2)))

    def _augmented_lagrangian(self, A: torch.Tensor) -> torch.Tensor:
        # Compute the loss ℓ(A) (if callback is provided; otherwise 0.0)
        loss_term = (
            self.loss_callback(A)
            if self.loss_callback is not None
            else torch.tensor(0.0, device=self.device)
        )
        h_value = self._h_function(A)
        lagrangian = (
            loss_term
            + (self.penalty_parameter / 2.0) * (h_value**2)
            + self.dual_variable * h_value
        )
        return lagrangian

    def optimize_tree(
        self, initial_tree: TreeStructure, embeddings: Optional[torch.Tensor] = None
    ) -> TreeStructure:
        # Get initial adjacency matrix A from the initial tree; ensure it's a float tensor on the proper device.
        A_init = initial_tree.adj_matrix.clone().detach().to(self.device).float()
        # Make A a parameter to be optimized.
        A_opt = A_init.clone().detach().requires_grad_(True)

        # Set up an LBFGS optimizer to update A.
        optimizer = optim.LBFGS([A_opt], lr=self.lbfgs_lr, max_iter=self.lbfgs_max_iter)

        # Outer iterative optimization loop using augmented Lagrangian method.
        for iteration in range(self.max_iterations):

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                L_aug = self._augmented_lagrangian(A_opt)
                L_aug.backward()
                return L_aug

            optimizer.step(closure)

            # After the LBFGS step, evaluate h(A)
            with torch.no_grad():
                h_val = self._h_function(A_opt).item()
            # Dual ascent update of the dual variable.
            self.dual_variable += self.penalty_parameter * h_val

            # Check convergence of h(A).
            if abs(h_val) < self.tolerance:
                # Convergence achieved.
                break

        # Post-processing: Threshold A to obtain a binary matrix.
        with torch.no_grad():
            A_thresholded = A_opt.clone()
            A_thresholded[torch.abs(A_thresholded) < self.threshold] = 0.0
            # Create binary matrix: set nonzero entries to 1.0.
            A_binary = (torch.abs(A_thresholded) >= self.threshold).float()

        # Convert the optimized binary adjacency matrix A_binary to a numpy array for edge list processing.
        A_np = A_binary.detach().cpu().numpy()
        num_nodes = A_np.shape[0]
        edge_list: List[Tuple[int, int]] = []
        gamma_dict: Dict[Tuple[int, int], torch.Tensor] = {}
        # To assign a unique parent for each node (except the root), we pick the first encountered edge
        parent_assigned: Dict[int, int] = {}
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if A_np[i, j] != 0:
                    # Assign i as parent of j if not already assigned.
                    if j not in parent_assigned:
                        parent_assigned[j] = i
                        edge_list.append((i, j))
                        # Set gamma for edge (i, j) as the corresponding optimized value,
                        # clamped to (-0.99, 0.99) and output as a tensor.
                        gamma_val = float(A_np[i, j])
                        gamma_val_clamped = max(min(gamma_val, 0.99), -0.99)
                        # For simplicity, we assume latent dimension 1 (or can be extended to vectorized gamma).
                        gamma_tensor = torch.tensor(
                            [gamma_val_clamped], dtype=torch.float32, device=self.device
                        )
                        gamma_dict[(i, j)] = gamma_tensor

        # Create a new TreeStructure with the optimized binary adjacency matrix, edge list, and gamma_dict.
        new_tree = TreeStructure(
            adj_matrix=A_binary.detach(), edge_list=edge_list, gamma_dict=gamma_dict
        )
        return new_tree
