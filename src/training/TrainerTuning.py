import time
from typing import Callable, Dict, List, Tuple, Optional
import optuna
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class TrainerTuning:
    """
    Lightweight trainer for Optuna hyperparameter tuning.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
    

    def _apply_random_mask(self, images: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        if mask_ratio <= 0.0:
            return images

        B, _, H, W = images.shape
        device = images.device
        random_ratios = torch.empty(B, device=device).uniform_(0.15, mask_ratio)
        mask_sizes = (random_ratios * H).long().clamp(min=1)
        top_coords = torch.randint(0, H, (B,), device=device)
        left_coords = torch.randint(0, W, (B,), device=device)
        top_coords = torch.minimum(top_coords, H - mask_sizes)
        left_coords = torch.minimum(left_coords, W - mask_sizes)

        for i in range(B):
            y1, y2 = top_coords[i], top_coords[i] + mask_sizes[i]
            x1, x2 = left_coords[i], left_coords[i] + mask_sizes[i]
            images[i, :, y1:y2, x1:x2] = 0.5
        return images

    def _train_epoch(self, model, loader, optimizer, loss_fn, mask_ratio):
        model.train()
        total_loss = 0.0

        for x_batch, target in tqdm(loader, desc="Train (tuning)", leave=False):
            x_batch = x_batch.to(self.device)
            target = target.to(self.device)

            x_masked = self._apply_random_mask(x_batch.clone(), mask_ratio)

            optimizer.zero_grad()
            out = model(x_masked)
            if isinstance(out, dict):
                recon = out["recon"]
                partial_loss = out.get("partial_loss", 0.0)
            else:
                recon = out
                partial_loss = 0.0

            loss = loss_fn(recon, target) + (partial_loss if torch.is_tensor(partial_loss) else torch.tensor(partial_loss, device=self.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate(self, model, loader, loss_fn):
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for x_batch, target in loader:
                x_batch = x_batch.to(self.device)
                target = target.to(self.device)
                out = model(x_batch)
                if isinstance(out, dict):
                    recon = out["recon"]
                    partial_loss = out.get("partial_loss", 0.0)
                else:
                    recon = out
                    partial_loss = 0.0

                val_loss = loss_fn(recon, target) + (partial_loss if torch.is_tensor(partial_loss) else torch.tensor(partial_loss, device=self.device))
                total_val_loss += val_loss.item()

        return total_val_loss / len(loader)


    def train_for_tuning(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        weight_decay: float,
        mask_ratio: float,
        loss_fn: Callable,
        trial: Optional["optuna.Trial"] = None,
        model_name: str = "geovae_tuning",
        save_dir: Optional[str] = None,
        patience: int = 5,             
        improvement_threshold: float = 1e-3,
    ) -> float:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val_loss = float("inf")
        val_losses = []

        for epoch in range(epochs):
            train_loss = self._train_epoch(model, train_loader, optimizer, loss_fn, mask_ratio)
            val_loss = self._validate(model, val_loader, loss_fn)

            val_losses.append(val_loss)

            if trial is not None:
                trial.report(val_loss, step=epoch)

                # --- NEW: dynamic pruning based on recent improvement ---
                # if len(val_losses) > patience:
                #     recent_improvement = np.mean(np.diff(val_losses[-patience:]))
                #     if abs(recent_improvement) < improvement_threshold:
                #         print(f"Pruning trial {trial.number} (stagnant improvement: {recent_improvement:.6f})")
                #         raise optuna.TrialPruned()

                if trial.should_prune():
                    print(f"Trial pruned at epoch {epoch+1} (val_loss={val_loss:.4f})")
                    raise optuna.TrialPruned()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir is not None:
                    torch.save(model.state_dict(), f"{save_dir}/{model_name}_best.pt")

            print(f"[Epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        return best_val_loss
