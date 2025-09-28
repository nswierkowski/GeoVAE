import time
from typing import Callable, Dict, List, Tuple
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import os
import mlflow
import mlflow.pytorch


class Trainer:
    """
    A class to handle training of Autoencoders and VAEs.
    """

    def _apply_random_mask(
        self, images: torch.Tensor, mask_ratio: float
    ) -> torch.Tensor:
        """
        Apply a random square mask with pixel value 0.5 to each image in the batch.
        - Random size sampled from [0.15, mask_size_ratio]
        - Random location per image
        Vectorized version — no for loop.
        """
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

    def _initialize_metrics(
        self,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        train_metrics = {
            "loss": [],
            "mse": [],
            "partial_loss": [],
            "step": [],
            "num_active_dims": [],
            "time": [],
        }
        val_metrics = {
            "loss": [],
            "mse": [],
            "partial_loss": [],
            "step": [],
            "num_active_dims": [],
            "time": []
        }
        return train_metrics, val_metrics

    def _train_epoch(self, model, train_loader, optimizer, loss_fn, train_metrics, mask_ratio):
        model.train()
        total_loss = total_mse = total_partial_loss = total_num_active_dims = 0.0

        for x_batch, target in tqdm(train_loader, desc="Training"):
            x_batch = x_batch.to(model.device)
            target = target.to(model.device)  # target = original image
            x_masked = x_batch.clone()
            x_masked = self._apply_random_mask(x_masked, mask_ratio)
            optimizer.zero_grad()

            out = model(x_masked)
            if isinstance(out, dict):
                recon = out["recon"]
                partial_loss = out.get("partial_loss", torch.tensor(0.0, device=x_masked.device))
                num_active_dims = out.get("num_active_dims", 0)
            else:
                recon = out
                partial_loss = torch.tensor(0.0, device=x_masked.device)

            recon_loss = loss_fn(recon, target)
            loss = recon_loss + partial_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss.item()
                total_mse += nn.functional.mse_loss(recon.view(target.size(0), -1), target.view(target.size(0), -1), reduction="mean").item()
                total_partial_loss += partial_loss.item() if isinstance(partial_loss, torch.Tensor) else float(partial_loss)
                total_num_active_dims += num_active_dims

        num_batches = len(train_loader)
        train_metrics["loss"].append(total_loss / num_batches)
        train_metrics["mse"].append(total_mse / num_batches)
        train_metrics["partial_loss"].append(total_partial_loss / num_batches)
        train_metrics["num_active_dims"].append(total_num_active_dims / num_batches)


    def _validate_model(self, model, val_loader, loss_fn, val_metrics, writer=None, n_epoch=0):
        model.eval()
        total_loss = total_mse = total_partial_loss = total_num_active_dims = 0.0
        batches = 0

        with torch.no_grad():
            for x_batch, target in val_loader:
                x_batch = x_batch.to(model.device)
                target = target.to(model.device) 

                out = model(x_batch)
                if isinstance(out, dict):
                    recon = out["recon"]
                    partial_loss = out.get("partial_loss", torch.tensor(0.0, device=x_batch.device))
                    num_active_dims = out.get("num_active_dims", 0)
                else:
                    recon = out
                    partial_loss = torch.tensor(0.0, device=x_batch.device)

                recon_loss = loss_fn(recon, target)
                mse = nn.functional.mse_loss(recon.view(target.size(0), -1), target.view(target.size(0), -1), reduction="mean")

                total_loss += (recon_loss + partial_loss).item()
                total_mse += mse.item()
                total_partial_loss += partial_loss.item()
                total_num_active_dims += num_active_dims
                batches += 1

        num_batches = batches
        val_metrics["loss"].append(total_loss / num_batches)
        val_metrics["mse"].append(total_mse / num_batches)
        val_metrics["partial_loss"].append(total_partial_loss / num_batches)
        val_metrics["num_active_dims"].append(total_num_active_dims / num_batches)

        if writer is not None:
            writer.add_scalar('val_loss', total_loss / num_batches, n_epoch)
            writer.add_scalar('val_mse', total_mse / num_batches, n_epoch)
            writer.add_scalar('val_partial_loss', total_partial_loss / num_batches, n_epoch)
            writer.add_scalar('val_num_active_dims', total_num_active_dims / num_batches, n_epoch)

        print(f"Validation - Loss: {total_loss / num_batches:.6f}, MSE: {total_mse / num_batches:.6f}, partial_loss: {total_partial_loss / num_batches:.6f}, num_active_dims: {total_num_active_dims / num_batches}")
        return total_loss / num_batches


    def plot_metrics(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]],
        vis_dir: str,
        model_name: str,
    ):
        fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=True)
        ax1, ax2, ax3, ax4 = axes

        ax1.plot(train_metrics["step"], train_metrics["loss"], label="train loss")
        ax1.plot(val_metrics["step"], val_metrics["loss"], label="val loss")
        ax1.set_ylabel("Loss")
        ax3.set_xlabel("Step")
        ax1.grid()
        ax1.legend()

        ax2.plot(train_metrics["step"], train_metrics["mse"], label="train mse")
        ax2.plot(val_metrics["step"], val_metrics["mse"], label="val mse")
        ax2.set_ylabel("MSE")
        ax3.set_xlabel("Step")
        ax2.grid()
        ax2.legend()

        ax3.plot(
            train_metrics["step"],
            train_metrics["partial_loss"],
            label="train partial loss",
        )
        ax3.plot(
            val_metrics["step"], val_metrics["partial_loss"], label="val partial loss"
        )
        ax3.set_ylabel("Partial Loss")
        ax3.set_xlabel("Step")
        ax3.grid()
        ax3.legend()

        ax4.plot(
            train_metrics["step"],
            train_metrics["num_active_dims"],
            label="train Perplexity",
        )
        ax4.plot(
            val_metrics["step"],
            val_metrics["num_active_dims"],
            label="val Perplexity",
        )
        ax4.set_ylabel("Perplexity")
        ax4.set_xlabel("Step")
        ax4.grid()
        ax4.legend()

        num_epochs = max(train_metrics["step"])

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticks(range(1, num_epochs + 1))

        fig.tight_layout()

        save_path = os.path.join(vis_dir, f"{model_name}_metrics.png")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(save_path)
        plt.close(fig)

    def train_supervised(
        self,
        model: nn.Module,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float,
        weight_decay: float,
        mask_ratio: float,
        loss_fn: Callable,
        model_name: str,
        save_dir: str,
        vis_dir: str,
        tensorboard: bool = False,
        use_mlflow: bool = True
    ) -> None:
        
        if use_mlflow:
            mlflow.start_run(run_name=model_name)
            mlflow.log_params({
                "lr": lr,
                "weight_decay": weight_decay,
                "mask_ratio": mask_ratio,
                "epochs": epochs,
                "model_name": model_name,
            })

        train_metrics, val_metrics = self._initialize_metrics()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        writer = SummaryWriter(
            log_dir=f"logs/{model_name}",
            flush_secs=30,
        ) if tensorboard else None

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            start_time = time.time()
            self._train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                train_metrics,
                mask_ratio,
            )
            train_metrics['time'].append(time.time() - start_time)
            start_val_time = time.time()
            loss = self._validate_model(model, val_loader, loss_fn, val_metrics, writer=writer, n_epoch=epoch)
            val_metrics['time'].append(time.time() - start_val_time)
            train_metrics["step"].append(epoch)
            val_metrics["step"].append(epoch)
            if loss < best_val_loss:
                best_val_loss = loss
                torch.save(model.state_dict(), f"{save_dir}/{model_name}.pt")
                
            if use_mlflow:
                mlflow.log_metric("train_loss", train_metrics["loss"][-1], step=epoch)
                mlflow.log_metric("val_loss", val_metrics["loss"][-1], step=epoch)
                mlflow.log_metric("train_mse", train_metrics["mse"][-1], step=epoch)
                mlflow.log_metric("val_mse", val_metrics["mse"][-1], step=epoch)

        self.plot_metrics(train_metrics, val_metrics, vis_dir, model_name)
        
        if use_mlflow:
            mlflow.pytorch.log_model(model, artifact_path="models")
            metrics_plot_path = os.path.join(vis_dir, f"{model_name}_metrics.png")
            mlflow.log_artifact(metrics_plot_path)
            mlflow.end_run()
