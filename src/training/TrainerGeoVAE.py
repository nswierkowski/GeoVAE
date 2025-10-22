import time
from typing import Callable, Dict, List, Tuple, Optional
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
from src.training.Trainer import Trainer


class TrainerGeoVAE(Trainer):
    """
    Trainer for GeoVAE with ELBO-based optimization, checkpointing and MLflow logging.
    """

    def _train_epoch(self, model, train_loader, optimizer, loss_fn, train_metrics, mask_ratio):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_num_active_dims = 0.0
        seen = 0

        for x_batch, target in tqdm(train_loader, desc="Training", leave=False):
            x_batch = x_batch.to(model.device)
            target = target.to(model.device)
            x_masked = self._apply_random_mask(x_batch.clone(), mask_ratio)

            optimizer.zero_grad()
            out = model(x_masked)

            recon = out["recon"]
            elbo = out["elbo"]
            mu = out.get("mu", None)
            sigma = out.get("sigma", None)

            # We maximize ELBO -> minimize negative ELBO
            loss = -elbo
            loss.backward()
            optimizer.step()

            # For reporting we use recon loss (MSE) as well
            recon_loss = loss_fn(recon, target)
            
            if hasattr(model, "optimizer_module"):
                model.optimizer_module.update_from_loss(loss, recon_loss)

            with torch.no_grad():
                batch_size = x_batch.size(0)
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                total_recon += recon_loss.item()
                seen += batch_size

                if mu is not None and sigma is not None:
                    logvar = torch.log(sigma.pow(2) + 1e-8)
                    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                    mean_kl_per_dim = kl_per_dim.mean(dim=0)
                    num_active_dims = (mean_kl_per_dim > 0.01).sum().item()
                    total_num_active_dims += num_active_dims

        # Avoid division by zero when loader empty
        if seen == 0:
            avg_loss = 0.0
            avg_recon = 0.0
            avg_active = 0.0
        else:
            avg_loss = total_loss / (len(train_loader) if len(train_loader) > 0 else 1)
            avg_recon = total_recon / (len(train_loader) if len(train_loader) > 0 else 1)
            avg_active = total_num_active_dims / (len(train_loader) if len(train_loader) > 0 else 1)

        train_metrics["loss"].append(avg_loss)
        train_metrics["mse"].append(avg_recon)
        train_metrics["partial_loss"].append(0.0)
        train_metrics["num_active_dims"].append(avg_active)

    def _validate_model(self, model, val_loader, loss_fn, val_metrics, writer=None, n_epoch=0):
        model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_num_active_dims = 0.0
        batches = 0

        with torch.no_grad():
            for x_batch, target in val_loader:
                x_batch = x_batch.to(model.device)
                target = target.to(model.device)

                out = model(x_batch)
                recon = out["recon"]
                elbo = out["elbo"]
                mu = out.get("mu", None)
                sigma = out.get("sigma", None)

                recon_loss = loss_fn(recon, target)
                # mse per-batch
                mse = nn.functional.mse_loss(recon.view(target.size(0), -1),
                                             target.view(target.size(0), -1),
                                             reduction="mean")

                total_loss += (-elbo).item() if isinstance(elbo, torch.Tensor) else float(-elbo)
                total_recon += mse.item()
                if mu is not None and sigma is not None:
                    logvar = torch.log(sigma.pow(2) + 1e-8)
                    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                    mean_kl_per_dim = kl_per_dim.mean(dim=0)
                    num_active_dims = (mean_kl_per_dim > 0.01).sum().item()
                    total_num_active_dims += num_active_dims

                batches += 1

        if batches == 0:
            avg_loss = 0.0
            avg_recon = 0.0
            avg_active = 0.0
        else:
            avg_loss = total_loss / batches
            avg_recon = total_recon / batches
            avg_active = total_num_active_dims / batches

        val_metrics["loss"].append(avg_loss)
        val_metrics["mse"].append(avg_recon)
        val_metrics["partial_loss"].append(0.0)
        val_metrics["num_active_dims"].append(avg_active)

        if writer is not None:
            writer.add_scalar("val_loss", float(avg_loss), n_epoch)
            writer.add_scalar("val_mse", float(avg_recon), n_epoch)
            writer.add_scalar("val_num_active_dims", float(avg_active), n_epoch)

        print(
            f"Validation - ELBO loss: {avg_loss:.6f}, MSE: {avg_recon:.6f}, num_active_dims: {avg_active}"
        )
        return avg_loss

    def _save_checkpoint(self, model, optimizer, train_metrics, val_metrics, epoch, save_dir, model_name):
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"{model_name}_ckpt_epoch{epoch}.pt")
        try:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
        except Exception as e:
            print(f"[WARN] Failed to save checkpoint {ckpt_path}: {e}")

    def _load_checkpoint(self, ckpt_path: str, model, optimizer: Optional[optim.Optimizer] = None):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=model.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"[WARN] Could not fully load optimizer state: {e}")
        print(f"Checkpoint loaded: {ckpt_path}")
        return checkpoint.get("epoch", 0), checkpoint.get("train_metrics", None), checkpoint.get("val_metrics", None)

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
        use_mlflow: bool = True,
        resume_from_ckpt: Optional[str] = None,
    ) -> None:

        # initialize metrics and optimizer
        train_metrics, val_metrics = self._initialize_metrics()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        start_epoch = 0

        # MLflow start BEFORE logging params (so that resume doesn't create duplicate runs unintentionally)
        if use_mlflow:
            mlflow.start_run(run_name=model_name)
            mlflow.log_params({
                "lr": lr,
                "weight_decay": weight_decay,
                "mask_ratio": mask_ratio,
                "epochs": epochs,
                "model_name": model_name,
            })

        # Resume if requested
        if resume_from_ckpt is not None:
            try:
                start_epoch, loaded_train, loaded_val = self._load_checkpoint(resume_from_ckpt, model, optimizer)
                # if metrics available in ckpt, use them
                if loaded_train is not None:
                    train_metrics = loaded_train
                if loaded_val is not None:
                    val_metrics = loaded_val
                start_epoch = int(start_epoch) + 1  # continue from next epoch
                print(f"Resuming from epoch {start_epoch}")
            except Exception as e:
                print(f"[WARN] Resume failed: {e}. Starting from scratch.")

        best_val_loss = float("inf")
        writer = SummaryWriter(log_dir=f"logs/{model_name}", flush_secs=30) if tensorboard else None

        # training loop
        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            t0 = time.time()

            # train + validate
            self._train_epoch(model, train_loader, optimizer, loss_fn, train_metrics, mask_ratio)
            self._validate_model(model, val_loader, loss_fn, val_metrics, writer=writer, n_epoch=epoch)

            # append steps (ensure X axis for plots)
            train_metrics["step"].append(epoch + 1)
            val_metrics["step"].append(epoch + 1)

            # checkpoint every epoch
            self._save_checkpoint(model, optimizer, train_metrics, val_metrics, epoch + 1, save_dir, model_name)

            # save best model
            val_loss = float(val_metrics["loss"][-1]) if len(val_metrics["loss"]) > 0 else float("inf")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, f"{model_name}_best.pt")
                try:
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved: {best_model_path}")
                except Exception as e:
                    print(f"[WARN] Failed to save best model: {e}")

            # MLflow metric logging (cast to float)
            if use_mlflow:
                try:
                    mlflow.log_metric("train_loss", float(train_metrics["loss"][-1]), step=epoch + 1)
                    mlflow.log_metric("val_loss", float(val_metrics["loss"][-1]), step=epoch + 1)
                    mlflow.log_metric("train_mse", float(train_metrics["mse"][-1]), step=epoch + 1)
                    mlflow.log_metric("val_mse", float(val_metrics["mse"][-1]), step=epoch + 1)
                    mlflow.log_metric("train_num_active_dims", float(train_metrics["num_active_dims"][-1]), step=epoch + 1)
                    mlflow.log_metric("val_num_active_dims", float(val_metrics["num_active_dims"][-1]), step=epoch + 1)
                except Exception as e:
                    print(f"[WARN] MLflow logging failed: {e}")

            print(f"Epoch time: {time.time()-t0:.1f}s")

        # ensure steps exist for plotting
        if len(train_metrics["step"]) == 0:
            train_metrics["step"] = list(range(1, len(train_metrics["loss"]) + 1))
        if len(val_metrics["step"]) == 0:
            val_metrics["step"] = list(range(1, len(val_metrics["loss"]) + 1))

        # plot & finalize
        self.plot_metrics(train_metrics, val_metrics, vis_dir, model_name)

        if use_mlflow:
            try:
                mlflow.pytorch.log_model(model, artifact_path="models")
                metrics_plot_path = os.path.join(vis_dir, f"{model_name}_metrics.png")
                if os.path.exists(metrics_plot_path):
                    mlflow.log_artifact(metrics_plot_path)
            except Exception as e:
                print(f"[WARN] MLflow artifact logging failed: {e}")
            mlflow.end_run()

        if writer is not None:
            writer.close()
