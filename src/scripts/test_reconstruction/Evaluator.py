from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torch.utils.data import TensorDataset, DataLoader


class Evaluator:
    def __init__(self, device=None, batch_size=64):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def _load_labeled_testset(self, dataset_name, base_split_dir="data/data_splits"):
        path = Path(base_split_dir) / dataset_name / dataset_name / "test_with_labels.pt"
        if not path.exists():
            print(f"[WARN] {path} not found → skipping clustering/latent viz for {dataset_name}")
            return None
        
        data = torch.load(path)
        if isinstance(data, tuple) and len(data) == 2:
            images, labels = data
            dataset = TensorDataset(images, labels)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        else:
            print(f"[WARN] {path} format unexpected, expected (images, labels) tuple")
            return None

    def _unpack_batch(self, batch):
        """Helper to unpack batches from dataloader consistently."""
        if isinstance(batch, (list, tuple)):
            return batch[0].to(self.device)
        else:
            return batch.to(self.device)

    def mse_reconstruction(self, model, dataloader):
        model.eval()
        mse_loss = nn.MSELoss(reduction="sum")
        total_loss, total_samples = 0.0, 0
        with torch.no_grad():
            for batch in dataloader:
                x = self._unpack_batch(batch)
                out = model(x)
                recon = out["recon"]
                total_loss += mse_loss(recon, x).item()
                total_samples += x.size(0)
        return total_loss / total_samples

    def active_units(self, model, dataloader, threshold=0.01):
        model.eval()
        mus = []
        with torch.no_grad():
            for batch in dataloader:
                x = self._unpack_batch(batch)
                mu, _ = model.encode(x)
                mus.append(mu.cpu())
        mus = torch.cat(mus, dim=0)
        vars = mus.var(dim=0)
        return (vars > threshold).sum().item()

    def _to_uint8_3ch(self, x):
        if x.min() < 0:
            x = (x + 1) / 2
        x = (x.clamp(0, 1) * 255).to(torch.uint8)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def sample_quality(self, model, real_loader, n_samples=256):
        model.eval()
        samples_list, reals_list = [], []
        collected = 0

        for batch in real_loader:
            x = self._unpack_batch(batch)
            batch_size = x.size(0)
            if x.size(1) == 1:  # grayscale -> RGB
                x = x.repeat(1,3,1,1)
            reals_list.append((x*255).to(torch.uint8))

            with torch.no_grad():
                z = torch.randn(batch_size, model.latent_dim, device=self.device)
                gen = model.decode(z)
                if gen.size(1) == 1:
                    gen = gen.repeat(1,3,1,1)
                samples_list.append((gen*255).to(torch.uint8))

            collected += batch_size
            if collected >= n_samples:
                break

        samples = torch.cat(samples_list)[:n_samples] # For optimalization reasons
        reals   = torch.cat(reals_list)[:n_samples]

        fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)
        kid_metric = KernelInceptionDistance(subset_size=min(256, n_samples), subsets=50).to(self.device)
        is_metric  = InceptionScore(normalize=False).to(self.device)

        # Update generated images with real=None
        fid_metric.update(samples, real=False)

        # Update all real images at once
        fid_metric.update(reals, real=True)

        # KID & IS still batch-wise
        batch_size = 32
        for i in range(0, n_samples, batch_size):
            kid_metric.update(samples[i:i+batch_size], real=False)
            kid_metric.update(reals[i:i+batch_size], real=True)
            is_metric.update(samples[i:i+batch_size])

        fid_value = fid_metric.compute().item()
        kid_mean, kid_std = kid_metric.compute()
        is_mean, is_std = is_metric.compute()

        return {
            "fid": fid_value,
            "kid_mean": kid_mean.item(),
            "kid_std": kid_std.item(),
            "is_mean": is_mean.item(),
            "is_std": is_std.item()
        }
        
    def sample_quality_vqvae(self, model, real_loader, n_samples=256):
        model.eval()
        samples_list, reals_list = [], []
        collected = 0

        for batch in real_loader:
            x = self._unpack_batch(batch)  # shape [B, C, H, W]
            batch_size = x.size(0)

            reals_list.append((x * 255).to(torch.uint8))

            with torch.no_grad():
                # forward pass
                out = model(x)
                gen = out["recon"]
                samples_list.append((gen * 255).to(torch.uint8))

            collected += batch_size
            if collected >= n_samples:
                break

        samples = torch.cat(samples_list)[:n_samples]
        reals   = torch.cat(reals_list)[:n_samples]

        # Convert to 3 channels for metrics if grayscale
        if samples.size(1) == 1:
            samples = samples.repeat(1, 3, 1, 1)
            reals   = reals.repeat(1, 3, 1, 1)

        # Initialize metrics
        fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)
        kid_metric = KernelInceptionDistance(subset_size=min(256, n_samples), subsets=50).to(self.device)
        is_metric  = InceptionScore(normalize=False).to(self.device)

        # FID
        fid_metric.update(samples, real=False)
        fid_metric.update(reals, real=True)

        # KID & IS batch-wise
        batch_size = 32
        for i in range(0, n_samples, batch_size):
            kid_metric.update(samples[i:i+batch_size], real=False)
            kid_metric.update(reals[i:i+batch_size], real=True)
            is_metric.update(samples[i:i+batch_size])

        fid_value = fid_metric.compute().item()
        kid_mean, kid_std = kid_metric.compute()
        is_mean, is_std = is_metric.compute()

        return {
            "fid": fid_value,
            "kid_mean": kid_mean.item(),
            "kid_std": kid_std.item(),
            "is_mean": is_mean.item(),
            "is_std": is_std.item()
        }




    def clustering_metrics(self, model, dataloader):
        model.eval()
        zs, labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(self.device)
                mu, _ = model.encode(x)
                zs.append(mu.cpu().numpy())
                labels.append(y.numpy())
        Z = np.concatenate(zs, axis=0)
        Y = np.concatenate(labels, axis=0)
        kmeans = KMeans(n_clusters=len(np.unique(Y)), n_init=10)
        pred = kmeans.fit_predict(Z)
        ari = adjusted_rand_score(Y, pred)
        nmi = normalized_mutual_info_score(Y, pred)
        return {"ari": ari, "nmi": nmi}

    def latent_visualization(self, model, dataloader, method="tsne", model_name="unknown", dataset_name="dataset", out_dir="../experiments"):
        model.eval()
        zs, labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(self.device)
                mu, _ = model.encode(x)
                zs.append(mu.cpu().numpy())
                labels.append(y.numpy())
        Z = np.concatenate(zs, axis=0)
        Y = np.concatenate(labels, axis=0)

        if method == "tsne":
            Z2d = TSNE(n_components=2, perplexity=30).fit_transform(Z)
        else:
            import umap
            Z2d = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(Z)

        # --- ensure output directory exists ---
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        save_path = out_path / f"{dataset_name}_{model_name}_{method}.png"

        plt.figure(figsize=(6,6))
        scatter = plt.scatter(Z2d[:,0], Z2d[:,1], c=Y, cmap="tab10", s=5, alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def evaluate(self, model, dataloader, real_loader=None, dataset_name=None, base_split_dir="data/data_splits", is_vqvae=False):
        metrics = {}
        metrics["recon_mse"] = self.mse_reconstruction(model, dataloader)
        metrics["active_units"] = self.active_units(model, dataloader)
        if real_loader is not None:
            if not is_vqvae:
                metrics.update(self.sample_quality(model, real_loader, n_samples=256))
            else:
                metrics.update(self.sample_quality_vqvae(model, real_loader, n_samples=256))

        return metrics
    
    def evaluate_only_clustering(self, model, dataset_name=None, base_split_dir="data/data_splits", model_name: str='unknown'):
        metrics = {}
        labeled_loader = None
        if dataset_name is not None:
            labeled_loader = self._load_labeled_testset(dataset_name, base_split_dir)

        if labeled_loader is not None:
            metrics.update(self.clustering_metrics(model, labeled_loader))
            metrics["latent_vis"] = self.latent_visualization(model, labeled_loader, method="tsne", model_name=model_name)
        else:
            metrics["ari"] = None
            metrics["nmi"] = None
            metrics["latent_vis"] = None
        
        return metrics
    
    
