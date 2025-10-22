import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

class BenchmarkLoader:
    def __init__(self, dataset_name: str, split_dir: str = "data/splits", image_size: int = 64,
                 batch_size: int = 256, num_workers: int = 4, seed: int = 42):
        self.dataset_name = dataset_name.lower()
        self.split_dir = os.path.join(split_dir, self.dataset_name)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        os.makedirs(self.split_dir, exist_ok=True)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _transform(self):
        """Return appropriate transform pipeline."""
        if self.dataset_name in ["mnist", "fmnist", "fashion-mnist"]:
            normalize = transforms.Normalize([0.5], [0.5])
        else:
            normalize = transforms.Normalize([0.5]*3, [0.5]*3)

        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            normalize
        ])

    def _load_cached(self):
        """Load cached .pt splits if they exist."""
        paths = [os.path.join(self.split_dir, f"{name}.pt") for name in ["train", "val", "test"]]
        if all(os.path.exists(p) for p in paths):
            train_images = torch.load(paths[0])
            val_images = torch.load(paths[1])
            test_images = torch.load(paths[2])

            train_ds = TensorDataset(train_images, train_images)
            val_ds = TensorDataset(val_images, val_images)
            test_ds = TensorDataset(test_images, test_images)
            print(f"[INFO] Loaded cached dataset from {self.split_dir}")
            return train_ds, val_ds, test_ds
        return None

    def _prepare_and_cache(self):
        cached = self._load_cached()
        if cached is not None:
            return cached

        if self.dataset_name == "mnist":
            dataset = datasets.MNIST("data/benchmark", train=True, download=True, transform=self._transform())

        elif self.dataset_name in ["fmnist", "fashion-mnist"]:
            dataset = datasets.FashionMNIST("data/benchmark", train=True, download=True, transform=self._transform())

        elif self.dataset_name == "stl10":
            dataset = datasets.STL10(
                "data/benchmark",
                split="unlabeled",
                download=True,
                transform=self._transform()
            )
            part_size = len(dataset) // 3
            dataset, _ = random_split(dataset, [part_size, len(dataset) - part_size])
            print(f"[INFO] Using PART of STL10 UNLABELED dataset: {len(dataset)} samples")

            images = torch.stack([dataset[i][0] for i in range(len(dataset))])
            dataset = TensorDataset(images, images)

        elif self.dataset_name == "celeba":
            dataset = datasets.CelebA(
                root="data/benchmark",
                split="train",
                download=True,
                transform=self._transform()
            )
            part_size = len(dataset) // 5  # use 20% for manageable memory
            dataset, _ = random_split(dataset, [part_size, len(dataset) - part_size])
            print(f"[INFO] Using PART of CelebA dataset: {len(dataset)} samples")

            images = torch.stack([dataset[i][0] for i in range(len(dataset))])
            dataset = TensorDataset(images, images)

        else:
            raise ValueError(f"Unsupported benchmark dataset: {self.dataset_name}")

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size],
                                                 generator=torch.Generator().manual_seed(self.seed))

        for name, subset in zip(["train", "val", "test"], [train_ds, val_ds, test_ds]):
            images = torch.stack([subset[i][0] for i in range(len(subset))])
            torch.save(images, os.path.join(self.split_dir, f"{name}.pt"))
            print(f"[INFO] Saved {name} split with {len(subset)} samples to {self.split_dir}")

        return train_ds, val_ds, test_ds

    def load(self):
        train_ds, val_ds, test_ds = self._prepare_and_cache()

        print(f"[DEBUG] Train dataset first sample shape: {train_ds[0][0].shape}")
        print(f"[DEBUG] Val dataset first sample shape: {val_ds[0][0].shape}")
        print(f"[DEBUG] Test dataset first sample shape: {test_ds[0][0].shape}")

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
