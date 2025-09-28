import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms


class BenchmarkLoader:
    """
    Prepares benchmark datasets (MNIST, Fashion-MNIST, STL-10) resized to 64x64.
    Returns DataLoaders for train/val/test. Ignores labels for reconstruction.
    """

    def __init__(
        self,
        dataset_name: str,
        split_dir: str = "data/splits",
        image_size: int = 64,
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        self.dataset_name = dataset_name.lower()
        self.split_dir = os.path.join(split_dir, self.dataset_name)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        os.makedirs(self.split_dir, exist_ok=True)

    def _transform(self):
        # Grayscale datasets: normalize to [0,1] then mean=0.5 std=0.5
        if self.dataset_name in ["mnist", "fmnist", "fashion-mnist"]:
            normalize = transforms.Normalize([0.5], [0.5])
            channels = 1
        else:
            normalize = transforms.Normalize([0.5]*3, [0.5]*3)
            channels = 3

        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    def _prepare_and_cache(self):
        # Load dataset
        if self.dataset_name == "mnist":
            dataset = datasets.MNIST("data/benchmark", train=True, download=True, transform=self._transform())
        elif self.dataset_name in ["fmnist", "fashion-mnist"]:
            dataset = datasets.FashionMNIST("data/benchmark", train=True, download=True, transform=self._transform())
        elif self.dataset_name == "stl10":
            dataset = datasets.STL10("data/benchmark", split="train", download=True, transform=self._transform())
        else:
            raise ValueError(f"Unsupported benchmark dataset: {self.dataset_name}")

        # Split train/val/test
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

        for name, subset in zip(["train", "val", "test"], [train_ds, val_ds, test_ds]):
            images = torch.stack([subset[i][0] for i in range(len(subset))])
            # Labels are ignored for reconstruction
            torch.save(images, os.path.join(self.split_dir, f"{name}.pt"))
            print(f"[INFO] Saved {name} split with {len(subset)} samples to {self.split_dir}")

    def _load_split(self, name, shuffle):
        path = os.path.join(self.split_dir, f"{name}.pt")
        if not os.path.exists(path):
            self._prepare_and_cache()
        images = torch.load(path)
        dataset = TensorDataset(images, images)  # input=target for reconstruction
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def load(self):
        train_loader = self._load_split("train", shuffle=True)
        val_loader = self._load_split("val", shuffle=False)
        test_loader = self._load_split("test", shuffle=False)
        return train_loader, val_loader, test_loader
