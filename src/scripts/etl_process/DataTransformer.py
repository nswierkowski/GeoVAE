import json
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch

class ListDataset(Dataset):
    """
    Custom Dataset loading images from a list of (path, label) tuples.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class DataTransformer:
    """
    This class provides methods to transform image datasets into PyTorch DataLoaders.
    It applies necessary transformations and prepares the data for training and evaluation.
    Attributes:
        image_size (int): Size to which images will be resized.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
    Methods:
        _get_transform(): Returns a composed transform for image preprocessing.
        get_dataloader(samples_path: str, shuffle: bool = True): Returns a DataLoader for the dataset.
    """

    def __init__(
        self, image_size: int = 64, batch_size: int = 256, num_workers: int = 4
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def get_dataloader(self, samples_path: str, shuffle: bool = True):
        with open(samples_path, "r") as f:
            samples = json.load(f)

        dataset = ListDataset(samples, transform=self._get_transform())
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        
    def get_pt_dataloader(self, pt_file, shuffle=True):
        images, labels = torch.load(pt_file)
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

