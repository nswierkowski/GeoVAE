import json
import os
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder


class DataSplitter:
    """
    This class provides methods to split an image dataset into training, validation, and test sets.
    It saves the splits as JSON files in a specified directory.
    Attributes:
        image_dir (str): Directory containing the images.
        split_dir (str): Directory where the split JSON files will be saved.
        ratios (tuple): Ratios for train, validation, and test splits.
        random_state (int): Random seed for reproducibility.
    Methods:
        split(): Splits the dataset into train, validation, and test sets and saves them as JSON files.
        _save_split(name, data): Saves a split to a JSON file.
    """

    def __init__(
        self,
        image_dir: str,
        split_dir: str = "data/splits",
        ratios=(0.7, 0.15, 0.15),
        random_state=42,
    ):
        self.image_dir = image_dir
        self.split_dir = split_dir
        self.ratios = ratios
        self.random_state = random_state
        os.makedirs(split_dir, exist_ok=True)

    def _save_split(self, name, data):
        with open(os.path.join(self.split_dir, f"{name}.json"), "w") as f:
            json.dump(data, f)

    def split(self):
        dataset = ImageFolder(self.image_dir)
        samples = [(os.path.normpath(path), label) for path, label in dataset.samples]

        train_val, test = train_test_split(
            samples, test_size=self.ratios[2], random_state=self.random_state
        )
        train, val = train_test_split(
            train_val,
            test_size=self.ratios[1] / sum(self.ratios[:2]),
            random_state=self.random_state,
        )

        self._save_split("train", train)
        self._save_split("val", val)
        self._save_split("test", test)
        return train, val, test
