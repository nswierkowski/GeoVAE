import json
import os
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder


class DataSplitter:
    """
    Splits an image dataset into training, validation, and test sets.
    Optionally downsamples large datasets like CelebA for faster experimentation.
    """

    def __init__(
        self,
        image_dir: str,
        split_dir: str = "data/splits",
        ratios=(0.7, 0.15, 0.15),
        random_state=42,
        celeba_subset_ratio: float = 0.2,  # use only 20% for CelebA
    ):
        self.image_dir = image_dir
        self.split_dir = split_dir
        self.ratios = ratios
        self.random_state = random_state
        self.celeba_subset_ratio = celeba_subset_ratio
        os.makedirs(split_dir, exist_ok=True)

    def _save_split(self, name, data):
        with open(os.path.join(self.split_dir, f"{name}.json"), "w") as f:
            json.dump(data, f)

    def split(self):
        dataset = ImageFolder(self.image_dir)
        samples = [(os.path.normpath(path), label) for path, label in dataset.samples]

        print(f'Before reducing celeba: {"celaba" in self.image_dir.lower()}')
        if any(x in self.image_dir.lower() for x in ["celeba", "celaba"]) and self.celeba_subset_ratio < 1.0:
            subset_size = int(len(samples) * self.celeba_subset_ratio)
            samples, _ = train_test_split(
                samples,
                train_size=subset_size,
                random_state=self.random_state,
            )
            print(f"[INFO] Detected CelebA dataset — using only {self.celeba_subset_ratio*100:.0f}% of samples ({len(samples)} total).")

        # 🔹 Standard train/val/test split
        train_val, test = train_test_split(
            samples, test_size=self.ratios[2], random_state=self.random_state
        )
        train, val = train_test_split(
            train_val,
            test_size=self.ratios[1] / sum(self.ratios[:2]),
            random_state=self.random_state,
        )

        # 🔹 Save splits
        self._save_split("train", train)
        self._save_split("val", val)
        self._save_split("test", test)

        return train, val, test
