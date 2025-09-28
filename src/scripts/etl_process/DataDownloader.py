import os
import shutil
import kagglehub


class DataDownloader:
    """
    This class provides methods to download datasets from Kaggle and prepare them for use in PyTorch.

    Attributes:
        dataset - name of dataset to be downloaded
        output_dir - path where dataset should be downloaded

    Methods:
        download() - Downloads and extracts the specified Kaggle dataset into the output directory.
    """

    def __init__(self, dataset: str, output_dir: str = "data/raw_data"):
        self.dataset = dataset
        self.output_dir = output_dir

    def download(self) -> str:
        cached_path = kagglehub.dataset_download(self.dataset)

        if os.path.exists(self.output_dir):
            cached_files = set(os.listdir(cached_path))
            output_files = set(os.listdir(self.output_dir))

            if cached_files == output_files:
                print(
                    f"[INFO] Dataset '{self.dataset}' already exists at {self.output_dir}, skipping copy."
                )
                return self.output_dir

        print(f"[INFO] Copying dataset '{self.dataset}' to {self.output_dir}...")
        os.makedirs(self.output_dir, exist_ok=True)

        for item in os.listdir(cached_path):
            s = os.path.join(cached_path, item)
            d = os.path.join(self.output_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

        print(f"[INFO] Dataset ready at {self.output_dir}")
        return self.output_dir
