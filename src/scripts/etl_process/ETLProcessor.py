import os
from src.scripts.etl_process.BenchmarkLoader import BenchmarkLoader
from src.scripts.etl_process.DataDownloader import DataDownloader
from src.scripts.etl_process.DataSplitter import DataSplitter
from src.scripts.etl_process.DataTransformer import DataTransformer


class ETLProcessor:
    """
    ETLProcessor orchestrates the ETL process for both Kaggle and benchmark datasets.
    - Kaggle datasets: downloaded, split into JSON, and transformed into DataLoaders.
    - Benchmark datasets (MNIST, fMNIST, STL-10, Reuters): prepared into .pt splits
      and loaded efficiently into DataLoaders.
    Attributes:
        dataset_name (str): Dataset identifier (Kaggle string or benchmark name).
        raw_dir (str): Directory where raw data will be stored.
        split_dir (str): Directory where split data will be stored.
    Methods:
        process(): Executes the ETL process, returning DataLoaders for train, val, and test.
    """

    def __init__(
        self,
        dataset_name: str,
        raw_dir: str = "data/raw_data",
        split_dir: str = "data/splits",
    ):
        self.dataset_name = dataset_name.lower()
        self.raw_dir = raw_dir
        self.split_dir = split_dir

    def process(self):
        if self.dataset_name in ["mnist", "fmnist", "fashion-mnist", "stl10", "reuters"]:
            benchmark_loader = BenchmarkLoader(self.dataset_name, split_dir=self.split_dir)
            return benchmark_loader.load()

        downloader = DataDownloader(self.dataset_name, self.raw_dir)
        image_dir = downloader.download()

        splitter = DataSplitter(image_dir=image_dir, split_dir=self.split_dir)
        if not all(
            os.path.exists(os.path.join(self.split_dir, f"{s}.json"))
            for s in ["train", "val", "test"]
        ):
            splitter.split()

        transformer = DataTransformer()
        train_loader = transformer.get_dataloader(
            os.path.join(self.split_dir, "train.json"), shuffle=True
        )
        val_loader = transformer.get_dataloader(
            os.path.join(self.split_dir, "val.json"), shuffle=False
        )
        test_loader = transformer.get_dataloader(
            os.path.join(self.split_dir, "test.json"), shuffle=False
        )

        return train_loader, val_loader, test_loader
