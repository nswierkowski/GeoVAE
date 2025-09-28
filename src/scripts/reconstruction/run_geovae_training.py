from pathlib import Path
import itertools
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.treevi.pyg_tree.PygTreeVAE import GeoVAE
from src.scripts.etl_process.ETLProcessor import ETLProcessor
from src.training.MaskDataset import MaskedDataset
from src.training.Trainer import Trainer


CONFIG = {
    "input_dim": 1,
    "hidden_dim": 128,
    "residual_hiddens": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mask_size": 0.0,
    "param_grid": {"num_residual_layers": [1,2], "latent_dim": [64,128]},
    "save_dir": Path("models/reconstruction/geovae/"),
    "dataset_config": {
        # Can be benchmark ("mnist", "fmnist", "stl10", "reuters") or Kaggle ("mahmudulhaqueshawon/cat-image")
        "dataset_name": "mnist",
        "raw_dir": "data/raw_data/MNIST/raw",
        "split_dir": "data/data_splits/mnist",
    },
    "use_mlflow": True
}

CONFIG["vis_dir"] = CONFIG["save_dir"] / "metrics"
CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["vis_dir"].mkdir(parents=True, exist_ok=True)


def main():
    dataset_name = CONFIG["dataset_config"]["dataset_name"]
    print("Preparing data...")
    print(f"Dataset: {dataset_name}")
    print(f"Using device: {CONFIG['device']}")

    etl = ETLProcessor(**CONFIG["dataset_config"])
    train_loader, val_loader, _ = etl.process()

    if "/" in dataset_name: 
        print("Applying masked dataset transformation for reconstruction...")
        masked_val_ds = MaskedDataset(val_loader.dataset, CONFIG["mask_size"])
        val_loader = DataLoader(
            masked_val_ds,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
            drop_last=val_loader.drop_last,
        )

    param_combinations = list(itertools.product(*CONFIG["param_grid"].values()))
    total_configs = len(param_combinations)

    print(f"Total configurations to run: {total_configs}")

    for i, (layers, latent_dim) in enumerate(param_combinations):
        model_name = f"GeoVAE_layers{layers}_latent{latent_dim}_dataset{dataset_name}".replace(".", "")
        print(f"\n[{i + 1}/{total_configs}] Running: {model_name}")

        model = GeoVAE(
            input_dim=CONFIG["input_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            residual_hiddens=CONFIG["residual_hiddens"],
            num_residual_layers=layers,
            latent_dim=latent_dim,
        ).to(CONFIG["device"])

        loss_fn = nn.MSELoss(reduction="sum")

        trainer = Trainer()
        trainer.train_supervised(
            model=model,
            epochs=CONFIG["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
            mask_ratio=CONFIG["mask_size"],
            loss_fn=loss_fn,
            model_name=model_name,
            save_dir=CONFIG["save_dir"],
            vis_dir=CONFIG["vis_dir"],
            use_mlflow=CONFIG["use_mlflow"]
        )

        config_out = {
            "model_name": model_name,
            "dataset": dataset_name,
            "config": CONFIG,
        }
        with open(CONFIG["save_dir"] / f"{model_name}_config.json", "w") as f:
            json.dump(config_out, f, indent=4)

    print("Training and visualization complete.")


if __name__ == "__main__":
    main()