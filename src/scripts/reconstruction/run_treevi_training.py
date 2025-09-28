from pathlib import Path
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.treevi.TreeVI_VAE import VAEWithVIStructure
from src.scripts.etl_process.ETLProcessor import ETLProcessor
from src.training.MaskDataset import MaskedDataset
from src.training.Trainer import Trainer


CONFIG = {
    "input_dim": 3,
    "hidden_dim": 128,
    "residual_hiddens": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mask_size": 0.35,
    "param_grid": {
        "num_residual_layers": [2],
        "latent_dim": [256, 512, 1024],  # , 2048],
    },
    "save_dir": Path("models/reconstruction/treevi/text/"),
    "dataset_config": {
        "kaggle_dataset": "mahmudulhaqueshawon/cat-image",
        "raw_dir": "data/raw_data",
        "split_dir": "data/data_splits",
    },
}

CONFIG["vis_dir"] = CONFIG["save_dir"] / "metrics"
CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["vis_dir"].mkdir(parents=True, exist_ok=True)


def main():
    print("Preparing data...")
    print(f"Using device: {CONFIG['device']}")
    etl = ETLProcessor(**CONFIG["dataset_config"])
    train_loader, val_loader, _ = etl.process()
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
        model_name = f"treevi_layers{layers}_latent{latent_dim}".replace(".", "")
        print(f"\n[{i + 1}/{total_configs}] Running: {model_name}")

        model = VAEWithVIStructure(
            input_dim=CONFIG["input_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            residual_hiddens=CONFIG["residual_hiddens"],
            num_residual_layers=layers,
            latent_dim=latent_dim,
        )
        model = model.to(CONFIG["device"])

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
        )

    print("Training and visualization complete.")


if __name__ == "__main__":
    main()
