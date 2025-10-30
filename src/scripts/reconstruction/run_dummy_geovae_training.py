from pathlib import Path
import itertools
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.dummy_geovae.dummy_geovae import DummyGeoVAE  
from src.scripts.etl_process.ETLProcessor import ETLProcessor
from src.training.MaskDataset import MaskedDataset
from src.training.Trainer import Trainer


DEFAULT_CONFIG = {
    "input_dim": 3,
    "hidden_dim": 128,
    "residual_hiddens": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    "image_size": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mask_size": 0.35,
    "param_grid": {"num_residual_layers": [1], "latent_dim": [64]},
    "save_dir": Path("models/reconstruction/dummy_geovae/"),
    "dataset_config": {
        "dataset_name": "mahmudulhaqueshawon/cat-image",
        "raw_dir": "data/raw_data/cats/raw",
        "split_dir": "data/data_splits/cats",
    },
    "use_mlflow": True,
    "mask_random_state": 42,

    "match_graph_config": {
        "hidden_dim_gnn": 64,
        "num_inst_gnn_layers": 2,
        "num_dim_gnn_layers": 2,
    },
}

DEFAULT_CONFIG["vis_dir"] = DEFAULT_CONFIG["save_dir"] / "metrics"
DEFAULT_CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
DEFAULT_CONFIG["vis_dir"].mkdir(parents=True, exist_ok=True)


def main(config):
    dataset_name = config["dataset_config"]["dataset_name"]
    print("Preparing data...")
    print(f"Dataset: {dataset_name}")
    print(f"Using device: {config['device']}")

    etl = ETLProcessor(**config["dataset_config"])
    train_loader, val_loader, _ = etl.process()

    if config["mask_size"] > 0:
        print(f"Applying masked dataset transformation with mask ratio = {config['mask_size']}")
        masked_train_ds = MaskedDataset(
            train_loader.dataset,
            config["mask_size"],
            random_state=config["mask_random_state"],
        )
        masked_val_ds = MaskedDataset(
            val_loader.dataset,
            config["mask_size"],
            random_state=config["mask_random_state"],
        )

        train_loader = DataLoader(
            masked_train_ds,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            masked_val_ds,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    param_combinations = list(itertools.product(*config["param_grid"].values()))
    total_configs = len(param_combinations)
    print(f"Total configurations to run: {total_configs}")

    trainer = Trainer()

    for i, (layers, latent_dim) in enumerate(param_combinations):
        model_name = (
            f"DummyGeoVAE_layers{layers}_latent{latent_dim}_dataset{dataset_name}"
            .replace(".", "")
            .replace("/", "_")
        )
        print(f"\n[{i + 1}/{total_configs}] Running: {model_name}")

        model = DummyGeoVAE(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            residual_hiddens=config["residual_hiddens"],
            num_residual_layers=layers,
            latent_dim=latent_dim,
            image_size=config["image_size"],
            match_graph_config=config.get("match_graph_config", None),  
        ).to(config["device"])
        
        
        loss_fn = nn.MSELoss(reduction="sum")

        
        trainer.train_supervised(
            model=model,
            epochs=config["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            mask_ratio=config["mask_size"],
            loss_fn=loss_fn,
            model_name=model_name,
            save_dir=config["save_dir"],
            vis_dir=config["vis_dir"],
            use_mlflow=config["use_mlflow"],
        )

        
        config_out = {
            "model_name": model_name,
            "dataset": dataset_name,
            "config": config,
        }
        with open(config["save_dir"] / f"{model_name}_config.json", "w") as f:
            json.dump(config_out, f, indent=4, default=str)

    print("Training and visualization complete.")


if __name__ == "__main__":
    main(DEFAULT_CONFIG)
