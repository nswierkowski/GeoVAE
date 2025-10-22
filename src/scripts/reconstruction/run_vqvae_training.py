from pathlib import Path
import itertools
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.vqvae.vqvae import VQVAE
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
    "image_size": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mask_size": 0.35,
    "param_grid": {
        "num_residual_layers": [1],
        "num_embeddings": [64],
        "embedding_dim": [32],
    },
    "save_dir": Path("models/reconstruction/vqvae/"),
    "dataset_config": {
        "dataset_name": "thetthetyee/celaba-face-recognition",
        "raw_dir": "data/raw_data/celaba/raw",
        "split_dir": "data/data_splits/celaba",
    },
    "use_mlflow": True,
    "mask_random_state": 42,
}

CONFIG["vis_dir"] = CONFIG["save_dir"] / "metrics"
CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["vis_dir"].mkdir(parents=True, exist_ok=True)
 

def main():
    dataset_name = CONFIG["dataset_config"]["dataset_name"]
    print("Preparing data...")
    print(f"Dataset: {dataset_name}")
    print(f"Using device: {CONFIG['device']}")

    # === ETL and Dataloaders ===
    etl = ETLProcessor(**CONFIG["dataset_config"])
    train_loader, val_loader, _ = etl.process()

    if CONFIG["mask_size"] > 0:
        print(f"Applying masked dataset transformation with mask ratio = {CONFIG['mask_size']}")
        masked_train_ds = MaskedDataset(
            train_loader.dataset, CONFIG["mask_size"], random_state=CONFIG["mask_random_state"]
        )
        masked_val_ds = MaskedDataset(
            val_loader.dataset, CONFIG["mask_size"], random_state=CONFIG["mask_random_state"]
        )

        train_loader = DataLoader(
            masked_train_ds,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            drop_last=train_loader.drop_last,
        )
        val_loader = DataLoader(
            masked_val_ds,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
            drop_last=val_loader.drop_last,
        )

    # === Grid Search ===
    param_combinations = list(itertools.product(*CONFIG["param_grid"].values()))
    total_configs = len(param_combinations)
    print(f"Total configurations to run: {total_configs}")

    trainer = Trainer()

    for i, (layers, num_embeddings, embedding_dim) in enumerate(param_combinations):
        model_name = (
            f"VQVAE_layers{layers}_ne{num_embeddings}_ed{embedding_dim}_dataset{dataset_name}"
            .replace(".", "")
            .replace("/", "_")
        )
        print(f"\n[{i + 1}/{total_configs}] Running: {model_name}")

        # === Model ===
        model = VQVAE(
            input_dim=CONFIG["input_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            residual_hiddens=CONFIG["residual_hiddens"],
            num_residual_layers=layers,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=0.25,
            device=CONFIG["device"],
            image_size=CONFIG["image_size"]
        ).to(CONFIG["device"])

        loss_fn = nn.MSELoss(reduction="sum")

        # === Resume Checkpoint ===
        ckpt_files = sorted(CONFIG["save_dir"].glob(f"{model_name}_ckpt_epoch*.pt"))
        resume_ckpt = ckpt_files[-1] if ckpt_files else None
        if resume_ckpt:
            print(f"Resuming training from checkpoint: {resume_ckpt.name}")

        # === Train ===
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
            use_mlflow=CONFIG["use_mlflow"],
            resume_from_cpkt=resume_ckpt
        )

        # === Save Configuration ===
        config_out = {
            "model_name": model_name,
            "dataset": dataset_name,
            "config": CONFIG,
        }
        with open(CONFIG["save_dir"] / f"{model_name}_config.json", "w") as f:
            json.dump(config_out, f, indent=4, default=str)

    print("Training and visualization complete.")


if __name__ == "__main__":
    main()
