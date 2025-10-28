from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState
from src.models.geovae.geovae import GeoVAE, GraphConvType
from src.scripts.etl_process.ETLProcessor import ETLProcessor
from src.training.MaskDataset import MaskedDataset
from src.training.TrainerTuning import TrainerTuning

CONFIG = {
    "input_dim": 3,
    "hidden_dim": 128,
    "residual_hiddens": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "image_size": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mask_size": 0.35,
    "save_dir": Path("models/reconstruction/geovae_optuna/"),
    "dataset_config": {
        "dataset_name": "mahmudulhaqueshawon/cat-image",
        "raw_dir": "data/raw_data/cats/raw",
        "split_dir": "data/data_splits/cats",
    },
    "use_mlflow": True,
    "mask_random_state": 42,
}

CONFIG["vis_dir"] = CONFIG["save_dir"] / "metrics"
CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["vis_dir"].mkdir(parents=True, exist_ok=True)

def get_free_device(trial):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_id = trial.number % num_gpus
        return f"cuda:{device_id}"
    else:
        return "cpu"

def objective(trial):
    device = get_free_device(trial)
    dataset_name = CONFIG["dataset_config"]["dataset_name"]

    latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128, 256, 512])
    num_residual_layers = trial.suggest_int("num_residual_layers", 1, 4)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, log=True)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    mask_size = CONFIG["mask_size"]

    graph_conv_map = {
        "GCN": GraphConvType.GCN,
        "GAT": GraphConvType.GAT,
        "SAGE": GraphConvType.GST,
    }
    graph_conv_str = trial.suggest_categorical("graph_conv", list(graph_conv_map.keys()))
    graph_conv = graph_conv_map[graph_conv_str]

    num_inst_gnn_layers = trial.suggest_int("num_inst_gnn_layers", 1, 4)
    num_dim_gnn_layers = trial.suggest_int("num_dim_gnn_layers", 1, 3)

    optimizer_config = {
        "hidden_dim": trial.suggest_int("opt_hidden_dim", 32, 256, log=True),
        "num_gnn_layers": trial.suggest_int("opt_num_gnn_layers", 1, 3),
        "graph_conv_type": trial.suggest_categorical("opt_graph_conv", ["GCN", "GAT", "GST"]),
    }

    etl = ETLProcessor(**CONFIG["dataset_config"])
    train_loader, val_loader, _ = etl.process()

    if mask_size > 0:
        masked_train_ds = MaskedDataset(
            train_loader.dataset, mask_size, random_state=CONFIG["mask_random_state"]
        )
        masked_val_ds = MaskedDataset(
            val_loader.dataset, mask_size, random_state=CONFIG["mask_random_state"]
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
            drop_last=True
        )

    model = GeoVAE(
        input_dim=CONFIG["input_dim"],
        hidden_dim=hidden_dim,
        residual_hiddens=CONFIG["residual_hiddens"],
        num_residual_layers=num_residual_layers,
        latent_dim=latent_dim,
        image_size=CONFIG["image_size"],
        graph_conv_type=graph_conv,
        optimizer_config=optimizer_config,
        num_dim_gnn_layers=num_dim_gnn_layers,
        num_inst_gnn_layers=num_inst_gnn_layers
    ).to(device)

    loss_fn = nn.MSELoss(reduction="sum")
    trainer = TrainerTuning(device=device)

    model_name = (
        f"GeoVAE_trial{trial.number}_latent{latent_dim}_h{hidden_dim}_mask{mask_size:.2f}_"
        f"optH{optimizer_config['hidden_dim']}_optL{optimizer_config['num_gnn_layers']}"
    ).replace(".", "")

    val_loss = trainer.train_for_tuning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG["epochs"],
        lr=lr,
        weight_decay=CONFIG["weight_decay"],
        mask_ratio=mask_size,
        loss_fn=loss_fn,
        trial=trial,
        model_name=model_name,
        save_dir=str(CONFIG["save_dir"]),
    )

    trial.report(val_loss, step=CONFIG["epochs"])
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss

def main():
    print("=== GeoVAE Bayesian Optimization (with pruning + parallel) ===")
    print(f"Device: {CONFIG['device']}")

    study_name = "GeoVAE_BayesOpt"
    storage = f"sqlite:///{CONFIG['save_dir']}/optuna_study.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=5),
    )

    study.optimize(
        objective,
        n_trials=20,
        n_jobs=1,
        timeout=None,
    )

    print("\n=== Optimization complete ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (val_loss): {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    best_config_path = CONFIG["save_dir"] / "best_trial_config.json"
    with open(best_config_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print(f"Saved best configuration to: {best_config_path}")

if __name__ == "__main__":
    main()
