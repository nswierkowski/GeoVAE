import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from src.models.geovae.geovae import GeoVAE
from src.models.dummy_geovae.dummy_geovae import DummyGeoVAE
from src.models.vae.vae import VAE
from src.scripts.test_reconstruction.Evaluator import Evaluator
from src.models.vqvae.vqvae import VQVAE

def parse_model_name(model_path: str):
    filename = Path(model_path).stem
    pattern = r"(?P<name>[A-Za-z]+)_layers(?P<layers>\d+)_latent(?P<latent>\d+)_dataset(?P<dataset>.+)"
    m = re.match(pattern, filename)
    if not m:
        raise ValueError(f"Model name {filename} does not match expected format.")
    return m.groupdict()


def dataset_defaults(dataset_name: str):
    if dataset_name.lower() in ["mnist", "fmnist"]:
        return 1, 28
    else:
        return 3, 64


def load_model(model_info, input_dim, hidden_dim=128, residual_hiddens=64, image_size=64, device="cpu"):
    layers = int(model_info["layers"])
    latent_dim = int(model_info["latent"])
    name = model_info["name"]

    if name == "GeoVAE":
        model = GeoVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            residual_hiddens=residual_hiddens,
            num_residual_layers=layers,
            latent_dim=latent_dim,
            image_size=image_size,
        )
    elif name == "DummyGeoVAE":
        model = DummyGeoVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            residual_hiddens=residual_hiddens,
            num_residual_layers=layers,
            latent_dim=latent_dim,
            image_size=image_size,
        )
    else:
        model = VAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            residual_hiddens=residual_hiddens,
            num_residual_layers=layers,
            latent_dim=latent_dim,
            image_size=image_size,
        )
    return model.to(device)

def evaluate_all_models(models_dir: Path, output_csv: Path):
    evaluator = Evaluator()
    results = []
    dataset_cache = {}

    for model_path in models_dir.rglob("*.pt"):
        try:
            model_info = parse_model_name(str(model_path))
        except ValueError as e:
            print(f"Skipping {model_path.name}: {e}")
            continue

        dataset_name = model_info["dataset"]
        input_dim, image_size = dataset_defaults(dataset_name)
        print(f"\nEvaluating {model_path.name} on {dataset_name} "
              f"(input_dim={input_dim}, image_size={image_size}, device={evaluator.device})")

        # --- dataset ---
        if dataset_name not in dataset_cache:
            test_path = Path(f"../data/data_splits/{dataset_name}/{dataset_name}/test.pt")
            if not test_path.exists():
                print(f"Skipping {dataset_name}, missing {test_path}")
                continue
            test_dataset = torch.load(test_path)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
            dataset_cache[dataset_name] = test_loader
        else:
            test_loader = dataset_cache[dataset_name]

        # --- model ---
        model = load_model(model_info, input_dim=input_dim, image_size=image_size, device=evaluator.device)
        model.load_state_dict(torch.load(model_path, map_location=evaluator.device))

        # --- evaluate ---
        metrics = evaluator.evaluate(model, test_loader, real_loader=test_loader, dataset_name=dataset_name)

        results.append({
            "model_file": model_path.name,
            "model_type": model_info["name"],
            "layers": int(model_info["layers"]),
            "latent_dim": int(model_info["latent"]),
            "dataset": dataset_name,
            "input_dim": input_dim,
            "image_size": image_size,
            "recon_mse": metrics.get("recon_mse"),
            "nll": metrics.get("nll"),
            "active_units": metrics.get("active_units"),
        })

    # --- save to CSV ---
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results → {output_csv}")
    

def evaluate_dataset(models_dir: Path, output_csv: Path, dataset: str):
    evaluator = Evaluator()
    results = []
    dataset_cache = {}

    for model_path in models_dir.rglob("*.pt"):
        try:
            model_info = parse_model_name(str(model_path))
        except ValueError as e:
            print(f"Skipping {model_path.name}: {e}")
            continue

        dataset_name = model_info["dataset"]
        print(f'dataset_name: {dataset_name}')
        if dataset != dataset_name:
            continue
        input_dim, image_size = dataset_defaults(dataset_name)
        print(f"\nEvaluating {model_path.name} on {dataset_name} "
              f"(input_dim={input_dim}, image_size={image_size}, device={evaluator.device})")

        # --- dataset ---
        if dataset_name not in dataset_cache:
            test_path = Path(f"../data/data_splits/{dataset_name}/{dataset_name}/test.pt")
            if not test_path.exists():
                print(f"Skipping {dataset_name}, missing {test_path}")
                continue
            
            test_images = torch.load(test_path)
            test_dataset = torch.utils.data.TensorDataset(test_images, test_images)  # (x, x)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

            dataset_cache[dataset_name] = test_loader
        else:
            test_loader = dataset_cache[dataset_name]

        # --- model ---
        model = load_model(model_info, input_dim=input_dim, image_size=image_size, device=evaluator.device)
        model.load_state_dict(torch.load(model_path, map_location=evaluator.device))

        # --- evaluate ---
        metrics = evaluator.evaluate(model, test_loader, real_loader=test_loader, dataset_name=dataset_name)

        results.append({
            "model_file": model_path.name,
            "model_type": model_info["name"],
            "layers": int(model_info["layers"]),
            "latent_dim": int(model_info["latent"]),
            "dataset": dataset_name,
            "input_dim": input_dim,
            "image_size": image_size,
            "recon_mse": metrics.get("recon_mse"),
            "nll": metrics.get("nll"),
            "active_units": metrics.get("active_units"),
        })

    # --- save to CSV ---
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results → {output_csv}")
    
    
def evaluate_all_models_clustering(models_dir: Path, output_csv: Path, base_split_dir: str):
    evaluator = Evaluator()
    results = []
    dataset_cache = {}

    for model_path in models_dir.rglob("*.pt"):
        try:
            model_info = parse_model_name(str(model_path))
        except ValueError as e:
            print(f"Skipping {model_path.name}: {e}")
            continue

        dataset_name = model_info["dataset"]
        input_dim, image_size = dataset_defaults(dataset_name)
        print(f"\n[Clustering] Evaluating {model_path.name} on {dataset_name} "
              f"(input_dim={input_dim}, image_size={image_size}, device={evaluator.device})")

        if dataset_name not in dataset_cache:
            test_path = Path(f"data/data_splits/{dataset_name}/{dataset_name}/test.pt")
            if not test_path.exists():
                print(f"[WARN] Skipping {dataset_name}, missing {test_path}")
                continue
            test_dataset = torch.load(test_path)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
            dataset_cache[dataset_name] = test_loader
        else:
            test_loader = dataset_cache[dataset_name]

        # --- model ---
        model = load_model(model_info, input_dim=input_dim, image_size=image_size, device=evaluator.device)
        model.load_state_dict(torch.load(model_path, map_location=evaluator.device))

        # --- evaluate clustering only ---
        metrics = evaluator.evaluate_only_clustering(model, dataset_name=dataset_name, base_split_dir=base_split_dir, model_name=model_path.name)

        results.append({
            "model_file": model_path.name,
            "model_type": model_info["name"],
            "layers": int(model_info["layers"]),
            "latent_dim": int(model_info["latent"]),
            "dataset": dataset_name,
            "input_dim": input_dim,
            "image_size": image_size,
            "ari": metrics.get("ari"),
            "nmi": metrics.get("nmi"),
            "latent_vis": metrics.get("latent_vis"),
        })

    # --- save to CSV ---
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved clustering results → {output_csv}")

def parse_vqvae_model_name(model_path: str):
    filename = Path(model_path).stem
    pattern = r"VQVAE_layers(?P<layers>\d+)_ne(?P<ne>\d+)_ed(?P<ed>\d+)_dataset(?P<dataset>.+)"
    m = re.match(pattern, filename)
    if not m:
        raise ValueError(f"VQVAE model name {filename} does not match expected format.")
    return m.groupdict()


def load_vqvae_model(model_info, input_dim, image_size, hidden_dim=128, residual_hiddens=64, device="cpu"):
    layers = int(model_info["layers"])
    num_embeddings = int(model_info["ne"])
    embedding_dim = int(model_info["ed"])
    
    model = VQVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        residual_hiddens=residual_hiddens,
        num_residual_layers=layers,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=0.25,
        image_size=image_size
    )
    return model.to(device)


def evaluate_all_vqvae(models_dir: Path, output_csv: Path):
    evaluator = Evaluator()
    results = []
    dataset_cache = {}

    for model_path in models_dir.rglob("*.pt"):
        try:
            model_info = parse_vqvae_model_name(str(model_path))
        except ValueError as e:
            print(f"Skipping {model_path.name}: {e}")
            continue

        dataset_name = model_info["dataset"]
        input_dim, image_size = (1,28) if dataset_name.lower() in ["mnist","fmnist"] else (3,96)
        print(f"\nEvaluating {model_path.name} on {dataset_name} "
              f"(input_dim={input_dim}, image_size={image_size})")

        # --- dataset ---
        if dataset_name not in dataset_cache:
            test_path = Path(f"data/data_splits/{dataset_name}/{dataset_name}/test.pt")
            if not test_path.exists():
                print(f"Skipping {dataset_name}, missing {test_path}")
                continue
            test_dataset = torch.load(test_path)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            dataset_cache[dataset_name] = test_loader
        else:
            test_loader = dataset_cache[dataset_name]

        # --- model ---
        model = load_vqvae_model(model_info, input_dim=input_dim, image_size=image_size)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        # --- evaluate ---
        metrics = evaluator.evaluate(model, test_loader, real_loader=test_loader, dataset_name=dataset_name, is_vqvae=True)

        results.append({
            "model_file": model_path.name,
            "layers": int(model_info["layers"]),
            "num_embeddings": int(model_info["ne"]),
            "embedding_dim": int(model_info["ed"]),
            "dataset": dataset_name,
            "input_dim": input_dim,
            "image_size": image_size,
            "recon_mse": metrics.get("recon_mse"),
            "active_units": metrics.get("active_units"),
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved VQVAE results → {output_csv}")


def evaluate_all_vqvae_clustering(models_dir: Path, output_csv: Path, base_split_dir: str):
    evaluator = Evaluator()
    results = []
    dataset_cache = {}

    for model_path in models_dir.rglob("*.pt"):
        try:
            model_info = parse_vqvae_model_name(str(model_path))
        except ValueError as e:
            print(f"Skipping {model_path.name}: {e}")
            continue

        dataset_name = model_info["dataset"]
        input_dim, image_size = (1,28) if dataset_name.lower() in ["mnist","fmnist"] else (3,96)
        print(f"\n[Clustering] Evaluating {model_path.name} on {dataset_name}")

        # --- dataset ---
        if dataset_name not in dataset_cache:
            test_path = Path(f"{base_split_dir}/{dataset_name}/{dataset_name}/test.pt")
            if not test_path.exists():
                print(f"[WARN] Skipping {dataset_name}, missing {test_path}")
                continue
            test_dataset = torch.load(test_path)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            dataset_cache[dataset_name] = test_loader
        else:
            test_loader = dataset_cache[dataset_name]

        # --- model ---
        model = load_vqvae_model(model_info, input_dim=input_dim, image_size=image_size)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        # --- clustering ---
        metrics = evaluator.evaluate_only_clustering(model, dataset_name=dataset_name, base_split_dir=base_split_dir, model_name=model_path.name)

        results.append({
            "model_file": model_path.name,
            "layers": int(model_info["layers"]),
            "num_embeddings": int(model_info["ne"]),
            "embedding_dim": int(model_info["ed"]),
            "dataset": dataset_name,
            "input_dim": input_dim,
            "image_size": image_size,
            "ari": metrics.get("ari"),
            "nmi": metrics.get("nmi"),
            "latent_vis": metrics.get("latent_vis"),
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)