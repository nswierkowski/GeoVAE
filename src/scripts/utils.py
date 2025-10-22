from src.scripts.etl_process.ETLProcessor import ETLProcessor
from src.training.MaskDataset import MaskedDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def get_test_loader(dataset_name="mahmudulhaqueshawon/cat-image",
        raw_dir="data/raw_data",
        split_dir="data/data_splits"):
    """
    Get the test DataLoader for the cat image dataset.
    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    etl = ETLProcessor(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        split_dir=split_dir,
    )
    _, _, test_loader = etl.process()
    mask_size = 0.3
    mask_test_dataset = MaskedDataset(test_loader.dataset, mask_size)
    test_loader = torch.utils.data.DataLoader(
        mask_test_dataset, batch_size=32, shuffle=False
    )

    return test_loader

def get_train_loader(dataset_name="mahmudulhaqueshawon/cat-image",
        raw_dir="data/raw_data",
        split_dir="data/data_splits"):
    """
    Get the test DataLoader for the cat image dataset.
    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    etl = ETLProcessor(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        split_dir=split_dir,
    )
    train, _, _ = etl.process()
    mask_size = 0.35
    mask_test_dataset = MaskedDataset(train.dataset, mask_size)
    train = torch.utils.data.DataLoader(
        mask_test_dataset, batch_size=32, shuffle=False
    )

    return train

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.
    Args:
        model (nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').
    Returns:
        tuple: (average MSE, average SSIM, average PSNR)
    """
    model.eval()
    model = model.to(device)

    total_mse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    num_samples = 0

    with torch.no_grad():
        for noisy, original in test_loader:
            noisy = noisy.to(device)
            original = original.to(device)

            out = model(noisy)
            reconstructed = out["recon"]

            original_np = ((original.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1)
            recon_np = ((reconstructed.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1)
            recon_np = np.clip(recon_np, 0, 1)

            batch_size = original_np.shape[0]

            for i in range(batch_size):
                mse = np.mean((original_np[i] - recon_np[i]) ** 2)
                ssim = compare_ssim(
                    original_np[i], recon_np[i], data_range=1.0, multichannel=True, channel_axis=-1
                )
                psnr = compare_psnr(original_np[i], recon_np[i], data_range=1.0)

                total_mse += mse
                total_ssim += ssim
                total_psnr += psnr

            num_samples += batch_size

    avg_mse = total_mse / num_samples
    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples

    print(f"Test MSE: {avg_mse:.4f}")
    print(f"Test SSIM: {avg_ssim:.4f}")
    print(f"Test PSNR: {avg_psnr:.4f}")

def show_images(original, noisy, reconstructed, n=8, should_denormalize=True):

    if should_denormalize:
        original = denormalize(original)
        noisy = denormalize(noisy)
        reconstructed = denormalize(reconstructed)
    n = min(n, original.size(0), noisy.size(0), reconstructed.size(0))

    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))

    if isinstance(axes, np.ndarray) and axes.ndim == 1:
        axes = axes.reshape(3, -1)

    for i in range(n):
        axes[0, i].imshow(original[i].permute(1, 2, 0).cpu().clip(0, 1).numpy())
        axes[0, i].axis("off")

        axes[1, i].imshow(noisy[i].permute(1, 2, 0).cpu().clip(0, 1).numpy())
        axes[1, i].axis("off")

        axes[2, i].imshow(reconstructed[i].permute(1, 2, 0).cpu().clip(0, 1).numpy())
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Noisy", fontsize=12)
    axes[2, 0].set_ylabel("Reconstructed", fontsize=12)

    plt.tight_layout()
    plt.show()
    
def denormalize(img: torch.Tensor) -> torch.Tensor:
    img = img * 0.5 + 0.5
    return img.clamp(0, 1)
