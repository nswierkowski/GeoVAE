from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
from torch.utils.data import DataLoader
import torch
import os
import json
from torch.utils.data import RandomSampler
from src.training.Trainer import Trainer
from src.training.MaskDataset import MaskedDataset


@dataclass
class SingleTrainEvent:
    """
    A single training event containing all necessary parameters for training a model.
    """

    epochs: int
    train_loader: DataLoader
    val_loader: DataLoader
    lr: float
    loss_fn: Callable
    loss_fn_args: Optional[Tuple[Any]]
    mask_size: float = 0.0
    save_dir: Optional[str] = None
    dropout: Optional[float] = None
    weight_decay = (0,)
    model_name = ("default",)
    vis_dir = "."


class TrainScheduler:
    """
    TrainScheduler orchestrates the training process for a model using multiple training events.
    It manages the training loop, including epochs, data loaders, and loss functions.
    Attributes:
        train_events (List[SingleTrainEvent]): List of training events to be executed.
        trainer (Trainer): Instance of the Trainer class to handle training logic.
        model (Any): The model to be trained.
    """

    def __init__(
        self,
        SingleTrainEvents: List[SingleTrainEvent],
        model: Any,
        model_name: str = "Model",
    ):
        self.train_events = SingleTrainEvents
        self.trainer: Trainer = Trainer()
        self.model = model
        self.model_name = model_name

    def start_training(self):
        for i, event in enumerate(self.train_events):
            print(f"Start {i} training event with {event.epochs} epochs.")
            self.run_event(event, index=i)
        print("All training events completed.")

    def prepare_event(self, event: SingleTrainEvent):
        print(
            f"[INFO] Preparing event: epochs={event.epochs}, lr={event.lr}, mask_ratio={event.mask_size}"
        )

        masked_train_ds = MaskedDataset(event.train_loader.dataset, event.mask_size)
        masked_val_ds = MaskedDataset(event.val_loader.dataset, event.mask_size)

        def clone_loader(orig_loader, new_ds):
            sampler = orig_loader.sampler
            has_shuffle = isinstance(sampler, RandomSampler)

            return DataLoader(
                new_ds,
                batch_size=orig_loader.batch_size,
                shuffle=has_shuffle,
                num_workers=orig_loader.num_workers,
                pin_memory=orig_loader.pin_memory,
                drop_last=orig_loader.drop_last,
            )

        self.train_loader = clone_loader(event.train_loader, masked_train_ds)
        self.val_loader = clone_loader(event.val_loader, masked_val_ds)

    def run_event(self, event: SingleTrainEvent, index: int):
        """Prepare data, train model, and save results."""
        self.prepare_event(event)

        save_dir = event.save_dir or os.path.join("data/training/", self.model_name)
        os.makedirs(save_dir, exist_ok=True)

        train_metrics, val_metrics = self.trainer.train_supervised(
            model=self.model,
            epochs=event.epochs,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            lr=event.lr,
            loss_fn=event.loss_fn,
            mask_ratio=event.mask_size,
            weight_decay=0,
            model_name="",
            save_dir=".",
            vis_dir=".",
        )

        model_path = os.path.join(save_dir, f"{self.model_name}_event{index}.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path} (event {index})")

        metrics = {"train": train_metrics, "val": val_metrics}
        metrics_path = os.path.join(
            save_dir, f"{self.model_name}_event{index}_metrics.json"
        )
        with open(metrics_path, "w") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"Metrics saved to {metrics_path}")

    def run_auto_event(self, target_event: SingleTrainEvent):
        print("[AUTO] Starting auto curriculum training")

        max_epochs = target_event.epochs
        final_noise = target_event.noise_level
        full_train_data = target_event.train_loader.dataset
        val_loader = target_event.val_loader
        test_loader = target_event.test_loader
        loss_fn = target_event.loss_fn
        lr = target_event.lr

        # Initial state
        current_mask = 0.05  # Start small
        current_data_frac = 0.1
        used_epochs = 0
        step_epochs = 10

        previous_acc = 0.0

        while used_epochs < max_epochs:
            print(f"\n[AUTO] Epochs used: {used_epochs}/{max_epochs}")

            # Create subset of dataset
            subset_size = int(len(full_train_data) * current_data_frac)
            indices = torch.randperm(len(full_train_data))[:subset_size]
            subset = torch.utils.data.Subset(full_train_data, indices)
            noisy_subset = MaskedDataset(subset, current_mask)

            train_loader = DataLoader(
                noisy_subset,
                batch_size=target_event.train_loader.batch_size,
                shuffle=True,
                num_workers=0,
            )

            # Train small event
            event = SingleTrainEvent(
                epochs=step_epochs,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                lr=lr,
                loss_fn=loss_fn,
                loss_fn_args=target_event.loss_fn_args,
                mask_size=current_mask,
                dropout=target_event.dropout,
            )
            self.run_event(event, index=used_epochs)

            # Evaluate model (basic val accuracy estimate)
            val_loss = self.trainer._validate_model(
                self.model, val_loader, loss_fn, {}, used_epochs
            )
            acc = 1.0 - val_loss if val_loss is not None else 0.0

            # Update epochs used
            used_epochs += step_epochs

            # Estimate improvement
            improvement = acc - previous_acc
            expected_progress = (
                current_mask / final_noise
            ) * 0.1  # simplistic heuristic

            print(
                f"[AUTO] Accuracy: {acc:.4f}, Improvement: {improvement:.4f}, Expected: {expected_progress:.4f}"
            )

            if acc < previous_acc + expected_progress:
                # Too slow -> smaller step
                current_data_frac = min(current_data_frac + 0.1, 1.0)
                current_mask = min(current_mask + 0.05, final_noise)
                step_epochs = min(15, max_epochs - used_epochs)
            else:
                # Doing well -> larger step
                current_data_frac = min(current_data_frac + 0.2, 1.0)
                current_mask = min(current_mask + 0.1, final_noise)
                step_epochs = min(25, max_epochs - used_epochs)

            previous_acc = acc

            if current_data_frac >= 1.0 and current_mask >= final_noise:
                print("[AUTO] Curriculum completed early.")
                break

        print("[AUTO] Final fine-tuning on full dataset")
        final_dataset = MaskedDataset(full_train_data, final_noise)
        full_train_loader = DataLoader(
            final_dataset,
            batch_size=target_event.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )

        final_event = SingleTrainEvent(
            epochs=max_epochs - used_epochs,
            train_loader=full_train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=lr,
            loss_fn=loss_fn,
            loss_fn_args=target_event.loss_fn_args,
            noise_level=final_noise,
            dropout=target_event.dropout,
        )
        self.run_event(final_event, index="final")
        print("[AUTO] Auto curriculum training completed.")
