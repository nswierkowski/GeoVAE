from torch.utils.data import Dataset
import random


class MaskedDataset(Dataset):
    def __init__(
        self, clean_dataset: Dataset, mask_size_ratio: float, random_state: int = 0
    ):
        """
        :param clean_dataset: Dataset providing clean images.
        :param mask_size_ratio: Size of square mask as a fraction of image size (e.g., 0.3 masks a 30% wide/tall square).
        :param fixed_masks: If True, masks positions are fixed across calls.
        """
        self.clean = clean_dataset
        self.mask_size_ratio = mask_size_ratio

        random.seed(random_state)

        self.masks = []
        for _ in range(len(self.clean)):
            H, W = self.clean[0][0].shape[1:3]
            random_ratio = random.uniform(0.15, self.mask_size_ratio)
            mask_size = int(min(H, W) * random_ratio)
            top = random.randint(0, H - mask_size)
            left = random.randint(0, W - mask_size)
            self.masks.append((top, left, mask_size))

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean_img, _ = self.clean[idx]
        masked_img = clean_img.clone()

        top, left, mask_size = self.masks[idx]

        masked_img[:, top : top + mask_size, left : left + mask_size] = 0.5

        return masked_img, clean_img
