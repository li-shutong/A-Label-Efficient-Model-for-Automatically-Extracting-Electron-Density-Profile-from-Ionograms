import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


def auto_crop_ionogram(img: Image.Image) -> Image.Image:
    gray = np.array(img.convert("L"))
    mask = gray < 250
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


def get_transforms(img_size: int, augment: bool = True):
    transforms_list = [
        T.Lambda(auto_crop_ionogram),
        T.Resize((img_size, img_size),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ]

    if augment:
        transforms_list.insert(2, T.ColorJitter(brightness=0.05, contrast=0.05,
                                                  saturation=0.02, hue=0.01))

    return T.Compose(transforms_list)


class IonogramDataset(Dataset):
    SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(self, root_A: str, root_B: str, img_size: int = 512,
                 augment: bool = True):
        self.transform = get_transforms(img_size, augment)
        self.files_A = self._load_paths(root_A)
        self.files_B = self._load_paths(root_B)
        print(f"[Dataset] Domain A (理论): {len(self.files_A)} 张")
        print(f"[Dataset] Domain B (真实): {len(self.files_B)} 张")
        self.length = max(len(self.files_A), len(self.files_B))

    def _load_paths(self, root: str):
        if not os.path.exists(root):
            raise FileNotFoundError(f"数据目录不存在: {root}")
        paths = [
            os.path.join(root, f) for f in sorted(os.listdir(root))
            if os.path.splitext(f)[1].lower() in self.SUPPORTED_EXT
        ]
        if len(paths) == 0:
            raise ValueError(f"目录中没有找到图像: {root}")
        return paths

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path_A = self.files_A[idx % len(self.files_A)]
        path_B = self.files_B[idx % len(self.files_B)]
        img_A = Image.open(path_A).convert("RGB")
        img_B = Image.open(path_B).convert("RGB")
        return {
            "A": self.transform(img_A),
            "B": self.transform(img_B),
            "path_A": path_A,
            "path_B": path_B,
        }


def get_dataloader(cfg, augment=True):
    dataset = IonogramDataset(
        root_A=cfg.TRAIN_A,
        root_B=cfg.TRAIN_B,
        img_size=cfg.IMG_SIZE,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=augment,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
