import json
import os
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


def load_rgb_tensor(img_path: str, input_size: int) -> Optional[torch.Tensor]:
    if sys.platform == "win32":
        file_bytes = np.fromfile(os.path.normpath(img_path), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(img_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1)


def parse_fof2(iono_params: dict) -> Optional[float]:
    foF2 = iono_params.get("foF2", None)
    if foF2 is None or foF2 == 0.0:
        return None
    return float(foF2)


class IonosphericDataset(Dataset):
    def __init__(self, data_dir: str, json_files: List[str],
                 input_size: int = 224, return_filename: bool = False):
        self.return_filename = return_filename
        self.images = []
        self.targets = []
        self.filenames = []

        print(f"加载数据集: {len(json_files)} 个样本")
        for json_file in tqdm(json_files, desc="加载数据"):
            json_file_decoded = os.fsdecode(json_file)
            img_path = os.path.join(data_dir, json_file_decoded.replace(".json", ".jpg"))
            if not os.path.exists(img_path):
                continue
            image = load_rgb_tensor(img_path, input_size)
            if image is None:
                continue
            json_path = os.path.join(data_dir, json_file)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            target = parse_fof2(data["主要电离层参数"])
            if target is None:
                continue
            self.images.append(image)
            self.targets.append(target)
            self.filenames.append(json_file)
        print(f"成功加载 {len(self.images)} 个样本")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].clone()
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        if self.return_filename:
            return image, target, self.filenames[idx]
        return image, target


class IonosphericLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, delta=0.5, min_val=1.0, max_val=20.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.min_val = min_val
        self.max_val = max_val
        self.mse_loss = 0.0
        self.mae_loss = 0.0
        self.mse_loss_normalized = 0.0
        self.mae_loss_normalized = 0.0
        self.range_loss = 0.0

    def forward(self, predictions, targets):
        predictions = predictions.squeeze(-1)
        targets = targets.squeeze(-1)
        mse_loss = nn.functional.mse_loss(predictions, targets)
        mae_loss = nn.functional.l1_loss(predictions, targets)
        center = (self.min_val + self.max_val) / 2
        scale = (self.max_val - self.min_val) / 2
        mse_loss_normalized = mse_loss / (scale ** 2)
        mae_loss_normalized = mae_loss / scale
        normalized_preds = (predictions - center) / scale
        range_loss = torch.mean(nn.functional.silu(torch.abs(normalized_preds) - 1) ** 2)
        total_loss = (
            self.alpha * mse_loss_normalized
            + self.beta * mae_loss_normalized
            + self.delta * range_loss
        )
        self.mse_loss = mse_loss.item()
        self.mae_loss = mae_loss.item()
        self.mse_loss_normalized = mse_loss_normalized.item()
        self.mae_loss_normalized = mae_loss_normalized.item()
        self.range_loss = range_loss.item()
        return total_loss
