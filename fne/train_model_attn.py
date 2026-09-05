import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from model import create_model
from dataset import IonosphericDataset, IonosphericLoss

MODEL_CHOICES = [
    "resnet18", "resnet34", "resnet50", "resnet101",
    "vit_base_patch16_224", "vit_base_patch16_384",
    "vit_large_patch16_224", "vit_small_patch16_224",
    "convnext_tiny", "convnext_small", "convnext_base",
    "convnext_large", "convnext_xlarge",
]


class ImprovedTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.avg_mse_loss = 0.0
        self.avg_mae_loss = 0.0
        self.avg_normalized_mse = 0.0
        self.avg_normalized_mae = 0.0
        self.avg_range_loss = 0.0
        self.avg_grad_norm = 0.0

        if config.get("use_different_lr", True):
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            self.optimizer = optim.Adam(
                [
                    {"params": backbone_params, "lr": config["learning_rate"] * 0.1},
                    {"params": head_params, "lr": config["learning_rate"]},
                ],
                weight_decay=config.get("weight_decay", 5e-5),
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config.get("weight_decay", 5e-5),
            )

        T_max = len(train_loader) * 50
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max, eta_min=1e-6)

        loss_config = config.get("loss", {})
        self.criterion = IonosphericLoss(
            alpha=loss_config.get("alpha", 1.0),
            beta=loss_config.get("beta", 1.0),
            delta=loss_config.get("delta", 0.5),
            min_val=loss_config.get("min_val", 1.0),
            max_val=loss_config.get("max_val", 20.0),
        )
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_mse_loss = 0
        total_mae_loss = 0
        total_normalized_mse = 0
        total_normalized_mae = 0
        total_range_loss = 0
        total_grad_norm = 0
        grad_count = 0
        batch_count = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc="训练中")
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device).unsqueeze(1)
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)
            loss.backward()
            if self.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                total_grad_norm += grad_norm.item()
                grad_count += 1
            self.optimizer.step()
            total_loss += loss.item()
            total_mse_loss += self.criterion.mse_loss
            total_mae_loss += self.criterion.mae_loss
            total_normalized_mse += self.criterion.mse_loss_normalized
            total_normalized_mae += self.criterion.mae_loss_normalized
            total_range_loss += self.criterion.range_loss
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        denom = batch_count if batch_count > 0 else 1
        self.avg_mse_loss = total_mse_loss / denom
        self.avg_mae_loss = total_mae_loss / denom
        self.avg_normalized_mse = total_normalized_mse / denom
        self.avg_normalized_mae = total_normalized_mae / denom
        self.avg_range_loss = total_range_loss / denom
        self.avg_grad_norm = total_grad_norm / grad_count if grad_count > 0 else 0
        return total_loss / denom if batch_count > 0 else 0

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device).unsqueeze(1)
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        scale = (self.criterion.max_val - self.criterion.min_val) / 2
        return total_loss / len(self.val_loader), {
            "mse": mse / (scale ** 2),
            "mae": mae / scale,
        }

    def train(self, epochs):
        print(f"开始训练，共 {epochs} 个epoch")
        print(f"使用设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"训练损失: {train_loss:.4f}")
            print(f"训练MSE损失: {self.avg_mse_loss:.4f} (归一化: {self.avg_normalized_mse:.4f})")
            print(f"训练MAE损失: {self.avg_mae_loss:.4f} (归一化: {self.avg_normalized_mae:.4f})")
            print(f"训练范围约束损失: {self.avg_range_loss:.4f}")
            print(f"梯度范数: {self.avg_grad_norm:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"验证MSE: {val_metrics['mse']:.4f}")
            print(f"验证MAE: {val_metrics['mae']:.4f}")
            print(f"学习率: {current_lr:.6f}")
            print(
                f"当前损失函数参数: alpha={self.criterion.alpha:.2f}, "
                f"beta={self.criterion.beta:.2f}, delta={self.criterion.delta:.2f}"
            )
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), "best_ionosphere_model.pth")
                print("保存最佳模型")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.get("patience", 15):
                    print(
                        f"早停触发，验证损失在 {self.config.get('patience', 15)} 个epoch内没有改善"
                    )
                    break
        return self.train_losses, self.val_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--val_dir", type=str, default="val")
    parser.add_argument("--model_type", type=str, default="convnext_small", choices=MODEL_CHOICES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if not os.path.exists(args.train_dir):
        print(f"错误: 训练目录 {args.train_dir} 不存在")
        raise SystemExit(1)
    if not os.path.exists(args.val_dir):
        print(f"错误: 验证目录 {args.val_dir} 不存在")
        raise SystemExit(1)

    train_json_files = [f for f in os.listdir(args.train_dir) if f.endswith(".json")]
    val_json_files = [f for f in os.listdir(args.val_dir) if f.endswith(".json")]
    print(f"训练样本: {len(train_json_files)}")
    print(f"验证样本: {len(val_json_files)}")
    if len(train_json_files) == 0 or len(val_json_files) == 0:
        print("错误: 没有找到训练或验证数据")
        raise SystemExit(1)

    print("创建数据集...")
    train_dataset = IonosphericDataset(args.train_dir, train_json_files, args.input_size)
    val_dataset = IonosphericDataset(args.val_dir, val_json_files, args.input_size)
    workers = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    print(f"创建模型: {args.model_type}")
    model = create_model(args.model_type, pretrained=args.pretrained).to(device)
    config = {
        "learning_rate": args.lr,
        "patience": args.patience,
        "weight_decay": 5e-5,
        "max_grad_norm": 0.5,
        "use_different_lr": True,
        "loss": {
            "alpha": args.alpha,
            "beta": args.beta,
            "delta": args.delta,
            "min_val": 1.0,
            "max_val": 20.0,
        },
    }
    print(f"\n训练配置:")
    print(f"- 模型类型: {args.model_type}")
    print(f"- 训练轮数: {args.epochs}")
    print(f"- 批处理大小: {args.batch_size}")
    print(f"- 学习率: {args.lr}")
    print(f"- 损失函数参数: alpha={args.alpha:.2f}, beta={args.beta:.2f}, delta={args.delta:.2f}")
    print(f"- 使用不同学习率: {config['use_different_lr']}")
    print(f"- 预训练权重: {args.pretrained}")
    print(f"- 数据增强: {args.augment}")

    trainer = ImprovedTrainer(model, train_loader, val_loader, device, config)
    train_losses, val_losses = trainer.train(args.epochs)
    torch.save(model.state_dict(), "final_ionosphere_model.pth")
    print("最终模型已保存")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()
    plt.title("训练历史")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失 (对数)")
    plt.yscale("log")
    plt.legend()
    plt.title("训练历史 (对数)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ionosphere_training_history.jpg", dpi=300, bbox_inches="tight")
    print("训练完成!")


if __name__ == "__main__":
    main()
