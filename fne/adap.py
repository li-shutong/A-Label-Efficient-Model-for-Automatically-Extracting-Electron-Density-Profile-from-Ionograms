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
from train_model_attn import MODEL_CHOICES


def freeze_backbone(model):
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False
    if hasattr(model, "regression_head"):
        for param in model.regression_head.parameters():
            param.requires_grad = True
    if hasattr(model, "coordinate_attn"):
        for param in model.coordinate_attn.parameters():
            param.requires_grad = True
    if hasattr(model, "vit_to_cam"):
        for param in model.vit_to_cam.parameters():
            param.requires_grad = True
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--val_dir", type=str, default="val")
    parser.add_argument("--model_type", type=str, default="convnext_small", choices=MODEL_CHOICES)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--weight_path", type=str, default="best_ionosphere_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if not os.path.exists(args.train_dir):
        print(f"错误: 训练目录 {args.train_dir} 不存在")
        return
    if not os.path.exists(args.val_dir):
        print(f"错误: 验证目录 {args.val_dir} 不存在")
        return

    train_json_files = [f for f in os.listdir(args.train_dir) if f.endswith(".json")]
    val_json_files = [f for f in os.listdir(args.val_dir) if f.endswith(".json")]
    print(f"训练样本: {len(train_json_files)}")
    print(f"验证样本: {len(val_json_files)}")
    if len(train_json_files) == 0 or len(val_json_files) == 0:
        print("错误: 没有找到训练或验证数据")
        return

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
    model = create_model(args.model_type, pretrained=False)
    if not os.path.exists(args.weight_path):
        print(f"错误: 权重文件 {args.weight_path} 不存在")
        return
    print(f"加载预训练权重: {args.weight_path}")
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model = freeze_backbone(model).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)")

    config = {
        "learning_rate": args.lr,
        "patience": args.patience,
        "weight_decay": 5e-5,
        "max_grad_norm": 0.5,
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
    print(f"- 数据增强: {args.augment}")
    print(f"- 冻结backbone: True")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 5e-5),
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-7
    )
    loss_config = config["loss"]
    criterion = IonosphericLoss(
        alpha=loss_config["alpha"],
        beta=loss_config["beta"],
        delta=loss_config["delta"],
        min_val=loss_config["min_val"],
        max_val=loss_config["max_val"],
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"开始训练，共 {args.epochs} 个epoch")
    print(f"使用设备: {device}")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        model.train()
        total_loss = 0
        batch_count = len(train_loader)
        progress_bar = tqdm(train_loader, desc="训练中")
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            if config.get("max_grad_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device).unsqueeze(1)
                predictions = model(images)
                loss = criterion(predictions, targets)
                total_val_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        scale = (criterion.max_val - criterion.min_val) / 2
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"验证损失: {avg_val_loss:.4f}")
        print(f"验证MSE: {mse / (scale ** 2):.4f}")
        print(f"验证MAE: {mae / scale:.4f}")
        print(f"学习率: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_ionosphere_model.pth")
            print("保存最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 15):
                print(f"早停触发，验证损失在 {config.get('patience', 15)} 个epoch内没有改善")
                break

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
    plt.savefig("transfer_learning_history.jpg", dpi=300, bbox_inches="tight")
    print("迁移学习完成!")


if __name__ == "__main__":
    main()
