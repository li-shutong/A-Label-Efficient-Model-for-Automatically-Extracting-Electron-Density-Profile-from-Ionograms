import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

from model import create_model
from dataset import IonosphericDataset
from train_model_attn import MODEL_CHOICES


def load_model(model_path, model_type, device):
    print(f"加载模型: {model_type}")
    model = create_model(model_type, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    all_filenames = []
    print("在测试集上进行推理...")
    with torch.no_grad():
        for images, targets, filenames in tqdm(test_loader, desc="测试"):
            images = images.to(device)
            predictions = model(images)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_filenames.extend(filenames)
    return {
        "predictions": np.concatenate(all_predictions, axis=0).flatten(),
        "targets": np.concatenate(all_targets, axis=0).flatten(),
        "filenames": all_filenames,
    }


def compute_metrics(targets, predictions):
    mse = mean_squared_error(targets, predictions)
    errors = predictions - targets
    abs_errors = np.abs(errors)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(targets, predictions),
        "R2": r2_score(targets, predictions),
        "Mean Error": np.mean(errors),
        "Std Error": np.std(errors),
        "Max Abs Error": np.max(abs_errors),
        "Min Abs Error": np.min(abs_errors),
        "Median Abs Error": np.median(abs_errors),
        "Target Mean": np.mean(targets),
        "Target Std": np.std(targets),
        "Prediction Mean": np.mean(predictions),
        "Prediction Std": np.std(predictions),
    }


def generate_report(results, metrics, output_dir="test_results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions = results["predictions"]
    targets = results["targets"]
    detail_df = pd.DataFrame({
        "filename": results["filenames"],
        "target": targets,
        "prediction": predictions,
        "error": predictions - targets,
        "abs_error": np.abs(predictions - targets),
    })
    detail_df.to_csv(f"{output_dir}/test_predictions_{timestamp}.csv", index=False)

    report_lines = [
        "=" * 60,
        "电离层参数 foF2 预测模型 - 测试报告",
        "=" * 60,
        f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"测试样本数: {len(targets)}",
        "",
        "-" * 40,
        "评估指标",
        "-" * 40,
        f"  MSE (均方误差):          {metrics['MSE']:.4f}",
        f"  RMSE (均方根误差):       {metrics['RMSE']:.4f}",
        f"  MAE (平均绝对误差):      {metrics['MAE']:.4f}",
        f"  R² (决定系数):           {metrics['R2']:.4f}",
        "",
        "-" * 40,
        "误差统计",
        "-" * 40,
        f"  平均误差:                {metrics['Mean Error']:.4f}",
        f"  误差标准差:              {metrics['Std Error']:.4f}",
        f"  最大绝对误差:            {metrics['Max Abs Error']:.4f}",
        f"  最小绝对误差:            {metrics['Min Abs Error']:.4f}",
        f"  中位数绝对误差:          {metrics['Median Abs Error']:.4f}",
        "",
        "-" * 40,
        "数据分布",
        "-" * 40,
        f"  目标值均值:              {metrics['Target Mean']:.4f}",
        f"  目标值标准差:            {metrics['Target Std']:.4f}",
        f"  预测值均值:              {metrics['Prediction Mean']:.4f}",
        f"  预测值标准差:            {metrics['Prediction Std']:.4f}",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)
    print(report_text)
    with open(f"{output_dir}/test_report_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    return detail_df, report_text


def plot_results(results, metrics, output_dir="test_results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions = results["predictions"]
    targets = results["targets"]
    errors = predictions - targets

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1 = axes[0, 0]
    ax1.scatter(targets, predictions, alpha=0.5, edgecolors="none", s=50)
    min_val = min(targets.min(), predictions.min()) - 0.5
    max_val = max(targets.max(), predictions.max()) + 0.5
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="理想预测线")
    ax1.set_xlabel("目标值 (foF2)", fontsize=12)
    ax1.set_ylabel("预测值 (foF2)", fontsize=12)
    ax1.set_title(f"预测值 vs 目标值 (R² = {metrics['R2']:.4f})", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    ax2 = axes[0, 1]
    ax2.hist(errors, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax2.axvline(x=0, color="r", linestyle="--", lw=2, label="零误差线")
    ax2.axvline(x=metrics["Mean Error"], color="orange", linestyle="-", lw=2,
                label=f"平均误差: {metrics['Mean Error']:.3f}")
    ax2.set_xlabel("预测误差", fontsize=12)
    ax2.set_ylabel("频次", fontsize=12)
    ax2.set_title(f"误差分布 (MAE = {metrics['MAE']:.4f})", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    sorted_indices = np.argsort(targets)
    x_plot = np.arange(len(targets))
    ax3.plot(x_plot, targets[sorted_indices], "b-", lw=1.5, label="目标值", alpha=0.8)
    ax3.plot(x_plot, predictions[sorted_indices], "r-", lw=1.5, label="预测值", alpha=0.8)
    ax3.fill_between(
        x_plot,
        targets[sorted_indices] - metrics["MAE"],
        targets[sorted_indices] + metrics["MAE"],
        alpha=0.2, color="gray", label="±MAE区间",
    )
    ax3.set_xlabel("样本索引 (按目标值排序)", fontsize=12)
    ax3.set_ylabel("foF2", fontsize=12)
    ax3.set_title("预测值与目标值对比", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.scatter(targets, np.abs(errors), alpha=0.5, edgecolors="none", s=50, c="coral")
    ax4.axhline(y=metrics["MAE"], color="r", linestyle="--", lw=2,
                label=f"MAE: {metrics['MAE']:.3f}")
    ax4.set_xlabel("目标值 (foF2)", fontsize=12)
    ax4.set_ylabel("绝对误差", fontsize=12)
    ax4.set_title("绝对误差 vs 目标值", fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/test_visualization_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/test_visualization_{timestamp}.pdf", bbox_inches="tight")
    plt.close()
    print(f"可视化图表已保存到: {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--model_path", type=str, default="best_ionosphere_model.pth")
    parser.add_argument("--model_type", type=str, default="convnext_small", choices=MODEL_CHOICES)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if not os.path.exists(args.test_dir):
        print(f"错误: 测试目录 {args.test_dir} 不存在")
        raise SystemExit(1)
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        raise SystemExit(1)

    test_json_files = [f for f in os.listdir(args.test_dir) if f.endswith(".json")]
    print(f"测试样本: {len(test_json_files)}")
    if len(test_json_files) == 0:
        print("错误: 没有找到测试数据")
        raise SystemExit(1)

    test_dataset = IonosphericDataset(
        args.test_dir, test_json_files, args.input_size, return_filename=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4 if device.type == "cuda" else 0, pin_memory=True,
    )
    model = load_model(args.model_path, args.model_type, device)
    results = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(results["targets"], results["predictions"])
    generate_report(results, metrics)
    plot_results(results, metrics)
    print("\n测试完成!")
    print("结果已保存到: test_results/")
    return metrics


if __name__ == "__main__":
    main()
