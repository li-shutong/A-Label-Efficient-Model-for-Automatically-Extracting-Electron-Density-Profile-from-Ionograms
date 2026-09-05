import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

import torch
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm

from dataset import auto_crop_ionogram
from config import cfg
from models import Generator


class MonitorConfig:
    def __init__(self):
        self.TRAIN_A_DIR = cfg.TRAIN_A
        self.TRAIN_B_DIR = cfg.TRAIN_B
        self.CHECKPOINT_DIR = cfg.CHECKPOINT_DIR
        self.MONITOR_DIR = "monitor"
        self.TSNE_SAMPLES = 300
        self.TSNE_PERPLEXITY = 40
        self.TSNE_MAX_ITER = 1000
        self.IMG_SIZE = cfg.IMG_SIZE
        self.IN_CHANNELS = cfg.IN_CHANNELS
        self.N_FEATURES = cfg.N_FEATURES
        self.N_RESIDUAL = cfg.N_RESIDUAL


class FeatureExtractor:
    def __init__(self, device, img_size=512):
        self.device = device
        self.img_size = img_size
        self.preprocess = T.Compose([
            T.Lambda(auto_crop_ionogram),
            T.Resize((img_size, img_size), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.to_inception = T.Compose([
            T.Resize((299, 299), T.InterpolationMode.BICUBIC),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("加载 InceptionV3 特征提取器...")
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.eval()
        self.model = self.model.to(device)
        self.features = []
        self.hook = self.model.avgpool.register_forward_hook(
            lambda m, i, o: self.features.append(o.flatten(start_dim=1).cpu().numpy())
        )

    def _preprocess_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(img)
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
        tensor = self.to_inception(tensor)
        return tensor

    def extract_features(self, paths, batch_size=32, desc=""):
        self.features = []
        n = len(paths)
        for start in range(0, n, batch_size):
            batch_paths = paths[start:start + batch_size]
            batch = torch.stack([self._preprocess_image(p) for p in batch_paths]).to(self.device)
            with torch.no_grad():
                self.model(batch)
            if desc and (start + batch_size) % 200 == 0:
                print(f"    {desc}: {min(start + batch_size, n)}/{n}")
        result = np.concatenate(self.features, axis=0)
        self.features = []
        return result

    def __del__(self):
        if hasattr(self, 'hook'):
            self.hook.remove()


def compute_mixing_score(coords, labels):
    fake_coords = coords[labels == 0]
    real_coords = coords[labels == 1]
    if len(fake_coords) == 0 or len(real_coords) == 0:
        return 0.0

    nbrs = NearestNeighbors(n_neighbors=5).fit(coords)
    fake_indices = np.where(labels == 0)[0]
    real_indices = np.where(labels == 1)[0]

    mixing_ratios = []
    for idx in fake_indices:
        distances, neighbor_indices = nbrs.kneighbors(coords[idx:idx + 1], n_neighbors=6)
        neighbor_indices = neighbor_indices[0][1:]
        real_count = np.sum([1 for nidx in neighbor_indices if nidx in real_indices])
        mixing_ratios.append(real_count / 5.0)

    for idx in real_indices:
        distances, neighbor_indices = nbrs.kneighbors(coords[idx:idx + 1], n_neighbors=6)
        neighbor_indices = neighbor_indices[0][1:]
        fake_count = np.sum([1 for nidx in neighbor_indices if nidx in fake_indices])
        mixing_ratios.append(fake_count / 5.0)

    return np.mean(mixing_ratios)


def fid_from_features(feat_real, feat_fake):
    mu_r, mu_f = feat_real.mean(0), feat_fake.mean(0)
    sig_r = np.cov(feat_real, rowvar=False)
    sig_f = np.cov(feat_fake, rowvar=False)
    diff = mu_r - mu_f
    cov_mean, _ = sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    return float(diff @ diff + np.trace(sig_r + sig_f - 2 * cov_mean))


class CycleGANMonitor:
    def __init__(self, config=None):
        self.config = config if config is not None else MonitorConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Monitor using device: {self.device}")
        os.makedirs(self.config.MONITOR_DIR, exist_ok=True)
        self.extractor = FeatureExtractor(self.device, self.config.IMG_SIZE)
        self.real_B_paths = self._load_real_B()
        print(f"真实 B 图像: {len(self.real_B_paths)} 张")
        self.history = {
            "epochs": [],
            "fid": [],
            "mixing_score": [],
            "timestamp": [],
        }
        self._load_history()

    def _load_real_B(self):
        paths = glob.glob(os.path.join(self.config.TRAIN_B_DIR, "*.*"))
        paths = [p for p in paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        return paths

    def _load_history(self):
        history_path = os.path.join(self.config.MONITOR_DIR, "history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    self.history = json.load(f)
                print(f"加载历史记录: {len(self.history['epochs'])} 次评估")
            except Exception:
                pass

    def _save_history(self):
        history_path = os.path.join(self.config.MONITOR_DIR, "history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _sample_paths(self, paths, n, seed=None):
        if len(paths) <= n:
            return paths
        if seed is not None:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(paths), n, replace=False)
        else:
            indices = np.random.choice(len(paths), n, replace=False)
        return [paths[i] for i in indices]

    def evaluate_epoch(self, epoch, generator):
        print(f"\n{'='*50}")
        print(f"评估 Epoch {epoch}")
        print(f"{'='*50}")

        generator.eval()
        fake_paths = self._generate_fake_images(epoch, generator)
        if not fake_paths:
            print("  无法生成 fake 图像，跳过评估")
            return None

        real_paths = self._sample_paths(
            self.real_B_paths,
            self.config.TSNE_SAMPLES,
            seed=epoch
        )

        print("  提取特征...")
        real_feats = self.extractor.extract_features(real_paths, desc="  Real B")
        fake_feats = self.extractor.extract_features(fake_paths, desc="  Fake B")

        fid = fid_from_features(real_feats, fake_feats)
        print(f"  FID: {fid:.2f}")

        combined = np.concatenate([fake_feats, real_feats], axis=0)
        labels = np.array([0] * len(fake_feats) + [1] * len(real_feats))

        print("  运行 t-SNE...")
        tsne = TSNE(
            n_components=2,
            perplexity=self.config.TSNE_PERPLEXITY,
            max_iter=self.config.TSNE_MAX_ITER,
            random_state=epoch,
            init='random'
        )
        coords = tsne.fit_transform(combined)

        mixing = compute_mixing_score(coords, labels)
        print(f"  混合度: {mixing:.3f} (0=完全分离, 1=完全混合)")

        self._save_tsne_plot(epoch, coords, labels, mixing)
        self.history["epochs"].append(epoch)
        self.history["fid"].append(fid)
        self.history["mixing_score"].append(mixing)
        self.history["timestamp"].append(datetime.now().isoformat())
        self._save_history()
        self._update_plots()
        return {"fid": fid, "mixing_score": mixing}

    def _generate_fake_images(self, epoch, generator):
        fixed_src_file = os.path.join(self.config.MONITOR_DIR, "fixed_src_paths.json")
        if os.path.exists(fixed_src_file):
            with open(fixed_src_file, 'r') as f:
                src_paths = json.load(f)
        else:
            all_src = glob.glob(os.path.join(self.config.TRAIN_A_DIR, "*.*"))
            all_src = [p for p in all_src if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
            src_paths = self._sample_paths(all_src, self.config.TSNE_SAMPLES, seed=42)
            with open(fixed_src_file, 'w') as f:
                json.dump(src_paths, f)
            print(f"  固定源图像: {len(src_paths)} 张")

        temp_dir = os.path.join(self.config.MONITOR_DIR, f"temp_epoch_{epoch:03d}")
        os.makedirs(temp_dir, exist_ok=True)

        fake_paths = []
        with torch.no_grad():
            for src_path in src_paths:
                src_name = os.path.basename(src_path)
                fake_path = os.path.join(temp_dir, f"fake_{src_name}")
                if not os.path.exists(fake_path):
                    img = Image.open(src_path).convert("RGB")
                    transform = T.Compose([
                        T.Lambda(auto_crop_ionogram),
                        T.Resize((self.config.IMG_SIZE, self.config.IMG_SIZE),
                                 T.InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ])
                    input_tensor = transform(img).unsqueeze(0).to(self.device)
                    output = generator(input_tensor)
                    save_image(output, fake_path, normalize=True, value_range=(-1, 1))
                fake_paths.append(fake_path)
        return fake_paths

    def _save_tsne_plot(self, epoch, coords, labels, mixing):
        fig, ax = plt.subplots(figsize=(6, 5))
        fake_c = coords[labels == 0]
        real_c = coords[labels == 1]
        ax.scatter(fake_c[:, 0], fake_c[:, 1], c="#2166AC", s=5, alpha=0.45, label="Fake B")
        ax.scatter(real_c[:, 0], real_c[:, 1], c="#CA0020", s=5, alpha=0.45, label="Real B")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Epoch {epoch} | Mixing Score: {mixing:.3f}", fontsize=12)
        ax.legend()
        if mixing > 0.3:
            ax.text(0.02, 0.98, f"✓ 混合良好 ({mixing:.2f})",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', color='green')
        else:
            ax.text(0.02, 0.98, f"✗ 混合不足 ({mixing:.2f})",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', color='red')
        plt.tight_layout()
        save_path = os.path.join(self.config.MONITOR_DIR, f"tsne_epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存 t-SNE 图: {save_path}")

    def _update_plots(self):
        if len(self.history["epochs"]) == 0:
            return
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        ax = axes[0]
        ax.plot(self.history["epochs"], self.history["fid"], 'b-o', markersize=4, linewidth=1.5)
        ax.axhline(y=50, color='g', linestyle='--', alpha=0.5, label="Good FID < 50")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("FID ↓")
        ax.set_title("FID Score (lower is better)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        ax.plot(self.history["epochs"], self.history["mixing_score"], 'r-o', markersize=4, linewidth=1.5)
        ax.axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label="Good mixing > 0.3")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mixing Score")
        ax.set_title("t-SNE Mixing Score (higher is better)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.config.MONITOR_DIR, "training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  更新曲线图: {save_path}")

    def evaluate_all_checkpoints(self):
        ckpt_files = glob.glob(os.path.join(self.config.CHECKPOINT_DIR, "epoch_*.pth"))
        ckpt_epochs = []
        for f in ckpt_files:
            name = os.path.basename(f)
            epoch_str = name.replace("epoch_", "").replace(".pth", "")
            try:
                epoch = int(epoch_str)
                ckpt_epochs.append((epoch, f))
            except ValueError:
                pass
        ckpt_epochs.sort()
        evaluated = set(self.history["epochs"])
        for epoch, ckpt_path in ckpt_epochs:
            if epoch in evaluated:
                print(f"Epoch {epoch} 已评估，跳过")
                continue
            print(f"\n加载模型: {ckpt_path}")
            generator = Generator(
                self.config.IN_CHANNELS,
                self.config.N_FEATURES,
                self.config.N_RESIDUAL
            ).to(self.device)
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            generator.load_state_dict(checkpoint["G_AB"])
            generator.eval()
            self.evaluate_epoch(epoch, generator)
        print("\n所有评估完成！")
        print(f"监控结果保存在: {self.config.MONITOR_DIR}")
        print(f"  - training_curves.png: 训练曲线")
        print(f"  - tsne_epoch_*.png: 各 epoch t-SNE 图")
        print(f"  - history.json: 历史数据")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--train-a", type=str, default=None)
    parser.add_argument("--train-b", type=str, default=None)
    args = parser.parse_args()
    if args.checkpoint_dir:
        cfg.CHECKPOINT_DIR = args.checkpoint_dir
    if args.train_a:
        cfg.TRAIN_A = args.train_a
    if args.train_b:
        cfg.TRAIN_B = args.train_b
    monitor = CycleGANMonitor()
    monitor.evaluate_all_checkpoints()


if __name__ == "__main__":
    main()
