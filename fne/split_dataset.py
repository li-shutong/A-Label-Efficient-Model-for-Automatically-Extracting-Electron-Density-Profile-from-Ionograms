import os
import json
import shutil
import argparse
from typing import List, Dict

import numpy as np


class DatasetSplitter:
    def __init__(self, source_dir: str = "imgs", json_dir: str = "json"):
        self.source_dir = source_dir
        self.json_dir = json_dir
        self.train_dir = "train"
        self.val_dir = "val"
        self.test_dir = "test"

    def rename_fake_prefix(self):
        for file in os.listdir(self.source_dir):
            old_path = os.path.join(self.source_dir, file)
            if os.path.isfile(old_path) and file.startswith("fake_"):
                new_name = file[len("fake_"):]
                new_path = os.path.join(self.source_dir, new_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                else:
                    print(f"警告: 目标文件已存在，跳过重命名 {file} -> {new_name}")

    def get_matching_files(self) -> List[Dict[str, str]]:
        self.rename_fake_prefix()
        image_map = {}
        for file in os.listdir(self.source_dir):
            full_path = os.path.join(self.source_dir, file)
            if os.path.isfile(full_path) and file.lower().endswith(".jpg"):
                image_map[file] = full_path
        json_map = {}
        for file in os.listdir(self.json_dir):
            full_path = os.path.join(self.json_dir, file)
            if os.path.isfile(full_path) and file.lower().endswith(".json"):
                json_map[file] = full_path

        matched_pairs = []
        for json_file, json_path in json_map.items():
            image_name = json_file.replace(".json", ".jpg")
            if image_name in image_map:
                matched_pairs.append({
                    "image_name": image_name,
                    "json_name": json_file,
                    "image_path": image_map[image_name],
                    "json_path": json_path,
                })
        matched_pairs.sort(key=lambda x: x["image_name"])
        print(f"找到 {len(matched_pairs)} 个匹配的图像-JSON对")
        return matched_pairs

    def create_directories(self):
        for dir_name in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(dir_name, exist_ok=True)
            print(f"创建目录: {dir_name}")

    def _stem(self, image_name: str) -> str:
        if image_name.endswith("_ionogram.jpg"):
            return image_name.replace("_ionogram.jpg", "")
        return os.path.splitext(image_name)[0]

    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        print("开始分割数据集...")
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        matched_pairs = self.get_matching_files()
        if len(matched_pairs) == 0:
            print("错误: 未找到匹配的图像-JSON对")
            return None

        total_samples = len(matched_pairs)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        self.create_directories()

        splits = {
            "train": (train_indices, self.train_dir),
            "val": (val_indices, self.val_dir),
            "test": (test_indices, self.test_dir),
        }
        for split_name, (split_indices, target_dir) in splits.items():
            print(f"\n处理 {split_name} 集 ({len(split_indices)} 个样本):")
            for i, idx in enumerate(split_indices):
                pair = matched_pairs[idx]
                base_name = self._stem(pair["image_name"])
                dst_image = os.path.join(target_dir, base_name + ".jpg")
                dst_json = os.path.join(target_dir, base_name + ".json")
                try:
                    shutil.copy2(pair["image_path"], dst_image)
                    shutil.copy2(pair["json_path"], dst_json)
                    if (i + 1) % 50 == 0 or i == len(split_indices) - 1:
                        print(f"  已处理: {i + 1}/{len(split_indices)}")
                except Exception as e:
                    print(f"  复制失败: {pair['image_name']} / {pair['json_name']} -> {e}")

        print("\n数据集分割完成!")
        print(f"训练集: {len(train_indices)} 样本 -> {self.train_dir}/")
        print(f"验证集: {len(val_indices)} 样本 -> {self.val_dir}/")
        print(f"测试集: {len(test_indices)} 样本 -> {self.test_dir}/")

        split_info = {
            "total_samples": total_samples,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "test_samples": len(test_indices),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "train_files": [self._stem(matched_pairs[i]["image_name"]) + ".jpg" for i in train_indices],
            "val_files": [self._stem(matched_pairs[i]["image_name"]) + ".jpg" for i in val_indices],
            "test_files": [self._stem(matched_pairs[i]["image_name"]) + ".jpg" for i in test_indices],
        }
        with open("dataset_split_info.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print("分割信息已保存到: dataset_split_info.json")
        return split_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default=r"E:\Sar_Syn\Fur_exp\data\epoch_070")
    parser.add_argument("--json_dir", type=str, default=r"E:\Sar_Syn\Fur_exp\data\22.7N-101.05E_2024_json")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    splitter = DatasetSplitter(args.source_dir, args.json_dir)
    splitter.split_dataset(args.train_ratio, args.val_ratio, args.test_ratio)
    print("\n数据集分割完成!")


if __name__ == "__main__":
    main()
