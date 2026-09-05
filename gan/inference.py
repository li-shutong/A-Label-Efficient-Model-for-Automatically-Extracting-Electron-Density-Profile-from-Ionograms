import os
import argparse
import torch
from PIL import Image
from torchvision.utils import save_image
from config import cfg
from models import Generator
from dataset import get_transforms, IonogramDataset
from utils import resolve_ckpt_path


DIRECTIONS = ("ab", "ba", "cycle")
DEFAULT_INPUT = {
    "ab": lambda: cfg.TRAIN_A,
    "ba": lambda: cfg.TRAIN_B,
    "cycle": lambda: cfg.TRAIN_A,
}
OUTPUT_PREFIX = {
    "ab": "fake_B",
    "ba": "fake_A",
    "cycle": "rec_A",
}
SAVE_NAME = {
    "ab": lambda name: name,
    "ba": lambda name: f"fake_{name}",
    "cycle": lambda name: f"recA_{name}",
}


def list_image_files(input_dir):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    files = [
        f for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in IonogramDataset.SUPPORTED_EXT
    ]
    if not files:
        raise ValueError(f"目录中没有找到图像: {input_dir}")
    return files


def default_output_dir(direction, checkpoint_path):
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    return os.path.join("results", f"{OUTPUT_PREFIX[direction]}_{stem}")


def run_inference(checkpoint_path, input_dir, result_dir, direction="ab"):
    if direction not in DIRECTIONS:
        raise ValueError(f"direction 必须是 {DIRECTIONS} 之一")

    device = cfg.DEVICE
    os.makedirs(result_dir, exist_ok=True)

    print(f"正在加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    G_AB = Generator(cfg.IN_CHANNELS, cfg.N_FEATURES, cfg.N_RESIDUAL).to(device)
    G_AB.load_state_dict(checkpoint["G_AB"])
    G_AB.eval()

    G_BA = None
    if direction in ("ba", "cycle"):
        G_BA = Generator(cfg.IN_CHANNELS, cfg.N_FEATURES, cfg.N_RESIDUAL).to(device)
        G_BA.load_state_dict(checkpoint["G_BA"])
        G_BA.eval()

    transform = get_transforms(cfg.IMG_SIZE, augment=False)
    image_files = list_image_files(input_dir)
    print(f"找到 {len(image_files)} 张待转换图片。")

    name_fn = SAVE_NAME[direction]
    with torch.no_grad():
        for i, filename in enumerate(image_files):
            img_path = os.path.join(input_dir, filename)
            raw_img = Image.open(img_path).convert("RGB")
            x = transform(raw_img).unsqueeze(0).to(device)
            if direction == "ab":
                out = G_AB(x)
            elif direction == "ba":
                out = G_BA(x)
            else:
                out = G_BA(G_AB(x))
            save_image(
                out,
                os.path.join(result_dir, name_fn(filename)),
                normalize=True, value_range=(-1, 1),
            )
            if (i + 1) % 10 == 0:
                print(f"已处理: {i + 1}/{len(image_files)}")

    print(f"全部转换完成！结果保存在: {result_dir}")


def build_parser(default_direction="ab"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, default=default_direction, choices=DIRECTIONS)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser


def main(default_direction="ab"):
    args = build_parser(default_direction).parse_args()
    ckpt_dir = args.checkpoint_dir or cfg.CHECKPOINT_DIR
    ckpt = resolve_ckpt_path(args.checkpoint, args.epoch, ckpt_dir)
    input_dir = args.input or DEFAULT_INPUT[args.direction]()
    output_dir = args.output or default_output_dir(args.direction, ckpt)
    print(f"direction={args.direction}")
    print(f"checkpoint={ckpt}")
    print(f"input={input_dir}")
    print(f"output={output_dir}")
    run_inference(ckpt, input_dir, output_dir, args.direction)


if __name__ == "__main__":
    main("ab")
