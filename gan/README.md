# Ionogram CycleGAN + WGAN-GP

频高图域迁移：理论频高图 → 伪真实频高图

## 项目结构

```
gan/
├── config.py              # 超参数
├── dataset.py             # 无配对双域数据 + 裁白边
├── models.py              # ResNet 生成器 + PatchGAN 判别器
├── losses.py              # WGAN-GP + 循环一致性 + 身份损失
├── train.py               # 训练
├── train_monitor.py       # FID / t-SNE 监控
├── utils.py               # 检查点 / 样本图 / 损失曲线
├── inference.py           # A→B / B→A / 循环重建
├── inference_inverse.py   # B→A 入口
├── rec.py                 # A→B→A 入口
├── requirements.txt
├── data/
│   ├── trainA/            # 理论频高图
│   └── trainB/            # 真实频高图
└── checkpoints/           # 权重，如 epoch_070.pth
```

## 环境安装

```bash
pip install -r requirements.txt
```

`torch` / `torchvision` 请按本机 CUDA 版本从 PyTorch 官网安装。

## 数据准备

- `data/trainA/`：理论频高图（PNG/JPG，任意分辨率）
- `data/trainB/`：真实频高图（PNG/JPG，任意分辨率）

图像会自动裁掉白边并缩放到 512×512。

## 训练

在 `gan/` 目录下执行：

```bash
python train.py
python train.py --resume 70
python train.py --resume checkpoints/epoch_070.pth
python train.py --train-a data/trainA --train-b data/trainB --no-monitor
```

| 参数 | 说明 |
|---|---|
| `--resume` | 检查点路径，或只写 epoch 数字 |
| `--train-a` / `--train-b` | 覆盖数据目录 |
| `--output-dir` | 训练样本图目录，默认 `outputs/` |
| `--checkpoint-dir` | 权重目录，默认 `checkpoints/` |
| `--epochs` | 覆盖总轮数 |
| `--no-monitor` | 不加载 Inception，不做 FID / t-SNE |

## 推理

```bash
python inference.py --epoch 70
python inference.py --direction ab --input <理论图目录> --output <输出目录> --epoch 70
python inference_inverse.py --epoch 70
python rec.py --checkpoint checkpoints/epoch_070.pth --input data/trainA
```

| 参数 | 说明 |
|---|---|
| `--direction` | `ab` 理论→伪真实；`ba` 真实→伪理论；`cycle` 循环重建 |
| `--input` | 输入目录。`ab`/`cycle` 默认 `data/trainA`，`ba` 默认 `data/trainB` |
| `--output` | 输出目录。默认 `results/fake_B_epoch_XXX` 等 |
| `--checkpoint` | 权重文件路径 |
| `--epoch` | 使用 `checkpoints/epoch_XXX.pth` |
| `--checkpoint-dir` | 覆盖权重目录 |

不写 `--checkpoint` / `--epoch` 时，使用 `checkpoints/` 里最新的权重。

单独评估已有权重：

```bash
python train_monitor.py
python train_monitor.py --checkpoint-dir checkpoints --train-a data/trainA --train-b data/trainB
```

## 输出

- `outputs/`：每 200 步一张对比图（真实 A | 生成 B | 重建 A）
- `checkpoints/`：每 10 个 epoch 存一次模型
- `outputs/loss_curve.png`：训练损失曲线
- `monitor/`：FID、t-SNE（未加 `--no-monitor` 时）
- `results/`：推理结果

## 关键超参数（config.py）

| 参数 | 默认值 | 说明 |
|---|---|---|
| LAMBDA_CYCLE | 10.0 | 循环一致性权重，越大越保留结构 |
| LAMBDA_IDENTITY | 5.0 | 身份损失权重，防止色调漂移 |
| LAMBDA_GP | 10.0 | 梯度惩罚权重 |
| N_CRITIC | 5 | 每训练 1 次 G，训练 D 的次数 |
| DECAY_EPOCH | 100 | 从该 epoch 起线性衰减学习率 |
| N_RESIDUAL | 9 | 生成器残差块数量 |

## 训练监控指标

- **Wasserstein Distance（W_A, W_B）**：越大说明判别器区分真假越清晰，正常应先增大后趋于稳定
- **Cycle Loss**：循环重建误差，应持续下降
- **Generator Loss**：应波动但整体稳定，不应持续增大

## 常见问题

**Q: 显存不足**  
A: 将 `config.py` 中 `IMG_SIZE` 改为 256，或 `N_RESIDUAL` 改为 6

**Q: 训练不稳定（Loss 爆炸）**  
A: 降低学习率到 `LR = 1e-4`，或增大 `LAMBDA_GP` 到 15

**Q: 生成图像模糊**  
A: 增大 `LAMBDA_CYCLE`，或减小 `LAMBDA_IDENTITY`
