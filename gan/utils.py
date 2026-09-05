import os
import glob
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from config import cfg


def resolve_ckpt_path(checkpoint=None, epoch=None, ckpt_dir=None):
    ckpt_dir = ckpt_dir or cfg.CHECKPOINT_DIR
    if checkpoint:
        path = checkpoint
    elif epoch is not None:
        path = os.path.join(ckpt_dir, f"epoch_{int(epoch):03d}.pth")
    else:
        files = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")))
        if not files:
            raise FileNotFoundError(f"未找到检查点: {ckpt_dir}")
        path = files[-1]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点不存在: {path}")
    return path


def resolve_resume_path(resume, ckpt_dir=None):
    if resume is None:
        return None
    if str(resume).isdigit():
        return resolve_ckpt_path(epoch=int(resume), ckpt_dir=ckpt_dir)
    return resolve_ckpt_path(checkpoint=resume, ckpt_dir=ckpt_dir)


def save_sample_images(G_AB, G_BA, batch, epoch, step, output_dir: str, device: str):
    G_AB.eval()
    G_BA.eval()

    with torch.no_grad():
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)

    row_A = torch.cat([real_A, fake_B, recov_A], dim=3)
    row_B = torch.cat([real_B, fake_A, recov_B], dim=3)
    grid = torch.cat([row_A, row_B], dim=2)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"epoch{epoch:03d}_step{step:05d}.png")
    vutils.save_image(grid, save_path, normalize=True, value_range=(-1, 1))

    G_AB.train()
    G_BA.train()
    return save_path


def save_checkpoint(epoch: int, G_AB, G_BA, D_A, D_B,
                    opt_G, opt_D, checkpoint_dir: str,
                    sched_G=None, sched_D=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
    payload = {
        "epoch": epoch,
        "G_AB": G_AB.state_dict(),
        "G_BA": G_BA.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
    }
    if sched_G is not None:
        payload["sched_G"] = sched_G.state_dict()
    if sched_D is not None:
        payload["sched_D"] = sched_D.state_dict()
    torch.save(payload, path)
    print(f"[Checkpoint] 保存到 {path}")


def _restore_scheduler(sched, key, epoch, ckpt):
    if sched is None:
        return
    if key in ckpt:
        sched.load_state_dict(ckpt[key])
        return
    for _ in range(epoch):
        sched.step()


def load_checkpoint(path: str, G_AB, G_BA, D_A, D_B, opt_G, opt_D, device: str,
                    sched_G=None, sched_D=None):
    ckpt = torch.load(path, map_location=device)
    G_AB.load_state_dict(ckpt["G_AB"])
    G_BA.load_state_dict(ckpt["G_BA"])
    D_A.load_state_dict(ckpt["D_A"])
    D_B.load_state_dict(ckpt["D_B"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    epoch = ckpt["epoch"]
    _restore_scheduler(sched_G, "sched_G", epoch, ckpt)
    _restore_scheduler(sched_D, "sched_D", epoch, ckpt)
    print(f"[Checkpoint] 从 epoch {epoch} 恢复训练")
    return epoch


class LossLogger:
    def __init__(self):
        self.history = {
            "D_A_loss": [], "D_B_loss": [],
            "G_loss": [], "cycle_loss": [], "identity_loss": [],
            "W_dist_A": [], "W_dist_B": [],
        }

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.history:
                self.history[k].append(v)

    def plot(self, save_path: str = "loss_curve.png"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Training Loss Curves", fontsize=14)

        axes[0, 0].plot(self.history["D_A_loss"], label="D_A")
        axes[0, 0].plot(self.history["D_B_loss"], label="D_B")
        axes[0, 0].set_title("Discriminator Loss")
        axes[0, 0].legend()

        axes[0, 1].plot(self.history["G_loss"], label="Generator")
        axes[0, 1].set_title("Generator Loss")
        axes[0, 1].legend()

        axes[1, 0].plot(self.history["cycle_loss"], label="Cycle")
        axes[1, 0].plot(self.history["identity_loss"], label="Identity")
        axes[1, 0].set_title("Auxiliary Losses")
        axes[1, 0].legend()

        axes[1, 1].plot(self.history["W_dist_A"], label="W_dist_A")
        axes[1, 1].plot(self.history["W_dist_B"], label="W_dist_B")
        axes[1, 1].set_title("Wasserstein Distance")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[Logger] 损失曲线保存到 {save_path}")
