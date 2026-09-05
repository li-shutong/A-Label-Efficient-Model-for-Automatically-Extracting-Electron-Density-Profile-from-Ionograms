import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from dataset import get_dataloader
from models import Generator, Discriminator, init_weights
from losses import WGANGPLoss, CycleLoss, IdentityLoss
from utils import (
    save_sample_images, save_checkpoint, load_checkpoint,
    LossLogger, resolve_resume_path,
)


def get_lr_scheduler(optimizer, start_decay_epoch: int, total_epochs: int):
    def rule(epoch):
        if epoch < start_decay_epoch:
            return 1.0
        return 1.0 - (epoch - start_decay_epoch) / (total_epochs - start_decay_epoch)
    return LambdaLR(optimizer, lr_lambda=rule)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--train-a", type=str, default=None)
    parser.add_argument("--train-b", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-monitor", action="store_true")
    return parser.parse_args()


def apply_args(args):
    if args.train_a:
        cfg.TRAIN_A = args.train_a
    if args.train_b:
        cfg.TRAIN_B = args.train_b
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.checkpoint_dir:
        cfg.CHECKPOINT_DIR = args.checkpoint_dir
    if args.epochs:
        cfg.EPOCHS = args.epochs


def train():
    args = parse_args()
    apply_args(args)

    device = cfg.DEVICE
    print(f"[Train] 使用设备: {device}")

    dataloader = get_dataloader(cfg, augment=True)

    G_AB = Generator(cfg.IN_CHANNELS, cfg.N_FEATURES, cfg.N_RESIDUAL).to(device)
    G_BA = Generator(cfg.IN_CHANNELS, cfg.N_FEATURES, cfg.N_RESIDUAL).to(device)
    D_A = Discriminator(cfg.IN_CHANNELS, cfg.N_FEATURES).to(device)
    D_B = Discriminator(cfg.IN_CHANNELS, cfg.N_FEATURES).to(device)

    for model in [G_AB, G_BA, D_A, D_B]:
        init_weights(model)

    monitor = None
    if not args.no_monitor:
        from train_monitor import CycleGANMonitor
        monitor = CycleGANMonitor()

    opt_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2)
    )
    opt_D = optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2)
    )

    wgan_gp = WGANGPLoss(lambda_gp=cfg.LAMBDA_GP)
    cycle_loss = CycleLoss(lambda_cycle=cfg.LAMBDA_CYCLE)
    id_loss = IdentityLoss(lambda_identity=cfg.LAMBDA_IDENTITY)

    sched_G = get_lr_scheduler(opt_G, cfg.DECAY_EPOCH, cfg.EPOCHS)
    sched_D = get_lr_scheduler(opt_D, cfg.DECAY_EPOCH, cfg.EPOCHS)

    start_epoch = 0
    if args.resume:
        resume_path = resolve_resume_path(args.resume, cfg.CHECKPOINT_DIR)
        start_epoch = load_checkpoint(
            resume_path, G_AB, G_BA, D_A, D_B, opt_G, opt_D, device,
            sched_G=sched_G, sched_D=sched_D,
        )

    logger = LossLogger()

    print("[Train] 开始训练...")

    for epoch in range(start_epoch, cfg.EPOCHS):
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        for step, batch in enumerate(dataloader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            for _ in range(cfg.N_CRITIC):
                with torch.no_grad():
                    fake_B = G_AB(real_A)
                    fake_A = G_BA(real_B)

                opt_D.zero_grad()
                d_A_info = wgan_gp.discriminator_loss(D_A, real_A, fake_A, device)
                d_B_info = wgan_gp.discriminator_loss(D_B, real_B, fake_B, device)
                d_loss = d_A_info["total"] + d_B_info["total"]
                d_loss.backward()
                opt_D.step()

            opt_G.zero_grad()

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)
            idt_A = G_BA(real_A)
            idt_B = G_AB(real_B)

            g_loss_AB = wgan_gp.generator_loss(D_B, fake_B)
            g_loss_BA = wgan_gp.generator_loss(D_A, fake_A)
            loss_cycle_A = cycle_loss(real_A, recov_A)
            loss_cycle_B = cycle_loss(real_B, recov_B)
            loss_id_A = id_loss(real_A, idt_A)
            loss_id_B = id_loss(real_B, idt_B)

            g_total = (g_loss_AB + g_loss_BA
                       + loss_cycle_A + loss_cycle_B
                       + loss_id_A + loss_id_B)
            g_total.backward()
            opt_G.step()

            logger.update(
                D_A_loss=d_A_info["total"].item(),
                D_B_loss=d_B_info["total"].item(),
                G_loss=(g_loss_AB + g_loss_BA).item(),
                cycle_loss=(loss_cycle_A + loss_cycle_B).item(),
                identity_loss=(loss_id_A + loss_id_B).item(),
                W_dist_A=d_A_info["wasserstein_dist"],
                W_dist_B=d_B_info["wasserstein_dist"],
            )

            if step % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{cfg.EPOCHS}] Step [{step}/{len(dataloader)}] | "
                    f"D_A: {d_A_info['total'].item():.3f} "
                    f"D_B: {d_B_info['total'].item():.3f} | "
                    f"G: {g_total.item():.3f} | "
                    f"Cycle: {(loss_cycle_A+loss_cycle_B).item():.3f} | "
                    f"W_A: {d_A_info['wasserstein_dist']:.3f} "
                    f"W_B: {d_B_info['wasserstein_dist']:.3f}"
                )

            if step % 200 == 0:
                save_sample_images(G_AB, G_BA, batch, epoch + 1, step,
                                   cfg.OUTPUT_DIR, device)

        sched_G.step()
        sched_D.step()

        current_lr = opt_G.param_groups[0]["lr"]
        print(f"[LR] epoch {epoch+1}: {current_lr:.6f}")

        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            save_checkpoint(epoch + 1, G_AB, G_BA, D_A, D_B,
                            opt_G, opt_D, cfg.CHECKPOINT_DIR,
                            sched_G=sched_G, sched_D=sched_D)
            logger.plot(os.path.join(cfg.OUTPUT_DIR, "loss_curve.png"))
            if monitor is not None:
                monitor.evaluate_epoch(epoch + 1, G_AB)

    print("[Train] 训练完成！")
    save_checkpoint(cfg.EPOCHS, G_AB, G_BA, D_A, D_B,
                    opt_G, opt_D, cfg.CHECKPOINT_DIR,
                    sched_G=sched_G, sched_D=sched_D)


if __name__ == "__main__":
    train()
