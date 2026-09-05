import subprocess


def run(cmd: str):
    print(f">> {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main():
    run("python split_dataset.py --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1 --seed 42")
    run("python train_model_attn.py --train_dir train --val_dir val")
    run("python test.py --test_dir test --model_path best_ionosphere_model.pth")


if __name__ == "__main__":
    main()
