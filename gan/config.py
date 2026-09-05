import torch


class Config:
    TRAIN_A = "data/trainA"
    TRAIN_B = "data/trainB"
    OUTPUT_DIR = "outputs"
    CHECKPOINT_DIR = "checkpoints"

    IMG_SIZE = 512
    IN_CHANNELS = 3

    EPOCHS = 200
    BATCH_SIZE = 1
    LR = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999
    DECAY_EPOCH = 100

    LAMBDA_CYCLE = 10.0
    LAMBDA_IDENTITY = 5.0
    LAMBDA_GP = 10.0

    N_CRITIC = 5

    N_RESIDUAL = 9
    N_FEATURES = 64

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SAVE_INTERVAL = 10


cfg = Config()
