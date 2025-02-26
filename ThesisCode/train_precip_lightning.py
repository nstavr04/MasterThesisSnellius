from root import ROOT_DIR

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch import loggers
import argparse
from models import unet_precip_regression_lightning as unet_regr
from models import SmaAT_UNet_VQ_lightning
from lightning.pytorch.tuner import Tuner


def train_regression(hparams, find_batch_size_automatically: bool = False):
    if hparams.model in ["SmaAT_UNet_VQ_MSE"]:
        hparams.vqmodel_recon_loss_type = "mse"
        net = SmaAT_UNet_VQ_lightning.SmaAT_UNet_VQ(hparams=hparams)
    elif hparams.model in ["SmaAT_UNet_VQ_MWAE"]:
        hparams.vqmodel_recon_loss_type = "mwae"
        net = SmaAT_UNet_VQ_lightning.SmaAT_UNet_VQ(hparams=hparams)
    elif hparams.model in ["SmaAT_UNet"]:
        net = unet_regr.UNet_Attention(hparams=hparams)
    elif hparams.model in ["UNet"]:
        net = unet_regr.UNet(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    default_save_path = ROOT_DIR / "lightning" / "precip_regression"

    # IMPORTANT: Update the monitor to "val_total_loss" (the sum of recon + VQ loss)
    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / net.__class__.__name__,
        filename=net.__class__.__name__ + "_rain_threshold_50_{epoch}-{val_total_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor="val_total_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)

    # Also update the early stopping monitor to "val_total_loss"
    # If the model has the same val_loss for patience times (the default now is 15), the model will stop training.
    earlystopping_callback = EarlyStopping(
        monitor="val_total_loss",
        mode="min",
        patience=hparams.es_patience,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # Executes only one batch, it is to debug basically that everything works quickly
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystopping_callback, lr_monitor],
        # 0.25 means 4 times validation per epoch, 1.0 means validation once per epoch.
        # If it's an int e.g 5 it means to run validation every 5 training steps
        val_check_interval=hparams.val_check_interval,
    )

    if find_batch_size_automatically:
        tuner = Tuner(trainer)
        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(net, mode="binsearch")

    # This can be used to speed up training with newer GPUs:
    # https://lightning.ai/docs/pytorch/stable/advanced/speed.html#low-precision-matrix-multiplication
    
    # I can try it sometime. Low means fast but less precision, high means high precision but slower.
    # torch.set_float32_matmul_precision('medium')

    trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = unet_regr.Precip_regression_base.add_model_specific_args(parser)

    parser.add_argument(
        "--dataset_folder",
        default=ROOT_DIR / "data" / "precipitation" / "RAD_NL25_RAC_5min_train_test_2016-2019.h5",
        type=str,
    )

    # default is 16
    # I had 8
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)

    args = parser.parse_args()

    # (Irrelevant comments and assignments are kept for context)
    # This probably means 12 input satellite images 
    args.n_channels = 12
    # args.gpus = 1

    args.lr_patience = 4
    args.es_patience = 15
    # args.val_check_interval = 0.25
    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    args.dataset_folder = (
        ROOT_DIR / "data" / "precipitation" / "train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50.h5"
    )
    # args.resume_from_checkpoint = f"lightning/precip_regression/{args.model}/UNetDS_Attention.ckpt"

     # train_regression(args, find_batch_size_automatically=False)

    # I can change these 2 if I want
    args.vq_num_embeddings = 32
    args.vq_commitment_cost = 0.75

    # Pick which models we want to train
    # args.model = "UNet"
    # args.model = "SmaAT_UNet"
    # args.model = "SmaAT_UNet_VQ_MWAE"
    # args.model = "SmaAT_UNet_VQ_MSE"
    for m in ["SmaAT_UNet_VQ_MWAE"]:
        args.model = m
        print(f"Start training model: {m}")
        train_regression(args, find_batch_size_automatically=False)

    # Use this if we want hyperparameter tuning on VQ models

    # Define the hyperparameter ranges:
    # For commitment cost, we use 5 values in logspace between 1e-2 and 1 (since log(0) is undefined)

    # commitment_cost_values = np.linspace(0, 1, num=5) # e.g. [0, 0.25, 0.5, 0.75, 1.0]
    # num_embeddings_values = [32, 64, 128, 256, 512]

    # for num_emb in num_embeddings_values:
    #     for comm_cost in commitment_cost_values:
    #         # Update hyperparameters:
    #         args.vq_num_embeddings = num_emb
    #         args.vq_commitment_cost = float(comm_cost)  # ensure it's a float
    #         args.model = "SmaAT_UNet_VQ"
    #         # Define a folder name that reflects the hyperparameter settings:
    #         args.save_folder = f"SmaAT-UNet-VQ-num{num_emb}-commitment{comm_cost:.4f}"
            
    #         print(f"Start training model: {args.model} with vq_num_embeddings={num_emb} and vq_commitment_cost={comm_cost:.4f}")
    #         train_regression(args, find_batch_size_automatically=False)
