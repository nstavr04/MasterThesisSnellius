import lightning.pytorch as pl
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import dataset_precip
import argparse
import numpy as np

###############################################################################
# Base Class for SmaAT_UNet and SmaAT_UNet_VQ
###############################################################################
class UNet_base(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # We only support the VQ version now.
        parser.add_argument(
            "--model",
            type=str,
            default="SmaAT_UNet_VQ_MSE",
            choices=["SmaAT_UNet_VQ_MSE", "SmaAT_UNet_VQ_MWAE", "SmaAT_UNet", "UNet"],
        )
        # Basic model arguments
        parser.add_argument("--n_channels", type=int, default=12)
        parser.add_argument("--n_classes", type=int, default=1)
        parser.add_argument("--kernels_per_layer", type=int, default=1)
        parser.add_argument("--bilinear", type=bool, default=True)
        parser.add_argument("--reduction_ratio", type=int, default=16)
        parser.add_argument("--lr_patience", type=int, default=5)
        # VQ-specific arguments:
        parser.add_argument("--vq_num_embeddings", type=int, default=512)
        parser.add_argument("--vq_commitment_cost", type=float, default=0.25)

        # Loss type for the VQ model variations
        parser.add_argument("--vqmodel_recon_loss_type", type=str, default="mse", choices=["mse", "mwae"])

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # Note: the forward() method should be implemented in your model subclass.
        # It is expected to return: (logits, vq_loss, loss_dict)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.1, patience=self.hparams.lr_patience
            ),
            "monitor": "val_total_loss",  # Monitor the total loss during validation.
        }
        return [opt], [scheduler]

    # MWAE loss function as defined in GPTCast paper
    # We use it only for the recon loss
    def mwae(self, x, y):
        sx = sigmoid(x)
        sy = sigmoid(y)
        return ((abs(sx - sy) * sx).sum()) / y_true.size(0)

    def mse(self, x, y):
        return nn.functional.mse_loss(x, y, reduction="sum") / y_true.size(0)

    def loss_func(self, y_pred, y_true):
        """
        Reconstruction (or regression) loss.
        We use the mean squared error averaged per image or the MWAE loss.
        """
        if self.hparams.vqmodel_recon_loss_type == 'mwae':
            return self.mwae(y_pred, y_true)
        else:
            return self.mse(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Unpack the model output; note that we expect the VQ model to return three items.
        logits, vq_loss, loss_dict = self(x)
        # Compute the reconstruction loss on the output.
        recon_loss = self.loss_func(logits.squeeze(), y)
        # Total loss includes both the reconstruction and the VQ losses.
        total_loss = recon_loss + vq_loss

        # Log losses for monitoring.
        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_vq_loss", vq_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_codebook_loss", loss_dict["codebook_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_commitment_loss", loss_dict["commitment_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, vq_loss, loss_dict = self(x)
        recon_loss = self.loss_func(logits.squeeze(), y)
        total_loss = recon_loss + vq_loss

        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_vq_loss", vq_loss, prog_bar=True)
        self.log("val_total_loss", total_loss, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, vq_loss, loss_dict = self(x)
        recon_loss = self.loss_func(logits.squeeze(), y)
        total_loss = recon_loss + vq_loss
        factor = 47.83
        loss_denorm = self.loss_func(logits.squeeze() * factor, y * factor)
        self.log("test_recon_loss", recon_loss)
        self.log("test_vq_loss", vq_loss)
        self.log("test_total_loss", total_loss)
        self.log("MSE_denormalized", loss_denorm)


###############################################################################
# Precipitation Regression Base Class
###############################################################################
class Precip_regression_base(UNet_base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        # Extend the basic UNet arguments with dataset-specific parameters.
        parent_parser = UNet_base.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=12)
        parser.add_argument("--num_output_images", type=int, default=6)
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.add_argument("--use_oversampled_dataset", type=bool, default=True)
        # Set n_channels and n_classes based on dataset parameters.
        parser.n_channels = parser.parse_args().num_input_images
        parser.n_classes = 1
        return parser

    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    def prepare_data(self):
        # Dataset transforms can be added if needed.
        train_transform = None
        valid_transform = None
        precip_dataset = (
            dataset_precip.precipitation_maps_oversampled_h5
            if self.hparams.use_oversampled_dataset
            else dataset_precip.precipitation_maps_h5
        )
        self.train_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            train=True,
            transform=train_transform,
        )
        self.valid_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            train=True,
            transform=valid_transform,
        )

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.hparams.valid_size * num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            pin_memory=True,
            # Tweaked accordingly to CPU
            num_workers=5,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.valid_sampler,
            pin_memory=True,
            # Tweaked accordingly to CPU
            num_workers=5,
            persistent_workers=True,
        )
