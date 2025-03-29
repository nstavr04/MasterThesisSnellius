import lightning.pytorch as pl
import torch
from torch import nn, optim, sigmoid, abs
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import dataset_precip
import argparse
import numpy as np
import bucket

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
            choices=["SmaAT_UNet_VQ_CE", "SmaAT_UNet_VQ_MSE", "SmaAT_UNet_VQ_MWAE", "SmaAT_UNet", "UNet"],
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
        parser.add_argument("--vqmodel_recon_loss_type", type=str, default="mse", choices=["ce", "mse", "mwae"])

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

    # Cross Entropy thresholding loss as defined in RainAI paper
    # We use it only for the recon loss
    def ce_recon_loss(self, y_pred, y_true):
        """ Computes a bucketed cross-entropy reconstruction loss.
        - y_pred: logits of shape [B, num_buckets, H, W]
        - y_true: continuous targets of shape [B, 1, H, W]
        Hard-coded bucket settings are used here. """
        # Set bucket boundaries based on the histogram data on view_dataset.ipynb.
        # These boundaries are chosen so that:
        # - Class 0: values < 0.00333
        # - Class 1: 0.00333 ≤ values < 0.00667
        # - Class 2: 0.00667 ≤ values < 0.01000
        # - Class 3: 0.01000 ≤ values < 0.01333
        # - Class 4: 0.01333 ≤ values < 0.01667
        # - Class 5: 0.01667 ≤ values < 0.02000
        # - Class 6: 0.02000 ≤ values < 0.02333
        # - Class 7: 0.02333 ≤ values < 0.02667
        # - Class 8: 0.02667 ≤ values < 0.03000
        # - Class 9: 0.03000 ≤ values < 0.03333
        # - Class 10: 0.03333 ≤ values < 0.03667
        # - Class 11: 0.03667 ≤ values < 0.04000
        # - Class 12: 0.04000 ≤ values < 0.04333
        # - Class 13: 0.04333 ≤ values < 0.04667
        # - Class 14: values ≥ 0.04667

        # Bin edges (14 values define 15 buckets)
        bucket_boundaries = buckets.bucket_boundaries(device=y_true.device)
    
        # Midpoint means of buckets
        bucket_means = buckets.get_bucket_means(device=y_true.device)

        # Weights (inverse frequency approx., can be tuned further)
        bucket_weights = buckets.bucket_weights(device=y_true.device)

        # Convert continuous target to bucket indices.
        # Assume y_true shape is [B, 1, H, W]; squeeze out the channel dimension.
        target_class = torch.bucketize(y_true, bucket_boundaries).squeeze(1).long()  # Shape: [B, H, W]

        # Compute cross-entropy loss per pixel (no reduction)
        ce_loss = nn.functional.cross_entropy(
            y_pred, target_class, weight=bucket_weights, reduction="none"
        )
        
        # Final loss: sum over all pixels and then divide by batch size.
        final_loss = ce_loss.sum() / y_true.size(0)

        # Optional: Compute a derived continuous forecast from bucket probabilities for logging.
        # This lets you obtain an MSE metric for comparison.
        p = torch.softmax(y_pred, dim=1)  # shape: [B, num_buckets, H, W]
        predicted_cont = torch.sum(p * bucket_means.view(1, -1, 1, 1), dim=1)  # shape: [B, H, W]
        mse_metric = nn.functional.mse_loss(
            predicted_cont, y_true.squeeze(1), reduction="sum"
        ) / y_true.size(0)
        self.log("train_mse_metric", mse_metric, on_step=True, on_epoch=True, prog_bar=True)

        return final_loss

    # MWAE loss function as defined in GPTCast paper
    # We use it only for the recon loss
    def mwae(self, x, y):
        sx = sigmoid(x)
        sy = sigmoid(y)
        return abs(sx - sy) * sx

    def loss_func(self, y_pred, y_true):
        """
        Reconstruction (or regression) loss.
        We use the mean squared error averaged per image or the MWAE loss.
        """
        if self.hparams.vqmodel_recon_loss_type == 'ce':
            return self.ce_recon_loss(y_pred, y_true)
        elif self.hparams.vqmodel_recon_loss_type == 'mwae':
            return self.mwae(y_pred, y_true).sum() / y_true.size(0)
            # return self.mwae(y_pred, y_true).mean()
        else:
            return nn.functional.mse_loss(y_pred, y_true, reduction="sum") / y_true.size(0)
            # return nn.functional.mse_loss(y_pred, y_true, reduction="mean")

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Unpack the model output; note that we expect the VQ model to return three items.
        logits, vq_loss, loss_dict = self(x)

        # Compute the reconstruction loss on the output.
        # For CE loss, do not squeeze logits so that the shape remains [B, num_buckets, H, W]
        if self.hparams.vqmodel_recon_loss_type == 'ce':
            recon_loss = self.loss_func(logits, y)
        else:
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

        # For CE loss, do not squeeze logits so that the shape remains [B, num_buckets, H, W]
        if self.hparams.vqmodel_recon_loss_type == 'ce':
            recon_loss = self.loss_func(logits, y)
        else:
            recon_loss = self.loss_func(logits.squeeze(), y)

        total_loss = recon_loss + vq_loss

        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_vq_loss", vq_loss, prog_bar=True)
        self.log("val_total_loss", total_loss, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, vq_loss, loss_dict = self(x)
        factor = 47.83

        if self.hparams.vqmodel_recon_loss_type == 'ce':
            recon_loss = self.loss_func(logits, y)
            # For denormalized loss, also avoid squeezing
            loss_denorm = self.loss_func(logits * factor, y * factor)
        else:
            recon_loss = self.loss_func(logits.squeeze(), y)
            loss_denorm = self.loss_func(logits.squeeze() * factor, y * factor)

        total_loss = recon_loss + vq_loss
        self.log("test_recon_loss", recon_loss)
        self.log("test_vq_loss", vq_loss)
        self.log("test_total_loss", total_loss)
        self.log("test_recon_loss_denormalized", loss_denorm)


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
        # parser.n_classes = 1
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
