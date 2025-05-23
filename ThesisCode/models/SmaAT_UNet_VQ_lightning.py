"""Overwrites basically the unet_precip_regression_lightning.py and adds VQ from vector_quantization.py"""

from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS, OutConv, MixDownDS, MixUpDS
from models.layers import CBAM
from models.regression_lightning import Precip_regression_base
from models.vector_quantization import VectorQuantizer

class SmaAT_UNet_VQ(Precip_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels

        # I think this is set as 1 in regression_lightning at UNet_Base model because we have regression
        # We set to 5 for CE because we have buckets
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        ### Encoder Components ###

        # DoubleConvDS defined in unet_parts_depthwise_separable.py
        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        # CBAM defined in layers.py
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        # DownDS defined in unet_parts_depthwise_separable.py
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = MixDownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = MixDownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        
        ### VQ module at the botleneck components ###
        
        # We need to add them as hparams at some point
        vq_num_embeddings = getattr(self.hparams, "vq_num_embeddings", 512)
        vq_commitment_cost = getattr(self.hparams, "vq_commitment_cost", 0.25)
        bottleneck_channels = 1024 // factor  # Must match the channel dimension from cbam5.
        self.vq = VectorQuantizer(
            num_embeddings=vq_num_embeddings,
            embedding_dim=bottleneck_channels,
            commitment_cost=vq_commitment_cost
        )

        ### Decoder Components ###

        # UpDS defined in unet_parts_depthwise_separable.py
        self.up1 = MixUpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = MixUpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        # OutConv defined in unet_parts_depthwise_separable.py
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        # Encoder.
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)

        # VQ Bottleneck.
        # 3rd argument here is a dict of both losses separately
        # I don't use it yet but we might need it
        x5Quantized, vq_loss, loss_dict = self.vq(x5Att)

        # Decoder using quantized features.
        x = self.up1(x5Quantized, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)

        # Return logits and vq_loss; the training loop should combine these losses.
        return logits, vq_loss, loss_dict
