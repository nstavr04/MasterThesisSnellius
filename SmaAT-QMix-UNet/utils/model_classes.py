from models import unet_precip_regression_lightning as unet_regr
from models import SmaAT_UNet_VQ_lightning
import lightning.pytorch as pl

# Used in the plotting and for the comparison etc. in testing
def get_model_class(model_file) -> tuple[type[pl.LightningModule], str]:
    
    if "SmaAT_UNet_VQ_MSE_PartialMixConv" in model_file:
        model_name = "SmaAT_UNet_VQ_MSE_PartialMixConv"
        model = SmaAT_UNet_VQ_lightning.SmaAT_UNet_MixConv_VQ
    elif "SmaAT_UNet_PartialMixConv_MSE" in model_file:
        model_name = "SmaAT_UNet_PartialMixConv_MSE"
        model = SmaAT_UNet_VQ_lightning.SmaAT_UNet_MixConv
    elif "SmaAT_UNet_VQ_MSE" in model_file:
        model_name = "SmaAT_UNet_VQ_MSE"
        model = SmaAT_UNet_VQ_lightning.SmaAT_UNet_VQ
    elif "SmaAT_UNet" in model_file:
        model_name = "SmaAT_UNet"
        model = unet_regr.UNetDS_Attention
    else:
        raise NotImplementedError("Model not found")
    return model, model_name
