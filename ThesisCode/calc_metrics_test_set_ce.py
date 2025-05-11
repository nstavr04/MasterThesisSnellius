# This code will work with the ce loss

"""Calculate different test metrics. We use MSE for evaluation"""
import torch
from torch import nn

from root import ROOT_DIR
from utils import dataset_precip, model_classes
from tqdm import tqdm
import os
import numpy as np
import json
from utils.buckets import get_bucket_means

def get_metrics_from_model(model, test_dl, threshold=0.01667, device: str = "cpu"):
    device = torch.device(device)
    loss_func = nn.functional.mse_loss
    with torch.no_grad():
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        loss = 0.0
        loss_denorm = 0.0
        total_vq_loss = 0.0
        total_codebook_loss = 0.0
        total_commitment_loss = 0.0

        for x, y_true in tqdm(test_dl, leave=True):
            # Move data to device
            x = x.to(device)
            y_true = y_true.to(device)

            # Get model output and unpack VQ losses if available
            output = model(x)
            if isinstance(output, tuple) and len(output) == 3:
                logits, vq_loss, loss_dict = output
            else:
                logits = output
                vq_loss = torch.tensor(0.0, device=device)
                loss_dict = {"codebook_loss": torch.tensor(0.0, device=device), 
                             "commitment_loss": torch.tensor(0.0, device=device)}

            # Process the model output:
            # If logits has more than one channel, assume CE branch and compute continuous prediction.
            if logits.ndim >= 2 and logits.shape[1] > 1:
                # Define bucket means as in ce_recon_loss
                bucket_means = get_bucket_means(device=logits.device)

                # Compute softmax over bucket dimension and derive continuous prediction
                p = torch.softmax(logits, dim=1)
                y_pred_cont = torch.sum(p * bucket_means.view(1, -1, 1, 1), dim=1)
            else:
                # Regression branch: assume logits are continuous predictions.
                y_pred_cont = logits.squeeze(1) if logits.ndim == 4 else logits

            # Calculate MSE loss using the continuous prediction
            loss += loss_func(y_pred_cont, y_true.squeeze(), reduction="sum")
            # Denormalize predictions and ground truth (same as in training)
            y_pred_adj = y_pred_cont * 47.83
            y_true_adj = y_true.squeeze() * 47.83
            loss_denorm += loss_func(y_pred_adj, y_true_adj, reduction="sum")
            
            # Accumulate VQ losses (will be zero if not applicable)
            total_vq_loss += vq_loss.item()
            total_codebook_loss += loss_dict["codebook_loss"].item()
            total_commitment_loss += loss_dict["commitment_loss"].item()

            # Convert from mm/5min to mm/h
            y_pred_adj *= 12
            y_true_adj *= 12
            # Create binary masks based on the threshold
            y_pred_mask = y_pred_adj > (threshold * 47.83 * 12)
            y_true_mask = y_true_adj > (threshold * 47.83 * 12)

            # Compute confusion matrix components using np.bincount
            tn, fp, fn, tp = np.bincount(
                y_true_mask.cpu().view(-1) * 2 + y_pred_mask.cpu().view(-1), minlength=4
            )
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

        mse_image = loss / len(test_dl)
        mse_denormalized_image = loss_denorm / len(test_dl)
        mse_pixel = mse_denormalized_image / torch.numel(y_true)
        avg_vq_loss = total_vq_loss / len(test_dl)
        avg_codebook_loss = total_codebook_loss / len(test_dl)
        avg_commitment_loss = total_commitment_loss / len(test_dl)

        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)
        csi = total_tp / (total_tp + total_fn + total_fp)
        far = total_fp / (total_tp + total_fp)
        hss = ((total_tp * total_tn) - (total_fn * total_fp)) / (
            (total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn)
        )

    return (
        mse_image.item(),
        mse_denormalized_image.item(),
        mse_pixel.item(),
        precision,
        recall,
        accuracy,
        f1,
        csi,
        far,
        hss,
        avg_vq_loss,
        avg_codebook_loss,
        avg_commitment_loss,
    )


def calculate_metrics_for_models(model_folder, threshold: float = 0.01667):
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=ROOT_DIR
        / "data"
        / "precipitation"
        / f"train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_{int(50)}.h5",
        num_input_images=12,
        num_output_images=6,
        train=False,
    )

    # Move both the model and the data to the same device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Adjust batch size and workers as needed.
    test_dl = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1, persistent_workers=True
    )

    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]

    model_metrics = {}
    for model_file in tqdm(models, desc="Models", leave=True):
        model, model_name = model_classes.get_model_class(model_file)
        model = model.load_from_checkpoint(model_folder / model_file)
        model.eval()
        model.to(device)

        (
            mse_image,
            mse_denormalized_image,
            mse_pixel,
            precision,
            recall,
            accuracy,
            f1,
            csi,
            far,
            hss,
            avg_vq_loss,
            avg_codebook_loss,
            avg_commitment_loss,
        ) = get_metrics_from_model(model, test_dl, threshold, device=device)
        model_metrics[model_name] = {
            "mse": mse_image,
            "mse_denormalized_image": mse_denormalized_image,
            "mse_pixel": mse_pixel,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1": f1,
            "CSI": csi,
            "FAR": far,
            "HSS": hss,
            "avg_vq_loss": avg_vq_loss,
            "avg_codebook_loss": avg_codebook_loss,
            "avg_commitment_loss": avg_commitment_loss,
        }
        print(model_name, model_metrics[model_name])
    return model_metrics


if __name__ == "__main__":
    load_metrics = False

    # Calculates metrics for all models in the checkpoints/comparison folder
    model_folder = ROOT_DIR / "checkpoints" / "comparison"
    threshold = 0.5

    test_metrics_file = model_folder / f"model_metrics_{threshold}mmh.txt"
    if load_metrics:
        with open(test_metrics_file) as f:
            model_metrics = json.loads(f.read())
    else:
        model_metrics = calculate_metrics_for_models(model_folder, threshold=threshold)
        with open(test_metrics_file, "w") as f:
            json.dump(model_metrics, f, indent=4)
    print(model_metrics)
