"""Class implementing the vector quantization part for the SmaAT-UNet"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        """
        Args:
            num_embeddings (int): Number of codebook vectors.
            embedding_dim (int): Dimensionality of each codebook vector.
            commitment_cost (float): Weight for the commitment loss.
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # This is b basically
        self.commitment_cost = commitment_cost

        # Initialize the embedding codebook with uniform random values.
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) where C == embedding_dim.
        
        Returns:
            quantized (Tensor): Quantized tensor of the same shape as x.
            loss (Tensor): Sum of codebook loss and commitment loss.
            loss_dict (dict): Dictionary containing individual loss terms.
        """
        # Permute to shape (B, H, W, C) and flatten to (B*H*W, C)
        # I think here that each pixel is a  
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        flat_x = x_perm.view(-1, self.embedding_dim)

        # Compute squared L2 distances between flat_x and each codebook vector.
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_x, self.embedding.weight.t())
        )  # shape: (B*H*W, num_embeddings)

        # For each latent vector, we want to find the index of the closest embedding.
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_x.dtype)

        # Quantize: replace each latent vector with its closest codebook vector.
        quantized_flat = torch.matmul(encodings, self.embedding.weight)  # shape: (B*H*W, embedding_dim)
        quantized = quantized_flat.view(x_perm.shape)  # (B, H, W, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # Compute the Codebook Loss:
        # This loss will update the codebook embeddings.
        codebook_loss = F.mse_loss(quantized, x.detach())

        # Compute the Commitment Loss:
        # This loss will update the encoder outputs to commit to the quantized vectors.
        commitment_loss = self.commitment_cost * F.mse_loss(x, quantized.detach())

        # Total VQ Loss:
        vq_loss = codebook_loss + commitment_loss

        # Apply the straight-through estimator.
        # During the backward pass, the gradient from quantized will be passed to x.
        quantized = x + (quantized - x).detach()

        return quantized, vq_loss, {'codebook_loss': codebook_loss, 'commitment_loss': commitment_loss}
