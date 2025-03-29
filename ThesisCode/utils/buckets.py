# buckets.py
import torch

def get_bucket_boundaries(device=None):
    """
    Returns the bucket boundaries tensor.
    These boundaries define the edges between buckets.
    """
    return torch.tensor([
        0.00333, 0.00667, 0.01000, 0.01333, 0.01667, 0.02000, 0.02333,
        0.02667, 0.03000, 0.03333, 0.03667, 0.04000, 0.04333, 0.04667
    ], device=device)

def get_bucket_means(device=None):
    """
    Returns the bucket means (midpoints) tensor.
    These values are used to convert discrete bucket predictions into continuous values.
    """
    return torch.tensor([
        0.00167, 0.00500, 0.00833, 0.01167, 0.01500, 0.01833, 0.02167,
        0.02500, 0.02833, 0.03167, 0.03500, 0.03833, 0.04167, 0.04500, 0.04833
    ], device=device)

def get_bucket_weights(device=None):
    """
    Returns the bucket weights tensor.
    These weights can be used to balance the loss contribution from each bucket.
    """
    # These go logarithmically inverse
    # The same thing is done in the RainAI paper as well
    return torch.tensor([
        0.5107,
        0.6014,
        0.6270,
        0.6295,
        0.6310,
        0.6359,
        0.6472,
        0.6667,
        0.6901,
        0.7298,
        0.7823,
        0.8428,
        0.9084,
        0.9617,
        1.0000
    ], device=device)
