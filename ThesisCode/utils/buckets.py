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

# def get_bucket_boundaries(device=None):
#     """
#     Returns the bucket boundaries tensor (in mm/5min) for 15 buckets.
#     These boundaries were computed from your analysis.
#     """
#     return torch.tensor([
#         0.00250889, # Bucket 0 upper bound / Bucket 1 lower bound
#         0.00355425, # Bucket 1 upper bound / Bucket 2 lower bound
#         0.00480870, # Bucket 2 upper bound / Bucket 3 lower bound
#         0.00564499, # Bucket 3 upper bound / Bucket 4 lower bound
#         0.00689944, # Bucket 4 upper bound / Bucket 5 lower bound
#         0.00794480, # Bucket 5 upper bound / Bucket 6 lower bound
#         0.00940832, # Bucket 6 upper bound / Bucket 7 lower bound
#         0.01128998, # Bucket 7 upper bound / Bucket 8 lower bound
#         0.01338072, # Bucket 8 upper bound / Bucket 9 lower bound
#         0.01609868, # Bucket 9 upper bound / Bucket 10 lower bound
#         0.02028016, # Bucket 10 upper bound / Bucket 11 lower bound
#         0.02487978, # Bucket 11 upper bound / Bucket 12 lower bound
#         0.03031570, # Bucket 12 upper bound / Bucket 13 lower bound
#         0.03951495, # Bucket 13 upper bound / Bucket 14 lower bound
#     ], device=device)

# def get_bucket_means(device=None):
#     """
#     Returns the bucket means tensor for 15 buckets.
#     These means are approximated here as arithmetic midpoints of the bucket boundaries.
#     In practice, you would compute the mean of the pixel values falling into each bucket.
#     """
#     return torch.tensor([
#         (0.00000000 + 0.00250889) / 2,  # Bucket 0
#         (0.00250889 + 0.00355425) / 2,  # Bucket 1
#         (0.00355425 + 0.00480870) / 2,  # Bucket 2
#         (0.00480870 + 0.00564499) / 2,  # Bucket 3
#         (0.00564499 + 0.00689944) / 2,  # Bucket 4
#         (0.00689944 + 0.00794480) / 2,  # Bucket 5
#         (0.00794480 + 0.00940832) / 2,  # Bucket 6
#         (0.00940832 + 0.01128998) / 2,  # Bucket 7
#         (0.01128998 + 0.01338072) / 2,  # Bucket 8
#         (0.01338072 + 0.01609868) / 2,  # Bucket 9
#         (0.01609868 + 0.02028016) / 2,  # Bucket 10
#         (0.02028016 + 0.02487978) / 2,  # Bucket 11
#         (0.02487978 + 0.03031570) / 2,  # Bucket 12
#         (0.03031570 + 0.03951495) / 2,  # Bucket 13
#         (0.03951495 + 0.47125235) / 2   # Bucket 14
#     ], device=device)

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

    # return torch.tensor([1.0000, 0.9617, 0.9084, 0.8428, 0.7823,
    #     0.7298, 0.6901, 0.6667, 0.6472, 0.6359,
    #     0.6310, 0.6295, 0.6270, 0.6014, 0.5107], device=device)

    # No weights gives very bad val loss
    # return torch.tensor([
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000,
    #     1.0000
    # ], device=device)

# def get_bucket_boundaries(device=None):
#     """
#     Returns the final bucket boundaries tensor for a 5-bucket quantile-based scheme.
#     Buckets are defined as:
#       Bucket 0: [0.0, 0.00062722)
#       Bucket 1: [0.00062722, 0.00104537)
#       Bucket 2: [0.00104537, 0.00167259)
#       Bucket 3: [0.00167259, 0.00334518)
#       Bucket 4: [0.00334518, 0.47125235]
#     """
#     return torch.tensor(
#         [0.0, 0.00062722, 0.00104537, 0.00167259, 0.00334518, 0.47125235],
#         device=device
#     )

# def get_bucket_means(device=None):
#     """
#     Returns the final bucket means tensor for a 5-bucket quantile-based scheme.
#     These means are estimated as the weighted average of values falling in each bucket:
#       Bucket 0: Mean ~ 0.00025
#       Bucket 1: Mean ~ 0.00075
#       Bucket 2: Mean ~ 0.00125
#       Bucket 3: Mean ~ 0.00237
#       Bucket 4: Mean ~ 0.00667
#     """
#     return torch.tensor(
#         [0.00025, 0.00075, 0.00125, 0.00237, 0.00667],
#         device=device
#     )

# def get_bucket_weights(device=None):
#     """
#     Returns the final bucket weights tensor for a 5-bucket quantile-based scheme.
#     Because nearly 50% of all pixels fall into bucket 0, we assign it a lower weight.
#     For a first pass you might use:
#       Bucket 0: 0.5
#       Buckets 1-4: 1.0 each
#     You can later experiment with different weighting schemes.
#     """
#     return torch.tensor(
#         [0.5, 1.0, 1.0, 1.0, 1.0],
#         device=device
#     )

