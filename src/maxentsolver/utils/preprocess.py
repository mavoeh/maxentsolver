import numpy as np
import torch


class NotBinaryError(Exception):
    pass

def check_adjust_binary(data):
    """
    Ensure data is binary (0/1) or (-1/1) and map to {-1, 1}.
    """
    data = torch.as_tensor(data)
    unique_values = torch.unique(data)
    if unique_values.numel() == 2:
        if torch.all((unique_values == 0) | (unique_values == 1)):
            return 2 * data.float() - 1
        elif torch.all((unique_values == -1) | (unique_values == 1)):
            return data.float()
    raise NotBinaryError("Data must be binary with values in {0,1} or {-1,1}.")


def binarize_data(data, z_score_threshold=0.0):
    """
    Binarize continuous data using z-score thresholding.
    Values above the threshold become 1, below become 0.
    """
    data = torch.as_tensor(data, dtype=torch.float)
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    assert torch.all(std > 0), "Standard deviation is zero for some features."
    z_scores = (data - mean) / std
    binary_data = (z_scores > z_score_threshold).float()
    binary_data = 2 * binary_data - 1  # Convert to {-1, 1}
    return binary_data