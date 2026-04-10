"""
Scoring rules for predictive distributions.

This module implements energy, kernel, and combined energy-kernel scoring rules
for evaluating probabilistic forecasts.
"""

import torch

BETA = 1 # For energym the paper defines that beta is in (0,2) and uses 1 in its experiments
GAMMA = 1 # For kernel, needs to be defined by us
ALPHA_ENERGY = 1.0 # Weight for energy-kernel
ALPHA_KERNEL = 1.0 # Weight for energy-kernel

def energy(P, y):
    """
    Compute the energy score.

    Implements the energy score formula (C.1.1).

    Args:
        P (torch.Tensor): The predictive distribution.
        y (torch.Tensor): The goal value observation.

    Returns:
        torch.Tensor: The computed energy score.
    """
    single_input = (P.dim() == 1)
    if single_input:
        P = P.unsqueeze(0)

    if y.dim() == 0:
        y = y.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    m = P.shape[1]

    term1 = (2 / m) * torch.sum(torch.abs(P - y) ** BETA, dim=1)

    differences = torch.abs(P.unsqueeze(1) - P.unsqueeze(2)) ** BETA # Absolute differences

    # Term 2 of the formula sums all the differences, except when j = k. So the diagonal should not be counted
    diagonal_sum = torch.diagonal(differences, dim1=1, dim2=2).sum(dim=1)
    off_diagonal_sum = differences.sum(dim=(1, 2)) - diagonal_sum

    term2 = (1 / (m * (m - 1))) * off_diagonal_sum

    score = term1 - term2

    if single_input:
        return score.squeeze(0)

    return score.mean()

def kernel(P, y):
    """
    Compute the kernel score.

    Implements the kernel score formula (C.1.2) using a Gaussian Kernel (B.2.2).

    Args:
        P (torch.Tensor): The predictive distribution.
        y (torch.Tensor): The goal value observation.

    Returns:
        torch.Tensor: The computed kernel score.
    """
    single_input = (P.dim() == 1)
    if single_input:
        P = P.unsqueeze(0)

    if y.dim() == 0:
        y = y.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    m = P.shape[1]

    k_term1 = torch.exp((-torch.abs(P.unsqueeze(1) - P.unsqueeze(2)) ** 2) / (2 * (GAMMA ** 2))) # Gaussian kernel

    # Create a mask for the differences where when j = k, set to false
    diagonal_sum = torch.diagonal(k_term1, dim1=1, dim2=2).sum(dim=1)
    off_diagonal_sum = k_term1.sum(dim=(1, 2)) - diagonal_sum

    term1 = (1 / (m * (m - 1))) * off_diagonal_sum

    k_term2 = torch.exp((-torch.abs(P - y) ** 2) / (2 * (GAMMA ** 2))) # Gaussian kernel

    term2 = (2 / m) * torch.sum(k_term2, dim=1)

    score = term1 - term2

    if single_input:
        return score.squeeze(0)

    return score.mean()

def energy_kernel(P, y):
    """
    Compute the weighted sum of energy and kernel scores.

    Based on Lemma 4.

    Args:
        P (torch.Tensor): The predictive distribution.
        y (torch.Tensor): The goal value observation.

    Returns:
        torch.Tensor: The computed combined energy-kernel score.
    """
    return (ALPHA_ENERGY * energy(P, y)) + (ALPHA_KERNEL * kernel(P, y))

if __name__ == '__main__':
    # We want the score minimized, lower it is, closer the distribution is to the value

    distribution = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    goal = torch.tensor(1.0)

    energy_score = energy(distribution, goal)
    kernel_score = kernel(distribution, goal)
    energy_kernel_score = energy_kernel(distribution, goal)

    print(energy_score)
    print(kernel_score)
    print(energy_kernel_score)