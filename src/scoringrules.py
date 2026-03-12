import torch

# I looked into some other scoring rules but I think they may not be applicable? idk

BETA = 1 # For energym the paper defines that beta is in (0,2) and uses 1 in its experiments
GAMMA = 1 # For kernel, needs to be defined by us
ALPHA_ENERGY = 1.0 # Weight for energy-kernel
ALPHA_KERNEL = 1.0 # Weight for energy-kernel

# Implementation of C.1.1
# P is the distribution, y is the goal value
def energy(P, y):
    m = P.shape[0]

    term1 = (2 / m) * torch.sum(torch.abs(P - y) ** BETA)

    differences = torch.abs(P.unsqueeze(0) - P.unsqueeze(1)) ** BETA # Absolute differences

    # Term 2 of the formula sums all the differences, except when j = k. So the diagonal should not be counted
    # Create a mask for the differences where when j = k, set to false
    boolean_mask = ~torch.eye(m, dtype = torch.bool)

    term2 = (1 / (m * (m - 1))) * (differences[boolean_mask].sum())

    return term1 - term2

# Implementation of C.1.2
# In B.2.2 the Gaussian Kernel k(x,y) is defined
def kernel(P, y):
    m = P.shape[0]

    k_term1 = torch.exp((-torch.abs(P.unsqueeze(0) - P.unsqueeze(1)) ** 2) / (2 * (GAMMA ** 2))) # Gaussian kernel

    # Create a mask for the differences where when j = k, set to false
    boolean_mask = ~torch.eye(m, dtype = torch.bool)

    term1 = (1 / (m * (m - 1))) * (k_term1[boolean_mask].sum())

    k_term2 = torch.exp((-torch.abs(P - y) ** 2) / (2 * (GAMMA ** 2))) # Gaussian kernel

    term2 = (2 / m) * torch.sum(k_term2)

    return term1 - term2

# Weighted sum of energy and kernel, by lemma 4
def energy_kernel(P, y):
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