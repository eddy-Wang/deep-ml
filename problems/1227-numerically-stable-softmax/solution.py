import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tenso`r
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    t_max = torch.max(t, dim = dim, keepdims = True).values
    t_exp = torch.exp(t-t_max)
    sum_exp = torch.sum(t_exp, dim = dim, keepdims = True)
    return t_exp/sum_exp