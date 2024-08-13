import torch


def clamp(x:torch.Tensor, min_value=0, max_value=1):
    return torch.clamp(x, min=min_value, max=max_value)


def inplace_clamp(x:torch.Tensor, min_value:float=0, max_value:float=1):
    return x.clamp_(min=min_value, max=max_value)
