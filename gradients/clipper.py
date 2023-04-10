import torch
from torch.nn.utils import clip_grad_value_


def base_clip(grad):
    clip_grad_value_(grad, clip_value=1)
    return grad

def fancy_clip(grad):
    return grad