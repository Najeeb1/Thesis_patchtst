#implementation of utils

import torch

class RevIN(torch.nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False, target_idx=-1):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.target_idx = target_idx
        if self.affine:
            self._init_params()

    def forward(self, x, mode):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else: raise AssertionError
        return x

    def _init_params(self):
        self.affine_weight = torch.nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = torch.nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev[:, :, self.target_idx]
        if self.subtract_last:
            x = x + self.last[:, :, self.target_idx]
        else:
            x = x + self.mean[:, :, self.target_idx]
        return x


class Transpose(torch.nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: 
            return x.transpose(*self.dims).contiguous()
        else: 
            return x.transpose(*self.dims)


def positional_encoding(q_len, d_model):
    W_pos = torch.empty((q_len, d_model))
    torch.nn.init.uniform_(W_pos, -0.02, 0.02)
    return torch.nn.Parameter(W_pos, requires_grad=True)