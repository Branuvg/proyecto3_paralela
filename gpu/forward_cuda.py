"""Propagacion hacia adelante de CNN1/CNN2 usando SOLO kernels propios (Numba CUDA)."""
import torch
import torch.nn as nn
from gpu import kernels_cuda as K


@torch.no_grad()
def forward(model, x):
    """x: tensor float32 CUDA (N,3,64,64). Dropout = identidad en inferencia."""
    h = x.contiguous()
    layers = [l for l in model if not isinstance(l, nn.Dropout)]
    i = 0
    while i < len(layers):
        l = layers[i]
        if isinstance(l, nn.Conv2d):                      # Conv2d + ReLU fusionados
            N, _, H, W = h.shape
            y = torch.empty((N, l.out_channels, H - 2, W - 2), device=h.device)
            K.conv3x3_relu(h, l.weight.data, l.bias.data, y); h = y; i += 2
        elif isinstance(l, nn.MaxPool2d):
            N, C, H, W = h.shape
            y = torch.empty((N, C, H // 2, W // 2), device=h.device)
            K.maxpool2(h, y); h = y; i += 1
        elif isinstance(l, nn.Flatten):
            h = h.reshape(h.shape[0], -1); i += 1
        elif isinstance(l, nn.Linear):
            relu = i + 1 < len(layers) and isinstance(layers[i + 1], nn.ReLU)
            y = torch.empty((h.shape[0], l.out_features), device=h.device)
            K.dense(h, l.weight.data, l.bias.data, y, relu); h = y; i += 2 if relu else 1
        else:
            raise ValueError(l)
    return h


@torch.no_grad()
def predict(model, X, bs=256):
    out = []
    for i in range(0, len(X), bs):
        logits = forward(model, X[i:i + bs])
        idx = torch.empty(len(logits), dtype=torch.int32, device=X.device)
        K.argmax(logits, idx)
        out.append(idx)
    return torch.cat(out).long()
