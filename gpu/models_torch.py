"""CNN 1 y CNN 2 (misma arquitectura que el notebook Keras) en PyTorch + bucle de entrenamiento."""
import time
import torch
import torch.nn as nn

NUM_CLASSES = 29


def build_cnn(name):
    if name == "CNN1":
        m = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 14 * 14, 128), nn.ReLU(), nn.Linear(128, NUM_CLASSES))
    else:
        m = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, NUM_CLASSES))
    for l in m:  # inicializacion de Keras: Glorot uniforme + sesgo 0
        if isinstance(l, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)
    return m


@torch.no_grad()
def evaluate(model, X, y, bs=256):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        correct += (model(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum()
    return (correct.float() / len(X)).item()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def train(model, Xtr, ytr, Xva, yva, epochs, device, bs=32, seed=42):
    """Mismo esquema que model.fit de Keras: Adam(1e-3, eps=1e-7), lote 32, barajado por epoca, validacion por epoca."""
    g = torch.Generator(device=device); g.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    times = []
    for ep in range(epochs):
        sync(device); t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, device=device, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        acc = evaluate(model, Xva, yva)   # .item() sincroniza
        sync(device); times.append(time.perf_counter() - t0)
    return times, acc
