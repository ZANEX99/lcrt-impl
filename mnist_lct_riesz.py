import os
import sys

# ---------------- 解决所有你遇到的错误 ----------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 🔥 关键：强制RTX5060使用sm_90兼容模式，绕过PTX编译错误
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6;9.0"
# -------------------------------------------
import math
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Utilities
# ============================================================

def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    data_root: str = "./data"
    batch_size: int = 128
    test_batch_size: int = 256
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    print_freq: int = 100
    resize_to: int = 32
    plugin_inner_ratio: float = 0.5
    use_plugin: bool = True
    lct_alpha: float = 1.0  # alpha=1, M=1, q=0 -> standard FT-like initialization
    lct_m: float = 1.0
    lct_q: float = 0.0


# ============================================================
# 2. LCT core
#    Based on the user's uploaded implementation, but rewritten
#    for batched tensor processing and stable training.
# ============================================================

def para_calculate(alpha: float, m: float, q: float) -> Tuple[float, float, float, float]:
    """
    Parameterization consistent with the uploaded code.
    alpha in fractional order style.
    """
    A = m * math.cos(alpha * math.pi / 2.0)
    B = m * math.sin(alpha * math.pi / 2.0)
    C = -q * m * math.cos(alpha * math.pi / 2.0) - math.sin(alpha * math.pi / 2.0) / m
    D = -q * m * math.sin(alpha * math.pi / 2.0) + math.cos(alpha * math.pi / 2.0) / m
    return A, B, C, D


def chirp_func_torch(length: int, theta: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = torch.arange(length, device=device, dtype=torch.float32)
    phase = (torch.pi / length) * theta * (n ** 2)
    return torch.exp(1j * phase).to(dtype)


def _complex_const(value: complex, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor(value, device=ref.device, dtype=ref.dtype)


def lct_1d_torch(f: torch.Tensor, a: float, b: float, c: float, d: float) -> torch.Tensor:
    """
    f: (..., N) complex tensor
    Returns same shape.
    """
    if not torch.is_complex(f):
        f = f.to(torch.complex64)

    N = f.shape[-1]
    device = f.device
    dtype = f.dtype
    eps = 1e-8

    if abs(b) < eps:
        if abs(a) > abs(d) + eps:
            chirp1 = chirp_func_torch(N, (c + 1.0) / (d + eps), device, dtype)
            chirp2 = chirp_func_torch(N, d, device, dtype)
            chirp3 = chirp_func_torch(N, 1.0 / (d + eps), device, dtype)
            f1 = f * chirp1
            F1 = torch.fft.fft(f1, dim=-1)
            F2 = F1 * chirp2
            f2 = torch.fft.ifft(F2, dim=-1)
            f3 = f2 * chirp3
            result = torch.fft.fft(f3, dim=-1) * torch.sqrt(_complex_const(-1j, f))
        elif abs(a) < abs(d) - eps:
            chirp1 = chirp_func_torch(N, -1.0 / (a + eps), device, dtype)
            chirp2 = chirp_func_torch(N, -a, device, dtype)
            chirp3 = chirp_func_torch(N, (c - 1.0) / (a + eps), device, dtype)
            f1 = torch.fft.ifft(f, dim=-1) * chirp1
            F1 = torch.fft.fft(f1, dim=-1)
            F2 = F1 * chirp2
            f2 = torch.fft.ifft(F2, dim=-1)
            f3 = f2 * chirp3
            result = f3 * torch.sqrt(_complex_const(1j, f))
        else:
            chirp = chirp_func_torch(N, c, device, dtype)
            result = chirp * f
    else:
        chirp1 = chirp_func_torch(N, (a - 1.0) / b, device, dtype)
        chirp2 = chirp_func_torch(N, -b, device, dtype)
        chirp3 = chirp_func_torch(N, (d - 1.0) / b, device, dtype)
        f1 = f * chirp1
        F1 = torch.fft.fft(f1, dim=-1)
        F2 = F1 * chirp2
        f2 = torch.fft.ifft(F2, dim=-1)
        result = f2 * chirp3

    return result


def lct_2d_torch(x: torch.Tensor, a: float, b: float, c: float, d: float) -> torch.Tensor:
    """
    x: (B, C, H, W), real or complex
    returns complex tensor of the same shape
    """
    if not torch.is_complex(x):
        x = x.to(torch.complex64)

    B, C, H, W = x.shape

    # along W
    xw = x.reshape(-1, W)
    xw = lct_1d_torch(xw, a, b, c, d)
    xw = xw.view(B, C, H, W)

    # along H
    xh = xw.permute(0, 1, 3, 2).contiguous().view(-1, H)
    xh = lct_1d_torch(xh, a, b, c, d)
    out = xh.view(B, C, W, H).permute(0, 1, 3, 2).contiguous()
    return out


def ilct_2d_torch(x: torch.Tensor, a: float, b: float, c: float, d: float) -> torch.Tensor:
    return lct_2d_torch(x, d, -b, -c, a)


class FixedLCT2D(nn.Module):
    def __init__(self, alpha: float = 1.0, m: float = 1.0, q: float = 0.0):
        super().__init__()
        a, b, c, d = para_calculate(alpha, m, q)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if inverse:
            return ilct_2d_torch(x, self.a, self.b, self.c, self.d)
        return lct_2d_torch(x, self.a, self.b, self.c, self.d)


# ============================================================
# 3. LCT-domain Riesz kernels and plug-in block
# ============================================================

def build_lcrt_riesz_kernels(
    H: int,
    W: int,
    bx: float,
    by: float,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build anisotropic Riesz kernels in LCT domain.
    Inspired by the user's MATLAB ker_lcrt function, but centered
    on fftfreq grids for numerical stability.
    """
    fy = torch.fft.fftfreq(H, d=1.0, device=device).view(H, 1)
    fx = torch.fft.fftfreq(W, d=1.0, device=device).view(1, W)

    bx = float(abs(bx)) + eps
    by = float(abs(by)) + eps

    ux = fx / bx
    vy = fy / by
    denom = torch.sqrt(ux ** 2 + vy ** 2 + eps)

    kx = (-1j * ux / denom).to(dtype)
    ky = (-1j * vy / denom).to(dtype)
    kx[0, 0] = 0.0 + 0.0j
    ky[0, 0] = 0.0 + 0.0j

    return kx[None, None, :, :], ky[None, None, :, :]


class SpatialResidualBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class LCTRieszResidualBlock(nn.Module):
    """
    input -> LCT -> Riesz kernel in LCT domain -> ILCT -> fuse
          -> spatial residual branch
    out = x + alpha * spatial + beta * spectral
    """
    def __init__(
        self,
        channels: int,
        inner_channels: int,
        lct_alpha: float = 1.0,
        lct_m: float = 1.0,
        lct_q: float = 0.0,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        self.lct = FixedLCT2D(alpha=lct_alpha, m=lct_m, q=lct_q)

        self.spec_fuse = nn.Sequential(
            nn.Conv2d(inner_channels * 4, inner_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.spa_branch = SpatialResidualBranch(channels)
        self.alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, H, W = x.shape

        xr = self.pre(x)
        z = self.lct(xr, inverse=False)

        bx = self.lct.b
        by = self.lct.b
        kx, ky = build_lcrt_riesz_kernels(
            H=H,
            W=W,
            bx=bx,
            by=by,
            device=x.device,
            dtype=z.dtype,
        )

        z_x = z * kx
        z_y = z * ky

        rx = self.lct(z_x, inverse=True).real
        ry = self.lct(z_y, inverse=True).real
        mag = torch.sqrt(rx * rx + ry * ry + 1e-6)

        spec_feat = torch.cat([xr, rx, ry, mag], dim=1)
        spec_out = self.spec_fuse(spec_feat)

        spa_out = self.spa_branch(x)

        out = identity + self.alpha * spa_out + self.beta * spec_out
        out = self.out_act(out)
        return out


# ============================================================
# 4. ResNet-16-style backbone for MNIST
# ============================================================
class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet16MNIST(nn.Module):
    def __init__(self, num_classes: int = 10, use_plugin: bool = True, cfg: Config = None):
        super().__init__()
        if cfg is None:
            cfg = Config()

        self.use_plugin = use_plugin
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
        )

        self.stage2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32, stride=1),
        )

        self.stage3 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64, stride=1),
        )

        if use_plugin:
            self.plugin1 = LCTRieszResidualBlock(
                channels=16,
                inner_channels=max(8, int(16 * cfg.plugin_inner_ratio)),
                lct_alpha=cfg.lct_alpha,
                lct_m=cfg.lct_m,
                lct_q=cfg.lct_q,
            )
            self.plugin2 = LCTRieszResidualBlock(
                channels=32,
                inner_channels=max(8, int(32 * cfg.plugin_inner_ratio)),
                lct_alpha=cfg.lct_alpha,
                lct_m=cfg.lct_m,
                lct_q=cfg.lct_q,
            )
        else:
            self.plugin1 = nn.Identity()
            self.plugin2 = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.plugin1(x)
        x = self.stage2(x)
        x = self.plugin2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


# ============================================================
# 5. Data, training, evaluation
# ============================================================

def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((cfg.resize_to, cfg.resize_to)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST(root=cfg.data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=cfg.data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    print_freq: int = 100,
) -> Tuple[float, float]:
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    count = 0

    use_amp = device.type == "cuda"

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        loss_sum += loss.item() * bs
        acc_sum += accuracy(outputs.detach(), targets) * bs
        count += bs

        if (i + 1) % print_freq == 0:
            print(
                f"Epoch [{epoch}] Step [{i+1}/{len(loader)}] "
                f"Loss: {loss_sum / count:.4f} Acc: {acc_sum / count:.2f}%"
            )

    return loss_sum / count, acc_sum / count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        bs = images.size(0)
        loss_sum += loss.item() * bs
        acc_sum += accuracy(outputs, targets) * bs
        count += bs

    return loss_sum / count, acc_sum / count


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    cfg = Config()
    seed_everything(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(cfg)

    model = ResNet16MNIST(num_classes=10, use_plugin=cfg.use_plugin, cfg=cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = 0.0
    total_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, epoch, cfg.print_freq
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
            f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.2f}% | "
            f"Time {elapsed:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "best_acc": best_acc,
                },
                "best_mnist_lct_riesz.pth",
            )
            print(f"Saved new best checkpoint with acc={best_acc:.2f}%")

    total_elapsed = time.time() - total_start
    print(f"Training finished. Best test acc: {best_acc:.2f}% | Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
