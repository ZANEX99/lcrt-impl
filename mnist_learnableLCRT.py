import math
import time
from typing import Tuple

import torch
import torch.nn as nn

import mnist_lct_riesz as fixed


def learnable_lct_matrix(
    alpha: torch.Tensor, log_m: torch.Tensor, q: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a trainable unit-determinant LCT matrix [[a, b], [c, d]]."""
    m = torch.exp(log_m)
    angle = alpha * (torch.pi / 2.0)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    a = m * cos_angle
    b = m * sin_angle
    c = -q * m * cos_angle - sin_angle / m
    d = -q * m * sin_angle + cos_angle / m
    return a, b, c, d


def lct_2d_anisotropic(
    x: torch.Tensor,
    matrix_x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    matrix_y: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply horizontal and vertical LCTs using separately trainable matrices."""
    if not torch.is_complex(x):
        x = x.to(torch.complex64)

    batch, channels, height, width = x.shape
    xw = fixed.lct_1d_torch(x.reshape(-1, width), *matrix_x)
    xw = xw.view(batch, channels, height, width)

    xh = xw.permute(0, 1, 3, 2).contiguous().view(-1, height)
    xh = fixed.lct_1d_torch(xh, *matrix_y)
    return xh.view(batch, channels, width, height).permute(0, 1, 3, 2).contiguous()


def inverse_matrix(
    matrix: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b, c, d = matrix
    return d, -b, -c, a


class LearnableLCT2D(nn.Module):
    """
    Two-dimensional LCT with trainable horizontal and vertical matrices.

    Each matrix is parameterized by (alpha, exp(log_m), q), preserving
    a*d - b*c = 1 throughout optimization.
    """

    def __init__(self, alpha: float = 1.0, m: float = 1.0, q: float = 0.0):
        super().__init__()
        log_m = math.log(m)
        self.alpha_x = nn.Parameter(torch.tensor(float(alpha)))
        self.log_m_x = nn.Parameter(torch.tensor(float(log_m)))
        self.q_x = nn.Parameter(torch.tensor(float(q)))
        self.alpha_y = nn.Parameter(torch.tensor(float(alpha)))
        self.log_m_y = nn.Parameter(torch.tensor(float(log_m)))
        self.q_y = nn.Parameter(torch.tensor(float(q)))

    def matrices(
        self,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        matrix_x = learnable_lct_matrix(self.alpha_x, self.log_m_x, self.q_x)
        matrix_y = learnable_lct_matrix(self.alpha_y, self.log_m_y, self.q_y)
        return matrix_x, matrix_y

    def b_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        matrix_x, matrix_y = self.matrices()
        return matrix_x[1], matrix_y[1]

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        matrix_x, matrix_y = self.matrices()
        if inverse:
            matrix_x = inverse_matrix(matrix_x)
            matrix_y = inverse_matrix(matrix_y)
        return lct_2d_anisotropic(x, matrix_x, matrix_y)


def build_learnable_lcrt_riesz_kernels(
    height: int,
    width: int,
    bx: torch.Tensor,
    by: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct Riesz multipliers directly from the current trainable b entries."""
    fy = torch.fft.fftfreq(height, d=1.0, device=device).view(height, 1)
    fx = torch.fft.fftfreq(width, d=1.0, device=device).view(1, width)

    # This is the differentiable analogue of the fixed version's abs(b) + eps.
    bx_magnitude = torch.sqrt(bx.square() + eps * eps)
    by_magnitude = torch.sqrt(by.square() + eps * eps)
    ux = fx / bx_magnitude
    vy = fy / by_magnitude
    denom = torch.sqrt(ux.square() + vy.square() + eps)

    dc_mask = torch.ones((height, width), device=device)
    dc_mask[0, 0] = 0.0
    kx = (-1j * ux / denom * dc_mask).to(dtype)
    ky = (-1j * vy / denom * dc_mask).to(dtype)
    return kx[None, None, :, :], ky[None, None, :, :]


class LearnableLCTRieszResidualBlock(nn.Module):
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
        self.lct = LearnableLCT2D(alpha=lct_alpha, m=lct_m, q=lct_q)
        self.spec_fuse = nn.Sequential(
            nn.Conv2d(inner_channels * 4, inner_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spa_branch = fixed.SpatialResidualBranch(channels)
        self.alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, height, width = x.shape
        xr = self.pre(x)
        matrix_x, matrix_y = self.lct.matrices()
        z = lct_2d_anisotropic(xr, matrix_x, matrix_y)
        bx, by = matrix_x[1], matrix_y[1]
        kx, ky = build_learnable_lcrt_riesz_kernels(
            height=height,
            width=width,
            bx=bx,
            by=by,
            device=x.device,
            dtype=z.dtype,
        )

        inverse_x = inverse_matrix(matrix_x)
        inverse_y = inverse_matrix(matrix_y)
        rx = lct_2d_anisotropic(z * kx, inverse_x, inverse_y).real
        ry = lct_2d_anisotropic(z * ky, inverse_x, inverse_y).real
        magnitude = torch.sqrt(rx.square() + ry.square() + 1e-6)
        spec_out = self.spec_fuse(torch.cat([xr, rx, ry, magnitude], dim=1))
        spa_out = self.spa_branch(x)
        return self.out_act(identity + self.alpha * spa_out + self.beta * spec_out)


class ResNet16MNISTLearnableLCRT(nn.Module):
    def __init__(self, num_classes: int = 10, use_plugin: bool = True, cfg: fixed.Config = None):
        super().__init__()
        if cfg is None:
            cfg = fixed.Config()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(fixed.BasicBlock(16, 16), fixed.BasicBlock(16, 16))
        self.stage2 = nn.Sequential(fixed.BasicBlock(16, 32, stride=2), fixed.BasicBlock(32, 32))
        self.stage3 = nn.Sequential(fixed.BasicBlock(32, 64, stride=2), fixed.BasicBlock(64, 64))
        if use_plugin:
            self.plugin1 = LearnableLCTRieszResidualBlock(
                channels=16,
                inner_channels=max(8, int(16 * cfg.plugin_inner_ratio)),
                lct_alpha=cfg.lct_alpha,
                lct_m=cfg.lct_m,
                lct_q=cfg.lct_q,
            )
            self.plugin2 = LearnableLCTRieszResidualBlock(
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
        x = self.plugin1(self.stage1(self.stem(x)))
        x = self.plugin2(self.stage2(x))
        x = self.stage3(x)
        return self.fc(self.pool(x).flatten(1))


def main() -> None:
    cfg = fixed.Config()
    fixed.seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    train_loader, test_loader = fixed.build_dataloaders(cfg)
    model = ResNet16MNISTLearnableLCRT(num_classes=10, use_plugin=cfg.use_plugin, cfg=cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    best_acc = 0.0
    total_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_loss, train_acc = fixed.train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, epoch, cfg.print_freq
        )
        test_loss, test_acc = fixed.evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
            f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.2f}% | Time {time.time() - start:.1f}s"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {"model": model.state_dict(), "config": cfg.__dict__, "best_acc": best_acc},
                "best_minist_learnableLCRT.pth",
            )
            print(f"Saved new best checkpoint with acc={best_acc:.2f}%")

    print(f"Training finished. Best test acc: {best_acc:.2f}% | Total time: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
