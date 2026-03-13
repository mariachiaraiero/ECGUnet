"""
ecg_unet.py
===========
Rete neurale U-Net 1D completamente convoluzionale per la segmentazione
delle onde ECG (P, QRS, T, Background).

Architettura:
  - Encoder: 4 blocchi (2×Conv1d + BatchNorm + ReLU) + MaxPool1d
  - Bottleneck: 1 blocco (2×Conv1d + BatchNorm + ReLU)
  - Decoder: 4 blocchi con skip connections (ConvTranspose1d + zero-pad + concat + 2×Conv1d)
  - Classificatore: Conv1d(kernel=1)

Uso:
  from ecg_unet import ECGUNet
  model = ECGUNet(in_channels=1, num_classes=4)
  out = model(x)  # x: (B, 1, L)  ->  out: (B, 4, L)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Blocco con 2 layer convoluzionali + BatchNorm + ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=9, padding=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ECGUNet(nn.Module):
    """
    U-Net 1D per segmentazione ECG.

    Parametri:
      in_channels  : canali di input (default: 1, singola derivazione)
      num_classes  : numero di classi di output (default: 4)
      base_filters : numero di filtri nel primo blocco encoder (default: 64)
    """

    def __init__(self, in_channels=1, num_classes=4, base_filters=4):
        super().__init__()

        f = base_filters  # 4

        # ===== Encoder =====
        self.enc1 = ConvBlock(in_channels, f)           # 1  -> 4
        self.enc2 = ConvBlock(f,     f * 2)             # 4  -> 8
        self.enc3 = ConvBlock(f * 2, f * 4)             # 8  -> 16
        self.enc4 = ConvBlock(f * 4, f * 8)             # 16 -> 32

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # ===== Bottleneck =====
        self.bottleneck = ConvBlock(f * 8, f * 16)      # 32 -> 64

        # ===== Decoder (deconvoluzioni) =====
        # UpConv non dimezza i canali come nella classica UNet (Paper Fig 2: 96, 48, 24, 12 out di concat)
        self.upconv4 = nn.ConvTranspose1d(f * 16, f * 16, kernel_size=8, stride=2, padding=3)
        self.dec4    = ConvBlock(f * 24, f * 8)         # 96 (64+32) -> 32

        self.upconv3 = nn.ConvTranspose1d(f * 8, f * 8, kernel_size=8, stride=2, padding=3)
        self.dec3    = ConvBlock(f * 12, f * 4)         # 48 (32+16) -> 16

        self.upconv2 = nn.ConvTranspose1d(f * 4, f * 4, kernel_size=8, stride=2, padding=3)
        self.dec2    = ConvBlock(f * 6, f * 2)          # 24 (16+8) -> 8

        self.upconv1 = nn.ConvTranspose1d(f * 2, f * 2, kernel_size=8, stride=2, padding=3)
        self.dec1    = ConvBlock(f * 3, f)              # 12 (8+4) -> 4

        # ===== Classificatore finale =====
        self.final_conv = nn.Conv1d(f, num_classes, kernel_size=1)

    @staticmethod
    def _pad_to_match(x, target_len):
        """
        Zero-padding (copy + zero pad) per allineare la dimensione temporale
        di x a target_len. Questo sostituisce il 'copy + crop' della U-Net
        originale, garantendo che output_size == input_size.
        """
        diff = target_len - x.size(2)
        if diff > 0:
            # Padding simmetrico: metà a sinistra, metà a destra
            pad_left = diff // 2
            pad_right = diff - pad_left
            x = F.pad(x, (pad_left, pad_right), mode='constant', value=0)
        elif diff < 0:
            # Se l'upsampled è più lungo, taglia (raro con kernel=8, stride=2, pad=3)
            x = x[:, :, :target_len]
        return x

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor di shape (batch, 1, L) — segnale ECG

        Returns:
            Tensor di shape (batch, 4, L) — score per classe per campione
        """
        # ===== Encoder =====
        e1 = self.enc1(x)               # (B, 4,   L)
        e2 = self.enc2(self.pool(e1))   # (B, 8,   L/2)
        e3 = self.enc3(self.pool(e2))   # (B, 16,  L/4)
        e4 = self.enc4(self.pool(e3))   # (B, 32,  L/8)

        # ===== Bottleneck =====
        b = self.bottleneck(self.pool(e4))  # (B, 64, L/16)

        # ===== Decoder con skip connections =====
        # Livello 4
        d4 = self.upconv4(b)                        # (B, 64, ~L/8)
        d4 = self._pad_to_match(d4, e4.size(2))     # zero-pad per allineare a e4
        d4 = torch.cat([d4, e4], dim=1)             # (B, 96, L/8)
        d4 = self.dec4(d4)                          # (B, 32,  L/8)

        # Livello 3
        d3 = self.upconv3(d4)                       # (B, 32, ~L/4)
        d3 = self._pad_to_match(d3, e3.size(2))
        d3 = torch.cat([d3, e3], dim=1)             # (B, 48, L/4)
        d3 = self.dec3(d3)                          # (B, 16, L/4)

        # Livello 2
        d2 = self.upconv2(d3)                       # (B, 16, ~L/2)
        d2 = self._pad_to_match(d2, e2.size(2))
        d2 = torch.cat([d2, e2], dim=1)             # (B, 24, L/2)
        d2 = self.dec2(d2)                          # (B, 8, L/2)

        # Livello 1
        d1 = self.upconv1(d2)                       # (B, 8, ~L)
        d1 = self._pad_to_match(d1, e1.size(2))
        d1 = torch.cat([d1, e1], dim=1)             # (B, 12, L)
        d1 = self.dec1(d1)                          # (B, 4,  L)

        # ===== Output =====
        out = self.final_conv(d1)                   # (B, 4, L)
        return out


# ============================================================
#  Test rapido dell'architettura
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGUNet(in_channels=1, num_classes=4).to(device)

    # Conta i parametri
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri totali:      {n_params:,}")
    print(f"Parametri allenabili:  {n_trainable:,}")

    # Test con un batch simulato
    x = torch.randn(2, 1, 5000).to(device)
    y = model(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (2, 4, 5000), f"Shape errata! Atteso (2, 4, 5000), ottenuto {y.shape}"
    print("\n[OK] Shape corretto: output_size == input_size")
