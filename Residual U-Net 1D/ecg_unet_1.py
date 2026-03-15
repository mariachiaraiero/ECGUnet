"""
ecg_unet_1.py
===========
Rete neurale ResU-Net 1D completamente convoluzionale per la segmentazione
delle onde ECG (P, QRS, T, Background).
Versione Migliorata:
- Aggiunte connessioni residue (ResBlock) all'encoder/decoder.
- Aumentati i filtri di base (base_filters=8) per maggiore capacità (mantenendo comunque il modello piccolo).
- Aggiunto Spatial Dropout per regolarizzazione.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    """
    Blocco Residuale con 2 layer convoluzionali + BatchNorm + ReLU.
    Aggiunge una skip connection (x + block(x)) per migliorare il flusso del gradiente.
    """

    def __init__(self, in_ch, out_ch, kernel_size=9, padding=4, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch)
        )
        
        # Se in_ch != out_ch, la skip connection ha bisogno di un layer 1x1 per allineare i canali
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.block(x) + self.shortcut(x)
        return F.relu(out, inplace=True)


class ECGUNet(nn.Module):
    """
    ResU-Net 1D per segmentazione ECG.

    Parametri:
      in_channels  : canali di input (default: 1, singola derivazione)
      num_classes  : numero di classi di output (default: 4)
      base_filters : numero di filtri nel primo blocco encoder (default: 8)
      dropout      : dropout rate nel blocco convoluzionale (default: 0.1)
    """

    def __init__(self, in_channels=1, num_classes=4, base_filters=8, dropout=0.1):
        super().__init__()

        f = base_filters  # 8

        # ===== Encoder =====
        self.enc1 = ResConvBlock(in_channels, f, dropout=dropout)           # 1  -> 8
        self.enc2 = ResConvBlock(f,     f * 2, dropout=dropout)             # 8  -> 16
        self.enc3 = ResConvBlock(f * 2, f * 4, dropout=dropout)             # 16 -> 32
        self.enc4 = ResConvBlock(f * 4, f * 8, dropout=dropout)             # 32 -> 64

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # ===== Bottleneck =====
        self.bottleneck = ResConvBlock(f * 8, f * 16, dropout=dropout)      # 64 -> 128

        # ===== Decoder (deconvoluzioni) =====
        self.upconv4 = nn.ConvTranspose1d(f * 16, f * 16, kernel_size=8, stride=2, padding=3)
        self.dec4    = ResConvBlock(f * 24, f * 8, dropout=dropout)         # (128+64)=192 -> 64

        self.upconv3 = nn.ConvTranspose1d(f * 8, f * 8, kernel_size=8, stride=2, padding=3)
        self.dec3    = ResConvBlock(f * 12, f * 4, dropout=dropout)         # (64+32)=96 -> 32

        self.upconv2 = nn.ConvTranspose1d(f * 4, f * 4, kernel_size=8, stride=2, padding=3)
        self.dec2    = ResConvBlock(f * 6, f * 2, dropout=dropout)          # (32+16)=48 -> 16

        self.upconv1 = nn.ConvTranspose1d(f * 2, f * 2, kernel_size=8, stride=2, padding=3)
        self.dec1    = ResConvBlock(f * 3, f, dropout=dropout)              # (16+8)=24 -> 8

        # ===== Classificatore finale =====
        self.final_conv = nn.Conv1d(f, num_classes, kernel_size=1)

    @staticmethod
    def _pad_to_match(x, target_len):
        """
        Zero-padding (copy + zero pad) per allineare la dimensione temporale
        di x a target_len.
        """
        diff = target_len - x.size(2)
        if diff > 0:
            pad_left = diff // 2
            pad_right = diff - pad_left
            x = F.pad(x, (pad_left, pad_right), mode='constant', value=0)
        elif diff < 0:
            x = x[:, :, :target_len]
        return x

    def forward(self, x):
        """
        Forward pass.
        x: Tensor di shape (batch, 1, L)
        Returns: Tensor di shape (batch, 4, L)
        """
        # ===== Encoder =====
        e1 = self.enc1(x)               # (B, 8,   L)
        e2 = self.enc2(self.pool(e1))   # (B, 16,  L/2)
        e3 = self.enc3(self.pool(e2))   # (B, 32,  L/4)
        e4 = self.enc4(self.pool(e3))   # (B, 64,  L/8)

        # ===== Bottleneck =====
        b = self.bottleneck(self.pool(e4))  # (B, 128, L/16)

        # ===== Decoder con skip connections =====
        d4 = self.upconv4(b)                        # (B, 128, ~L/8)
        d4 = self._pad_to_match(d4, e4.size(2))     
        d4 = torch.cat([d4, e4], dim=1)             # (B, 192, L/8)
        d4 = self.dec4(d4)                          # (B, 64,  L/8)

        d3 = self.upconv3(d4)                       # (B, 64, ~L/4)
        d3 = self._pad_to_match(d3, e3.size(2))
        d3 = torch.cat([d3, e3], dim=1)             # (B, 96, L/4)
        d3 = self.dec3(d3)                          # (B, 32, L/4)

        d2 = self.upconv2(d3)                       # (B, 32, ~L/2)
        d2 = self._pad_to_match(d2, e2.size(2))
        d2 = torch.cat([d2, e2], dim=1)             # (B, 48, L/2)
        d2 = self.dec2(d2)                          # (B, 16, L/2)

        d1 = self.upconv1(d2)                       # (B, 16, ~L)
        d1 = self._pad_to_match(d1, e1.size(2))
        d1 = torch.cat([d1, e1], dim=1)             # (B, 24, L)
        d1 = self.dec1(d1)                          # (B, 8,  L)

        # ===== Output =====
        out = self.final_conv(d1)                   # (B, 4, L)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGUNet(in_channels=1, num_classes=4, base_filters=8).to(device)

    # Conta i parametri
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {n_params:,}")

    # Test
    x = torch.randn(2, 1, 5000).to(device)
    y = model(x)
    assert y.shape == (2, 4, 5000), f"Shape errata! Atteso (2, 4, 5000), ottenuto {y.shape}"
    print("\n[OK] Shape corretto: output_size == input_size")
