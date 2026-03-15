"""
ecg_unet_2.py
===========
Rete neurale ResU-Net 1D con Meccanismo di Attenzione (Squeeze-and-Excitation).
Versione Sperimentale (V2):
- Mantiene le connessioni residue (ResBlock) della V1.
- Aggiunge SEBlock (Squeeze-and-Excitation) per far concentrare la rete sulle feature rilevanti.
- Aumenta i filtri di base (base_filters=12) per maggiore capacità estrattiva.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block per segnali 1D.
    Impara a ricalibrare i pesi dei canali (Attention Mechanism).
    """
    def __init__(self, channel, reduction=4):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResConvBlockSE(nn.Module):
    """
    Blocco Residuale con 2 layer convoluzionali + BatchNorm + ReLU + SE Block (Attenzione).
    """
    def __init__(self, in_ch, out_ch, kernel_size=9, padding=4, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            SEBlock1D(out_ch, reduction=4)  # <-- Attention Mechanism
        )
        
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


class ECGUNet2(nn.Module):
    """
    ResU-Net 1D + SE Attention per segmentazione ECG estrema.

    Parametri:
      in_channels  : canali di input (default: 1)
      num_classes  : numero di classi di output (default: 4)
      base_filters : filtri base aumentati a 12 (default: 12)
    """

    def __init__(self, in_channels=1, num_classes=4, base_filters=12, dropout=0.1):
        super().__init__()

        f = base_filters

        # ===== Encoder =====
        self.enc1 = ResConvBlockSE(in_channels, f, dropout=dropout)           # 1  -> 12
        self.enc2 = ResConvBlockSE(f,     f * 2, dropout=dropout)             # 12 -> 24
        self.enc3 = ResConvBlockSE(f * 2, f * 4, dropout=dropout)             # 24 -> 48
        self.enc4 = ResConvBlockSE(f * 4, f * 8, dropout=dropout)             # 48 -> 96

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # ===== Bottleneck =====
        self.bottleneck = ResConvBlockSE(f * 8, f * 16, dropout=dropout)      # 96 -> 192

        # ===== Decoder (deconvoluzioni) =====
        self.upconv4 = nn.ConvTranspose1d(f * 16, f * 16, kernel_size=8, stride=2, padding=3)
        self.dec4    = ResConvBlockSE(f * 24, f * 8, dropout=dropout)         

        self.upconv3 = nn.ConvTranspose1d(f * 8, f * 8, kernel_size=8, stride=2, padding=3)
        self.dec3    = ResConvBlockSE(f * 12, f * 4, dropout=dropout)         

        self.upconv2 = nn.ConvTranspose1d(f * 4, f * 4, kernel_size=8, stride=2, padding=3)
        self.dec2    = ResConvBlockSE(f * 6, f * 2, dropout=dropout)          

        self.upconv1 = nn.ConvTranspose1d(f * 2, f * 2, kernel_size=8, stride=2, padding=3)
        self.dec1    = ResConvBlockSE(f * 3, f, dropout=dropout)              

        # ===== Classificatore finale =====
        self.final_conv = nn.Conv1d(f, num_classes, kernel_size=1)

    @staticmethod
    def _pad_to_match(x, target_len):
        diff = target_len - x.size(2)
        if diff > 0:
            pad_left = diff // 2
            pad_right = diff - pad_left
            x = F.pad(x, (pad_left, pad_right), mode='constant', value=0)
        elif diff < 0:
            x = x[:, :, :target_len]
        return x

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)               
        e2 = self.enc2(self.pool(e1))   
        e3 = self.enc3(self.pool(e2))   
        e4 = self.enc4(self.pool(e3))   

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  

        # Decoder 
        d4 = self.upconv4(b)                        
        d4 = self._pad_to_match(d4, e4.size(2))     
        d4 = torch.cat([d4, e4], dim=1)             
        d4 = self.dec4(d4)                          

        d3 = self.upconv3(d4)                       
        d3 = self._pad_to_match(d3, e3.size(2))
        d3 = torch.cat([d3, e3], dim=1)             
        d3 = self.dec3(d3)                          

        d2 = self.upconv2(d3)                       
        d2 = self._pad_to_match(d2, e2.size(2))
        d2 = torch.cat([d2, e2], dim=1)             
        d2 = self.dec2(d2)                          

        d1 = self.upconv1(d2)                       
        d1 = self._pad_to_match(d1, e1.size(2))
        d1 = torch.cat([d1, e1], dim=1)             
        d1 = self.dec1(d1)                          

        # Output
        out = self.final_conv(d1)                   
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGUNet2(in_channels=1, num_classes=4, base_filters=12).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {n_params:,}")
    x = torch.randn(2, 1, 5000).to(device)
    y = model(x)
    assert y.shape == (2, 4, 5000), "Shape errata!"
    print("[OK] Shape corretto con ResUNet-SE.")
