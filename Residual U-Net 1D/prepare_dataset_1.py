"""
prepare_dataset_1.py
==================
Prepara il dataset LUDB in formato tensore PyTorch.
Versione Migliorata: Aggiunto un filtro Passa-Banda (Bandpass Filtering) 0.5Hz - 45Hz
per eliminare il rumore di base e le interferenze di rete elettrica.
"""

import wfdb
import numpy as np
import os
import sys
import argparse
import glob
import torch
from collections import Counter
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

# Configurazioni e Variabili Globali come nell'originale
LEADS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
CLASS_BG, CLASS_P, CLASS_QRS, CLASS_T, NUM_CLASSES = 0, 1, 2, 3, 4
CLASS_NAMES = {0: 'Background', 1: 'Onda P', 2: 'QRS', 3: 'Onda T'}
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15


def setup_args():
    parser = argparse.ArgumentParser(description="Prepara dataset tensore LUDB migliorato")
    parser.add_argument("-d", "--dir", default="data", help="Directory WFDB")
    parser.add_argument("-o", "--output", default="ludb_tensors_1", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    return parser.parse_args()


def build_segmentation_mask(ann, signal_len):
    """Costruisce la segmentazione basata sulle etichette umane."""
    mask = np.zeros(signal_len, dtype=np.int64)
    symbols, samples = ann.symbol, ann.sample
    n = len(symbols)
    PEAK_TO_CLASS = {'p': CLASS_P, 'N': CLASS_QRS, 't': CLASS_T}
    
    i = 0
    while i < n:
        sym = str(symbols[i]).strip()
        if sym == '(':
            onset = int(samples[i])
            wave_class, offset = None, None
            j = i + 1
            while j < n:
                sj = str(symbols[j]).strip()
                if sj in PEAK_TO_CLASS:
                    wave_class = PEAK_TO_CLASS[sj]
                elif sj == ')':
                    offset = int(samples[j])
                    break
                j += 1
            
            if wave_class is not None and offset is not None:
                onset_c, offset_c = max(0, onset), min(signal_len - 1, offset)
                if onset_c <= offset_c:
                    mask[onset_c:offset_c + 1] = wave_class
            i = j + 1 if offset is not None else i + 1
        else:
            i += 1
    return mask

def apply_bandpass_filter(signal, fs, lowcut=0.5, highcut=45.0, order=3):
    """
    Filtro passa-banda Butterworth per rimuovere il baseline wander (<0.5Hz)
    e ad alta frequenza/interferenza rumore (>45Hz).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # filtfilt applica andata e ritorno per evitare lo shift di fase (zero phase)
    y = filtfilt(b, a, signal)
    return y

def resample_signal(signal, mask, original_fs, target_fs=500):
    """Interpolazione Spline Cubica + Step mask"""
    if original_fs == target_fs:
        return signal, mask
        
    n = len(signal)
    T = n / original_fs
    t = np.array([(2 * i - 1) * T / (2 * n) for i in range(1, n + 1)])
    cs = CubicSpline(t, signal)
    
    m = int(target_fs * T)
    t_new = np.array([(2 * i - 1) * T / (2 * m) for i in range(1, m + 1)])
    new_signal = cs(t_new)
    
    indices = np.round(np.linspace(0, n - 1, m)).astype(int)
    new_mask = mask[indices]
    return new_signal, new_mask


def normalize_signal(signal):
    """Z-score normalizzazione dopo il filtraggio"""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        return signal - mean
    return (signal - mean) / std


def compute_class_weights(Y):
    total = Y.numel()
    counts = torch.bincount(Y.flatten(), minlength=NUM_CLASSES).float()
    counts[counts == 0] = 1.0
    weights = total / (NUM_CLASSES * counts)
    return weights


def main():
    args = setup_args()
    print("=" * 60)
    print("  LUDB -> Tensor Dataset Miorato [Bandpass Filtered]")
    print("=" * 60)
    
    hea_files = glob.glob(os.path.join(args.dir, "*.hea"))
    record_names = sorted(list(set([os.path.basename(f).split('.')[0] for f in hea_files if os.path.basename(f).count('.') == 1])), key=lambda x: int(x))
    
    if not record_names:
        print("[ERRORE] Nessun record trovato!")
        sys.exit(1)
    
    all_signals, all_masks, all_record_ids, all_lead_names = [], [], [], []
    processed, skipped = 0, 0
    
    for idx, name in enumerate(record_names):
        record_path = os.path.join(args.dir, name)
        try:
            record = wfdb.rdrecord(record_path)
            signal_len = record.p_signal.shape[0]
            
            for lead in LEADS:
                try:
                    ann = wfdb.rdann(record_path, lead)
                    sig_col_idx = next((i for i, sn in enumerate(record.sig_name) if sn.upper() == lead.upper()), None)
                    if sig_col_idx is None:
                        skipped += 1
                        continue
                        
                    signal = record.p_signal[:, sig_col_idx].astype(np.float32)
                    fs = record.fs
                    
                    # 1. Filtro Passa banda sul segnale originale
                    signal = apply_bandpass_filter(signal, fs)
                    
                    mask = build_segmentation_mask(ann, signal_len)
                    
                    # 2. Resampling
                    signal, mask = resample_signal(signal, mask, fs, target_fs=500)
                    
                    # 3. Z-score normalization
                    signal = normalize_signal(signal).astype(np.float32)
                    
                    all_signals.append(signal)
                    all_masks.append(mask)
                    all_record_ids.append(int(name))
                    all_lead_names.append(lead.upper())
                    processed += 1
                except Exception:
                    skipped += 1
        except Exception as e:
            pass

    X = torch.tensor(np.stack(all_signals), dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(np.stack(all_masks), dtype=torch.long)
    
    rng = np.random.RandomState(args.seed)
    unique_records = sorted(list(set(all_record_ids)))
    shuffled = rng.permutation(unique_records)
    
    n_unique = len(unique_records)
    n_train = int(n_unique * TRAIN_RATIO)
    n_val = int(n_unique * VAL_RATIO)
    
    train_records = set(shuffled[:n_train].tolist())
    val_records = set(shuffled[n_train:n_train + n_val].tolist())
    test_records = set(shuffled[n_train + n_val:].tolist())
    
    train_indices = [i for i, rid in enumerate(all_record_ids) if rid in train_records]
    val_indices = [i for i, rid in enumerate(all_record_ids) if rid in val_records]
    test_indices = [i for i, rid in enumerate(all_record_ids) if rid in test_records]
    
    class_weights = compute_class_weights(Y)
    
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "ludb_dataset.pt")
    
    dataset = {
        'X': X, 'Y': Y, 'record_ids': all_record_ids, 'lead_names': all_lead_names,
        'train_indices': train_indices, 'val_indices': val_indices, 'test_indices': test_indices,
        'class_weights': class_weights, 'class_names': CLASS_NAMES, 'fs': 500,
        'num_classes': NUM_CLASSES, 'split_seed': args.seed,
    }
    torch.save(dataset, out_path)
    print(f"\n[SAVE] Salvato in {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB)")

if __name__ == "__main__":
    main()
