"""
prepare_dataset.py
==================
Prepara il dataset LUDB in formato tensore PyTorch per l'addestramento
di un estrattore di features ECG per ogni derivazione.

Output:
  - ludb_dataset.pt  -> dizionario con tensori X, Y, metadata e split indices

Uso:
  python prepare_dataset.py [-d data] [-o ludb_tensors]
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


# ============================================================
# Configurazione
# ============================================================
LEADS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
LEAD_NAMES_UPPER = [l.upper() for l in LEADS]

# Classi di segmentazione
CLASS_BG   = 0  # Background (nessuna onda)
CLASS_P    = 1  # Onda P
CLASS_QRS  = 2  # Complesso QRS
CLASS_T    = 3  # Onda T
NUM_CLASSES = 4
CLASS_NAMES = {0: 'Background', 1: 'Onda P', 2: 'QRS', 3: 'Onda T'}

# Split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


def setup_args():
    parser = argparse.ArgumentParser(
        description="Prepara dataset tensore LUDB per segmentazione ECG"
    )
    parser.add_argument("-d", "--dir", default="data",
                        help="Directory con i dati WFDB (default: data)")
    parser.add_argument("-o", "--output", default="ludb_tensors",
                        help="Directory di output per i tensori (default: ludb_tensors)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed per riproducibilità dello split (default: 42)")
    return parser.parse_args()


def build_segmentation_mask(ann, signal_len):
    """
    Costruisce una mappa di segmentazione [signal_len] dove ogni campione
    e' etichettato con la classe corrispondente (0=BG, 1=P, 2=QRS, 3=T).
    
    Il formato annotazioni LUDB usa il campo 'symbol':
      '(' = onset di un'onda
      'p' = picco onda P
      'N' = picco QRS (o altri simboli QRS)
      't' = picco onda T
      ')' = offset di un'onda
    
    Il pattern tipico e': ( p ) ( N ) ( t ) ripetuto per ogni battito.
    """
    mask = np.zeros(signal_len, dtype=np.int64)
    
    symbols = ann.symbol
    samples = ann.sample
    n = len(symbols)
    
    # Mappa simboli di picco -> classe
    PEAK_TO_CLASS = {
        'p': CLASS_P,
        'N': CLASS_QRS,
        't': CLASS_T,
    }
    
    i = 0
    while i < n:
        sym = str(symbols[i]).strip()
        
        # Quando troviamo '(', cerchiamo il tipo di onda e la ')'
        if sym == '(':
            onset = int(samples[i])
            wave_class = None
            offset = None
            
            # Scorri avanti per trovare il picco e l'offset
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
                # Clip ai limiti del segnale
                onset_c = max(0, onset)
                offset_c = min(signal_len - 1, offset)
                if onset_c <= offset_c:
                    mask[onset_c:offset_c + 1] = wave_class
            
            # Avanza dopo la ')'
            if offset is not None:
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    
    return mask


def resample_signal(signal, mask, original_fs, target_fs=500):
    """
    Interpolazione tramite Spline Cubica come descritto nel paper (Sezione 2.1).
    Il segnale originale a 'original_fs' viene portato a 'target_fs'.
    """
    if original_fs == target_fs:
        return signal, mask
        
    n = len(signal)
    T = n / original_fs
    
    # Original time points
    t = np.array([(2 * i - 1) * T / (2 * n) for i in range(1, n + 1)])
    cs = CubicSpline(t, signal)
    
    # Target time points
    m = int(target_fs * T)
    t_new = np.array([(2 * i - 1) * T / (2 * m) for i in range(1, m + 1)])
    
    # Resample signal with cubic spline
    new_signal = cs(t_new)
    
    # Nearest neighbor array scaling for Mask
    # Costruiamo una funzione di interpolazione a gradino (costante a tratti)
    indices = np.round(np.linspace(0, n - 1, m)).astype(int)
    new_mask = mask[indices]
    
    return new_signal, new_mask


def normalize_signal(signal):
    """Normalizzazione z-score del segnale."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        return signal - mean  # segnale piatto, evita divisione per zero
    return (signal - mean) / std


def compute_class_weights(Y):
    """
    Calcola i pesi per le classi in base alla frequenza inversa.
    Utile per la Focal Loss / Weighted Cross-Entropy.
    """
    total = Y.numel()
    counts = torch.bincount(Y.flatten(), minlength=NUM_CLASSES).float()
    # Evita divisione per zero
    counts[counts == 0] = 1.0
    weights = total / (NUM_CLASSES * counts)
    return weights


def main():
    args = setup_args()
    
    print("=" * 60)
    print("  LUDB -> Tensor Dataset per Segmentazione ECG")
    print("=" * 60)
    
    # 1. Trova tutti i record
    hea_files = glob.glob(os.path.join(args.dir, "*.hea"))
    record_names = sorted(
        list(set([
            os.path.basename(f).split('.')[0]
            for f in hea_files
            if os.path.basename(f).count('.') == 1
        ])),
        key=lambda x: int(x)
    )
    
    n_records = len(record_names)
    print(f"\n[INFO] Trovati {n_records} record in '{args.dir}'")
    
    if n_records == 0:
        print("[ERRORE] Nessun record trovato! Controlla la directory dei dati.")
        sys.exit(1)
    
    # 2. Preallocazione liste
    all_signals = []      # List[np.ndarray] di shape (5000,)
    all_masks = []         # List[np.ndarray] di shape (5000,)
    all_record_ids = []    # List[int]
    all_lead_names = []    # List[str]
    
    skipped_leads = 0
    processed_leads = 0
    
    # 3. Elaborazione di ogni record
    for idx, name in enumerate(record_names):
        record_path = os.path.join(args.dir, name)
        
        try:
            record = wfdb.rdrecord(record_path)
            signal_len = record.p_signal.shape[0]
            
            for lead_idx, lead in enumerate(LEADS):
                try:
                    # Leggi annotazioni per questa derivazione
                    ann = wfdb.rdann(record_path, lead)
                    
                    # Estrai il segnale per questa derivazione
                    # Trova la colonna corrispondente nel record
                    sig_col_idx = None
                    for ci, sn in enumerate(record.sig_name):
                        if sn.upper() == lead.upper():
                            sig_col_idx = ci
                            break
                    
                    if sig_col_idx is None:
                        skipped_leads += 1
                        continue
                    
                    signal = record.p_signal[:, sig_col_idx].astype(np.float32)
                    
                    # Frequenza di campionamento originale
                    fs = record.fs
                    
                    # Costruisci la mappa di segmentazione prima di un eventuale upsampling
                    mask = build_segmentation_mask(ann, signal_len)
                    
                    # Esegue il preprocessing con spline cubica se fs != 500 (o anche se 500 per completezza)
                    signal, mask = resample_signal(signal, mask, fs, target_fs=500)
                    
                    # Normalizza il segnale
                    signal = normalize_signal(signal).astype(np.float32)
                    
                    all_signals.append(signal)
                    all_masks.append(mask)
                    all_record_ids.append(int(name))
                    all_lead_names.append(lead.upper())
                    processed_leads += 1
                    
                except Exception:
                    skipped_leads += 1
                    continue
            
            if (idx + 1) % 20 == 0 or idx == n_records - 1:
                print(f"  [OK] Elaborati {idx + 1}/{n_records} record "
                      f"({processed_leads} derivazioni ok, {skipped_leads} saltate)")
                
        except Exception as e:
            print(f"  [WARN] Errore record {name}: {e}")
            continue
    
    # 4. Conversione in tensori
    print(f"\n[...] Conversione in tensori PyTorch...")
    
    X = torch.tensor(np.stack(all_signals), dtype=torch.float32)  # (N, 5000)
    X = X.unsqueeze(1)  # (N, 1, 5000) -> 1 canale per la Conv1D
    
    Y = torch.tensor(np.stack(all_masks), dtype=torch.long)       # (N, 5000)
    
    N = X.shape[0]
    print(f"  X shape: {X.shape}  (N campioni, 1 canale, 5000 timesteps)")
    print(f"  Y shape: {Y.shape}  (N campioni, 5000 labels)")
    
    # 5. Split train/val/test per record (non per derivazione!)
    # Questo evita data leakage: tutte le derivazioni di un paziente
    # sono nello stesso split
    print(f"\n[SPLIT] Train/Val/Test per record (seed={args.seed})...")
    
    unique_records = sorted(list(set(all_record_ids)))
    n_unique = len(unique_records)
    
    rng = np.random.RandomState(args.seed)
    shuffled = rng.permutation(unique_records)
    
    n_train = int(n_unique * TRAIN_RATIO)
    n_val   = int(n_unique * VAL_RATIO)
    # Il resto va al test
    
    train_records = set(shuffled[:n_train].tolist())
    val_records   = set(shuffled[n_train:n_train + n_val].tolist())
    test_records  = set(shuffled[n_train + n_val:].tolist())
    
    train_indices = [i for i, rid in enumerate(all_record_ids) if rid in train_records]
    val_indices   = [i for i, rid in enumerate(all_record_ids) if rid in val_records]
    test_indices  = [i for i, rid in enumerate(all_record_ids) if rid in test_records]
    
    print(f"  Train: {len(train_records)} record -> {len(train_indices)} campioni")
    print(f"  Val:   {len(val_records)} record -> {len(val_indices)} campioni")
    print(f"  Test:  {len(test_records)} record -> {len(test_indices)} campioni")
    
    # 6. Statistiche distribuzione classi
    print(f"\n[STATS] Distribuzione classi (su tutto il dataset):")
    
    Y_flat = Y.flatten()
    total_samples = Y_flat.numel()
    
    for cls_id in range(NUM_CLASSES):
        count = (Y_flat == cls_id).sum().item()
        pct = count / total_samples * 100
        print(f"  {CLASS_NAMES[cls_id]:>12}: {count:>10,} campioni ({pct:5.1f}%)")
    
    # Calcola pesi per le classi
    class_weights = compute_class_weights(Y)
    print(f"\n[PESI] Pesi suggeriti per le classi (inverso della frequenza):")
    for cls_id in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[cls_id]:>12}: {class_weights[cls_id]:.4f}")
    
    # 7. Salvataggio
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "ludb_dataset.pt")
    
    dataset = {
        'X': X,                               # (N, 1, 5000) float32
        'Y': Y,                               # (N, 5000) long
        'record_ids': all_record_ids,          # List[int]
        'lead_names': all_lead_names,          # List[str]
        'train_indices': train_indices,        # List[int]
        'val_indices': val_indices,            # List[int]
        'test_indices': test_indices,          # List[int]
        'class_weights': class_weights,        # (4,) float32
        'class_names': CLASS_NAMES,            # dict
        'fs': 500,                             # Frequenza di campionamento
        'num_classes': NUM_CLASSES,
        'split_seed': args.seed,
    }
    
    torch.save(dataset, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n[SAVE] Dataset salvato in: {os.path.abspath(output_path)}")
    print(f"       Dimensione file: {file_size_mb:.1f} MB")
    
    # 8. Report finale
    print("\n" + "=" * 60)
    print("  RIEPILOGO DATASET")
    print("=" * 60)
    print(f"  Record totali:       {n_unique}")
    print(f"  Derivazioni totali:  {N}")
    print(f"  Campioni per ECG:    {X.shape[2]}")
    print(f"  Frequenza (Hz):      500")
    print(f"  Classi:              {NUM_CLASSES} ({', '.join(CLASS_NAMES.values())})")
    print(f"  Tensore X:           {X.shape} float32")
    print(f"  Tensore Y:           {Y.shape} long")
    print(f"  Train/Val/Test:      {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
    print(f"  File:                {output_path}")
    print("=" * 60)
    
    # 9. Esempio: come caricare il dataset
    print("\n[USAGE] Per caricare il dataset in un training loop:\n")
    print("  import torch")
    print("  from torch.utils.data import TensorDataset, DataLoader, Subset\n")
    print("  data = torch.load('ludb_tensors/ludb_dataset.pt')")
    print("  ds = TensorDataset(data['X'], data['Y'])")
    print("  train_ds = Subset(ds, data['train_indices'])")
    print("  val_ds   = Subset(ds, data['val_indices'])")
    print("  test_ds  = Subset(ds, data['test_indices'])")
    print("  train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)")
    print()


if __name__ == "__main__":
    main()
