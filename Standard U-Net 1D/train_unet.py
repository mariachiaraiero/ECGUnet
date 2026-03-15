""" ciao 
train_unet.py
=============
Script di addestramento per la rete U-Net 1D su dataset LUDB.

Carica ludb_dataset.pt, addestra il modello ECGUNet e salva il miglior modello.

Uso:
  python train_unet.py                          # Training con parametri di default
  python train_unet.py --epochs 100 --lr 1e-3   # Personalizzato
  python train_unet.py --device cuda             # Forza GPU
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from ecg_unet import ECGUNet


# ============================================================
#  Argomenti
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Addestramento U-Net 1D per segmentazione ECG"
    )
    parser.add_argument("--dataset", type=str, default="ludb_tensors/ludb_dataset.pt",
                        help="Percorso al file dataset .pt")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Numero di epoche (default: 80)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Dimensione del batch (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate iniziale (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay per Adam (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Patience per early stopping (default: 15)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda' o 'auto' (default: auto)")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory per il salvataggio modelli (default: checkpoints)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Num workers per DataLoader (default: 0)")
    return parser.parse_args()


# ============================================================
#  Metriche AAMI (150ms Tolerance)
# ============================================================
def extract_segments(mask_1d, class_id):
    """Estrae (onset, offset) di tutti i segmenti contigui di una certa classe"""
    is_class = (mask_1d == class_id).astype(int)
    diff = np.diff(np.pad(is_class, (1, 1), constant_values=0))
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0] - 1
    return list(zip(onsets, offsets))


def compute_metrics(preds_tensor, targets_tensor, num_classes=4, fs=500, tolerance_ms=150):
    """
    Calcola le metriche di accuratezza AAMI-style come descritte in f9.pdf.
    La tolleranza è 150ms.
    """
    preds = preds_tensor.numpy()
    targets = targets_tensor.numpy()
    
    tol_samples = int((tolerance_ms / 1000.0) * fs)
    
    results = {
        c: {'onset': {'TP': 0, 'FP': 0, 'FN': 0, 'errors': []}, 
            'offset': {'TP': 0, 'FP': 0, 'FN': 0, 'errors': []}}
        for c in range(1, num_classes)
    }

    for i in range(len(preds)):
        pred_mask = preds[i]
        true_mask = targets[i]
        
        for c in range(1, num_classes):
            true_segs = extract_segments(true_mask, c)
            pred_segs = extract_segments(pred_mask, c)
            
            for is_offset, idx in [(False, 0), (True, 1)]:
                ptype = 'offset' if is_offset else 'onset'
                
                true_pts = [s[idx] for s in true_segs]
                pred_pts = [s[idx] for s in pred_segs]
                
                matched_preds = set()
                for t_pt in true_pts:
                    valid_preds = [p for p in pred_pts if abs(p - t_pt) <= tol_samples and p not in matched_preds]
                    if valid_preds:
                        best_p = min(valid_preds, key=lambda p: abs(p - t_pt))
                        matched_preds.add(best_p)
                        results[c][ptype]['TP'] += 1
                        error_ms = (best_p - t_pt) / fs * 1000.0
                        results[c][ptype]['errors'].append(error_ms)
                    else:
                        results[c][ptype]['FN'] += 1
                        
                fp_count = len(pred_pts) - len(matched_preds)
                results[c][ptype]['FP'] += fp_count

    metrics_summary = {'f1_list': []}
    f1_list = []
    
    for c in range(1, num_classes):
        for ptype in ['onset', 'offset']:
            tp = results[c][ptype]['TP']
            fp = results[c][ptype]['FP']
            fn = results[c][ptype]['FN']
            errs = results[c][ptype]['errors']
            
            se = tp / (tp + fn + 1e-8)
            ppv = tp / (tp + fp + 1e-8)
            f1 = 2 * se * ppv / (se + ppv + 1e-8)
            
            metrics_summary[f"{c}_{ptype}_Se"] = se * 100.0
            metrics_summary[f"{c}_{ptype}_PPV"] = ppv * 100.0
            metrics_summary[f"{c}_{ptype}_F1"] = f1 * 100.0
            metrics_summary[f"{c}_{ptype}_m"] = np.mean(errs) if errs else 0.0
            metrics_summary[f"{c}_{ptype}_std"] = np.std(errs) if errs else 0.0
            
            f1_list.append(f1)
            
    metrics_summary['f1_macro'] = np.mean(f1_list) * 100.0
    metrics_summary['accuracy'] = (preds == targets).mean()
    metrics_summary['iou_mean'] = 0.0  # Dummy
    metrics_summary['f1_per_class'] = [0] + [metrics_summary[f"{c}_onset_F1"] for c in range(1, num_classes)] 
    metrics_summary['iou_per_class'] = [0, 0, 0, 0] # Dummy
    
    return metrics_summary


# ============================================================
#  Training e Validazione (una epoca)
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X_batch, Y_batch in loader:
        # DATA AUGMENTATION: Estrazione crop 4 secondi da [2s, 8s] (campioni 1000-4000)
        if X_batch.size(2) >= 5000:
            start_idx = np.random.randint(1000, 2001)
            X_batch = X_batch[:, :, start_idx:start_idx+2000]
            Y_batch = Y_batch[:, start_idx:start_idx+2000]
            
        X_batch = X_batch.to(device)    # (B, 1, 2000)
        Y_batch = Y_batch.to(device)    # (B, 2000)

        optimizer.zero_grad()
        logits = model(X_batch)         # (B, 4, 2000)
        loss = criterion(logits, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)    # (B, 2000)
        all_preds.append(preds.cpu())
        all_targets.append(Y_batch.cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X_batch, Y_batch in loader:
        # VALIDAZIONE: Limitiamo ai secondi [2s, 8s] per evitare cicli incompleti ai bordi
        if X_batch.size(2) >= 5000:
            X_batch = X_batch[:, :, 1000:4000]
            Y_batch = Y_batch[:, 1000:4000]
            
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, Y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(Y_batch.cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss
    return metrics


# ============================================================
#  Funzione di stampa metriche
# ============================================================
def print_metrics(phase, metrics, class_names, epoch=None, total_epochs=None):
    header = f"  [{phase}]"
    if epoch is not None:
        header = f"  Epoca {epoch}/{total_epochs} [{phase}]"

    print(f"{header}  Loss: {metrics['loss']:.4f}  |  "
          f"Acc(px): {metrics['accuracy']:.4f}  |  "
          f"F1(AAMI) macro: {metrics['f1_macro']:.2f}%")

    # Dettaglio per classe
    for c in range(1, len(class_names)):
        name = class_names[c]
        print(f"       {name:>12} Onset : Se={metrics[f'{c}_onset_Se']:5.1f}%  "
              f"PPV={metrics[f'{c}_onset_PPV']:5.1f}%  "
              f"F1={metrics[f'{c}_onset_F1']:5.1f}%  m±std: {metrics[f'{c}_onset_m']:5.1f}±{metrics[f'{c}_onset_std']:.1f}ms")
              
        print(f"       {name:>12} Offset: Se={metrics[f'{c}_offset_Se']:5.1f}%  "
              f"PPV={metrics[f'{c}_offset_PPV']:5.1f}%  "
              f"F1={metrics[f'{c}_offset_F1']:5.1f}%  m±std: {metrics[f'{c}_offset_m']:5.1f}±{metrics[f'{c}_offset_std']:.1f}ms")


# ============================================================
#  Main
# ============================================================
def main():
    args = parse_args()

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  ECG U-Net 1D — Training")
    print(f"{'='*60}")
    print(f"  Device:     {device}")

    # --- Caricamento dataset ---
    print(f"\n[LOAD] Caricamento dataset da '{args.dataset}'...")
    data = torch.load(args.dataset, weights_only=False)

    X = data['X']                       # (N, 1, 5000) float32
    Y = data['Y']                       # (N, 5000) long
    class_weights = data['class_weights'].to(device)
    class_names = data['class_names']
    num_classes = data['num_classes']

    print(f"  X: {X.shape}   Y: {Y.shape}")
    print(f"  Classi: {num_classes}  ({', '.join(class_names.values())})")
    print(f"  Pesi classi: {class_weights.tolist()}")

    # --- DataLoaders ---
    full_ds = TensorDataset(X, Y)

    train_ds = Subset(full_ds, data['train_indices'])
    val_ds   = Subset(full_ds, data['val_indices'])
    test_ds  = Subset(full_ds, data['test_indices'])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    print(f"\n  Train: {len(train_ds)} campioni ({len(train_loader)} batch)")
    print(f"  Val:   {len(val_ds)} campioni ({len(val_loader)} batch)")
    print(f"  Test:  {len(test_ds)} campioni ({len(test_loader)} batch)")

    # --- Modello ---
    model = ECGUNet(in_channels=1, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] ECGUNet — {n_params:,} parametri allenabili")

    # --- Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    print(f"  Optimizer: Adam (lr={args.lr}, wd={args.weight_decay})")
    print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=7)")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Epoche: {args.epochs}")

    # --- Directory output ---
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "best_ecg_unet.pt")

    # ===== Training Loop =====
    print(f"\n{'='*60}")
    print(f"  INIZIO ADDESTRAMENTO")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Training
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validazione
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_metrics['loss'])

        # Salva nella history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_f1'].append(val_metrics['f1_macro'])

        epoch_time = time.time() - epoch_start

        # Stampa metriche
        print(f"\n--- Epoca {epoch}/{args.epochs} ({epoch_time:.1f}s) "
              f"--- LR: {optimizer.param_groups[0]['lr']:.2e} ---")
        print_metrics("TRAIN", train_metrics, class_names)
        print_metrics("VAL  ", val_metrics, class_names)

        # Early stopping check
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_no_improve = 0
            # Salva il miglior modello
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_f1_macro': val_metrics['f1_macro'],
                'val_iou_mean': val_metrics['iou_mean'],
                'class_names': class_names,
                'num_classes': num_classes,
            }, best_model_path)
            print(f"  [*] Nuovo miglior modello salvato! (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  [!] Nessun miglioramento per {epochs_no_improve}/{args.patience} epoche")

        if epochs_no_improve >= args.patience:
            print(f"\n[STOP] Early stopping attivato dopo {epoch} epoche.")
            break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ADDESTRAMENTO COMPLETATO in {total_time/60:.1f} minuti")
    print(f"  Miglior val_loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    # ===== Valutazione su Test Set =====
    print(f"\n[TEST] Caricamento del miglior modello e valutazione su test set...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"  RISULTATI SUL TEST SET")
    print(f"{'='*60}")
    print_metrics("TEST", test_metrics, class_names)

    print(f"\n  Modello salvato in: {os.path.abspath(best_model_path)}")
    print(f"  (Epoca del best model: {checkpoint['epoch']})")

    # Salva la history per eventuali plot futuri
    history_path = os.path.join(args.output_dir, "training_history.pt")
    torch.save(history, history_path)
    print(f"  History salvata in: {os.path.abspath(history_path)}")
    print()


if __name__ == "__main__":
    main()
