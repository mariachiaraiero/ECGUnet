"""
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
#  Metriche
# ============================================================
def compute_metrics(preds, targets, num_classes=4):
    """
    Calcola accuracy, F1 per classe e IoU per classe.

    Args:
        preds:   (N,) tensor di predizioni (long)
        targets: (N,) tensor di labels (long)

    Returns:
        dict con accuracy, f1_per_class, iou_per_class, f1_macro, iou_mean
    """
    accuracy = (preds == targets).float().mean().item()

    f1_scores = []
    iou_scores = []

    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)

        tp = (pred_c & true_c).sum().float()
        fp = (pred_c & ~true_c).sum().float()
        fn = (~pred_c & true_c).sum().float()

        # F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1.item())

        # IoU (Jaccard)
        iou = tp / (tp + fp + fn + 1e-8)
        iou_scores.append(iou.item())

    return {
        'accuracy': accuracy,
        'f1_per_class': f1_scores,
        'iou_per_class': iou_scores,
        'f1_macro': np.mean(f1_scores),
        'iou_mean': np.mean(iou_scores),
    }


# ============================================================
#  Training e Validazione (una epoca)
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)    # (B, 1, L)
        Y_batch = Y_batch.to(device)    # (B, L)

        optimizer.zero_grad()
        logits = model(X_batch)         # (B, 4, L)
        loss = criterion(logits, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)    # (B, L)
        all_preds.append(preds.cpu().flatten())
        all_targets.append(Y_batch.cpu().flatten())

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
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, Y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().flatten())
        all_targets.append(Y_batch.cpu().flatten())

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
          f"Acc: {metrics['accuracy']:.4f}  |  "
          f"F1 macro: {metrics['f1_macro']:.4f}  |  "
          f"mIoU: {metrics['iou_mean']:.4f}")

    # Dettaglio per classe
    for c in range(len(metrics['f1_per_class'])):
        name = class_names.get(c, f"Classe {c}")
        print(f"       {name:>12}: F1={metrics['f1_per_class'][c]:.4f}  "
              f"IoU={metrics['iou_per_class'][c]:.4f}")


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
        optimizer, mode='min', factor=0.5, patience=7, verbose=True
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
