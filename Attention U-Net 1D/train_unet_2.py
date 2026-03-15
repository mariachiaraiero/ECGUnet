"""
train_unet_2.py
=============
Script di addestramento V2 (ResU-Net 1D + SE Attention + Focal Loss).
Modifiche rispetto alla V1:
- Sostituisce la standard CrossEntropy con la Focal Loss (all'interno della combinazione con la Dice).
  La Focal Loss forza la rete a concentrarsi **unicamente** sugli errori difficili da imparare
  (i falsi positivi sulle Onde P e T), spingendo ulteriormente il PPV.
- Usa ecg_unet_2.py con il nuovo modulo SE.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
from ecg_unet_2 import ECGUNet2  # Importa il modello V2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ludb_tensors_2/ludb_dataset.pt", help="Dataset file")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="checkpoints_2")
    return parser.parse_args()


# ============================================================
#  Combined Loss: FocalLoss + DiceLoss
# ============================================================
class FocalDiceLoss(nn.Module):
    """
    Combinazione di Focal Loss e Multi-class Dice Loss.
    La Focal Loss sostituisce la CrossEntropy, punendo maggiormente 
    le predizioni errate su sui il modello è confuso (es. P wave boundaries).
    """
    def __init__(self, weight=None, gamma=2.0, num_classes=4):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # 1. Focal Loss (Calcolata pixel per pixel e poi media)
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # probabilità della classe vera predetta dal modello
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        # 2. Multi-class Dice Loss
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 2, 1).float()
        
        dice_loss = 0.0
        for c in range(1, self.num_classes):
            p = probs[:, c, :]
            t = targets_one_hot[:, c, :]
            intersection = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            dice = (2. * intersection + 1e-8) / (union + 1e-8)
            dice_loss += (1 - dice.mean())
            
        dice_loss = dice_loss / (self.num_classes - 1)

        return focal_loss + dice_loss

# ============================================================
#  Data Augmentation
# ============================================================
def apply_data_augmentation(X_batch):
    B, C, L = X_batch.shape
    device = X_batch.device
    
    scale = 1.0 + (torch.rand((B, 1, 1), device=device) * 0.3 - 0.15)
    X_batch = X_batch * scale
    
    noise_std = 0.05
    noise = torch.randn_like(X_batch) * noise_std
    X_batch = X_batch + noise
    
    shift = (torch.rand((B, 1, 1), device=device) * 0.2 - 0.1)
    X_batch = X_batch + shift
    
    return X_batch

def extract_segments(mask_1d, class_id):
    is_class = (mask_1d == class_id).astype(int)
    diff = np.diff(np.pad(is_class, (1, 1), constant_values=0))
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0] - 1
    return list(zip(onsets, offsets))

# ============================================================
# Logica Custom (Post-Processing Base) in Computazione Metriche
# ============================================================
def compute_metrics(preds_tensor, targets_tensor, num_classes=4, fs=500, tolerance_ms=150):
    preds, targets = preds_tensor.numpy(), targets_tensor.numpy()
    tol_samples = int((tolerance_ms / 1000.0) * fs)
    results = {c: {'onset': {'TP': 0, 'FP': 0, 'FN': 0, 'errors': []}, 'offset': {'TP': 0, 'FP': 0, 'FN': 0, 'errors': []}} for c in range(1, num_classes)}

    for i in range(len(preds)):
        p_mask, t_mask = preds[i], targets[i]
        
        # --- POTENZIALE POST-PROCESSING: Eliminare onde insensatamente corte (< 20ms) ---
        # Qui potremmo riscrivere p_mask prima di analizzarlo... per ora calcoliamolo raw.
        
        for c in range(1, num_classes):
            true_segs, pred_segs = extract_segments(t_mask, c), extract_segments(p_mask, c)
            for is_offset, idx in [(False, 0), (True, 1)]:
                ptype = 'offset' if is_offset else 'onset'
                true_pts, pred_pts = [s[idx] for s in true_segs], [s[idx] for s in pred_segs]
                matched_preds = set()
                for t_pt in true_pts:
                    valid_preds = [p for p in pred_pts if abs(p - t_pt) <= tol_samples and p not in matched_preds]
                    if valid_preds:
                        best_p = min(valid_preds, key=lambda p: abs(p - t_pt))
                        matched_preds.add(best_p)
                        results[c][ptype]['TP'] += 1
                        results[c][ptype]['errors'].append((best_p - t_pt) / fs * 1000.0)
                    else:
                        results[c][ptype]['FN'] += 1
                results[c][ptype]['FP'] += len(pred_pts) - len(matched_preds)

    m_sum = {'f1_list': []}
    f1_list = []
    for c in range(1, num_classes):
        for ptype in ['onset', 'offset']:
            d = results[c][ptype]
            se = d['TP'] / (d['TP'] + d['FN'] + 1e-8)
            ppv = d['TP'] / (d['TP'] + d['FP'] + 1e-8)
            f1 = 2 * se * ppv / (se + ppv + 1e-8)
            m_sum[f"{c}_{ptype}_Se"], m_sum[f"{c}_{ptype}_PPV"], m_sum[f"{c}_{ptype}_F1"] = se*100, ppv*100, f1*100
            m_sum[f"{c}_{ptype}_m"] = np.mean(d['errors']) if d['errors'] else 0.0
            m_sum[f"{c}_{ptype}_std"] = np.std(d['errors']) if d['errors'] else 0.0
            f1_list.append(f1)
            
    m_sum['f1_macro'], m_sum['accuracy'] = np.mean(f1_list)*100, (preds==targets).mean()
    return m_sum


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []
    for X, Y in loader:
        if X.size(2) >= 5000:
            start_idx = np.random.randint(1000, 2001)
            X, Y = X[:, :, start_idx:start_idx+2000], Y[:, start_idx:start_idx+2000]
        
        X, Y = X.to(device), Y.to(device)
        X = apply_data_augmentation(X)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(Y.cpu())
        
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics['loss'] = avg_loss
    return metrics

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    for X, Y in loader:
        if X.size(2) >= 5000:
            X, Y = X[:, :, 1000:4000], Y[:, 1000:4000]
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = criterion(logits, Y)
        total_loss += loss.item() * X.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(Y.cpu())
        
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics

def print_metrics(phase, m, c_names, ep=None, t_ep=None):
    h = f"  [{phase}]" if ep is None else f"  Epoca {ep}/{t_ep} [{phase}]"
    print(f"{h} Loss: {m['loss']:.4f} | Acc(px): {m['accuracy']:.4f} | F1(AAMI) macro: {m['f1_macro']:.2f}%")
    for c in range(1, len(c_names)):
        for p in ['onset', 'offset']:
            print(f"       {c_names[c]:>12} {p.capitalize():6}: Se={m[f'{c}_{p}_Se']:5.1f}% PPV={m[f'{c}_{p}_PPV']:5.1f}% F1={m[f'{c}_{p}_F1']:5.1f}% m±std: {m[f'{c}_{p}_m']:5.1f}±{m[f'{c}_{p}_std']:.1f}ms")

def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    if not os.path.exists(args.dataset):
        print(f"[ERRORE] Dataset {args.dataset} mancante. Riavvia prepare_dataset_2.py prima.")
        return

    data = torch.load(args.dataset, weights_only=False)
    X, Y = data['X'], data['Y']
    train_dl = DataLoader(Subset(TensorDataset(X, Y), data['train_indices']), batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(Subset(TensorDataset(X, Y), data['val_indices']), batch_size=args.batch_size, shuffle=False)
    test_dl  = DataLoader(Subset(TensorDataset(X, Y), data['test_indices']), batch_size=args.batch_size, shuffle=False)
    
    print(f"Training V2 su {device} - Dataset {X.shape}")
    
    model = ECGUNet2(in_channels=1, num_classes=4, base_filters=12).to(device)
    
    # Nuova FocalDice loss!
    criterion = FocalDiceLoss(weight=data['class_weights'].to(device), gamma=2.0, num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_ecg_unet_2.pt")
    
    best_f1 = 0.0 
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        m_t = train_one_epoch(model, train_dl, criterion, optimizer, device)
        m_v = evaluate(model, val_dl, criterion, device)
        scheduler.step() 

        # Salva nella history
        history['train_loss'].append(m_t['loss'])
        history['val_loss'].append(m_v['loss'])
        history['train_f1'].append(m_t['f1_macro'])
        history['val_f1'].append(m_v['f1_macro'])

        print(f"\n--- Epoca {epoch}/{args.epochs} ({(time.time()-t0):.1f}s) --- LR: {optimizer.param_groups[0]['lr']:.2e}")
        print_metrics("TRAIN", m_t, data['class_names'])
        print_metrics("VAL  ", m_v, data['class_names'])

        if m_v['f1_macro'] > best_f1:
            best_f1 = m_v['f1_macro']
            no_improve = 0
            torch.save({'model_state_dict': model.state_dict()}, best_path)
            print(f"  [*] Nuovo miglior modello salvato! (val_f1={best_f1:.2f}%)")
        else:
            no_improve += 1
            print(f"  [!] Nessun miglioramento per {no_improve}/{args.patience} epoche")

        if no_improve >= args.patience:
            print(f"\n[STOP] Early stopping attivato dopo {epoch} epoche.")
            break

    # ===== Valutazione su Test Set =====
    print(f"\n[TEST] Caricamento del miglior modello e valutazione su test set...")
    checkpoint = torch.load(best_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_dl, criterion, device)

    print(f"\n{'='*60}")
    print(f"  RISULTATI SUL TEST SET")
    print(f"{'='*60}")
    print_metrics("TEST", test_metrics, data['class_names'])

    # Salva la history per plot
    history_path = os.path.join(args.output_dir, "training_history_2.pt")
    torch.save(history, history_path)
    print(f"  History salvata in: {os.path.abspath(history_path)}")

if __name__ == "__main__":
    main()
