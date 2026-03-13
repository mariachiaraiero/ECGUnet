"""
inspect_dataset.py
==================
Apre e visualizza il contenuto del dataset tensore LUDB.

Uso:
  python inspect_dataset.py
"""

import torch

# 1. Carica il file
data = torch.load('ludb_tensors/ludb_dataset.pt', weights_only=False)

# 2. Mostra tutte le chiavi
print("=" * 60)
print("  CONTENUTO DI ludb_dataset.pt")
print("=" * 60)

print(f"\n--- TENSORI PRINCIPALI ---")
print(f"  X (segnali):  {data['X'].shape}  dtype={data['X'].dtype}")
print(f"  Y (labels):   {data['Y'].shape}  dtype={data['Y'].dtype}")

print(f"\n--- PARAMETRI ---")
print(f"  Frequenza:     {data['fs']} Hz")
print(f"  Num classi:    {data['num_classes']}")
print(f"  Classi:        {data['class_names']}")
print(f"  Seed split:    {data['split_seed']}")

print(f"\n--- SPLIT ---")
print(f"  Train:  {len(data['train_indices'])} campioni")
print(f"  Val:    {len(data['val_indices'])} campioni")
print(f"  Test:   {len(data['test_indices'])} campioni")

print(f"\n--- PESI CLASSI (per loss bilanciata) ---")
for i in range(data['num_classes']):
    print(f"  {data['class_names'][i]:>12}: {data['class_weights'][i]:.4f}")

print(f"\n--- DISTRIBUZIONE CLASSI ---")
Y_flat = data['Y'].flatten()
total = Y_flat.numel()
for i in range(data['num_classes']):
    count = (Y_flat == i).sum().item()
    pct = count / total * 100
    print(f"  {data['class_names'][i]:>12}: {count:>10,} campioni ({pct:5.1f}%)")

print(f"\n--- PRIMI 5 CAMPIONI ---")
for idx in range(min(5, len(data['record_ids']))):
    rec = data['record_ids'][idx]
    lead = data['lead_names'][idx]
    n_p   = (data['Y'][idx] == 1).sum().item()
    n_qrs = (data['Y'][idx] == 2).sum().item()
    n_t   = (data['Y'][idx] == 3).sum().item()
    print(f"  [{idx}] Record {rec:>3}, Lead {lead:>3}  "
          f"| P={n_p:>3} QRS={n_qrs:>3} T={n_t:>3} campioni")

print(f"\n--- ESEMPIO SEGNALE (campione 0, primi 10 valori) ---")
print(f"  Segnale: {data['X'][0, 0, :10].tolist()}")
print(f"  Labels:  {data['Y'][0, :10].tolist()}")

print("\n" + "=" * 60)
