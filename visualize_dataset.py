"""
visualize_dataset.py
====================
Visualizza graficamente i tensori del dataset LUDB.
Mostra il segnale ECG con le regioni P, QRS, T colorate.

Uso:
  python visualize_dataset.py
  python visualize_dataset.py --record 1 --lead II
  python visualize_dataset.py --idx 0
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse


def setup_args():
    parser = argparse.ArgumentParser(description="Visualizza dataset LUDB")
    parser.add_argument("--idx", type=int, default=None,
                        help="Indice del campione da visualizzare (0-2399)")
    parser.add_argument("--record", type=int, default=None,
                        help="ID del record (1-200)")
    parser.add_argument("--lead", type=str, default=None,
                        help="Nome derivazione (I, II, III, AVR, AVL, AVF, V1-V6)")
    parser.add_argument("--all-leads", action="store_true",
                        help="Mostra tutte le 12 derivazioni di un record")
    return parser.parse_args()


def find_sample_idx(data, record_id, lead_name):
    """Trova l'indice del campione dato record e derivazione."""
    for i in range(len(data['record_ids'])):
        if data['record_ids'][i] == record_id and data['lead_names'][i] == lead_name.upper():
            return i
    return None


def plot_single_lead(data, idx, save_path=None):
    """Visualizza un singolo campione con segmentazione colorata."""
    signal = data['X'][idx, 0, :].numpy()
    labels = data['Y'][idx, :].numpy()
    record = data['record_ids'][idx]
    lead = data['lead_names'][idx]
    fs = data['fs']

    time = np.arange(len(signal)) / fs  # in secondi

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # --- Pannello 1: Segnale con regioni colorate ---
    ax1.plot(time, signal, color='black', linewidth=0.5, label='ECG')

    # Colora le regioni
    colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#FFD93D'}  # P=rosso, QRS=verde, T=giallo
    names = {1: 'Onda P', 2: 'QRS', 3: 'Onda T'}

    for cls_id, color in colors.items():
        mask = labels == cls_id
        if mask.any():
            ax1.fill_between(time, signal.min(), signal.max(),
                           where=mask, alpha=0.3, color=color, label=names[cls_id])

    ax1.set_ylabel('Ampiezza (normalizzata)')
    ax1.set_title(f'Record {record} - Derivazione {lead}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Pannello 2: Mappa di segmentazione ---
    cmap = plt.cm.colors.ListedColormap(['#E8E8E8', '#FF6B6B', '#4ECDC4', '#FFD93D'])
    ax2.imshow(labels.reshape(1, -1), aspect='auto', cmap=cmap,
               extent=[0, time[-1], 0, 1], vmin=0, vmax=3)
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Classe')
    ax2.set_yticks([])

    # Legenda
    patches = [
        mpatches.Patch(color='#E8E8E8', label='Background'),
        mpatches.Patch(color='#FF6B6B', label='Onda P'),
        mpatches.Patch(color='#4ECDC4', label='QRS'),
        mpatches.Patch(color='#FFD93D', label='Onda T'),
    ]
    ax2.legend(handles=patches, loc='upper right', ncol=4, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Salvato: {save_path}")
    plt.show()


def plot_all_leads(data, record_id, save_path=None):
    """Visualizza tutte le 12 derivazioni di un record."""
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axes = plt.subplots(6, 2, figsize=(18, 20), sharex=True)
    fig.suptitle(f'Record {record_id} - Tutte le derivazioni', fontsize=16, fontweight='bold')

    colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#FFD93D'}

    for i, lead in enumerate(leads):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        idx = find_sample_idx(data, record_id, lead)
        if idx is None:
            ax.text(0.5, 0.5, f'{lead}: non trovato', transform=ax.transAxes,
                   ha='center', va='center')
            continue

        signal = data['X'][idx, 0, :].numpy()
        labels = data['Y'][idx, :].numpy()
        time = np.arange(len(signal)) / data['fs']

        ax.plot(time, signal, color='black', linewidth=0.5)

        for cls_id, color in colors.items():
            mask = labels == cls_id
            if mask.any():
                ax.fill_between(time, signal.min(), signal.max(),
                              where=mask, alpha=0.3, color=color)

        ax.set_ylabel(lead, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if row == 5:
            ax.set_xlabel('Tempo (s)')

    # Legenda comune
    patches = [
        mpatches.Patch(color='#FF6B6B', label='Onda P'),
        mpatches.Patch(color='#4ECDC4', label='QRS'),
        mpatches.Patch(color='#FFD93D', label='Onda T'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Salvato: {save_path}")
    plt.show()


def main():
    args = setup_args()

    print("Caricamento dataset...")
    data = torch.load('ludb_tensors/ludb_dataset.pt', weights_only=False)
    print(f"Caricati {data['X'].shape[0]} campioni\n")

    # Determina cosa visualizzare
    if args.all_leads:
        record_id = args.record if args.record else data['record_ids'][0]
        print(f"Visualizzo tutte le derivazioni del Record {record_id}")
        plot_all_leads(data, record_id, save_path=f"ecg_record_{record_id}_all.png")

    elif args.idx is not None:
        if 0 <= args.idx < len(data['record_ids']):
            print(f"Visualizzo campione [{args.idx}]: "
                  f"Record {data['record_ids'][args.idx]}, "
                  f"Lead {data['lead_names'][args.idx]}")
            plot_single_lead(data, args.idx, save_path=f"ecg_sample_{args.idx}.png")
        else:
            print(f"Indice {args.idx} non valido (0-{len(data['record_ids'])-1})")

    elif args.record and args.lead:
        idx = find_sample_idx(data, args.record, args.lead)
        if idx is not None:
            print(f"Visualizzo Record {args.record}, Lead {args.lead} (indice {idx})")
            plot_single_lead(data, idx, save_path=f"ecg_rec{args.record}_{args.lead}.png")
        else:
            print(f"Record {args.record}, Lead {args.lead} non trovato!")

    else:
        # Default: mostra il primo campione
        print("Visualizzo il primo campione (usa --help per le opzioni)")
        plot_single_lead(data, 0, save_path="ecg_sample_0.png")


if __name__ == "__main__":
    main()
