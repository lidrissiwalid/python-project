"""
=============================================================
Mini-Projet : Comparaison des datasets augmentés (CTGAN, TGAN, TVAE)
Étudiant 2 - Expert ML/DL : Entraînement & Benchmarking

CSV Usage:
  original.csv          → Split 80/20 : train de référence + test set partagé
  ctgan_augmented.csv   → Entraînement uniquement  (testé sur original test set)
  tgan_augmented.csv    → Entraînement uniquement  (testé sur original test set)
  tvae_augmented.csv    → Entraînement uniquement  (testé sur original test set)

ML  : Logistic Regression (Linear) + Random Forest (Non-linear)
DL  : MLP + LSTM
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              precision_score, confusion_matrix,
                              ConfusionMatrixDisplay)

# ──────────────────────────────────────────────
# 0. CONFIG
# ──────────────────────────────────────────────

DATASETS = {
    "Original": "original_dataset.csv",
    "CTGAN":    "ctgan_augmented_dataset.csv",
    "TGAN":     "tgan_augmented_dataset.csv",
    "TVAE":     "tvae_augmented_dataset.csv",
}

LABEL_COL = "Label"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device PyTorch : {DEVICE}\n")

MODEL_COLORS = {
    "Logistic Regression": "#2196F3",
    "Random Forest":       "#4CAF50",
    "MLP":                 "#FF5722",
    "LSTM":                "#FF9800",
}

# ──────────────────────────────────────────────
# 1. CHARGEMENT
# ──────────────────────────────────────────────

def load_dataset(path):
    """Load CSV, encode label column, return X and y."""
    df = pd.read_csv(path)
    le = LabelEncoder()
    df[LABEL_COL] = le.fit_transform(df[LABEL_COL])
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]
    return X, y

# ──────────────────────────────────────────────
# 2. SHARED REAL TEST SET  (from original.csv only)
# ──────────────────────────────────────────────
#
#   original.csv  ──80%──►  X_orig_train / y_orig_train   (baseline train)
#                 ──20%──►  X_test_real  / y_test_real    (shared test — never touched by augmented sets)
#
print("=" * 60)
print("Création du test set réel à partir de original.csv (20 %)")
print("=" * 60)

X_orig, y_orig = load_dataset(DATASETS["Original"])

X_orig_train, X_test_real, y_orig_train, y_test_real = train_test_split(
    X_orig, y_orig,
    test_size=0.2,
    random_state=42,
    stratify=y_orig
)

print(f"  original  → train : {len(X_orig_train):>6}  |  test (partagé) : {len(X_test_real)}\n")

# ──────────────────────────────────────────────
# 3. DEFINE WHICH DATA EACH DATASET USES FOR TRAINING
# ──────────────────────────────────────────────
#
#  ┌─────────────┬──────────────────────────────┬──────────────────────┐
#  │  Dataset    │  Train set                   │  Test set            │
#  ├─────────────┼──────────────────────────────┼──────────────────────┤
#  │  Original   │  80 % of original.csv        │  20 % of original    │
#  │  CTGAN      │  100 % of ctgan_augmented    │  20 % of original    │
#  │  TGAN       │  100 % of tgan_augmented     │  20 % of original    │
#  │  TVAE       │  100 % of tvae_augmented     │  20 % of original    │
#  └─────────────┴──────────────────────────────┴──────────────────────┘

def get_train_data(ds_name):
    """
    Return (X_train, y_train) according to the usage rules above.
    Original  → use the 80 % split already created.
    Augmented → load the full augmented file (train only).
    """
    if ds_name == "Original":
        return X_orig_train, y_orig_train

    path = DATASETS[ds_name]
    X_aug, y_aug = load_dataset(path)
    print(f"  {ds_name:<8} → train size : {len(X_aug)} échantillons (fichier complet)")
    return X_aug, y_aug

# ──────────────────────────────────────────────
# 4. SKLEARN ML MODELS
# ──────────────────────────────────────────────

def get_sklearn_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1, C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ),
    }

# ──────────────────────────────────────────────
# 5. DEEP LEARNING ARCHITECTURES (PyTorch)
# ──────────────────────────────────────────────

class MLPNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),          nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


class LSTMNet(nn.Module):
    """LSTM — each feature treated as one step in a sequence."""
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),  nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(2))
        return self.fc(out[:, -1, :]).squeeze(1)

# ──────────────────────────────────────────────
# 6. DL TRAINING LOOP
# ──────────────────────────────────────────────

def train_dl_model(model, X_train, y_train, X_test,
                   epochs=50, batch_size=64, lr=1e-3):
    model.to(DEVICE)

    Xtr = torch.tensor(X_train,         dtype=torch.float32)
    ytr = torch.tensor(y_train.values,  dtype=torch.float32)
    Xte = torch.tensor(X_test,          dtype=torch.float32)

    loader    = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_loss, patience, wait = float('inf'), 10, 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss - 1e-4:
            best_loss, wait = avg_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.eval()
    with torch.no_grad():
        preds = model(Xte.to(DEVICE)).cpu().numpy()
    return (preds >= 0.5).astype(int)

# ──────────────────────────────────────────────
# 7. MAIN TRAINING & EVALUATION LOOP
# ──────────────────────────────────────────────

results, cms = {}, {}

for ds_name in DATASETS:
    print(f"\n{'─'*60}")
    print(f"  Dataset : {ds_name}")
    print(f"{'─'*60}")

    # ── Get training data (rule-based, see section 3) ──
    try:
        X_train, y_train = get_train_data(ds_name)
    except FileNotFoundError:
        print(f"  [!] Fichier introuvable → {ds_name} ignoré")
        continue

    # ── Shared test set (always the 20 % split from original.csv) ──
    X_test, y_test = X_test_real, y_test_real

    print(f"  Train : {len(X_train)} | Test (original 20%) : {len(X_test)}")

    # ── Feature scaling (fit on train, apply to test) ──
    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train)
    Xte_sc  = scaler.transform(X_test)

    results[ds_name] = {}
    cms[ds_name]     = {}
    input_dim        = Xtr_sc.shape[1]

    # ── Sklearn ML models ──
    for model_name, model in get_sklearn_models().items():
        print(f"  → Training {model_name}...", end=" ", flush=True)
        model.fit(Xtr_sc, y_train)
        y_pred = model.predict(Xte_sc)
        print("done.")

        results[ds_name][model_name] = {
            "Accuracy":  round(accuracy_score (y_test, y_pred)                    * 100, 2),
            "F1-Score":  round(f1_score        (y_test, y_pred, average='weighted') * 100, 2),
            "Recall":    round(recall_score    (y_test, y_pred, average='weighted') * 100, 2),
            "Precision": round(precision_score (y_test, y_pred, average='weighted') * 100, 2),
        }
        cms[ds_name][model_name] = confusion_matrix(y_test, y_pred)
        m = results[ds_name][model_name]
        print(f"    [{ds_name:8s}] {model_name:22s} → Acc={m['Accuracy']}%  F1={m['F1-Score']}%")

    # ── PyTorch DL models ──
    for model_name, model in {
        "MLP":  MLPNet(input_dim),
        "LSTM": LSTMNet(input_dim),
    }.items():
        print(f"  → Training {model_name}...", end=" ", flush=True)
        y_pred = train_dl_model(model, Xtr_sc, y_train, Xte_sc)
        print("done.")

        results[ds_name][model_name] = {
            "Accuracy":  round(accuracy_score (y_test, y_pred)                    * 100, 2),
            "F1-Score":  round(f1_score        (y_test, y_pred, average='weighted') * 100, 2),
            "Recall":    round(recall_score    (y_test, y_pred, average='weighted') * 100, 2),
            "Precision": round(precision_score (y_test, y_pred, average='weighted') * 100, 2),
        }
        cms[ds_name][model_name] = confusion_matrix(y_test, y_pred)
        m = results[ds_name][model_name]
        print(f"    [{ds_name:8s}] {model_name:22s} → Acc={m['Accuracy']}%  F1={m['F1-Score']}%")

# ──────────────────────────────────────────────
# 8. VISUALISATIONS
# ──────────────────────────────────────────────

metrics_list = ["Accuracy", "F1-Score", "Recall", "Precision"]
ds_names     = list(results.keys())
model_names  = list(list(results.values())[0].keys())

# ── Tidy DataFrame used by all plots ──
rows_viz = []
for ds in ds_names:
    for model in model_names:
        row = {"Dataset": ds, "Model": model}
        row.update(results[ds][model])
        rows_viz.append(row)
df_results = pd.DataFrame(rows_viz)

# ════════════════════════════════════════════════════════════════════════
# PLOT 1 — Grouped bar chart (2×2 grid, one panel per metric)
#   X-axis = dataset, bar groups = models
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    "Comparaison ML/DL — Original vs Augmentés\n"
    "(évalués sur le même test set réel — 20 % de original.csv)",
    fontsize=14, fontweight='bold', y=1.01
)
x     = np.arange(len(ds_names))
width = 0.18

for ax, metric in zip(axes.flatten(), metrics_list):
    for i, model_name in enumerate(model_names):
        vals   = [results[ds][model_name][metric] for ds in ds_names]
        offset = (i - len(model_names) / 2) * width + width / 2
        bars   = ax.bar(x + offset, vals, width,
                        label=model_name, color=MODEL_COLORS[model_name],
                        alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_xticks(x);  ax.set_xticklabels(ds_names, fontsize=9)
    ax.set_ylim(0, 118);  ax.set_ylabel("%", fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("plot1_grouped_bars.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Plot 1 sauvegardé : plot1_grouped_bars.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 2 — Stacked bar chart (one panel per metric)
#   Shows how each model contributes to the total score per dataset.
#   Useful to visually compare the "weight" of each model across datasets.
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    "Stacked Bar — Contribution des modèles par dataset",
    fontsize=14, fontweight='bold'
)

for ax, metric in zip(axes.flatten(), metrics_list):
    bottoms = np.zeros(len(ds_names))
    for model_name in model_names:
        vals = np.array([results[ds][model_name][metric] for ds in ds_names])
        bars = ax.bar(ds_names, vals, bottom=bottoms,
                      label=model_name, color=MODEL_COLORS[model_name],
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, v, b in zip(bars, vals, bottoms):
            if v > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                        f'{v:.1f}', ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')
        bottoms += vals

    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_ylabel("Cumulative %", fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("plot2_stacked_bars.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 2 sauvegardé : plot2_stacked_bars.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 3 — Histogram of score distributions (per metric)
#   Each bar = one (dataset, model) combination.
#   Shows the spread / concentration of performance scores.
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "Histogramme — Distribution des scores par métrique\n"
    "(toutes combinaisons dataset × modèle)",
    fontsize=14, fontweight='bold'
)

for ax, metric in zip(axes.flatten(), metrics_list):
    all_scores = [results[ds][m][metric] for ds in ds_names for m in model_names]
    ax.hist(all_scores, bins=12, color='#5C6BC0', edgecolor='white',
            alpha=0.85, linewidth=0.6)
    ax.axvline(np.mean(all_scores), color='red', linestyle='--',
               linewidth=1.5, label=f'Moyenne = {np.mean(all_scores):.1f}%')
    ax.axvline(np.median(all_scores), color='orange', linestyle=':',
               linewidth=1.5, label=f'Médiane = {np.median(all_scores):.1f}%')
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_xlabel("Score (%)", fontsize=9)
    ax.set_ylabel("Fréquence", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("plot3_histograms.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 3 sauvegardé : plot3_histograms.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 4 — Heatmap : Metric × Model, one panel per dataset
#   Rows = metrics, Columns = models
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 5))
if len(ds_names) == 1:
    axes = [axes]
fig.suptitle("Heatmap Métriques × Modèles (par dataset)",
             fontsize=14, fontweight='bold')

for ax, ds_name in zip(axes, ds_names):
    matrix = np.array([[results[ds_name][m][met] for m in model_names]
                        for met in metrics_list])
    im = ax.imshow(matrix, cmap='YlGn', vmin=50, vmax=100, aspect='auto')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(metrics_list)))
    ax.set_yticklabels(metrics_list, fontsize=9)
    ax.set_title(ds_name, fontsize=11, fontweight='bold')
    for i in range(len(metrics_list)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{matrix[i, j]:.1f}",
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='black' if matrix[i, j] < 85 else 'white')
    plt.colorbar(im, ax=ax, shrink=0.8, label="%")

plt.tight_layout()
plt.savefig("plot4_heatmap_per_dataset.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 4 sauvegardé : plot4_heatmap_per_dataset.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 5 — Heatmap : Dataset × Model, one panel per metric
#   Rows = datasets, Columns = models
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Heatmap Dataset × Modèle (par métrique)",
             fontsize=14, fontweight='bold')

for ax, metric in zip(axes.flatten(), metrics_list):
    matrix = np.array([[results[ds][m][metric] for m in model_names]
                        for ds in ds_names])
    im = ax.imshow(matrix, cmap='Blues', vmin=50, vmax=100, aspect='auto')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(ds_names)))
    ax.set_yticklabels(ds_names, fontsize=9)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    for i in range(len(ds_names)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{matrix[i, j]:.1f}",
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='black' if matrix[i, j] < 85 else 'white')
    plt.colorbar(im, ax=ax, shrink=0.8, label="%")

plt.tight_layout()
plt.savefig("plot5_heatmap_per_metric.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 5 sauvegardé : plot5_heatmap_per_metric.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 6 — Full summary heatmap (all dataset × model combos × metrics)
#   Rows = (dataset / model) label, Cols = metrics
#   Horizontal white lines separate dataset groups
# ════════════════════════════════════════════════════════════════════════
index_labels = [f"{ds} / {m}" for ds in ds_names for m in model_names]
matrix_full  = np.array([[results[ds][m][met] for met in metrics_list]
                          for ds in ds_names for m in model_names])

fig, ax = plt.subplots(figsize=(10, 0.55 * len(index_labels) + 2))
im = ax.imshow(matrix_full, cmap='RdYlGn', vmin=50, vmax=100, aspect='auto')
ax.set_xticks(range(len(metrics_list)))
ax.set_xticklabels(metrics_list, fontsize=10, fontweight='bold')
ax.set_yticks(range(len(index_labels)))
ax.set_yticklabels(index_labels, fontsize=8)
ax.set_title("Heatmap Récapitulatif — Tous modèles × Tous datasets",
             fontsize=12, fontweight='bold', pad=12)

for i in range(len(index_labels)):
    for j in range(len(metrics_list)):
        ax.text(j, i, f"{matrix_full[i, j]:.1f}",
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='black' if 60 < matrix_full[i, j] < 88 else 'white')

for k in range(1, len(ds_names)):
    ax.axhline(k * len(model_names) - 0.5, color='white', linewidth=2)

plt.colorbar(im, ax=ax, label="%", shrink=0.6)
plt.tight_layout()
plt.savefig("plot6_summary_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 6 sauvegardé : plot6_summary_heatmap.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 7 — Δ Heatmap : gain / loss vs Original baseline (per augmented)
#   Rows = augmented datasets, Cols = (model × metric)
#   Green = improvement, Red = degradation vs Original
# ════════════════════════════════════════════════════════════════════════
aug_names    = [d for d in ds_names if d != "Original"]
col_labels   = [f"{m}\n{met}" for m in model_names for met in metrics_list]
delta_matrix = np.array([
    [results[ds][m][met] - results["Original"][m][met]
     for m in model_names for met in metrics_list]
    for ds in aug_names
])

fig, ax = plt.subplots(figsize=(max(12, len(col_labels) * 0.9), 3.5))
vmax = max(abs(delta_matrix).max(), 1)
im   = ax.imshow(delta_matrix, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha='right')
ax.set_yticks(range(len(aug_names)))
ax.set_yticklabels(aug_names, fontsize=10)
ax.set_title("Δ Heatmap — Gain / Perte vs Original (en points de %)",
             fontsize=12, fontweight='bold', pad=12)

for i in range(len(aug_names)):
    for j in range(len(col_labels)):
        v = delta_matrix[i, j]
        ax.text(j, i, f"{v:+.1f}",
                ha='center', va='center', fontsize=7.5, fontweight='bold',
                color='black' if abs(v) < vmax * 0.6 else 'white')

for k in range(1, len(model_names)):
    ax.axvline(k * len(metrics_list) - 0.5, color='white', linewidth=2)

plt.colorbar(im, ax=ax, label="Δ %", shrink=0.8)
plt.tight_layout()
plt.savefig("plot7_delta_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 7 sauvegardé : plot7_delta_heatmap.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 8 — Radar / Spider chart (one panel per dataset)
#   Each spoke = one metric, each coloured polygon = one model
# ════════════════════════════════════════════════════════════════════════
angles = np.linspace(0, 2 * np.pi, len(metrics_list), endpoint=False).tolist()
angles += angles[:1]   # close the polygon

fig, axes = plt.subplots(1, len(ds_names),
                          figsize=(5 * len(ds_names), 5),
                          subplot_kw=dict(polar=True))
if len(ds_names) == 1:
    axes = [axes]
fig.suptitle("Radar Chart — Profil métrique par dataset",
             fontsize=14, fontweight='bold')

for ax, ds_name in zip(axes, ds_names):
    for model_name in model_names:
        vals = [results[ds_name][model_name][m] for m in metrics_list]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8,
                label=model_name, color=MODEL_COLORS[model_name])
        ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS[model_name])
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_list, fontsize=9)
    ax.set_ylim(50, 100)
    ax.set_title(ds_name, fontsize=11, fontweight='bold', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=7)
    ax.grid(color='grey', alpha=0.3)

plt.tight_layout()
plt.savefig("plot8_radar_per_dataset.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 8 sauvegardé : plot8_radar_per_dataset.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 9 — Line chart : metric evolution across datasets (per model)
#   X-axis = dataset order, Y-axis = score, one line per model
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Évolution des métriques à travers les datasets (par modèle)",
             fontsize=14, fontweight='bold')

for ax, metric in zip(axes.flatten(), metrics_list):
    for model_name in model_names:
        vals = [results[ds][model_name][metric] for ds in ds_names]
        ax.plot(ds_names, vals, marker='o', linewidth=2,
                label=model_name, color=MODEL_COLORS[model_name])
        for x_pos, v in zip(ds_names, vals):
            ax.annotate(f'{v:.1f}', (x_pos, v),
                        textcoords="offset points", xytext=(0, 7),
                        ha='center', fontsize=7)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_ylim(50, 108);  ax.set_ylabel("%", fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("plot9_line_evolution.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 9 sauvegardé : plot9_line_evolution.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 10 — Best model per dataset (horizontal bar chart)
#   For each dataset, rank models by mean score across all 4 metrics.
#   Shows clearly which model wins on each training set.
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 5),
                          sharey=False)
if len(ds_names) == 1:
    axes = [axes]
fig.suptitle("Classement des modèles par dataset (score moyen sur 4 métriques)",
             fontsize=13, fontweight='bold')

for ax, ds_name in zip(axes, ds_names):
    means  = {m: np.mean([results[ds_name][m][met] for met in metrics_list])
               for m in model_names}
    sorted_models = sorted(means, key=means.get, reverse=True)
    sorted_vals   = [means[m] for m in sorted_models]
    colors_sorted = [MODEL_COLORS[m] for m in sorted_models]

    bars = ax.barh(sorted_models, sorted_vals, color=colors_sorted,
                   alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, sorted_vals):
        ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height() / 2,
                f'{v:.2f}%', va='center', ha='right',
                fontsize=9, fontweight='bold', color='white')

    ax.set_title(ds_name, fontsize=11, fontweight='bold')
    ax.set_xlabel("Score moyen (%)", fontsize=9)
    ax.set_xlim(50, 102)
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    # Gold star on the winner
    ax.annotate("★", xy=(sorted_vals[0], 0),
                xytext=(sorted_vals[0] + 0.2, 0),
                fontsize=14, color='gold', va='center')

plt.tight_layout()
plt.savefig("plot10_best_model_ranking.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 10 sauvegardé : plot10_best_model_ranking.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 11 — Per-model histogram : score distribution across datasets
#   One subplot per model — shows how consistent a model is
#   across Original / CTGAN / TGAN / TVAE
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(model_names),
                          figsize=(5 * len(model_names), 4), sharey=True)
if len(model_names) == 1:
    axes = [axes]
fig.suptitle("Distribution des scores par modèle (toutes métriques et datasets)",
             fontsize=13, fontweight='bold')

for ax, model_name in zip(axes, model_names):
    scores = [results[ds][model_name][met]
               for ds in ds_names for met in metrics_list]
    ax.hist(scores, bins=8, color=MODEL_COLORS[model_name],
            edgecolor='white', alpha=0.85, linewidth=0.6)
    ax.axvline(np.mean(scores), color='black', linestyle='--', linewidth=1.5,
               label=f'μ={np.mean(scores):.1f}%')
    ax.set_title(model_name, fontsize=10, fontweight='bold')
    ax.set_xlabel("Score (%)", fontsize=8)
    ax.set_ylabel("Fréquence", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("plot11_per_model_histogram.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 11 sauvegardé : plot11_per_model_histogram.png")

# ════════════════════════════════════════════════════════════════════════
# PLOT 12 — Confusion matrices (one figure per model, all datasets)
# ════════════════════════════════════════════════════════════════════════
for model_name in model_names:
    fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 4))
    if len(ds_names) == 1:
        axes = [axes]
    fig.suptitle(
        f"Matrices de confusion — {model_name}\n"
        "(test set réel partagé — 20 % de original.csv)",
        fontsize=13, fontweight='bold'
    )
    for ax, ds_name in zip(axes, ds_names):
        disp = ConfusionMatrixDisplay(cms[ds_name][model_name],
                                      display_labels=["Goodware", "Ransomware"])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(ds_name, fontsize=10, fontweight='bold')
    plt.tight_layout()
    fname = f"plot12_confusion_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot 12 — Matrice de confusion : {fname}")

print("\n✓ Toutes les visualisations ont été générées (12 plots).")

# ──────────────────────────────────────────────
# 9. SUMMARY TABLE + CSV EXPORT
# ──────────────────────────────────────────────

print("\n" + "=" * 80)
print("TABLEAU RÉCAPITULATIF (test set réel partagé — 20 % de original.csv)")
print("=" * 80)
print(f"{'Dataset':<10} {'Modèle':<24} {'Accuracy':>10} {'F1-Score':>10} {'Recall':>8} {'Precision':>10}")
print("-" * 80)
for ds in ds_names:
    for model in model_names:
        m = results[ds][model]
        print(f"{ds:<10} {model:<24} {m['Accuracy']:>9}% {m['F1-Score']:>9}% "
              f"{m['Recall']:>7}% {m['Precision']:>9}%")
print("=" * 80)

rows = [{"Dataset": ds, "Modèle": model, **results[ds][model]}
        for ds in ds_names for model in model_names]
pd.DataFrame(rows).to_csv("results_summary.csv", index=False)
print("\n✓ Résultats exportés : results_summary.csv")