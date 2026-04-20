"""
=============================================================
MDF KALİTE TAHMİN SİSTEMİ - MODÜL 3: MODEL DEĞERLENDİRME
=============================================================
Gebze MDF Üretim Hattı - Fire & Kalite Tahmini
Bitirme Projesi | 2024-2025

Bu modül:
- Confusion matrix
- ROC & PR eğrileri
- Özellik önemi (XGBoost & RF)
- Threshold optimizasyonu
- Tüm grafikleri PNG olarak kaydeder
- PDF rapor oluşturur
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR  = "outputs"
GRAFIK_DIR  = "outputs/grafikler"
os.makedirs(GRAFIK_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# ─────────────────────────────────────────
# VERİ & MODEL YÜKLEME
# ─────────────────────────────────────────
print("=" * 60)
print("  MDF KALİTE TAHMİN — DEĞERLENDİRME MODÜLÜ")
print("=" * 60)

def yukle(dosya):
    with open(f"{OUTPUT_DIR}/{dosya}", "rb") as f:
        return pickle.load(f)

X_test         = yukle("X_test.pkl")
y_test         = yukle("y_test.pkl")
X_test_scaled  = yukle("X_test_scaled.pkl")
features       = yukle("feature_names.pkl")
model_xgb      = yukle("model_xgboost.pkl")
model_rf       = yukle("model_rf.pkl")
model_ann      = yukle("model_ann.pkl")

print("\n✅ Modeller ve test verisi yüklendi")

# Tahminler
modeller = {
    'XGBoost':      (model_xgb,  X_test,        'tab:blue'),
    'RandomForest': (model_rf,   X_test,        'tab:green'),
    'ANN':          (model_ann,  X_test_scaled, 'tab:red'),
}

tahminler = {}
for isim, (mdl, X, renk) in modeller.items():
    y_pred  = mdl.predict(X)
    y_proba = mdl.predict_proba(X)[:, 1]
    tahminler[isim] = {
        'y_pred':  y_pred,
        'y_proba': y_proba,
        'renk':    renk,
        'model':   mdl,
        'X':       X
    }

# ══════════════════════════════════════════
# GRAFİK 1: Confusion Matrix (3 model)
# ══════════════════════════════════════════
print("\n📊 Confusion Matrix çiziliyor...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrix — Tüm Modeller', fontsize=15, fontweight='bold', y=1.02)

for ax, (isim, d) in zip(axes, tahminler.items()):
    cm = confusion_matrix(y_test, d['y_pred'])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Kaliteli (0)', 'Hatalı (1)'],
        yticklabels=['Kaliteli (0)', 'Hatalı (1)'],
        linewidths=0.5, linecolor='gray',
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    acc = (cm[0,0] + cm[1,1]) / cm.sum()
    ax.set_title(f'{isim}\nAccuracy: %{acc*100:.2f}', fontweight='bold')
    ax.set_xlabel('Tahmin Edilen', fontsize=10)
    ax.set_ylabel('Gerçek', fontsize=10)

plt.tight_layout()
plt.savefig(f"{GRAFIK_DIR}/01_confusion_matrix.png", bbox_inches='tight')
plt.close()
print(f"   ✅ 01_confusion_matrix.png kaydedildi")

# ══════════════════════════════════════════
# GRAFİK 2: ROC Eğrileri
# ══════════════════════════════════════════
print("📊 ROC eğrileri çiziliyor...")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Rastgele (AUC=0.50)')

for isim, d in tahminler.items():
    fpr, tpr, _ = roc_curve(y_test, d['y_proba'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2.5, color=d['renk'],
            label=f"{isim} (AUC = {roc_auc:.4f})")

ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
ax.set_title('ROC Eğrisi Karşılaştırması', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.01])

plt.tight_layout()
plt.savefig(f"{GRAFIK_DIR}/02_roc_egrisi.png", bbox_inches='tight')
plt.close()
print(f"   ✅ 02_roc_egrisi.png kaydedildi")

# ══════════════════════════════════════════
# GRAFİK 3: Precision-Recall Eğrileri
# ══════════════════════════════════════════
print("📊 PR eğrileri çiziliyor...")

fig, ax = plt.subplots(figsize=(8, 6))

for isim, d in tahminler.items():
    prec, rec, _ = precision_recall_curve(y_test, d['y_proba'])
    ap = average_precision_score(y_test, d['y_proba'])
    ax.plot(rec, prec, lw=2.5, color=d['renk'],
            label=f"{isim} (AP = {ap:.4f})")

baseline = y_test.mean()
ax.axhline(y=baseline, color='k', linestyle='--', lw=1.2,
           alpha=0.5, label=f'Baseline (AP={baseline:.3f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Eğrisi Karşılaştırması', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.01])

plt.tight_layout()
plt.savefig(f"{GRAFIK_DIR}/03_pr_egrisi.png", bbox_inches='tight')
plt.close()
print(f"   ✅ 03_pr_egrisi.png kaydedildi")

# ══════════════════════════════════════════
# GRAFİK 4: Özellik Önemi (XGBoost)
# ══════════════════════════════════════════
print("📊 XGBoost özellik önemi çiziliyor...")

xgb_imp = pd.Series(
    model_xgb.feature_importances_,
    index=features
).sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(xgb_imp)))
bars = ax.barh(xgb_imp.index[::-1], xgb_imp.values[::-1], color=colors[::-1], edgecolor='white')

for bar, val in zip(bars, xgb_imp.values[::-1]):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='left', fontsize=9)

ax.set_xlabel('Önem Skoru (Gain)', fontsize=12)
ax.set_title('XGBoost — Top 20 Özellik Önemi', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(right=xgb_imp.max() * 1.15)

plt.tight_layout()
plt.savefig(f"{GRAFIK_DIR}/04_xgb_feature_importance.png", bbox_inches='tight')
plt.close()
print(f"   ✅ 04_xgb_feature_importance.png kaydedildi")

# ══════════════════════════════════════════
# GRAFİK 5: Özellik Önemi (Random Forest)
# ══════════════════════════════════════════
print("📊 Random Forest özellik önemi çiziliyor...")

rf_imp = pd.Series(
    model_rf.feature_importances_,
    index=features
).sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
colors_rf = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(rf_imp)))
bars_rf = ax.barh(rf_imp.index[::-1], rf_imp.values[::-1], color=colors_rf[::-1], edgecolor='white')

for bar, val in zip(bars_rf, rf_imp.values[::-1]):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='left', fontsize=9)

ax.set_xlabel('Önem Skoru (Gini Impurity)', fontsize=12)
ax.set_title('Random Forest — Top 20 Özellik Önemi', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(right=rf_imp.max() * 1.15)

plt.tight_layout()
plt.savefig(f"{GRAFIK_DIR}/05_rf_feature_importance.png", bbox_inches='tight')
plt.close()
print(f"   ✅ 05_rf_feature_importance.png kaydedildi")

# ══════════════════════════════════════════
# GRAFİK 6: Threshold Analizi (XGBoost için)
# ══════════════════════════════════════════
print("📊 Threshold analizi çiziliyor...")

thresholds = np.linspace(0.1, 0.9, 80)
f1_scores, prec_scores, rec_scores = [], [], []

for t in thresholds:
    yp = (tahminler['XGBoost']['y_proba'] >= t).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1_scores.append(f1_score(y_test, yp, zero_division=0))
    prec_scores.append(precision_score(y_test, yp, zero_division=0))
    rec_scores.append(recall_score(y_test, yp, zero_division=0))

opt_idx  = np.argmax(f1_scores)
opt_thr  = thresholds[opt_idx]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, f1_scores,   color='tab:blue',   lw=2.5, label='F1-Score')
ax.plot(thresholds, prec_scores, color='tab:orange',  lw=2,   label='Precision', linestyle='--')
ax.plot(thresholds, rec_scores,  color='tab:green',   lw=2,   label='Recall',    linestyle=':')
ax.axvline(x=opt_thr, color='red', linestyle='--', lw=1.5,
           label=f'Optimal Threshold = {opt_thr:.2f}')
ax.axvline(x=0.5, color='gray', linestyle=':', lw=1.2, alpha=0.6, label='Default (0.50)')

ax.set_xlabel('Karar Eşiği (Threshold)', fontsize=12)
ax.set_ylabel('Skor', fontsize=12)
ax.set_title('XGBoost — Threshold Optimizasyonu', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0.1, 0.9])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(f"{GRAFIK_DIR}/06_threshold_analizi.png", bbox_inches='tight')
plt.close()
print(f"   ✅ 06_threshold_analizi.png kaydedildi")
print(f"   🎯 Optimal Threshold (XGBoost): {opt_thr:.3f}")

# Optimal threshold'u kaydet
with open(f"{OUTPUT_DIR}/optimal_threshold.pkl", "wb") as f:
    pickle.dump(opt_thr, f)

# ══════════════════════════════════════════
# GRAFİK 7: Model Karşılaştırma Bar Plot
# ══════════════════════════════════════════
print("📊 Model karşılaştırma grafiği çiziliyor...")

karsilastirma = pd.read_csv(f"{OUTPUT_DIR}/model_karsilastirma.csv", index_col=0)

metrikler = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
x = np.arange(len(metrikler))
genislik = 0.25
renkler = ['tab:blue', 'tab:green', 'tab:red']

fig, ax = plt.subplots(figsize=(12, 6))
for i, (isim, renk) in enumerate(zip(karsilastirma.index, renkler)):
    vals = [karsilastirma.loc[isim, m] for m in metrikler]
    bars = ax.bar(x + i * genislik, vals, genislik,
                  label=isim, color=renk, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(x + genislik)
ax.set_xticklabels(['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC'], fontsize=11)
ax.set_ylabel('Skor', fontsize=12)
ax.set_title('Model Performans Karşılaştırması', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{GRAFIK_DIR}/07_model_karsilastirma.png", bbox_inches='tight')
plt.close()
print(f"   ✅ 07_model_karsilastirma.png kaydedildi")

# ══════════════════════════════════════════
# CLASSIFICATION REPORT (metin)
# ══════════════════════════════════════════
print("\n\n" + "=" * 60)
print("  📋 DETAYLI KLASİFİKASYON RAPORU")
print("=" * 60)

with open(f"{OUTPUT_DIR}/classification_report.txt", "w", encoding='utf-8') as rapor:
    rapor.write("MDF KALİTE TAHMİN SİSTEMİ — KLASİFİKASYON RAPORU\n")
    rapor.write("=" * 60 + "\n\n")
    for isim, d in tahminler.items():
        baslik = f"MODEL: {isim}"
        print(f"\n{baslik}")
        print("-" * 40)
        cr = classification_report(
            y_test, d['y_pred'],
            target_names=['Kaliteli (0)', 'Hatalı (1)'],
            digits=4
        )
        print(cr)
        rapor.write(f"{baslik}\n" + "-"*40 + "\n" + cr + "\n\n")
        rapor.write(f"Optimal Threshold: {opt_thr:.3f}\n\n")

print(f"\n💾 classification_report.txt kaydedildi")
print(f"\n✅ Değerlendirme tamamlandı! Sıradaki adım: 04_tahmin_servisi.py")
print("=" * 60)
