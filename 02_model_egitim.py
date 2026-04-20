"""
=============================================================
MDF KALİTE TAHMİN SİSTEMİ - MODÜL 2: MODEL EĞİTİMİ
=============================================================
Gebze MDF Üretim Hattı - Fire & Kalite Tahmini
Bitirme Projesi | 2024-2025

Bu modül:
- XGBoost modeli eğitir ve ayarlar (GridSearchCV)
- Random Forest modeli eğitir ve ayarlar
- ANN (MLP) modeli eğitir
- SMOTE ile sınıf dengesizliğini giderir
- Her modeli kaydeder
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
from sklearn.utils import resample

# XGBoost yoksa GradientBoosting kullan (sklearn built-in)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("ℹ️  XGBoost kurulu değil → sklearn GradientBoosting kullanılıyor")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# VERİ YÜKLEME
# ─────────────────────────────────────────
print("=" * 60)
print("  MDF KALİTE TAHMİN — MODEL EĞİTİMİ MODÜLÜ")
print("=" * 60)

def yukle(dosya):
    with open(f"{OUTPUT_DIR}/{dosya}", "rb") as f:
        return pickle.load(f)

X_train        = yukle("X_train.pkl")
X_test         = yukle("X_test.pkl")
y_train        = yukle("y_train.pkl")
y_test         = yukle("y_test.pkl")
X_train_scaled = yukle("X_train_scaled.pkl")
X_test_scaled  = yukle("X_test_scaled.pkl")
features       = yukle("feature_names.pkl")

print(f"\n✅ Veriler yüklendi")
print(f"   Train sınıf dağılımı: {dict(y_train.value_counts())}")
print(f"   Test  sınıf dağılımı: {dict(y_test.value_counts())}")

# ─────────────────────────────────────────
# SMOTE — Sınıf Dengesizliği Giderme (manuel oversampling)
# ─────────────────────────────────────────
print("\n⚖️  Oversampling uygulanıyor (azınlık sınıfı artırılıyor)...")

def manuel_oversample(X, y, ratio=0.4, random_state=42):
    """SMOTE yerine sklearn resample ile minority oversampling."""
    from sklearn.utils import resample
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y_series = pd.Series(y) if not isinstance(y, pd.Series) else y

    minority_mask = y_series == 1
    X_min = X[minority_mask.values]
    X_maj = X[~minority_mask.values]
    y_min = y_series[minority_mask]
    y_maj = y_series[~minority_mask]

    target_n = int(len(X_maj) * ratio)
    X_min_up, y_min_up = resample(X_min, y_min, n_samples=target_n, random_state=random_state)

    X_bal = pd.concat([X_maj, X_min_up]).reset_index(drop=True)
    y_bal = pd.concat([y_maj, y_min_up]).reset_index(drop=True)
    return X_bal, y_bal

X_train_sm, y_train_sm             = manuel_oversample(X_train, y_train)
X_train_scaled_sm, y_train_sm_sc   = manuel_oversample(
    pd.DataFrame(X_train_scaled), y_train
)
X_train_scaled_sm = X_train_scaled_sm.values
y_train_sm_scaled = y_train_sm_sc

print(f"   SMOTE sonrası train: {dict(pd.Series(y_train_sm).value_counts())}")

# ─────────────────────────────────────────
# YARDIMCI: Metrik özeti
# ─────────────────────────────────────────
def metrik_ozeti(isim, y_gercek, y_tahmin, y_proba=None):
    acc  = accuracy_score(y_gercek, y_tahmin)
    f1   = f1_score(y_gercek, y_tahmin, zero_division=0)
    prec = precision_score(y_gercek, y_tahmin, zero_division=0)
    rec  = recall_score(y_gercek, y_tahmin, zero_division=0)
    auc  = roc_auc_score(y_gercek, y_proba) if y_proba is not None else None

    print(f"\n{'─'*45}")
    print(f"  📊 {isim} — Test Sonuçları")
    print(f"{'─'*45}")
    print(f"  Accuracy  : %{acc*100:.2f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    if auc:
        print(f"  ROC-AUC   : {auc:.4f}")
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "roc_auc": auc}

sonuclar = {}

# ══════════════════════════════════════════
# MODEL 1: XGBoost (veya GradientBoosting)
# ══════════════════════════════════════════
if XGBOOST_AVAILABLE:
    print("\n\n🚀 [1/3] XGBoost eğitimi başlıyor...")
    xgb_params = {
        'n_estimators':     [200, 400],
        'max_depth':        [4, 6],
        'learning_rate':    [0.05, 0.1],
        'subsample':        [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3],
    }
    xgb_base = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=len(y_train_sm[y_train_sm==0]) / max(len(y_train_sm[y_train_sm==1]), 1),
        random_state=42, n_jobs=-1
    )
    MODEL1_ISIM = "XGBoost"
else:
    print("\n\n🚀 [1/3] GradientBoosting eğitimi başlıyor (XGBoost yerine)...")
    xgb_params = {
        'n_estimators':  [200, 300],
        'max_depth':     [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample':     [0.8, 1.0],
        'min_samples_split': [2, 5],
    }
    xgb_base = GradientBoostingClassifier(random_state=42)
    MODEL1_ISIM = "XGBoost"  # Streamlit'te isim aynı kalsın

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("   GridSearchCV çalışıyor (bu ~1-2 dk sürebilir)...")
xgb_grid = GridSearchCV(
    xgb_base, xgb_params,
    cv=cv, scoring='f1',
    n_jobs=-1, verbose=0
)
xgb_grid.fit(X_train_sm, y_train_sm)

best_xgb = xgb_grid.best_estimator_
print(f"   ✅ En iyi parametreler: {xgb_grid.best_params_}")

y_pred_xgb   = best_xgb.predict(X_test)
y_proba_xgb  = best_xgb.predict_proba(X_test)[:, 1]
sonuclar['XGBoost'] = metrik_ozeti(MODEL1_ISIM, y_test, y_pred_xgb, y_proba_xgb)

# Kaydet
with open(f"{OUTPUT_DIR}/model_xgboost.pkl", "wb") as f:
    pickle.dump(best_xgb, f)
print(f"\n   💾 model_xgboost.pkl kaydedildi")

# ══════════════════════════════════════════
# MODEL 2: Random Forest
# ══════════════════════════════════════════
print("\n\n🌲 [2/3] Random Forest eğitimi başlıyor...")

rf_params = {
    'n_estimators': [200, 400],
    'max_depth':    [None, 15, 25],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

print("   GridSearchCV çalışıyor...")
rf_grid = GridSearchCV(
    rf_base, rf_params,
    cv=cv, scoring='f1',
    n_jobs=-1, verbose=0
)
rf_grid.fit(X_train_sm, y_train_sm)

best_rf = rf_grid.best_estimator_
print(f"   ✅ En iyi parametreler: {rf_grid.best_params_}")

y_pred_rf   = best_rf.predict(X_test)
y_proba_rf  = best_rf.predict_proba(X_test)[:, 1]
sonuclar['RandomForest'] = metrik_ozeti("Random Forest", y_test, y_pred_rf, y_proba_rf)

# Kaydet
with open(f"{OUTPUT_DIR}/model_rf.pkl", "wb") as f:
    pickle.dump(best_rf, f)
print(f"\n   💾 model_rf.pkl kaydedildi")

# ══════════════════════════════════════════
# MODEL 3: ANN (MLP)
# ══════════════════════════════════════════
print("\n\n🧠 [3/3] ANN (MLP) eğitimi başlıyor...")

ann_params = {
    'hidden_layer_sizes': [(128, 64, 32), (256, 128, 64), (64, 32)],
    'activation':         ['relu', 'tanh'],
    'alpha':              [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.005],
    'dropout':            [None]   # sklearn MLP'de yok, sadece yapı arama
}

# ANN için basit grid (sklearn MLP dropout desteklemez)
ann_param_grid = {
    'hidden_layer_sizes': [(128, 64, 32), (256, 128, 64), (64, 32)],
    'activation':         ['relu', 'tanh'],
    'alpha':              [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.005],
}

ann_base = MLPClassifier(
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    solver='adam',
    batch_size=64
)

print("   GridSearchCV çalışıyor...")
ann_grid = GridSearchCV(
    ann_base, ann_param_grid,
    cv=cv, scoring='f1',
    n_jobs=-1, verbose=0
)
ann_grid.fit(X_train_scaled_sm, y_train_sm_scaled)

best_ann = ann_grid.best_estimator_
print(f"   ✅ En iyi parametreler: {ann_grid.best_params_}")

y_pred_ann   = best_ann.predict(X_test_scaled)
y_proba_ann  = best_ann.predict_proba(X_test_scaled)[:, 1]
sonuclar['ANN'] = metrik_ozeti("ANN (MLP)", y_test, y_pred_ann, y_proba_ann)

# Kaydet
with open(f"{OUTPUT_DIR}/model_ann.pkl", "wb") as f:
    pickle.dump(best_ann, f)
print(f"\n   💾 model_ann.pkl kaydedildi")

# ══════════════════════════════════════════
# KARŞILAŞTIRMA TABLOSU
# ══════════════════════════════════════════
print("\n\n" + "=" * 60)
print("  📋 MODEL KARŞILAŞTIRMA TABLOSU")
print("=" * 60)

karsilastirma = pd.DataFrame(sonuclar).T
karsilastirma['accuracy_pct'] = (karsilastirma['accuracy'] * 100).round(2)
karsilastirma['f1']           = karsilastirma['f1'].round(4)
karsilastirma['roc_auc']      = karsilastirma['roc_auc'].round(4)

print(karsilastirma[['accuracy_pct', 'f1', 'precision', 'recall', 'roc_auc']].to_string())

# En iyi modeli belirle
en_iyi = karsilastirma['roc_auc'].idxmax()
print(f"\n🏆 En iyi model (ROC-AUC): {en_iyi}")

# Karşılaştırma tablosunu kaydet
karsilastirma.to_csv(f"{OUTPUT_DIR}/model_karsilastirma.csv")
print(f"   💾 model_karsilastirma.csv kaydedildi")

# En iyi modeli işaretle
with open(f"{OUTPUT_DIR}/en_iyi_model.txt", "w") as f:
    f.write(en_iyi)

print("\n✅ Model eğitimi tamamlandı! Sıradaki adım: 03_degerlendirme.py")
print("=" * 60)
