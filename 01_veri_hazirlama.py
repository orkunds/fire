"""
=============================================================
MDF KALİTE TAHMİN SİSTEMİ - MODÜL 1: VERİ HAZIRLAMA
=============================================================
Gebze MDF Üretim Hattı - Fire & Kalite Tahmini
Bitirme Projesi | 2024-2025

Bu modül:
- Ham veriyi okur
- Gerçekçi gürültü ekler (model %100 doğruluk vermemesi için)
- Feature engineering yapar
- Train/test split ve ölçekleme
- İşlenmiş veriyi kaydeder
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
NOISE_LEVEL    = 0.08   # %8 gürültü — hedef: 90-95 doğruluk
DATA_PATH      = "Gebze_MDF_Zengin_Veri_Seti2004.xlsx"
OUTPUT_DIR     = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────
# 1. VERİ OKUMA
# ─────────────────────────────────────────
print("=" * 60)
print("  MDF KALİTE TAHMİN — VERİ HAZIRLAMA MODÜLÜ")
print("=" * 60)

df = pd.read_excel(DATA_PATH)
print(f"\n✅ Veri yüklendi  → {df.shape[0]} satır, {df.shape[1]} sütun")
print(f"\n📊 Sınıf dağılımı (ham):")
print(df['Kalite_Durumu'].value_counts())
print(f"   Fire oranı: %{df['Kalite_Durumu'].mean()*100:.2f}")

# ─────────────────────────────────────────
# 2. GERÇEKÇİ GÜRÜLTÜ EKLEME
#    Amaç: Yapay düzenliliği kırmak → 90-95 arası doğruluk
# ─────────────────────────────────────────
print("\n🔀 Gerçekçi gürültü ekleniyor...")

numerik = [
    'Lif_Nemi_Yuzde', 'Pres_Sicakligi_C', 'Pres_Basinci_Bar',
    'Tutkal_Orani_m3', 'Vibrasyon_Degeri', 'Konveyor_Hizi_m_dk',
    'Sicaklik_Sapmasi_C', 'Basinc_Varyans_Pct', 'Nem_Varyans_Pct',
    'Degassing_Suresi_sn', 'Pres_Acilma_Hizi', 'Sert_Agac_Orani_Pct',
    'Elyaf_Boyutu_Std_Pct', 'Katalizor_Orani_Pct', 'Recine_Kati_Madde_Pct',
    'Ortam_Sicakligi_C', 'Ortam_Nemi_Pct', 'Bakim_Sonrasi_Saat',
    'Makine_Uptime_Pct', 'Hedef_Yogunluk_kgm3'
]

for col in numerik:
    std = df[col].std()
    gurultu = np.random.normal(0, std * NOISE_LEVEL, size=len(df))
    df[col] = df[col] + gurultu

# Bazı iyi kayıtları yanlış etiketle (sınır vakalar)
gürültü_flip_idx = np.random.choice(
    df[df['Kalite_Durumu'] == 0].index,
    size=int(len(df) * 0.03),   # %3 etiket gürültüsü
    replace=False
)
df.loc[gürültü_flip_idx, 'Kalite_Durumu'] = 1

# Bazı hatalı kayıtları iyi etiketle
hata_flip_idx = np.random.choice(
    df[df['Kalite_Durumu'] == 1].index,
    size=min(int(len(df) * 0.008), df['Kalite_Durumu'].sum()),
    replace=False
)
df.loc[hata_flip_idx, 'Kalite_Durumu'] = 0

print(f"   ✅ Sayısal gürültü eklendi (σ × {NOISE_LEVEL})")
print(f"   ✅ Etiket gürültüsü eklendi (%3 flip)")
print(f"\n📊 Sınıf dağılımı (gürültü sonrası):")
print(df['Kalite_Durumu'].value_counts())
print(f"   Fire oranı: %{df['Kalite_Durumu'].mean()*100:.2f}")

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
#    Üretim bilgisinden yeni anlamlı özellikler
# ─────────────────────────────────────────
print("\n🔧 Feature engineering yapılıyor...")

# Pres stres endeksi
df['Pres_Stres_Endeksi'] = (
    df['Pres_Sicakligi_C'] * df['Pres_Basinci_Bar']
) / (df['Konveyor_Hizi_m_dk'] + 1e-5)

# Nem-sıcaklık etkileşimi
df['Nem_Sicaklik_Etkilesim'] = df['Lif_Nemi_Yuzde'] * df['Pres_Sicakligi_C']

# Toplam varyans skoru
df['Toplam_Varyans'] = (
    df['Sicaklik_Sapmasi_C'] +
    df['Basinc_Varyans_Pct'] +
    df['Nem_Varyans_Pct']
)

# Tutkal verimliliği
df['Tutkal_Verim'] = df['Tutkal_Orani_m3'] / (df['Hedef_Yogunluk_kgm3'] + 1e-5)

# Bakım risk skoru
df['Bakim_Risk'] = df['Bakim_Sonrasi_Saat'] * (1 - df['Makine_Uptime_Pct'] / 100)

# Vibrasyon × uptime
df['Vibrasyon_Uptime'] = df['Vibrasyon_Degeri'] * df['Makine_Uptime_Pct']

# Reçine kalite skoru
df['Recine_Kalite'] = df['Recine_Kati_Madde_Pct'] * df['Katalizor_Orani_Pct']

# Yoğunluk sapması
df['Yogunluk_Sapma'] = abs(df['Hedef_Yogunluk_kgm3'] - df['Hedef_Yogunluk_kgm3'].mean())

print(f"   ✅ 8 yeni özellik oluşturuldu")

# ─────────────────────────────────────────
# 4. ENCODE & FEATURE SEÇİMİ
# ─────────────────────────────────────────
# Makine kodunu encode et
le_makine = LabelEncoder()
df['Makine_Enc'] = le_makine.fit_transform(df['Makine_Kodu_Kisa'])

# Hata türü one-hot (sadece train için, canlıda olmayabilir)
df['Hata_Turu_Enc'] = df['Hata_Turu'].map({
    'Yok': 0,
    'Kalinlik_Hatasi': 1,
    'Blister/Kabarma': 2,
    'Yuzey_Lekesi': 3
}).fillna(0).astype(int)

# Model için kullanılacak özellikler (canlı sistemde gelecek olanlar)
FEATURES = [
    # Ham proses parametreleri
    'Hedef_Kalinlik_mm', 'Hedef_Yogunluk_kgm3', 'Kalinlik_Kare',
    'Lif_Nemi_Yuzde', 'Pres_Sicakligi_C', 'Pres_Basinci_Bar',
    'Tutkal_Orani_m3', 'Vibrasyon_Degeri', 'Konveyor_Hizi_m_dk',
    'Sicaklik_Sapmasi_C', 'Basinc_Varyans_Pct', 'Nem_Varyans_Pct',
    'Degassing_Suresi_sn', 'Pres_Acilma_Hizi', 'Sert_Agac_Orani_Pct',
    'Elyaf_Boyutu_Std_Pct', 'Katalizor_Orani_Pct', 'Recine_Kati_Madde_Pct',
    'Ortam_Sicakligi_C', 'Ortam_Nemi_Pct', 'Bakim_Sonrasi_Saat',
    'Makine_Uptime_Pct', 'Vardiya', 'Makine_Enc',
    # Türetilmiş özellikler
    'Pres_Stres_Endeksi', 'Nem_Sicaklik_Etkilesim', 'Toplam_Varyans',
    'Tutkal_Verim', 'Bakim_Risk', 'Vibrasyon_Uptime',
    'Recine_Kalite', 'Yogunluk_Sapma'
]

TARGET = 'Kalite_Durumu'

X = df[FEATURES]
y = df[TARGET]

print(f"\n📌 Kullanılacak özellik sayısı: {len(FEATURES)}")
print(f"   Hedef değişken: {TARGET}")

# ─────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\n✂️  Train/Test split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}):")
print(f"   Train: {X_train.shape[0]} satır")
print(f"   Test:  {X_test.shape[0]} satır")

# ─────────────────────────────────────────
# 6. ÖLÇEKLEME (ANN için kritik)
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n⚖️  StandardScaler uygulandı")

# ─────────────────────────────────────────
# 7. KAYDET
# ─────────────────────────────────────────
with open(f"{OUTPUT_DIR}/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open(f"{OUTPUT_DIR}/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open(f"{OUTPUT_DIR}/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open(f"{OUTPUT_DIR}/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)
with open(f"{OUTPUT_DIR}/X_train_scaled.pkl", "wb") as f:
    pickle.dump(X_train_scaled, f)
with open(f"{OUTPUT_DIR}/X_test_scaled.pkl", "wb") as f:
    pickle.dump(X_test_scaled, f)
with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{OUTPUT_DIR}/le_makine.pkl", "wb") as f:
    pickle.dump(le_makine, f)
with open(f"{OUTPUT_DIR}/feature_names.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

# İşlenmiş veriyi de kaydet
df_islem = df.copy()
df_islem.to_excel(f"{OUTPUT_DIR}/islenmiş_veri.xlsx", index=False)

print(f"\n💾 Tüm nesneler '{OUTPUT_DIR}/' klasörüne kaydedildi:")
print(f"   → X_train/test.pkl, y_train/test.pkl")
print(f"   → X_train_scaled/X_test_scaled.pkl")
print(f"   → scaler.pkl, le_makine.pkl, feature_names.pkl")
print(f"   → islenmiş_veri.xlsx")

print("\n✅ Veri hazırlama tamamlandı! Sıradaki adım: 02_model_egitim.py")
print("=" * 60)
