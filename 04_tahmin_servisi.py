"""
=============================================================
MDF KALİTE TAHMİN SİSTEMİ - MODÜL 4: TAHMİN SERVİSİ
=============================================================
Gebze MDF Üretim Hattı - Fire & Kalite Tahmini
Bitirme Projesi | 2024-2025

Bu modül:
- Eğitilmiş modelleri kullanarak anlık tahmin yapar
- Canlı sistemden gelen tek satır / toplu veri alır
- Olasılık skoru ve risk seviyesi döndürür
- Streamlit entegrasyonu için API katmanı sağlar
- Önleyici aksiyon önerisi üretir

KULLANIM:
    from tahmin_servisi import MDFKaliteTahminServisi
    servis = MDFKaliteTahminServisi()
    sonuc = servis.tahmin_et(veri_dict)
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"

# ─────────────────────────────────────────
# YARDIMCI: Risk Seviyesi
# ─────────────────────────────────────────
def risk_seviyesi(olasilik: float) -> dict:
    """Fire olasılığına göre renk kodlu risk döndür."""
    if olasilik < 0.30:
        return {"seviye": "DÜŞÜK",   "renk": "green",  "emoji": "✅",
                "mesaj": "Normal üretim devam edebilir."}
    elif olasilik < 0.55:
        return {"seviye": "ORTA",    "renk": "orange", "emoji": "⚠️",
                "mesaj": "Parametreleri gözlemleyin, gerekirse ayar yapın."}
    elif olasilik < 0.75:
        return {"seviye": "YÜKSEK",  "renk": "red",    "emoji": "🔴",
                "mesaj": "Üretim parametrelerini kontrol edin, bakım öneriliyor."}
    else:
        return {"seviye": "KRİTİK",  "renk": "darkred","emoji": "🚨",
                "mesaj": "ÜRETİMİ DURDURUN — Acil bakım ve ayar gerekli!"}

# ─────────────────────────────────────────
# YARDIMCI: Aksiyon Önerisi
# ─────────────────────────────────────────
def aksiyon_onerisi(veri: dict, olasilik: float) -> list:
    """Kritik parametrelere göre aksiyon önerisi üret."""
    oneriler = []

    if veri.get('Sicaklik_Sapmasi_C', 0) > 5.0:
        oneriler.append("🌡️ Pres sıcaklık sapması yüksek → Sıcaklık kontrolcüsünü kalibre edin.")
    if veri.get('Vibrasyon_Degeri', 0) > 4.0:
        oneriler.append("🔧 Vibrasyon yüksek → Makine rulmanlarını kontrol edin.")
    if veri.get('Lif_Nemi_Yuzde', 0) > 10.0:
        oneriler.append("💧 Lif nemi yüksek → Kurutma süresini artırın.")
    if veri.get('Basinc_Varyans_Pct', 0) > 8.0:
        oneriler.append("📊 Basınç varyansı yüksek → Hidrolik sistem kontrolü gerekli.")
    if veri.get('Bakim_Sonrasi_Saat', 0) > 80:
        oneriler.append("🔩 Bakım arası uzun → Periyodik bakım planlanmalı.")
    if veri.get('Makine_Uptime_Pct', 100) < 90:
        oneriler.append("⚙️ Makine verimliliği düşük → Teknik ekibi bilgilendirin.")
    if veri.get('Nem_Varyans_Pct', 0) > 5.0:
        oneriler.append("💨 Nem varyansı yüksek → Ortam nemi kontrolünü artırın.")
    if veri.get('Tutkal_Orani_m3', 0) < 80:
        oneriler.append("🧪 Tutkal oranı düşük → Formülasyonu kontrol edin.")

    if not oneriler:
        oneriler.append("✅ Parametreler normal sınırlar içinde.")

    return oneriler


# ═══════════════════════════════════════════════════════════
# ANA SINIF: MDFKaliteTahminServisi
# ═══════════════════════════════════════════════════════════
class MDFKaliteTahminServisi:
    """
    Eğitilmiş modelleri yükleyip canlı veri ile tahmin yapan servis.

    Özellikler:
    - 3 model (XGBoost, RF, ANN) → ensemble oylama
    - Risk seviyesi ve renk kodu
    - Otomatik feature engineering
    - Toplu tahmin (batch) desteği
    - Tahmin geçmişi loglama
    """

    MAKINE_MAP = {'MDF1': 0, 'MDF2': 1, 'ZMPR': 2}

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self._modelleri_yukle()
        self.tahmin_gecmisi = []
        print("✅ MDFKaliteTahminServisi başlatıldı.")

    def _modelleri_yukle(self):
        """Pickle'dan model ve yardımcı nesneleri yükle."""
        def yukle(dosya):
            with open(f"{self.output_dir}/{dosya}", "rb") as f:
                return pickle.load(f)

        self.model_xgb       = yukle("model_xgboost.pkl")
        self.model_rf        = yukle("model_rf.pkl")
        self.model_ann       = yukle("model_ann.pkl")
        self.scaler          = yukle("scaler.pkl")
        self.le_makine       = yukle("le_makine.pkl")
        self.features        = yukle("feature_names.pkl")
        self.opt_threshold   = yukle("optimal_threshold.pkl")
        print(f"   📦 Modeller yüklendi | Optimal Threshold: {self.opt_threshold:.3f}")

    # ─────────────────────────────────────
    # FEATURE ENGINEERING (tek satır)
    # ─────────────────────────────────────
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Makine encode
        if 'Makine_Kodu_Kisa' in df.columns:
            df['Makine_Enc'] = df['Makine_Kodu_Kisa'].map(self.MAKINE_MAP).fillna(0).astype(int)
        elif 'Makine_Enc' not in df.columns:
            df['Makine_Enc'] = 0

        # Türetilmiş özellikler
        df['Pres_Stres_Endeksi']    = df['Pres_Sicakligi_C'] * df['Pres_Basinci_Bar'] / (df['Konveyor_Hizi_m_dk'] + 1e-5)
        df['Nem_Sicaklik_Etkilesim']= df['Lif_Nemi_Yuzde'] * df['Pres_Sicakligi_C']
        df['Toplam_Varyans']        = df['Sicaklik_Sapmasi_C'] + df['Basinc_Varyans_Pct'] + df['Nem_Varyans_Pct']
        df['Tutkal_Verim']          = df['Tutkal_Orani_m3'] / (df['Hedef_Yogunluk_kgm3'] + 1e-5)
        df['Bakim_Risk']            = df['Bakim_Sonrasi_Saat'] * (1 - df['Makine_Uptime_Pct'] / 100)
        df['Vibrasyon_Uptime']      = df['Vibrasyon_Degeri'] * df['Makine_Uptime_Pct']
        df['Recine_Kalite']         = df['Recine_Kati_Madde_Pct'] * df['Katalizor_Orani_Pct']
        df['Yogunluk_Sapma']        = abs(df['Hedef_Yogunluk_kgm3'] - df['Hedef_Yogunluk_kgm3'].mean())

        # Kalinlik_Kare yoksa hesapla
        if 'Kalinlik_Kare' not in df.columns:
            df['Kalinlik_Kare'] = df['Hedef_Kalinlik_mm'] ** 2

        return df[self.features]

    # ─────────────────────────────────────
    # TEK SATIR TAHMİN
    # ─────────────────────────────────────
    def tahmin_et(self, veri: dict, model: str = "ensemble") -> dict:
        """
        Tek bir üretim kaydını tahmin et.

        Parametreler:
            veri   : dict — proses parametreleri
            model  : 'xgboost' | 'rf' | 'ann' | 'ensemble' (varsayılan)

        Döndürür:
            dict — fire_olasılığı, kalite_durumu, risk, öneriler
        """
        df_satir = pd.DataFrame([veri])
        df_fe    = self._feature_engineering(df_satir)
        df_scl   = self.scaler.transform(df_fe)

        # Model bazlı tahmin
        if model == "xgboost":
            olasilik = float(self.model_xgb.predict_proba(df_fe)[:, 1][0])
        elif model == "rf":
            olasilik = float(self.model_rf.predict_proba(df_fe)[:, 1][0])
        elif model == "ann":
            olasilik = float(self.model_ann.predict_proba(df_scl)[:, 1][0])
        else:  # ensemble — ağırlıklı ortalama
            p_xgb = float(self.model_xgb.predict_proba(df_fe)[:, 1][0])
            p_rf  = float(self.model_rf.predict_proba(df_fe)[:, 1][0])
            p_ann = float(self.model_ann.predict_proba(df_scl)[:, 1][0])
            # XGBoost'a biraz daha ağırlık ver (genellikle en iyi)
            olasilik = 0.40 * p_xgb + 0.35 * p_rf + 0.25 * p_ann

        kalite = int(olasilik >= self.opt_threshold)
        risk   = risk_seviyesi(olasilik)
        oneri  = aksiyon_onerisi(veri, olasilik)

        sonuc = {
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model":            model,
            "fire_olasiligi":   round(olasilik, 4),
            "fire_yuzde":       f"%{olasilik*100:.2f}",
            "kalite_durumu":    kalite,
            "kalite_etiketi":   "🔴 HATALI" if kalite == 1 else "✅ KALİTELİ",
            "risk_seviyesi":    risk["seviye"],
            "risk_rengi":       risk["renk"],
            "risk_emoji":       risk["emoji"],
            "risk_mesaji":      risk["mesaj"],
            "aksiyon_onerileri": oneri,
            # Bireysel model olasılıkları (ensemble ise)
            "model_detay": {
                "xgboost": round(float(self.model_xgb.predict_proba(df_fe)[:, 1][0]), 4),
                "rf":       round(float(self.model_rf.predict_proba(df_fe)[:, 1][0]), 4),
                "ann":      round(float(self.model_ann.predict_proba(df_scl)[:, 1][0]), 4),
            } if model == "ensemble" else None
        }

        # Loglama
        self.tahmin_gecmisi.append({**veri, **{
            "timestamp": sonuc["timestamp"],
            "fire_olasiligi": sonuc["fire_olasiligi"],
            "kalite_durumu": kalite
        }})

        return sonuc

    # ─────────────────────────────────────
    # TOPLU TAHMİN (DataFrame)
    # ─────────────────────────────────────
    def toplu_tahmin(self, df: pd.DataFrame, model: str = "ensemble") -> pd.DataFrame:
        """
        Birden fazla satır için toplu tahmin.
        Canlı sistemden gelen batch veri için kullanılır.
        """
        df_fe  = self._feature_engineering(df)
        df_scl = self.scaler.transform(df_fe)

        if model == "ensemble":
            p_xgb = self.model_xgb.predict_proba(df_fe)[:, 1]
            p_rf  = self.model_rf.predict_proba(df_fe)[:, 1]
            p_ann = self.model_ann.predict_proba(df_scl)[:, 1]
            olasiliklar = 0.40 * p_xgb + 0.35 * p_rf + 0.25 * p_ann
        elif model == "xgboost":
            olasiliklar = self.model_xgb.predict_proba(df_fe)[:, 1]
        elif model == "rf":
            olasiliklar = self.model_rf.predict_proba(df_fe)[:, 1]
        else:
            olasiliklar = self.model_ann.predict_proba(df_scl)[:, 1]

        sonuc_df = df.copy()
        sonuc_df['Fire_Olasiligi']  = olasiliklar.round(4)
        sonuc_df['Fire_Yuzde']      = (olasiliklar * 100).round(2)
        sonuc_df['Kalite_Tahmini']  = (olasiliklar >= self.opt_threshold).astype(int)
        sonuc_df['Risk_Seviyesi']   = sonuc_df['Fire_Olasiligi'].apply(
            lambda p: risk_seviyesi(p)['seviye']
        )
        sonuc_df['Tahmin_Zamani']   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return sonuc_df

    # ─────────────────────────────────────
    # GEÇMİŞ LOGLARI
    # ─────────────────────────────────────
    def gecmis_kaydet(self, dosya_adi: str = None):
        """Tahmin geçmişini CSV'ye kaydet."""
        if not self.tahmin_gecmisi:
            print("⚠️ Kayıt için tahmin geçmişi bulunamadı.")
            return
        if dosya_adi is None:
            dosya_adi = f"{self.output_dir}/tahmin_gecmisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(self.tahmin_gecmisi).to_csv(dosya_adi, index=False)
        print(f"💾 Tahmin geçmişi kaydedildi → {dosya_adi}")

    # ─────────────────────────────────────
    # ÖZET İSTATİSTİK
    # ─────────────────────────────────────
    def ozet_istatistik(self) -> dict:
        """Son tahminlerin özet istatistiklerini döndür."""
        if not self.tahmin_gecmisi:
            return {}
        df = pd.DataFrame(self.tahmin_gecmisi)
        return {
            "toplam_tahmin":    len(df),
            "hatalı_sayisi":    int(df['kalite_durumu'].sum()),
            "fire_orani":       f"%{df['kalite_durumu'].mean()*100:.2f}",
            "ort_fire_olasılığı": f"%{df['fire_olasiligi'].mean()*100:.2f}",
            "max_fire_olasılığı": f"%{df['fire_olasiligi'].max()*100:.2f}",
        }


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM (doğrudan çalıştırıldığında)
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  MDFKaliteTahminServisi — ÖRNEK KULLANIM")
    print("=" * 60)

    servis = MDFKaliteTahminServisi()

    # ── Örnek 1: Normal üretim verisi ──
    normal_veri = {
        'Hedef_Kalinlik_mm': 18,
        'Hedef_Yogunluk_kgm3': 720.0,
        'Lif_Nemi_Yuzde': 8.0,
        'Pres_Sicakligi_C': 210.0,
        'Pres_Basinci_Bar': 330.0,
        'Tutkal_Orani_m3': 100.0,
        'Vibrasyon_Degeri': 2.5,
        'Konveyor_Hizi_m_dk': 17.0,
        'Sicaklik_Sapmasi_C': 2.5,
        'Basinc_Varyans_Pct': 3.0,
        'Nem_Varyans_Pct': 0.4,
        'Degassing_Suresi_sn': 7.5,
        'Pres_Acilma_Hizi': 0.5,
        'Sert_Agac_Orani_Pct': 50.0,
        'Elyaf_Boyutu_Std_Pct': 8.5,
        'Katalizor_Orani_Pct': 0.65,
        'Recine_Kati_Madde_Pct': 11.0,
        'Ortam_Sicakligi_C': 17.0,
        'Ortam_Nemi_Pct': 55.0,
        'Bakim_Sonrasi_Saat': 40.0,
        'Makine_Uptime_Pct': 96.0,
        'Vardiya': 2,
        'Makine_Kodu_Kisa': 'MDF1',
    }

    print("\n🔹 Test 1: Normal Üretim Koşulları")
    sonuc1 = servis.tahmin_et(normal_veri)
    print(f"   Fire Olasılığı : {sonuc1['fire_yuzde']}")
    print(f"   Kalite Durumu  : {sonuc1['kalite_etiketi']}")
    print(f"   Risk           : {sonuc1['risk_emoji']} {sonuc1['risk_seviyesi']}")
    print(f"   Mesaj          : {sonuc1['risk_mesaji']}")
    if sonuc1['model_detay']:
        d = sonuc1['model_detay']
        print(f"   Model Detay    : XGB={d['xgboost']:.4f} | RF={d['rf']:.4f} | ANN={d['ann']:.4f}")

    # ── Örnek 2: Riskli üretim verisi ──
    riskli_veri = {**normal_veri,
        'Sicaklik_Sapmasi_C': 8.5,
        'Vibrasyon_Degeri': 5.2,
        'Lif_Nemi_Yuzde': 12.5,
        'Basinc_Varyans_Pct': 11.0,
        'Bakim_Sonrasi_Saat': 95.0,
        'Makine_Uptime_Pct': 82.0,
    }

    print("\n🔸 Test 2: Riskli Üretim Koşulları")
    sonuc2 = servis.tahmin_et(riskli_veri)
    print(f"   Fire Olasılığı : {sonuc2['fire_yuzde']}")
    print(f"   Kalite Durumu  : {sonuc2['kalite_etiketi']}")
    print(f"   Risk           : {sonuc2['risk_emoji']} {sonuc2['risk_seviyesi']}")
    print(f"   Mesaj          : {sonuc2['risk_mesaji']}")
    print(f"   Öneriler:")
    for o in sonuc2['aksiyon_onerileri']:
        print(f"      {o}")

    # ── Özet ──
    print("\n📊 Oturum Özeti:")
    ozet = servis.ozet_istatistik()
    for k, v in ozet.items():
        print(f"   {k}: {v}")

    servis.gecmis_kaydet()
    print("\n✅ Tahmin servisi hazır — Streamlit'e bağlamak için: 05_streamlit_app.py")
    print("=" * 60)
