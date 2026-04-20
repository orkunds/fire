"""
=============================================================
MDF KALİTE TAHMİN SİSTEMİ - MODÜL 0: TAM PİPELINE ÇALIŞTIRICI
=============================================================
Gebze MDF Üretim Hattı - Fire & Kalite Tahmini
Bitirme Projesi | 2024-2025

Bu script tüm pipeline'ı sırayla çalıştırır.
Tek komutla bütün sistemi kurar.

KULLANIM:
    python 00_pipeline_calistir.py

ADIMLAR:
    1. Veri hazırlama   → 01_veri_hazirlama.py
    2. Model eğitimi    → 02_model_egitim.py
    3. Değerlendirme    → 03_degerlendirme.py
    4. Tahmin servisi   → 04_tahmin_servisi.py (test)
    5. Streamlit        → streamlit run 05_streamlit_app.py
=============================================================
"""

import subprocess
import sys
import os
import time

def calistir(komut, aciklama):
    print(f"\n{'='*60}")
    print(f"  ▶ {aciklama}")
    print(f"{'='*60}")
    baslangic = time.time()
    sonuc = subprocess.run(
        [sys.executable, komut],
        capture_output=False,
        text=True
    )
    sure = time.time() - baslangic
    if sonuc.returncode != 0:
        print(f"\n❌ HATA: {komut} başarısız oldu!")
        sys.exit(1)
    print(f"\n⏱️  Süre: {sure:.1f}s")
    return sonuc

if __name__ == "__main__":
    print("\n" + "🏭" * 30)
    print("  MDF KALİTE TAHMİN SİSTEMİ — TAM PİPELİNE")
    print("🏭" * 30)

    # Veri dosyasını kontrol et
    if not os.path.exists("Gebze_MDF_Zengin_Veri_Seti2004.xlsx"):
        print("\n❌ Hata: 'Gebze_MDF_Zengin_Veri_Seti2004.xlsx' bulunamadı!")
        print("   Excel dosyasını bu klasöre kopyalayın.")
        sys.exit(1)

    calistir("01_veri_hazirlama.py", "ADIM 1/4 — Veri Hazırlama")
    calistir("02_model_egitim.py",   "ADIM 2/4 — Model Eğitimi")
    calistir("03_degerlendirme.py",  "ADIM 3/4 — Değerlendirme ve Grafikler")
    calistir("04_tahmin_servisi.py", "ADIM 4/4 — Tahmin Servisi Testi")

    print("\n" + "="*60)
    print("  ✅ TÜM ADIMLAR TAMAMLANDI!")
    print("="*60)
    print("\n📂 Çıktılar: outputs/ klasöründe")
    print("📊 Grafikler: outputs/grafikler/ klasöründe")
    print("\n🚀 Streamlit'i başlatmak için:")
    print("   streamlit run 05_streamlit_app.py")
    print("="*60)
