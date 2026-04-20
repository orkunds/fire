"""
=============================================================
MDF KALİTE TAHMİN SİSTEMİ - MODÜL 5: STREAMLİT DASHBOARD
=============================================================
Gebze MDF Üretim Hattı - Fire & Kalite Tahmini
Bitirme Projesi | 2024-2025

ÇALIŞTIRMA:
    streamlit run 05_streamlit_app.py

Bu uygulama:
- Manuel parametre girişi ile anlık tahmin
- Canlı simülasyon modu (otomatik veri akışı)
- CSV yükleme ile toplu tahmin
- Gerçek zamanlı grafik izleme
- Risk skoru ve aksiyon önerileri
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────
st.set_page_config(
    page_title="MDF Kalite Tahmin Sistemi",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CSS TEMA
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .main-title {
        color: #e94560;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .sub-title {
        color: #a8b2d8;
        font-size: 0.95rem;
        margin-top: 5px;
    }
    .metric-card {
        background: #1e2a3a;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #e94560;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .risk-dusuk  { color: #00e676; font-weight: bold; font-size: 1.3rem; }
    .risk-orta   { color: #ffa726; font-weight: bold; font-size: 1.3rem; }
    .risk-yuksek { color: #ef5350; font-weight: bold; font-size: 1.3rem; }
    .risk-kritik { color: #b71c1c; font-weight: bold; font-size: 1.6rem; animation: blink 1s step-start infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    .stProgress > div > div { background-color: #e94560; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# OUTPUT_DIR (göreceli ya da mutlak)
# ─────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ─────────────────────────────────────────
# MODEL YÜKLEME (cache ile hızlı)
# ─────────────────────────────────────────
@st.cache_resource
def modelleri_yukle():
    def yukle(dosya):
        with open(f"{OUTPUT_DIR}/{dosya}", "rb") as f:
            return pickle.load(f)
    return {
        "xgb":       yukle("model_xgboost.pkl"),
        "rf":        yukle("model_rf.pkl"),
        "ann":       yukle("model_ann.pkl"),
        "scaler":    yukle("scaler.pkl"),
        "features":  yukle("feature_names.pkl"),
        "threshold": yukle("optimal_threshold.pkl"),
    }

# ─────────────────────────────────────────
# FEATURE ENGINEERING (streamlit için)
# ─────────────────────────────────────────
MAKINE_MAP = {'MDF1': 0, 'MDF2': 1, 'ZMPR': 2}

def feature_engineering(df):
    df = df.copy()
    if 'Makine_Kodu_Kisa' in df.columns:
        df['Makine_Enc'] = df['Makine_Kodu_Kisa'].map(MAKINE_MAP).fillna(0).astype(int)
    elif 'Makine_Enc' not in df.columns:
        df['Makine_Enc'] = 0

    df['Pres_Stres_Endeksi']    = df['Pres_Sicakligi_C'] * df['Pres_Basinci_Bar'] / (df['Konveyor_Hizi_m_dk'] + 1e-5)
    df['Nem_Sicaklik_Etkilesim']= df['Lif_Nemi_Yuzde'] * df['Pres_Sicakligi_C']
    df['Toplam_Varyans']        = df['Sicaklik_Sapmasi_C'] + df['Basinc_Varyans_Pct'] + df['Nem_Varyans_Pct']
    df['Tutkal_Verim']          = df['Tutkal_Orani_m3'] / (df['Hedef_Yogunluk_kgm3'] + 1e-5)
    df['Bakim_Risk']            = df['Bakim_Sonrasi_Saat'] * (1 - df['Makine_Uptime_Pct'] / 100)
    df['Vibrasyon_Uptime']      = df['Vibrasyon_Degeri'] * df['Makine_Uptime_Pct']
    df['Recine_Kalite']         = df['Recine_Kati_Madde_Pct'] * df['Katalizor_Orani_Pct']
    df['Yogunluk_Sapma']        = abs(df['Hedef_Yogunluk_kgm3'] - df['Hedef_Yogunluk_kgm3'].mean())
    if 'Kalinlik_Kare' not in df.columns:
        df['Kalinlik_Kare'] = df['Hedef_Kalinlik_mm'] ** 2
    return df

def tahmin_yap(veri_dict, modeller, secili_model="ensemble"):
    df = pd.DataFrame([veri_dict])
    df_fe  = feature_engineering(df)
    df_scl = modeller['scaler'].transform(df_fe[modeller['features']])

    p_xgb = float(modeller['xgb'].predict_proba(df_fe[modeller['features']])[:, 1][0])
    p_rf  = float(modeller['rf'].predict_proba(df_fe[modeller['features']])[:, 1][0])
    p_ann = float(modeller['ann'].predict_proba(df_scl)[:, 1][0])

    if secili_model == "XGBoost":        olasilik = p_xgb
    elif secili_model == "Random Forest": olasilik = p_rf
    elif secili_model == "ANN":           olasilik = p_ann
    else:  # ensemble
        olasilik = 0.40 * p_xgb + 0.35 * p_rf + 0.25 * p_ann

    thr = modeller['threshold']
    return {
        "olasilik": olasilik,
        "kalite":   int(olasilik >= thr),
        "p_xgb": p_xgb, "p_rf": p_rf, "p_ann": p_ann,
        "threshold": thr
    }

def risk_renk(olasilik):
    if olasilik < 0.30:   return "risk-dusuk",  "DÜŞÜK",  "✅"
    elif olasilik < 0.55: return "risk-orta",   "ORTA",   "⚠️"
    elif olasilik < 0.75: return "risk-yuksek", "YÜKSEK", "🔴"
    else:                 return "risk-kritik", "KRİTİK", "🚨"

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if 'canli_gecmis' not in st.session_state:
    st.session_state.canli_gecmis = []
if 'canli_aktif' not in st.session_state:
    st.session_state.canli_aktif = False

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <p class="main-title">🏭 MDF Kalite Tahmin Sistemi</p>
    <p class="sub-title">Gebze Tesisi | Yapay Zeka Destekli Fire & Kalite Öngörü Platformu | Bitirme Projesi 2025</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/factory.png", width=80)
    st.title("⚙️ Sistem Ayarları")

    try:
        modeller = modelleri_yukle()
        st.success("✅ Modeller yüklendi")
    except Exception as e:
        st.error(f"❌ Model yüklenemedi: {e}")
        st.info("Önce 01→02→03 pipeline çalıştırın.")
        st.stop()

    st.divider()
    sayfa = st.radio("📂 Sayfa", [
        "🔬 Manuel Tahmin",
        "📡 Canlı İzleme",
        "📤 CSV Toplu Tahmin",
        "📊 Model Bilgisi"
    ])

    st.divider()
    secili_model = st.selectbox("🤖 Aktif Model", ["Ensemble (Önerilen)", "XGBoost", "Random Forest", "ANN"])
    model_adi = secili_model.split(" ")[0] if "Ensemble" not in secili_model else "ensemble"

    st.info(f"⚡ Threshold: **{modeller['threshold']:.3f}**")

# ═══════════════════════════════════════════════════════════
# SAYFA 1: MANUEL TAHMİN
# ═══════════════════════════════════════════════════════════
if "Manuel" in sayfa:
    st.subheader("🔬 Manuel Parametre Girişi ile Tahmin")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔩 Pres Parametreleri**")
        pres_sicaklik = st.slider("Pres Sıcaklığı (°C)", 160.0, 240.0, 210.0, 0.5)
        pres_basinc   = st.slider("Pres Basıncı (Bar)", 280.0, 380.0, 330.0, 1.0)
        konveyor_hiz  = st.slider("Konveyör Hızı (m/dk)", 14.0, 22.0, 17.0, 0.1)
        degassing     = st.slider("Degassing Süresi (sn)", 5.0, 12.0, 7.5, 0.1)
        pres_acilma   = st.slider("Pres Açılma Hızı", 0.3, 0.8, 0.5, 0.01)

    with col2:
        st.markdown("**🌿 Lif & Reçine**")
        lif_nemi      = st.slider("Lif Nemi (%)", 5.0, 14.0, 8.0, 0.1)
        tutkal        = st.slider("Tutkal Oranı (m³)", 70.0, 130.0, 100.0, 0.5)
        sert_agac     = st.slider("Sert Ağaç Oranı (%)", 40.0, 65.0, 50.0, 0.5)
        elyaf_boyut   = st.slider("Elyaf Boyutu Std (%)", 5.0, 14.0, 8.5, 0.1)
        katalizor     = st.slider("Katalizör Oranı (%)", 0.4, 0.9, 0.65, 0.01)
        recine        = st.slider("Reçine Katı Madde (%)", 8.0, 14.0, 11.0, 0.1)

    with col3:
        st.markdown("**🏭 Makine & Ortam**")
        vibrasyon     = st.slider("Vibrasyon Değeri", 1.0, 6.0, 2.5, 0.1)
        sicaklik_sap  = st.slider("Sıcaklık Sapması (°C)", 1.0, 10.0, 2.5, 0.1)
        basinc_var    = st.slider("Basınç Varyans (%)", 0.5, 15.0, 3.0, 0.5)
        nem_var       = st.slider("Nem Varyans (%)", 0.1, 3.0, 0.4, 0.05)
        bakim_saat    = st.slider("Bakım Sonrası (Saat)", 0.0, 120.0, 40.0, 1.0)
        makine_uptime = st.slider("Makine Uptime (%)", 75.0, 100.0, 96.0, 0.5)

    col_a, col_b = st.columns(2)
    with col_a:
        hedef_kalinlik = st.selectbox("Hedef Kalınlık (mm)", [8, 12, 16, 18, 20, 25], index=3)
        hedef_yogunluk = st.number_input("Hedef Yoğunluk (kg/m³)", 650.0, 800.0, 720.0, 1.0)
    with col_b:
        makine = st.selectbox("Makine", ['MDF1', 'MDF2', 'ZMPR'])
        vardiya = st.selectbox("Vardiya", [1, 2, 3])
        ortam_sicaklik = st.slider("Ortam Sıcaklığı (°C)", 10.0, 30.0, 17.0, 0.5)
        ortam_nem = st.slider("Ortam Nemi (%)", 35.0, 75.0, 55.0, 1.0)

    veri = {
        'Hedef_Kalinlik_mm': hedef_kalinlik,
        'Hedef_Yogunluk_kgm3': hedef_yogunluk,
        'Kalinlik_Kare': hedef_kalinlik ** 2,
        'Lif_Nemi_Yuzde': lif_nemi,
        'Pres_Sicakligi_C': pres_sicaklik,
        'Pres_Basinci_Bar': pres_basinc,
        'Tutkal_Orani_m3': tutkal,
        'Vibrasyon_Degeri': vibrasyon,
        'Konveyor_Hizi_m_dk': konveyor_hiz,
        'Sicaklik_Sapmasi_C': sicaklik_sap,
        'Basinc_Varyans_Pct': basinc_var,
        'Nem_Varyans_Pct': nem_var,
        'Degassing_Suresi_sn': degassing,
        'Pres_Acilma_Hizi': pres_acilma,
        'Sert_Agac_Orani_Pct': sert_agac,
        'Elyaf_Boyutu_Std_Pct': elyaf_boyut,
        'Katalizor_Orani_Pct': katalizor,
        'Recine_Kati_Madde_Pct': recine,
        'Ortam_Sicakligi_C': ortam_sicaklik,
        'Ortam_Nemi_Pct': ortam_nem,
        'Bakim_Sonrasi_Saat': bakim_saat,
        'Makine_Uptime_Pct': makine_uptime,
        'Vardiya': vardiya,
        'Makine_Kodu_Kisa': makine,
    }

    if st.button("🚀 TAHMİN ET", type="primary", use_container_width=True):
        with st.spinner("Tahmin hesaplanıyor..."):
            sonuc = tahmin_yap(veri, modeller, model_adi)
            olasilik = sonuc['olasilik']
            css_cls, risk_txt, emoji = risk_renk(olasilik)

        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Fire Olasılığı", f"%{olasilik*100:.2f}")
        with r2:
            st.metric("Karar (Threshold)", f"{modeller['threshold']:.2f}")
        with r3:
            durum = "🔴 HATALI ÜRÜ" if sonuc['kalite'] == 1 else "✅ KALİTELİ"
            st.metric("Kalite Durumu", durum)
        with r4:
            st.markdown(f"<p class='{css_cls}'>{emoji} {risk_txt}</p>", unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=olasilik * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fire Riski (%)", 'font': {'size': 20}},
            delta={'reference': 30, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkred" if olasilik > 0.75 else "red" if olasilik > 0.55 else "orange" if olasilik > 0.30 else "green"},
                'steps': [
                    {'range': [0, 30],  'color': '#e8f5e9'},
                    {'range': [30, 55], 'color': '#fff3e0'},
                    {'range': [55, 75], 'color': '#fce4ec'},
                    {'range': [75, 100],'color': '#b71c1c'},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': modeller['threshold'] * 100
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Model detayı
        if model_adi == "ensemble":
            st.markdown("**🤖 Model Detayı (Ensemble)**")
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("XGBoost",      f"%{sonuc['p_xgb']*100:.2f}", delta=None)
            dc2.metric("Random Forest", f"%{sonuc['p_rf']*100:.2f}",  delta=None)
            dc3.metric("ANN",           f"%{sonuc['p_ann']*100:.2f}", delta=None)

        # Oturum geçmişi
        st.session_state.canli_gecmis.append({
            "zaman": datetime.now().strftime("%H:%M:%S"),
            "fire_olasiligi": olasilik,
            "kalite": sonuc['kalite']
        })

# ═══════════════════════════════════════════════════════════
# SAYFA 2: CANLI İZLEME
# ═══════════════════════════════════════════════════════════
elif "Canlı" in sayfa:
    st.subheader("📡 Canlı Üretim İzleme Simülasyonu")
    st.info("Bu mod, canlı sistemden gelen veriyi simüle eder. Gerçek entegrasyonda bu veriler OPC-UA/MQTT ile otomatik gelir.")

    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("▶️ Simülasyonu Başlat", type="primary"):
        st.session_state.canli_aktif = True
    if col_btn2.button("⏹️ Durdur"):
        st.session_state.canli_aktif = False

    gecmis_placeholder = st.empty()
    grafik_placeholder = st.empty()

    if st.session_state.canli_aktif:
        for _ in range(30):  # 30 simülasyon adımı
            if not st.session_state.canli_aktif:
                break

            # Rastgele veri üret (gerçek sistemde sensor verisi gelir)
            is_risky = np.random.random() < 0.15
            veri_sim = {
                'Hedef_Kalinlik_mm': np.random.choice([12, 16, 18]),
                'Hedef_Yogunluk_kgm3': np.random.uniform(690, 750),
                'Kalinlik_Kare': 324,
                'Lif_Nemi_Yuzde': np.random.uniform(10, 14) if is_risky else np.random.uniform(6, 10),
                'Pres_Sicakligi_C': np.random.uniform(185, 200) if is_risky else np.random.uniform(200, 225),
                'Pres_Basinci_Bar': np.random.uniform(300, 320) if is_risky else np.random.uniform(320, 350),
                'Tutkal_Orani_m3': np.random.uniform(85, 100),
                'Vibrasyon_Degeri': np.random.uniform(4, 6) if is_risky else np.random.uniform(1.5, 3.5),
                'Konveyor_Hizi_m_dk': np.random.uniform(15, 20),
                'Sicaklik_Sapmasi_C': np.random.uniform(6, 10) if is_risky else np.random.uniform(1, 4),
                'Basinc_Varyans_Pct': np.random.uniform(8, 12) if is_risky else np.random.uniform(1, 5),
                'Nem_Varyans_Pct': np.random.uniform(1, 3),
                'Degassing_Suresi_sn': np.random.uniform(6, 10),
                'Pres_Acilma_Hizi': np.random.uniform(0.4, 0.7),
                'Sert_Agac_Orani_Pct': np.random.uniform(45, 55),
                'Elyaf_Boyutu_Std_Pct': np.random.uniform(7, 10),
                'Katalizor_Orani_Pct': np.random.uniform(0.55, 0.75),
                'Recine_Kati_Madde_Pct': np.random.uniform(10, 12),
                'Ortam_Sicakligi_C': np.random.uniform(15, 22),
                'Ortam_Nemi_Pct': np.random.uniform(45, 65),
                'Bakim_Sonrasi_Saat': np.random.uniform(20, 90) if is_risky else np.random.uniform(5, 60),
                'Makine_Uptime_Pct': np.random.uniform(78, 88) if is_risky else np.random.uniform(92, 100),
                'Vardiya': np.random.choice([1, 2, 3]),
                'Makine_Kodu_Kisa': np.random.choice(['MDF1', 'MDF2', 'ZMPR']),
            }

            sonuc_sim = tahmin_yap(veri_sim, modeller, "ensemble")
            olasilik_sim = sonuc_sim['olasilik']
            _, risk_txt_sim, emoji_sim = risk_renk(olasilik_sim)

            st.session_state.canli_gecmis.append({
                "zaman": datetime.now().strftime("%H:%M:%S"),
                "fire_olasiligi": olasilik_sim,
                "kalite": sonuc_sim['kalite'],
                "makine": veri_sim['Makine_Kodu_Kisa'],
                "risk": risk_txt_sim
            })

            # Son 20 kaydı göster
            df_gec = pd.DataFrame(st.session_state.canli_gecmis[-20:])

            with gecmis_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Son Fire Olasılığı", f"%{olasilik_sim*100:.2f}")
                c2.metric("Risk", f"{emoji_sim} {risk_txt_sim}")
                c3.metric("Toplam Kayıt", len(st.session_state.canli_gecmis))

            with grafik_placeholder.container():
                fig_canli = go.Figure()
                fig_canli.add_trace(go.Scatter(
                    x=df_gec['zaman'],
                    y=df_gec['fire_olasiligi'] * 100,
                    mode='lines+markers',
                    name='Fire Olasılığı (%)',
                    line=dict(color='#e94560', width=2),
                    marker=dict(
                        size=10,
                        color=df_gec['kalite'].map({0: 'green', 1: 'red'}),
                        symbol=df_gec['kalite'].map({0: 'circle', 1: 'x'})
                    )
                ))
                fig_canli.add_hline(
                    y=modeller['threshold']*100,
                    line_dash="dash", line_color="orange", line_width=2,
                    annotation_text=f"Threshold (%{modeller['threshold']*100:.0f})"
                )
                fig_canli.update_layout(
                    title="Anlık Fire Riski Takibi",
                    xaxis_title="Zaman",
                    yaxis_title="Fire Olasılığı (%)",
                    yaxis_range=[0, 100],
                    height=400,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font_color='white'
                )
                st.plotly_chart(fig_canli, use_container_width=True)

            time.sleep(1.5)

# ═══════════════════════════════════════════════════════════
# SAYFA 3: CSV TOPLU TAHMİN
# ═══════════════════════════════════════════════════════════
elif "CSV" in sayfa:
    st.subheader("📤 CSV Dosyası ile Toplu Tahmin")
    yukle_dosya = st.file_uploader("CSV veya Excel dosyası yükleyin", type=['csv', 'xlsx'])

    if yukle_dosya:
        if yukle_dosya.name.endswith('.csv'):
            df_yukle = pd.read_csv(yukle_dosya)
        else:
            df_yukle = pd.read_excel(yukle_dosya)

        st.info(f"✅ {len(df_yukle)} satır yüklendi")
        st.dataframe(df_yukle.head())

        if st.button("🚀 Toplu Tahmin Yap"):
            with st.spinner("Tahminler hesaplanıyor..."):
                df_fe  = feature_engineering(df_yukle)
                df_scl = modeller['scaler'].transform(df_fe[modeller['features']])

                p_xgb = modeller['xgb'].predict_proba(df_fe[modeller['features']])[:, 1]
                p_rf  = modeller['rf'].predict_proba(df_fe[modeller['features']])[:, 1]
                p_ann = modeller['ann'].predict_proba(df_scl)[:, 1]
                proba = 0.40 * p_xgb + 0.35 * p_rf + 0.25 * p_ann

                df_sonuc = df_yukle.copy()
                df_sonuc['Fire_Olasiligi_Pct'] = (proba * 100).round(2)
                df_sonuc['Kalite_Tahmini']     = (proba >= modeller['threshold']).astype(int)
                df_sonuc['Risk_Seviyesi']       = pd.cut(
                    proba, bins=[0, 0.30, 0.55, 0.75, 1.0],
                    labels=['DÜŞÜK', 'ORTA', 'YÜKSEK', 'KRİTİK']
                )

            st.success(f"✅ Tahmin tamamlandı!")
            st.dataframe(df_sonuc[['Fire_Olasiligi_Pct', 'Kalite_Tahmini', 'Risk_Seviyesi']].head(50))

            fire_sayisi = df_sonuc['Kalite_Tahmini'].sum()
            st.metric("Hatalı Tahmin Sayısı", fire_sayisi)
            st.metric("Fire Oranı", f"%{fire_sayisi/len(df_sonuc)*100:.2f}")

            csv_out = df_sonuc.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Sonuçları İndir (CSV)", csv_out, "tahmin_sonuclari.csv", "text/csv")

# ═══════════════════════════════════════════════════════════
# SAYFA 4: MODEL BİLGİSİ
# ═══════════════════════════════════════════════════════════
elif "Model" in sayfa:
    st.subheader("📊 Model Performans Bilgisi")

    try:
        karsilastirma = pd.read_csv(f"{OUTPUT_DIR}/model_karsilastirma.csv", index_col=0)
        st.dataframe(karsilastirma.style.highlight_max(axis=0, color='lightgreen'))

        metrikler  = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        fig_bar = px.bar(
            karsilastirma[metrikler].reset_index().melt(id_vars='index'),
            x='variable', y='value', color='index', barmode='group',
            title='Model Karşılaştırması',
            labels={'variable': 'Metrik', 'value': 'Skor', 'index': 'Model'},
            color_discrete_map={
                'XGBoost': '#1f77b4', 'RandomForest': '#2ca02c', 'ANN': '#d62728'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    except FileNotFoundError:
        st.warning("Model değerlendirme sonuçları bulunamadı. 03_degerlendirme.py çalıştırın.")

    st.info(f"🎯 Optimal Threshold: **{modeller['threshold']:.3f}** (F1 optimize)")
    st.info(f"📌 Özellik Sayısı: **{len(modeller['features'])}**")
    st.code("\n".join(modeller['features']), language="text")
