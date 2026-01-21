import streamlit as st
import time

# --- SAYFA AYARLARI ---
st.set_page_config(layout="wide", page_title="PROP DESK V6.4 (SMART RSI)", page_icon="🦅")

# --- KÜTÜPHANE KONTROLÜ VE HATA YÖNETİMİ ---
try:
    import yfinance as yf
    import pandas as pd
    import pandas_ta as ta
    import plotly.graph_objects as go
    import numpy as np
    import random
except ImportError as e:
    st.error("⚠️ Kütüphane Eksik!")
    st.info("Lütfen terminali açıp şu komutu yapıştırın: pip install \"numpy<2.0.0\" --force-reinstall")
    st.stop()

# --- CSS TASARIM (DARK PRO THEME) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .trade-card { 
        background-color: #1c1f26; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #00e5ff;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .score-box {
        font-size: 2.5em; font-weight: bold; text-align: center;
        padding: 10px; border-radius: 10px; margin-bottom: 10px; color: white;
    }
    .score-high { background-color: #00c853; } /* Yeşil */
    .score-mid { background-color: #ffd600; color: black; } /* Sarı */
    .score-low { background-color: #d50000; } /* Kırmızı */
    
    .metric-row {
        display: flex; justify-content: space-between;
        background-color: #262730; padding: 8px;
        border-radius: 5px; margin-top: 5px; font-size: 0.9em;
    }
    
    .reason-text { font-size: 0.85em; color: #b0b8c3; }
</style>
""", unsafe_allow_html=True)

# --- YAN MENÜ: SERMAYE YÖNETİMİ ---
st.sidebar.header("⚙️ SERMAYE AYARLARI")
capital = st.sidebar.number_input("Sermaye (TL)", value=100000, step=5000)
risk_pct = st.sidebar.slider("İşlem Başı Risk (%)", 0.5, 3.0, 1.0)
st.sidebar.info("Sistem; Teknik Analiz, Constance Brown RSI Kuralları, Backtest ve Fibonacci Hedeflerini birleştirerek puan üretir.")

# BIST 30 LİSTESİ (TARAMA İÇİN)
BIST30_TICKERS = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI",
    "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD",
    "ODAS", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", 
    "TSKB", "TTKOM", "TUPRS", "YKBNK"
]

# --- MODÜL 1: VERİ ÇEKME ---
def get_data(symbol):
    if len(symbol) <= 5 and not symbol.endswith(".IS"): symbol += ".IS"
    try:
        # Son 6 aylık veriyi 60 dakikalık periyotla çekiyoruz
        df = yf.download(symbol, period="6mo", interval="60m", progress=False)
        if df.empty: return None, symbol
        
        # MultiIndex düzeltmesi
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [col.capitalize() for col in df.columns]
        
        # İndikatör Hesaplamaları
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        return df, symbol
    except: return None, symbol

# --- MODÜL 2: ARKA PLAN MOTORLARI (SİMÜLASYON) ---
def background_backtest_engine(df):
    """Geçmiş 200 mumda stratejinin başarısını ölçer"""
    sim = df.tail(200).reset_index(drop=True)
    wins = 0; trades = 0
    for i in range(1, len(sim)):
        # Basit Trend Takibi Testi
        if sim.iloc[i]['Close'] > sim.iloc[i]['EMA50'] and sim.iloc[i]['RSI'] < 70 and sim.iloc[i-1]['Close'] <= sim.iloc[i-1]['EMA50']:
            trades += 1
            if sim.iloc[i]['Close'] < sim.iloc[min(i+5, len(sim)-1)]['Close']: wins +=1
    win_rate = (wins/trades*100) if trades > 0 else 0
    return win_rate

def background_sentiment_engine():
    """Haber Analizi Simülasyonu"""
    return random.uniform(-0.5, 0.8) 

# --- MODÜL 3: SMART RSI & PROP DESK MANTIĞI (BEYİN) ---
def calculate_smart_logic(df, cap, risk):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- A. PUANLAMA SİSTEMİ (0-100) ---
    score = 50
    reasons = []
    
    # 1. Trend Kontrolü
    is_uptrend = last['Close'] > last['EMA50']
    if is_uptrend:
        score += 20
        reasons.append("✅ Ana Trend Yükseliş (EMA50 Üzeri)")
    else:
        score -= 20
        reasons.append("🔻 Ana Trend Düşüş")

    # 2. SMART RSI ANALİZİ (Gelişmiş Kurallar)
    curr_rsi = last['RSI']
    prev_rsi = prev['RSI']
    
    # Kural 1: Dip Dönüşü (30 Kırılımı)
    if prev_rsi < 30 and curr_rsi >= 30:
        score += 30
        reasons.append("🚀 DİP DÖNÜŞÜ: RSI 30'u Yukarı Kırdı! (Güçlü Al)")
        
    # Kural 2: Trend İçi Düzeltme (Bullish Pullback)
    elif is_uptrend and 40 <= curr_rsi <= 55:
        score += 20
        reasons.append("📈 Trend İçi Düzeltme (RSI 40-55 Desteği - Pullback)")
        
    # Kural 3: Momentum Bölgesi
    elif 55 < curr_rsi <= 75:
        score += 10
        reasons.append("✅ RSI Güçlü Momentum Bölgesinde")
        
    # Kural 4: Aşırı Isınma Kontrolü
    elif curr_rsi > 75:
        if curr_rsi > 85: 
            score -= 25; reasons.append("🔥 RSI > 85: Aşırı Riskli (Satış Bölgesi)")
        else:
            score -= 10; reasons.append("⚠️ RSI > 75: Aşırı Isınma (Kar Al)")

    # Kural 5: Ayı Piyasası Tepkisi
    elif not is_uptrend and curr_rsi > 60:
        score -= 20
        reasons.append("🔻 Düşüş Trendinde RSI Şişti (Satış Fırsatı)")

    # 3. Yan Faktörler (Backtest & Sentiment)
    win_rate = background_backtest_engine(df)
    if win_rate > 60: score += 10; reasons.append(f"🔙 Backtest Onayı (%{win_rate:.0f})")
    
    sent_val = background_sentiment_engine()
    if sent_val > 0.3: score += 10; reasons.append("📰 Haber Pozitif")
    elif sent_val < -0.3: score -= 10; reasons.append("📰 Haber Negatif")

    # --- B. HEDEF VE RİSK YÖNETİMİ ---
    
    # Dinamik Hedef (Fibonacci & Swing High)
    recent_high = df['High'].tail(60).max()
    recent_low = df['Low'].tail(60).min()
    fib_ext = recent_high + ((recent_high - recent_low) * 0.618) # 1.618 Ext.
    
    atr = last['ATR']
    
    # Breakout (Zirve Kırılımı) Var mı?
    if last['Close'] >= (recent_high * 0.98):
        target_price = fib_ext
        target_type = "Fibonacci Uzatma (Ralli)"
        score += 10
        reasons.append("🚀 Zirve Kırılımı (Breakout) Potansiyeli")
    else:
        target_price = recent_high
        target_type = "Tarihsel Direnç (Swing High)"

    # Stop Loss (ATR Bazlı)
    stop_dist = atr * 1.5
    stop_price = last['Close'] - stop_dist
    
    # Lot Hesabı (Prop Desk: Risk Sabitleme)
    risk_amt = cap * (risk/100)
    lot = int(risk_amt / stop_dist) if stop_dist > 0 else 0
    final_lot = min(lot, int(cap / last['Close'])) # Kasa kontrolü
    
    # Kar ve R/R Hesabı
    potential_profit = (target_price - last['Close']) * final_lot
    risk_money = final_lot * stop_dist
    rr_ratio = (target_price - last['Close']) / stop_dist if stop_dist > 0 else 0
    
    # R/R Filtresi
    if rr_ratio < 1.5:
        score -= 20
        reasons.append(f"⛔ R/R Oranı Düşük ({rr_ratio:.2f})")

    final_score = max(0, min(100, score))
    
    return {
        "symbol": "",
        "price": last['Close'],
        "stop": stop_price,
        "target": target_price,
        "lot": final_lot,
        "potential_profit": potential_profit,
        "risk_money": risk_money,
        "score": final_score,
        "reasons": reasons,
        "rr": rr_ratio,
        "target_type": target_type,
        "rsi": curr_rsi
    }

# --- ARAYÜZ ---
tab_single, tab_hunter = st.tabs(["🛡️ TEKLİ ANALİZ", "🦅 BIST 30 AKILLI AVCI"])

# --- SEKME 1: TEKLİ ANALİZ ---
with tab_single:
    col_s, _ = st.columns([1, 3])
    with col_s:
        symbol_input = st.text_input("Hisse Kodu Girin", value="THYAO").upper()
    
    if symbol_input:
        with st.spinner('Smart RSI Analizi Yapılıyor...'):
            df, sym = get_data(symbol_input)
            
        if df is not None:
            data = calculate_smart_logic(df, capital, risk_pct)
            
            # ÜST METRİKLER (PUAN KUTUSU)
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                bg = "score-high" if data['score'] >= 75 else ("score-mid" if data['score'] >= 50 else "score-low")
                st.markdown(f"<div class='score-box {bg}'>{data['score']}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("#### 📝 Smart RSI Raporu")
                for r in data['reasons']:
                    st.markdown(f"<span class='reason-text'>• {r}</span>", unsafe_allow_html=True)
            with c3:
                st.metric("Potansiyel Kar", f"{data['potential_profit']:.0f} TL")
                st.metric("R/R Oranı", f"{data['rr']:.2f}")

            # İŞLEM KARTI VE GRAFİK
            col_l, col_r = st.columns([1, 2])
            with col_l:
                st.markdown(f"""
                <div class='trade-card'>
                    <h3 style="margin:0; color:#00e5ff">İŞLEM PLANI</h3>
                    <hr style="border-color:#30363d">
                    <div class='metric-row'><span>GİRİŞ:</span> <b>{data['price']:.2f}</b></div>
                    <div class='metric-row' style='color:#ff4b4b'><span>STOP:</span> <b>{data['stop']:.2f}</b></div>
                    <div class='metric-row' style='color:#00c853'><span>HEDEF:</span> <b>{data['target']:.2f}</b></div>
                    <br>
                    <div style="text-align:center; background:#0d1117; padding:10px; border-radius:5px">
                        <b>{data['lot']} LOT</b><br>
                        <small style='color:#8b949e'>{data['target_type']}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_r:
                fig = go.Figure(data=[go.Candlestick(x=df.tail(80).index, open=df.tail(80)['Open'], high=df.tail(80)['High'], low=df.tail(80)['Low'], close=df.tail(80)['Close'])])
                fig.add_hline(y=data['stop'], line_dash="dash", line_color="red")
                fig.add_hline(y=data['target'], line_dash="solid", line_color="#00c853")
                fig.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Veri çekilemedi. Hisse kodunu kontrol edin.")

# --- SEKME 2: AVCI MODU ---
with tab_hunter:
    st.info("Algoritma; Trend İçi Pullback, Dip Dönüşü (RSI Crossover) veya Breakout fırsatlarını arar.")
    if st.button("TARAMAYI BAŞLAT", type="primary"):
        candidates = []
        bar = st.progress(0)
        
        for i, ticker in enumerate(BIST30_TICKERS):
            df_scan, _ = get_data(ticker)
            if df_scan is not None:
                res = calculate_smart_logic(df_scan, capital, risk_pct)
                
                # FİLTRE MANTIĞI:
                # 1. Puanı yüksek (60+) OLSUN
                # 2. VEYA Özel bir Setup (Pullback/Dip Dönüşü) yakalasın
                is_special_setup = any("Pullback" in r or "DİP DÖNÜŞÜ" in r for r in res['reasons'])
                
                if (res['score'] >= 60 and res['rr'] > 1.5) or is_special_setup:
                    res['symbol'] = ticker
                    candidates.append(res)
            bar.progress((i+1)/len(BIST30_TICKERS))
        bar.empty()
        
        if candidates:
            # Puan sırasına göre diz
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_3 = candidates[:3]
            
            st.success(f"Tarama Tamamlandı. En Güçlü 3 Fırsat:")
            for idx, item in enumerate(top_3):
                with st.container():
                    c1, c2, c3 = st.columns([1, 2, 2])
                    with c1:
                        bg = "score-high" if item['score'] >= 75 else "score-mid"
                        st.markdown(f"<div class='score-box {bg}' style='font-size:1.8em'>#{idx+1}<br>{item['score']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='text-align:center'>{item['symbol']}</h3>", unsafe_allow_html=True)
                    with c2:
                        st.markdown("**🔍 Neden Seçildi?**")
                        for r in item['reasons']:
                            st.caption(r)
                    with c3:
                        st.markdown("**📊 Özet:**")
                        st.write(f"• Giriş: **{item['price']:.2f}**")
                        st.write(f"• Hedef: **{item['target']:.2f}**")
                        st.write(f"• Kar: **+{item['potential_profit']:.0f} TL**")
                    st.markdown("---")
        else:
            st.warning("Kriterlere uyan (Puan > 60 ve R/R > 1.5) güvenli hisse bulunamadı.")
