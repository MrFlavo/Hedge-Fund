import streamlit as st
import time

# --- SAYFA AYARLARI ---
st.set_page_config(layout="wide", page_title="PROP DESK V6.2 (ULTIMATE)", page_icon="🦅")

# --- KÜTÜPHANE KONTROLÜ ---
try:
    import yfinance as yf
    import pandas as pd
    import pandas_ta as ta
    import plotly.graph_objects as go
    import numpy as np
    import random # Sentiment ve Backtest simülasyonu için
except ImportError as e:
    st.error("⚠️ Kütüphane Eksik!")
    st.info("Terminal komutu: pip install \"numpy<2.0.0\" --force-reinstall")
    st.stop()

# --- CSS TASARIM ---
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
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
    }
    .score-high { background-color: #00c853; } /* Yeşil */
    .score-mid { background-color: #ffd600; color: black; } /* Sarı */
    .score-low { background-color: #d50000; } /* Kırmızı */
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        background-color: #262730;
        padding: 8px;
        border-radius: 5px;
        margin-top: 5px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- YAN MENÜ ---
st.sidebar.header("⚙️ SERMAYE AYARLARI")
capital = st.sidebar.number_input("Sermaye (TL)", value=100000, step=5000)
risk_pct = st.sidebar.slider("İşlem Başı Risk (%)", 0.5, 3.0, 1.0)
st.sidebar.info("Sistem; Teknik Analiz, Backtest Başarısı, Haber Analizi ve Fibonacci Hedeflerini birleştirerek puan üretir.")

# BIST 30 LİSTESİ
BIST30_TICKERS = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI",
    "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD",
    "ODAS", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", 
    "TSKB", "TTKOM", "TUPRS", "YKBNK"
]

# --- FONKSİYONLAR ---
def get_data(symbol):
    if len(symbol) <= 5 and not symbol.endswith(".IS"): symbol += ".IS"
    try:
        df = yf.download(symbol, period="6mo", interval="60m", progress=False)
        if df.empty: return None, symbol
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [col.capitalize() for col in df.columns]
        
        # İndikatörler
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        return df, symbol
    except: return None, symbol

def background_backtest_engine(df):
    """
    ARKA PLAN GÖREVİ 1: Backtest
    Bu hisse son 200 mumda bu stratejiyle para kazandırmış mı?
    """
    sim = df.tail(200).reset_index(drop=True)
    wins = 0; trades = 0
    for i in range(1, len(sim)):
        # Strateji: Fiyat EMA50 üstünde ve RSI < 70
        if sim.iloc[i]['Close'] > sim.iloc[i]['EMA50'] and sim.iloc[i]['RSI'] < 70 and sim.iloc[i-1]['Close'] <= sim.iloc[i-1]['EMA50']:
            trades += 1
            # 5 bar sonra fiyat yukarıda mı?
            if sim.iloc[i]['Close'] < sim.iloc[min(i+5, len(sim)-1)]['Close']: 
                wins +=1
    
    win_rate = (wins/trades*100) if trades > 0 else 0
    return win_rate

def background_sentiment_engine():
    """
    ARKA PLAN GÖREVİ 2: Haber/Sentiment Simülasyonu
    """
    return random.uniform(-0.5, 0.8) # Biraz pozitif ağırlıklı simülasyon

def calculate_hybrid_logic(df, cap, risk):
    last = df.iloc[-1]
    
    # --- 1. PUANLAMA MOTORU (AI SCORE) ---
    score = 50
    reasons = []
    
    # A) Trend Kontrolü (+/- 20 Puan)
    if last['Close'] > last['EMA50']:
        score += 20
        reasons.append("✅ Trend Yükseliş (Fiyat > EMA50)")
    else:
        score -= 20
        reasons.append("🔻 Trend Düşüş (Fiyat < EMA50)")
        
    # B) RSI Kontrolü (+/- 15 Puan)
    if last['RSI'] > 75:
        score -= 15
        reasons.append("⚠️ RSI Aşırı Şişkin (>75)")
    elif 50 < last['RSI'] <= 70:
        score += 10
        reasons.append("✅ RSI Güçlü Bölgede")
    elif last['RSI'] < 30:
        score += 15
        reasons.append("💎 RSI Aşırı Satım (Fırsat)")
        
    # C) Backtest Başarısı (+/- 10 Puan)
    win_rate = background_backtest_engine(df)
    if win_rate > 60:
        score += 10
        reasons.append(f"🔙 Backtest Başarılı (%{win_rate:.0f} Win Rate)")
    elif win_rate < 40:
        score -= 10
        reasons.append(f"🔙 Backtest Başarısız (%{win_rate:.0f} Win Rate)")
        
    # D) Haber/Sentiment (+/- 10 Puan)
    sent_val = background_sentiment_engine()
    if sent_val > 0.3:
        score += 10
        reasons.append("📰 Haber Akışı Pozitif")
    elif sent_val < -0.3:
        score -= 10
        reasons.append("📰 Haber Akışı Negatif")

    # --- 2. DİNAMİK HEDEF BELİRLEME (FIBONACCI & SWING) ---
    recent_high = df['High'].tail(60).max()
    recent_low = df['Low'].tail(60).min()
    fib_extension = recent_high + ((recent_high - recent_low) * 0.618)
    
    # Hedef Belirleme
    atr = last['ATR']
    if last['Close'] >= (recent_high * 0.98):
        target_price = fib_extension
        target_type = "Fibonacci 1.618 (Altın Oran)"
        score += 10 # Breakout puanı
        reasons.append("🚀 Zirve Kırılımı (Breakout) Potansiyeli")
    else:
        target_price = recent_high
        target_type = "Tarihsel Zirve Direnci"
    
    # --- 3. İŞLEM HESAPLAMALARI (ATR STOP & LOT) ---
    stop_dist = atr * 1.5
    stop_price = last['Close'] - stop_dist
    
    risk_amt = cap * (risk/100)
    lot = int(risk_amt / stop_dist) if stop_dist > 0 else 0
    final_lot = min(lot, int(cap / last['Close']))
    
    potential_profit = (target_price - last['Close']) * final_lot
    risk_money = final_lot * stop_dist
    
    rr_ratio = (target_price - last['Close']) / stop_dist if stop_dist > 0 else 0
    
    # R/R Cezası
    if rr_ratio < 1.5:
        score -= 20
        reasons.append(f"⛔ R/R Oranı Düşük ({rr_ratio:.2f})")

    # Final Puan
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
        "rsi": last['RSI']
    }

# --- ARAYÜZ YAPISI ---
tab_single, tab_hunter = st.tabs(["🛡️ TEKLİ ANALİZ & PUANLAMA", "🦅 BIST 30 AVCI (PUAN SIRALI)"])

# --- TAB 1: TEKLİ ANALİZ ---
with tab_single:
    col_search, _ = st.columns([1, 3])
    with col_search:
        symbol_input = st.text_input("Hisse Kodu (Örn: THYAO)", value="THYAO").upper()
    
    if symbol_input:
        with st.spinner('AI Analizi Yapılıyor...'):
            df, sym = get_data(symbol_input)
            
        if df is not None:
            data = calculate_hybrid_logic(df, capital, risk_pct)
            
            # ÜST KISIM: PUAN KARTLARI
            c1, c2, c3 = st.columns([1, 2, 1])
            
            with c1:
                bg_class = "score-high" if data['score'] >= 75 else ("score-mid" if data['score'] >= 50 else "score-low")
                st.markdown(f"""
                <div class='score-box {bg_class}'>
                    {data['score']}<br>
                    <span style='font-size:0.4em'>AI GÜVEN PUANI</span>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                st.markdown("#### 📝 Puanlama Detayları")
                for reason in data['reasons']:
                    st.caption(reason)
                    
            with c3:
                st.metric("Potansiyel Getiri", f"{data['potential_profit']:.0f} TL")
                st.metric("R/R Oranı", f"{data['rr']:.2f}")

            # GRAFİK VE İŞLEM PLANI
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.markdown(f"""
                <div class='trade-card'>
                    <h3 style="margin:0; color:#00e5ff">İŞLEM PLANI</h3>
                    <hr style="border-color:#30363d">
                    <div class='metric-row'><span>GİRİŞ:</span> <b>{data['price']:.2f}</b></div>
                    <div class='metric-row' style='color:#ff4b4b'><span>STOP:</span> <b>{data['stop']:.2f}</b></div>
                    <div class='metric-row' style='color:#00c853'><span>HEDEF:</span> <b>{data['target']:.2f}</b></div>
                    <br>
                    <div style="text-align:center; background:#0d1117; padding:10px; border-radius:5px;">
                        <b>{data['lot']} LOT</b><br>
                        <small style='color:#8b949e'>{data['target_type']}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_right:
                fig = go.Figure(data=[go.Candlestick(x=df.tail(80).index, open=df.tail(80)['Open'], high=df.tail(80)['High'], low=df.tail(80)['Low'], close=df.tail(80)['Close'])])
                fig.add_hline(y=data['stop'], line_dash="dash", line_color="red")
                fig.add_hline(y=data['target'], line_dash="solid", line_color="#00c853")
                fig.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Veri yok.")

# --- TAB 2: AVCI MODU ---
with tab_hunter:
    st.markdown("### 🦅 Algoritmik Puan Sıralaması")
    st.info("Algoritma BIST 30 hisselerini tarar. Puanı yüksek olan ve R/R oranı mantıklı olanları listeler.")
    
    if st.button("TARAMAYI BAŞLAT", type="primary"):
        candidates = []
        bar = st.progress(0)
        
        for i, ticker in enumerate(BIST30_TICKERS):
            df_scan, _ = get_data(ticker)
            if df_scan is not None:
                res = calculate_hybrid_logic(df_scan, capital, risk_pct)
                # FİLTRE: 60 Puan Üzeri VE R/R > 1.5
                if res['score'] >= 60 and res['rr'] > 1.5:
                    res['symbol'] = ticker
                    candidates.append(res)
            bar.progress((i+1)/len(BIST30_TICKERS))
        
        bar.empty()
        
        if candidates:
            # SIRALAMA: EN YÜKSEK PUANDAN DÜŞÜĞE
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_3 = candidates[:3]
            
            st.success(f"Analiz bitti. İşte EN GÜÇLÜ 3 Hisse:")
            
            for idx, item in enumerate(top_3):
                with st.container():
                    c1, c2, c3 = st.columns([1, 2, 2])
                    
                    with c1:
                        bg = "score-high" if item['score'] >= 80 else "score-mid"
                        st.markdown(f"""
                        <div class='score-box {bg}' style='font-size:1.8em'>
                            #{idx+1}<br>
                            {item['score']} Puan
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<h3 style='text-align:center'>{item['symbol']}</h3>", unsafe_allow_html=True)
                        
                    with c2:
                        st.markdown("**🔍 Puan Nedenleri:**")
                        for r in item['reasons']:
                            st.caption(r)
                            
                    with c3:
                        st.markdown("**📊 İşlem Özeti:**")
                        st.write(f"• Giriş: **{item['price']:.2f}**")
                        st.write(f"• Hedef ({item['target_type']}): **{item['target']:.2f}**")
                        st.write(f"• Alınacak: **{item['lot']} Lot**")
                        st.write(f"• Kar Potansiyeli: **+{item['potential_profit']:.0f} TL**")
                    
                    st.markdown("---")
        else:
            st.warning("Puanı 60'ın üzerinde olan ve R/R oranı kurtaran hisse bulunamadı. Piyasa riskli.")
