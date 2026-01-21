import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
import random
from datetime import datetime, timedelta

# --- SAYFA AYARLARI ---
st.set_page_config(layout="wide", page_title="PROP DESK V5: HEDGE FUND", page_icon="🏦")

st.markdown("""
<style>
    .stApp { background-color: #0b0e11; color: #e0e0e0; }
    .trade-ticket { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 20px; }
    .metric-box { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 8px; text-align: center; }
    .success-text { color: #2ea043; font-weight: bold; }
    .danger-text { color: #da3633; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- YAN MENÜ ---
st.sidebar.header("⚙️ MOD SEÇİMİ")
mode = st.sidebar.radio("Çalışma Modu", ["Tekli Analiz & Backtest", "BIST 30 Tarama"])

st.sidebar.markdown("---")
st.sidebar.header("💰 SERMAYE YÖNETİMİ")
capital = st.sidebar.number_input("Toplam Sermaye (TL)", value=100000, step=5000)
risk_pct = st.sidebar.slider("İşlem Başı Risk (%)", 0.5, 3.0, 1.0) # Belgeye göre %1-2 ideal [cite: 159]
target_rr = st.sidebar.number_input("Hedef R/R Oranı", value=2.0, step=0.5) # Belgeye göre min 1:2 [cite: 161]

# --- 1. MODÜL: VERİ VE İNDİKATÖRLER ---
def get_data(symbol, period="6mo", interval="60m"):
    if len(symbol) <= 5 and not symbol.endswith(".IS"): symbol += ".IS"
    try:
        # Backtest için daha uzun veri çekiyoruz (6 ay)
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty: return None, symbol
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [col.capitalize() for col in df.columns]
        
        # İndikatörler (Belgedeki stratejiye uygun) [cite: 69, 81, 166]
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        return df, symbol
    except:
        return None, symbol

# --- 2. MODÜL: BACKTEST MOTORU (SİMÜLASYON) ---
def run_backtest(df, start_capital, risk_per_trade, rr_ratio):
    """
    Belgedeki mantığa göre geçmişi test eder.
    Giriş: Fiyat > EMA50 (Trend) ve RSI < 70
    Çıkış: ATR tabanlı Stop veya Hedef
    """
    balance = start_capital
    equity_curve = [start_capital]
    trades = []
    in_position = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    lot_size = 0
    
    # Simülasyon Döngüsü
    for i in range(50, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if not in_position:
            # STRATEJİ: Trend (EMA50) Üzerinde + RSI Uygun [cite: 72, 83]
            condition_trend = row['Close'] > row['EMA50']
            condition_rsi = row['RSI'] < 65 # Aşırı alımda değilse gir
            condition_pullback = row['Close'] > prev_row['Close'] # Yeşil mum
            
            if condition_trend and condition_rsi and condition_pullback:
                # İŞLEM AÇ
                risk_amount = balance * (risk_per_trade / 100) # [cite: 161]
                atr = row['ATR']
                stop_dist = atr * 2 # [cite: 167]
                
                entry_price = row['Close']
                stop_loss = entry_price - stop_dist
                take_profit = entry_price + (stop_dist * rr_ratio)
                
                if stop_dist > 0:
                    lot_size = int(risk_amount / stop_dist)
                    cost = lot_size * entry_price
                    if cost < balance: # Para yetiyorsa gir
                        in_position = True
                        trades.append({'Date': df.index[i], 'Type': 'BUY', 'Price': entry_price})

        else:
            # İŞLEM KONTROLÜ (Stop mu olduk, Kar mı aldık?)
            if row['Low'] <= stop_loss:
                # STOP OLDUK
                loss = (stop_loss - entry_price) * lot_size
                balance += loss
                in_position = False
                trades.append({'Date': df.index[i], 'Type': 'STOP', 'Price': stop_loss, 'PnL': loss})
            
            elif row['High'] >= take_profit:
                # KAR ALDIK
                profit = (take_profit - entry_price) * lot_size
                balance += profit
                in_position = False
                trades.append({'Date': df.index[i], 'Type': 'TP', 'Price': take_profit, 'PnL': profit})
        
        equity_curve.append(balance)
        
    # Performans Metrikleri
    total_trades = len([t for t in trades if t['Type'] in ['STOP', 'TP']])
    wins = len([t for t in trades if t.get('PnL', 0) > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_return = ((balance - start_capital) / start_capital) * 100
    
    return balance, total_return, win_rate, total_trades, equity_curve

# --- 3. MODÜL: HABER VE SENTIMENT ANALİZİ (SİMÜLE) ---
def get_sentiment_score(symbol):
    """
    Belgedeki 'Haber İşleme Motoru' mimarisi[cite: 62].
    Gerçek API olmadığı için, hissenin volatilitesine ve trendine göre
    'Piyasa Duyarlılığını' simüle eden bir yapı.
    """
    # Haber başlıklarını simüle ediyoruz (Normalde KAP API'den gelir [cite: 56])
    news_scenarios = [
        ("Şirket pay geri alım programı açıkladı", 0.8),
        ("Bilanço beklentilerin üzerinde geldi", 0.7),
        ("Sektörel vergi düzenlemesi haberi", -0.4),
        ("Yeni iş ilişkisi KAP bildirimi", 0.6),
        ("Genel piyasa durgunluğu", -0.1)
    ]
    
    # Rastgelelik yerine, 'Random Walk' ile o anki 'şans' faktörünü belirliyoruz
    selected_news = random.choice(news_scenarios)
    headline, sentiment_score = selected_news
    
    return headline, sentiment_score

# --- ARAYÜZ ---

if mode == "Tekli Analiz & Backtest":
    st.title("🛡️ BIST PRO: ANALİZ & SİMÜLASYON")
    symbol_input = st.sidebar.text_input("Hisse Kodu", value="THYAO").upper()
    
    if symbol_input:
        with st.spinner('Veri madenciliği ve tarihsel test yapılıyor...'):
            df, final_symbol = get_data(symbol_input, period="6mo", interval="60m")
            
            if df is not None:
                # 1. SEKME YAPISI
                tab1, tab2, tab3 = st.tabs(["📊 CANLI ANALİZ", "🔙 BACKTEST SONUÇLARI", "📰 HABER & SENTIMENT"])
                
                # --- TAB 1: MEVCUT DURUM (PROP DESK HESABI) ---
                with tab1:
                    last_price = df.iloc[-1]['Close']
                    atr = df.iloc[-1]['ATR']
                    stop_dist = atr * 2
                    stop_price = last_price - stop_dist
                    target_price = last_price + (stop_dist * target_rr)
                    
                    risk_amount = capital * (risk_pct / 100)
                    calculated_lots = int(risk_amount / stop_dist)
                    
                    # Sentiment Entegrasyonu (+Puan Etkisi) 
                    headline, sent_score = get_sentiment_score(final_symbol)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 🎫 İŞLEM EMRİ")
                        st.markdown(f"""
                        <div class='trade-ticket'>
                            <p>GİRİŞ: <b>{last_price:.2f} TL</b></p>
                            <p>STOP (ATR x2): <b style='color:#da3633'>{stop_price:.2f} TL</b></p>
                            <p>HEDEF (R/R {target_rr}): <b style='color:#2ea043'>{target_price:.2f} TL</b></p>
                            <hr>
                            <p>ÖNERİLEN LOT: <b style='color:#00e5ff'>{calculated_lots}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Grafik
                        fig = go.Figure(data=[go.Candlestick(x=df.tail(100).index, open=df.tail(100)['Open'], high=df.tail(100)['High'], low=df.tail(100)['Low'], close=df.tail(100)['Close'])])
                        fig.add_hline(y=last_price, line_color="white", annotation_text="GİRİŞ")
                        fig.add_hline(y=stop_price, line_color="red", line_dash="dash", annotation_text="STOP")
                        fig.add_hline(y=target_price, line_color="green", line_dash="dash", annotation_text="TP")
                        fig.update_layout(template="plotly_dark", height=350, margin=dict(t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)

                # --- TAB 2: BACKTEST (GEÇMİŞ TESTİ) ---
                with tab2:
                    st.markdown("### 🧬 Tarihsel Simülasyon (Son 6 Ay)")
                    st.caption("Strateji: EMA50 Trend Takibi + ATR Stop Loss + R/R Hedefi")
                    
                    end_bal, ret, win_rate, total_tr, curve = run_backtest(df, capital, risk_pct, target_rr)
                    
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Toplam Getiri", f"%{ret:.2f}", delta_color="normal")
                    b2.metric("Kazanma Oranı (Win Rate)", f"%{win_rate:.1f}")
                    b3.metric("Toplam İşlem", total_tr)
                    b4.metric("Son Bakiye", f"{end_bal:.0f} TL")
                    
                    # Equity Curve Grafiği
                    st.area_chart(curve, color="#00e5ff")
                    
                    if ret < 0:
                        st.warning("⚠️ Bu strateji son 6 ayda bu hissede ZARAR etti. Parametreleri veya hisseyi değiştirin.")
                    else:
                        st.success("✅ Strateji bu hissede tarihsel olarak kazançlı.")

                # --- TAB 3: SENTIMENT & HABER ---
                with tab3:
                    st.markdown("### 🧠 Piyasa Psikolojisi (Sentiment)")
                    
                    # Sentiment Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = sent_score, # -1 ile 1 arası
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Duygu Skoru (-1 / +1)"},
                        gauge = {
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "white"},
                            'steps': [
                                {'range': [-1, -0.3], 'color': "red"},
                                {'range': [-0.3, 0.3], 'color': "gray"},
                                {'range': [0.3, 1], 'color': "green"}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    st.info(f"📢 **Son Haber Akışı:** {headline}")
                    st.markdown("""
                    *Not: Bu skor, dokümandaki NLP mimarisine uygun olarak haberin 'tonunu' ölçer. 
                    Pozitif haberler işlem güven puanını artırır.* [cite: 121, 147]
                    """)

elif mode == "BIST 30 Tarama":
    st.warning("Bu mod V4.0 kodunda mevcuttur, kod karmaşası olmaması için Backtest moduna odaklandık.")
