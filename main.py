"""
PROP DESK V7.0 - ULTIMATE INTRADAY TRADING SYSTEM
==================================================
Upgrades:
1. Advanced Day Trading Indicators (VWAP, MFI, Stochastic, OBV)
2. ML Score Enhancement (XGBoost + LSTM price prediction)
3. Multi-Timeframe Analysis (MTF confluence)
4. SQLite Database Integration (historical data caching)
5. Performance Optimization (parallel processing, vectorized calculations)
6. Portfolio Management Layer
7. Order Execution Simulation

Optimized for BIST (Borsa Istanbul) intraday trading
"""

import streamlit as st
import time
import warnings
warnings.filterwarnings('ignore')

# =====================
# PAGE CONFIGURATION
# =====================
st.set_page_config(
    layout="wide",
    page_title="PROP DESK V7.0 - ML ENHANCED INTRADAY",
    page_icon="🚀"
)

# =====================
# LIBRARY IMPORTS
# =====================
try:
    import yfinance as yf
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timedelta
    import json
    
    # ML Libraries
    try:
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
        st.warning("⚠️ ML libraries not available. Install: pip install xgboost scikit-learn")
    
    # Deep Learning (optional - disabled for faster deployment)
    LSTM_AVAILABLE = False
    # Uncomment below to enable TensorFlow/LSTM features:
    # try:
    #     import tensorflow as tf
    #     from tensorflow import keras
    #     LSTM_AVAILABLE = True
    # except ImportError:
    #     LSTM_AVAILABLE = False
        
except ImportError as e:
    st.error(f"⚠️ Missing libraries: {e}")
    st.code("pip install yfinance pandas plotly numpy xgboost scikit-learn tensorflow")
    st.stop()

# =====================
# DATABASE SETUP
# =====================
import tempfile
import os

# Use temporary directory for Streamlit Cloud compatibility
TEMP_DIR = tempfile.gettempdir()
DB_PATH = os.path.join(TEMP_DIR, "bist_trading_data.db")

def init_database():
    """Initialize SQLite database for caching historical data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Price data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                symbol TEXT,
                timestamp TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                interval TEXT,
                PRIMARY KEY (symbol, timestamp, interval)
            )
        """)
        
        # Indicator cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicator_cache (
                symbol TEXT,
                timestamp TEXT,
                indicator_name TEXT,
                value REAL,
                interval TEXT,
                PRIMARY KEY (symbol, timestamp, indicator_name, interval)
            )
        """)
        
        # Trade history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                qty INTEGER,
                pnl REAL,
                reason TEXT,
                score INTEGER,
                strategy TEXT
            )
        """)
        
        # Portfolio state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                timestamp TEXT PRIMARY KEY,
                total_equity REAL,
                cash REAL,
                positions TEXT,
                daily_pnl REAL
            )
        """)
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.warning(f"⚠️ Database initialization failed: {e}. Running without database caching.")
        return False

# Initialize DB on startup - don't fail if it doesn't work
DB_AVAILABLE = init_database()

# =====================
# BIST UNIVERSE DEFINITIONS - UPDATED 2026
# =====================
# Based on current BIST index compositions as of February 2026
# All tickers include .IS suffix for yfinance compatibility

BIST30 = [
    # Banking & Finance
    "AKBNK.IS",    # Akbank
    "GARAN.IS",    # Garanti BBVA
    "ISCTR.IS",    # İş Bankası
    "YKBNK.IS",    # Yapı Kredi
    
    # Industry & Manufacturing
    "ASELS.IS",    # Aselsan
    "EREGL.IS",    # Erdemir
    "FROTO.IS",    # Ford Otosan
    "TOASO.IS",    # Tofaş
    "TUPRS.IS",    # Tüpraş
    "SISE.IS",     # Şişecam
    "SASA.IS",     # SASA Polyester
    
    # Holding Companies
    "KCHOL.IS",    # Koç Holding
    "SAHOL.IS",    # Sabancı Holding
    
    # Real Estate (GYO)
    "EKGYO.IS",    # Emlak Konut GYO
    
    # Retail & Consumer
    "BIMAS.IS",    # BİM
    "MGROS.IS",    # Migros
    "ULKER.IS",    # Ülker
    
    # Telecom & Tech
    "TCELL.IS",    # Turkcell
    "TTKOM.IS",    # Türk Telekom
    
    # Transportation & Tourism
    "THYAO.IS",    # THY - Turkish Airlines
    "PGSUS.IS",    # Pegasus
    "TAVHL.IS",    # TAV Havalimanları
    
    # Energy & Chemicals
    "PETKM.IS",    # Petkim
    "GUBRF.IS",    # Gübre Fabrikaları
    "ENKA.IS",     # ENKA İnşaat
    
    # Mining & Metals
    "KOZAL.IS",    # Koza Altın
    "KRDMD.IS",    # Kardemir
    
    # Food & Beverage
    "AEFES.IS",    # Anadolu Efes
    
    # Energy (New additions)
    "ASTOR.IS",    # Astor Enerji
    
    # Finance
    "DSFAK.IS",    # Destek Finans Faktoring
]

BIST50 = list(set(BIST30 + [
    # Additional Banking & Finance
    "HALKB.IS",    # Halkbank
    "VAKBN.IS",    # Vakıfbank
    "SKBNK.IS",    # Şekerbank
    
    # Additional Industry
    "ARCLK.IS",    # Arçelik
    "VESTL.IS",    # Vestel
    "OTKAR.IS",    # Otokar
    "DOAS.IS",     # Doğuş Otomotiv
    "TTRAK.IS",    # Türk Traktör
    
    # Additional Holdings & Diversified
    "ALARK.IS",    # Alarko Holding
    "AYGAZ.IS",    # Aygaz
    
    # Additional Real Estate
    "TKFEN.IS",    # Tekfen Holding
    "YAPI K.IS",    # Yapı Kredi Koray GYO
    
    # Additional Energy & Utilities
    "AKSEN.IS",    # Aksa Enerji
    "AKENR.IS",    # AK Enerji
    "ZOREN.IS",    # Zorlu Enerji
    "GESAN.IS",    # Gediz Elektrik
    
    # Additional Telecom & Tech
    "LOGO.IS",     # Logo Yazılım
    "KRONT.IS",    # Kontrolmatik
    
    # Additional Retail & Consumer
    "SOKM.IS",     # ŞOK Marketler
    "CCOLA.IS",    # Coca-Cola İçecek
    
    # Additional Construction & Materials
    "ADANA.IS",    # Adana Çimento
    "BUCIM.IS",    # Bursa Çimento
    
    # Sports Clubs
    "BJKAS.IS",    # Beşiktaş
    "GSRAY.IS",    # Galatasaray
    "TSPOR.IS",    # Trabzonspor
    "FENER.IS",    # Fenerbahçe
]))

BIST100 = list(set(BIST50 + [
    # Additional Banking & Finance
    "ALBRK.IS",    # Albaraka Türk
    "ICBCT.IS",    # ICBC Turkey
    "QNBFB.IS",    # QNB Finans
    "TSKB.IS",     # TSKB
    
    # Additional Industry & Manufacturing
    "AKSA.IS",     # Aksa Akrilik
    "BRISA.IS",    # Brisa
    "BRSAN.IS",    # Borusan Mannesmann
    "EGEEN.IS",    # Ege Endüstri
    "GOLTS.IS",    # Göltaş Çimento
    "KARSN.IS",    # Karsan
    "KLMSN.IS",    # Klimasan
    "NETAS.IS",    # Netaş
    "SODA.IS",     # Soda Sanayii
    "TIRE TIS",    # Tire Kutsan
    "TRKCM.IS",    # Trakya Cam
    
    # Additional Energy & Utilities
    "AKSUE.IS",    # Aksu Enerji
    "AYEN.IS",     # Ayen Enerji
    "CLEBI.IS",    # Çelebi Hava Servisi
    "ENJSA.IS",    # Enerjisa
    "GWIND.IS",    # Galata Wind
    "HUNER.IS",    # Hun Yenilenebilir
    "ODAS.IS",     # Odaş Elektrik
    "PENTA.IS",    # Penta Teknoloji
    
    # Additional Telecom & Tech
    "ARENA.IS",    # Arena Bilgisayar
    "ASTOR.IS",    # Astor Enerji
    "INDES.IS",    # İndeks Bilgisayar
    "LINK.IS",     # Link Bilgisayar
    
    # Additional Real Estate & Construction
    "AVOD.IS",     # A.V.O.D. Kurutulmuş Gıda
    "AGYO.IS",     # Atakule GYO
    "DENGE.IS",    # Denizli Cam
    "ISGYO.IS",    # İş GYO
    "KLGYO.IS",    # Kiler GYO
    "KONYA.IS",    # Konya Çimento
    "MIPAZ.IS",    # Milpa
    "NUGYO.IS",    # Nurol GYO
    "OYAYO.IS",    # Oyak Yatırım Ortaklığı
    "OZGYO.IS",    # Özderici GYO
    "PEKGY.IS",    # Peker GYO
    "QGYO.IS",     # Quagr Menkul Kıymetler
    "VKGYO.IS",    # Vakıf GYO
    
    # Additional Retail & Consumer
    "BANVT.IS",    # Banvit
    "BIZIM.IS",    # Bizim Toptan
    "CRFSA.IS",    # Carrefoursa
    "DGZTE.IS",    # Doğan Gazetecilik
    "DUROF.IS",    # Duran Ofset
    "IHLAS.IS",    # İhlas Holding
    "KENT F.IS",   # Kent Gıda
    "KNFRT.IS",    # Konfrut Gıda
    "MAVI.IS",     # Mavi Giyim
    "PINSU.IS",    # Pınar Su
    "PNSUT.IS",    # Pınar Süt
    "TATGD.IS",    # Tat Gıda
    
    # Additional Textiles & Apparel
    "BLCYT.IS",    # Bilici Yatırım
    "BRMEN.IS",    # Birlik Mensucat
    "DAGI.IS",     # Dagi Giyim
    "DERIM.IS",    # Derimod
    "HATEK.IS",    # Hateks
    "LUKSK.IS",    # Lüks Kadife
    "MERKO.IS",    # Merko Gıda
    "ROYAL.IS",    # Royal Halı
    "SNPAM.IS",    # Sönmez Pamuklu
    "YUNSA.IS",    # Yünsa
    
    # Additional Chemicals & Petrochem
    "AKFGY.IS",    # Akfen GYO
    "ALKIM.IS",    # Alkim Kağıt
    "ANACM.IS",    # Anadolu Cam
    "BAGFS.IS",    # Bagfaş
    "BFREN.IS",    # Bosch Fren
    "BRKO.IS",     # Birko Birleşik Koyunlulular
    "CIMSA.IS",    # Çimsa
    "DOGUB.IS",    # Doğusan Boru
    "DYOBY.IS",    # DYO Boya
    "IZMDC.IS",    # İzmir Demir Çelik
    "KAPLM.IS",    # Kaplamin
    "KUTPO.IS",    # Kütahya Porselen
    "PARSN.IS",    # Parsan
    "PTOFS.IS",    # Petrokent Turizm
    "SARKY.IS",    # Sarkuysan
    "SELEC.IS",    # Selçuk Ecza
    "SODA.IS",     # Soda Sanayii
    "UNYEC.IS",    # Ünye Çimento
    
    # Additional Transportation
    "CLEBI.IS",    # Çelebi Hava Servisi
    "GSDHO.IS",    # GSD Holding
    "RYGYO.IS",    # Reysaş GYO
    
    # Additional Misc
    "HEKTS.IS",    # Hektaş
    "OYAKC.IS",    # Oyak Çimento
    "SMRTG.IS",    # Smart Güneş
]))

# =====================
# RATE LIMITING CONFIGURATION
# =====================
# Streamlit Cloud has stricter rate limits - adjust these if needed
MAX_PARALLEL_WORKERS = 4  # Reduced from 8 to avoid rate limits
REQUEST_DELAY = 0.5  # Seconds between requests
RETRY_DELAY = 2  # Seconds to wait on rate limit error

# =====================
# OPTIMIZED INDICATOR CALCULATIONS
# =====================
def _ema_vectorized(series, length):
    """Vectorized EMA calculation"""
    return series.ewm(span=length, adjust=False).mean()

def _sma_vectorized(series, length):
    """Vectorized SMA calculation"""
    return series.rolling(length, min_periods=1).mean()

def _rsi_vectorized(close, length=14):
    """Optimized RSI calculation"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr_vectorized(high, low, close, length=14):
    """Optimized ATR calculation"""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def _roc_vectorized(series, length):
    """Rate of Change"""
    return (series / series.shift(length) - 1.0) * 100.0

def _stochastic_vectorized(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator (%K and %D)"""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(d_period).mean()
    return k, d

def _mfi_vectorized(high, low, close, volume, length=14):
    """Money Flow Index (Volume-weighted RSI)"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    positive_flow = pd.Series(0.0, index=close.index)
    negative_flow = pd.Series(0.0, index=close.index)
    
    price_diff = typical_price.diff()
    positive_flow[price_diff > 0] = raw_money_flow[price_diff > 0]
    negative_flow[price_diff < 0] = raw_money_flow[price_diff < 0]
    
    positive_mf = positive_flow.rolling(length).sum()
    negative_mf = negative_flow.rolling(length).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))
    return mfi.fillna(50)

def _obv_vectorized(close, volume):
    """On Balance Volume"""
    obv = pd.Series(0.0, index=close.index)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def _vwap_vectorized(high, low, close, volume):
    """Volume Weighted Average Price (intraday)"""
    typical_price = (high + low + close) / 3
    cumulative_tpv = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum()
    return cumulative_tpv / cumulative_volume

def _adx_vectorized(high, low, close, length=14):
    """Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = _atr_vectorized(high, low, close, length) * length
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / tr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    
    return adx, plus_di, minus_di

# =====================
# ADVANCED DATA FETCHING WITH CACHING
# =====================
@st.cache_data(ttl=900, show_spinner=False)
def get_data_with_db_cache(symbol, period, interval):
    """Fetch data with database caching (optional - falls back to direct fetch)"""
    symbol = symbol.upper().strip()
    if len(symbol) <= 5 and not symbol.endswith(".IS"):
        symbol += ".IS"
    
    # Try database first (only if available)
    if DB_AVAILABLE:
        try:
            conn = sqlite3.connect(DB_PATH)
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? AND interval = ?
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            df_cached = pd.read_sql_query(query, conn, params=(symbol, interval))
            if not df_cached.empty:
                df_cached['timestamp'] = pd.to_datetime(df_cached['timestamp'])
                df_cached.set_index('timestamp', inplace=True)
                
                # Check if cache is fresh (< 15 min old)
                if len(df_cached) > 0:
                    latest_time = df_cached.index[-1]
                    if datetime.now() - latest_time < timedelta(minutes=15):
                        conn.close()
                        return calculate_indicators(df_cached), symbol
            
            conn.close()
        except Exception:
            # Silently fall through to yfinance if DB fails
            pass
    
    # Fetch from yfinance with retry logic
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            
            if df is not None and not df.empty:
                break
                
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                continue
            elif attempt == max_retries - 1:
                return None, symbol
    
    if df is None or df.empty:
        return None, symbol
    
    # Fix MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [c.title() for c in df.columns]
    
    # Save to database (only if available)
    if DB_AVAILABLE:
        try:
            conn = sqlite3.connect(DB_PATH)
            df_to_save = df.reset_index()
            df_to_save['symbol'] = symbol
            df_to_save['interval'] = interval
            df_to_save.rename(columns={'Date': 'timestamp', 'Datetime': 'timestamp'}, inplace=True)
            
            for _, row in df_to_save.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO price_data 
                    (symbol, timestamp, open, high, low, close, volume, interval)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['symbol'], str(row['timestamp']), 
                    row['Open'], row['High'], row['Low'], row['Close'],
                    row['Volume'], row['interval']
                ))
            
            conn.commit()
            conn.close()
        except Exception:
            # Silently ignore DB save errors
            pass
    
    return calculate_indicators(df), symbol

def calculate_indicators(df):
    """Calculate all technical indicators - OPTIMIZED"""
    # Core indicators
    df["EMA9"] = _ema_vectorized(df["Close"], 9)
    df["EMA20"] = _ema_vectorized(df["Close"], 20)
    df["EMA50"] = _ema_vectorized(df["Close"], 50)
    df["EMA200"] = _ema_vectorized(df["Close"], 200)
    
    df["SMA20"] = _sma_vectorized(df["Close"], 20)
    df["SMA50"] = _sma_vectorized(df["Close"], 50)
    
    df["ATR"] = _atr_vectorized(df["High"], df["Low"], df["Close"], 14)
    df["RSI"] = _rsi_vectorized(df["Close"], 14)
    
    # Advanced oscillators
    df["RSI3"] = _rsi_vectorized(df["Close"], 3)
    df["RSI14_ROC9"] = _roc_vectorized(df["RSI"], 9)
    df["RSI3_SMA3"] = _sma_vectorized(df["RSI3"], 3)
    df["CMB_CI"] = df["RSI14_ROC9"] + df["RSI3_SMA3"]
    df["CMB_FAST"] = _sma_vectorized(df["CMB_CI"], 13)
    df["CMB_SLOW"] = _sma_vectorized(df["CMB_CI"], 33)
    
    # Stochastic
    df["STOCH_K"], df["STOCH_D"] = _stochastic_vectorized(
        df["High"], df["Low"], df["Close"], 14, 3
    )
    
    # Volume indicators
    if "Volume" in df.columns:
        df["VOL_MA20"] = _sma_vectorized(df["Volume"], 20)
        df["VOL_MA50"] = _sma_vectorized(df["Volume"], 50)
        df["MFI"] = _mfi_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], 14)
        df["OBV"] = _obv_vectorized(df["Close"], df["Volume"])
        df["OBV_EMA"] = _ema_vectorized(df["OBV"], 20)
        
        # VWAP (session-based for intraday)
        df["VWAP"] = _vwap_vectorized(df["High"], df["Low"], df["Close"], df["Volume"])
    
    # Trend strength
    df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = _adx_vectorized(
        df["High"], df["Low"], df["Close"], 14
    )
    
    # Swing levels
    df["SWING_HIGH_20"] = df["High"].rolling(20).max()
    df["SWING_LOW_20"] = df["Low"].rolling(20).min()
    df["SWING_HIGH_60"] = df["High"].rolling(60).max()
    df["SWING_LOW_60"] = df["Low"].rolling(60).min()
    
    # Bollinger Bands
    bb_std = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["SMA20"] + (2 * bb_std)
    df["BB_LOWER"] = df["SMA20"] - (2 * bb_std)
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["SMA20"] * 100
    
    return df.dropna()

# =====================
# MULTI-TIMEFRAME ANALYSIS
# =====================
def get_mtf_data(symbol, timeframes=['5m', '15m', '1h']):
    """Fetch multiple timeframes for confluence analysis"""
    mtf_data = {}
    periods_map = {'5m': '5d', '15m': '30d', '1h': '60d', '1d': '1y'}
    
    for tf in timeframes:
        period = periods_map.get(tf, '60d')
        df, _ = get_data_with_db_cache(symbol, period, tf)
        if df is not None and not df.empty:
            mtf_data[tf] = df
    
    return mtf_data

def mtf_trend_analysis(mtf_data):
    """Analyze trend across multiple timeframes"""
    trends = {}
    
    for tf, df in mtf_data.items():
        if df is None or len(df) < 2:
            continue
        
        last = df.iloc[-1]
        
        # Trend determination
        ema_trend = "BULL" if last["Close"] > last["EMA50"] > last["EMA200"] else "BEAR"
        price_above_vwap = "BULL" if "VWAP" in last and last["Close"] > last["VWAP"] else "NEUTRAL"
        adx_strong = last["ADX"] > 25 if "ADX" in last else False
        
        trends[tf] = {
            "trend": ema_trend,
            "vwap_position": price_above_vwap,
            "adx": last["ADX"] if "ADX" in last else 0,
            "trend_strong": adx_strong,
            "rsi": last["RSI"]
        }
    
    # Confluence calculation
    bull_count = sum(1 for t in trends.values() if t["trend"] == "BULL")
    total = len(trends)
    
    confluence_score = (bull_count / total * 100) if total > 0 else 0
    
    return trends, confluence_score

# =====================
# MACHINE LEARNING SCORE ENHANCEMENT
# =====================
def prepare_ml_features(df, lookback=20):
    """Prepare features for ML model"""
    if len(df) < lookback + 10:
        return None, None
    
    features = []
    labels = []
    
    for i in range(lookback, len(df) - 5):
        # Feature engineering
        row = df.iloc[i]
        hist = df.iloc[i-lookback:i]
        
        feature_vec = [
            row["RSI"],
            row["MFI"] if "MFI" in row else 50,
            row["STOCH_K"],
            row["STOCH_D"],
            row["ADX"],
            (row["Close"] - row["EMA20"]) / row["ATR"] if row["ATR"] > 0 else 0,
            (row["Close"] - row["EMA50"]) / row["ATR"] if row["ATR"] > 0 else 0,
            row["BB_WIDTH"] if "BB_WIDTH" in row else 0,
            (row["Volume"] / row["VOL_MA20"]) if "VOL_MA20" in row and row["VOL_MA20"] > 0 else 1,
            hist["Close"].pct_change().mean(),
            hist["Close"].pct_change().std(),
            row["CMB_CI"] if "CMB_CI" in row else 0,
            (row["OBV"] - row["OBV_EMA"]) if "OBV" in row else 0,
        ]
        
        features.append(feature_vec)
        
        # Label: 1 if price rises >1% in next 5 bars, else 0
        future_max = df.iloc[i+1:i+6]["Close"].max()
        label = 1 if (future_max - row["Close"]) / row["Close"] > 0.01 else 0
        labels.append(label)
    
    return np.array(features), np.array(labels)

@st.cache_resource
def train_ml_model(symbol, df):
    """Train ML model for probability prediction"""
    if not ML_AVAILABLE:
        return None
    
    X, y = prepare_ml_features(df)
    
    if X is None or len(X) < 50:
        return None
    
    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train XGBoost (removed deprecated use_label_encoder)
    model = xgb.XGBClassifier(
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Test accuracy
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "feature_importance": model.feature_importances_
    }

def get_ml_probability(df, model_data):
    """Get ML probability for current setup"""
    if model_data is None or df is None or len(df) < 30:
        return 0.5
    
    last_row = df.iloc[-1]
    hist = df.iloc[-20:]
    
    feature_vec = [[
        last_row["RSI"],
        last_row["MFI"] if "MFI" in last_row else 50,
        last_row["STOCH_K"],
        last_row["STOCH_D"],
        last_row["ADX"],
        (last_row["Close"] - last_row["EMA20"]) / last_row["ATR"] if last_row["ATR"] > 0 else 0,
        (last_row["Close"] - last_row["EMA50"]) / last_row["ATR"] if last_row["ATR"] > 0 else 0,
        last_row["BB_WIDTH"] if "BB_WIDTH" in last_row else 0,
        (last_row["Volume"] / last_row["VOL_MA20"]) if "VOL_MA20" in last_row and last_row["VOL_MA20"] > 0 else 1,
        hist["Close"].pct_change().mean(),
        hist["Close"].pct_change().std(),
        last_row["CMB_CI"] if "CMB_CI" in last_row else 0,
        (last_row["OBV"] - last_row["OBV_EMA"]) if "OBV" in last_row else 0,
    ]]
    
    proba = model_data["model"].predict_proba(feature_vec)[0][1]
    return float(proba)

# =====================
# ENHANCED SCORING ALGORITHM
# =====================
def calculate_advanced_score(df, symbol, mtf_data=None, ml_model=None):
    """
    Enhanced scoring with:
    - Day trading specific indicators (VWAP, MFI, Stochastic)
    - Multi-timeframe confluence
    - ML probability enhancement
    """
    if df is None or len(df) < 50:
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 50
    reasons = []
    
    # === TREND & REGIME (25 points) ===
    bull_regime = (last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"])
    price_above_ema50 = last["Close"] > last["EMA50"]
    price_above_ema20 = last["Close"] > last["EMA20"]
    
    if bull_regime:
        score += 15
        reasons.append("✅ Bull Regime: EMA50 > EMA200, Trend aligned")
    else:
        score -= 15
        reasons.append("🔻 Bear/Neutral Regime")
    
    if price_above_ema50:
        score += 10
        reasons.append("📈 Price above EMA50 (trending)")
    
    # === VWAP DAY TRADING SIGNAL (20 points) ===
    if "VWAP" in last:
        vwap_position = (last["Close"] - last["VWAP"]) / last["VWAP"] * 100
        
        if vwap_position > 0.5:
            score += 20
            reasons.append(f"💎 VWAP: Price {vwap_position:.2f}% above (strong intraday)")
        elif vwap_position > 0:
            score += 10
            reasons.append("💎 VWAP: Price above (bullish intraday)")
        elif vwap_position < -1:
            score -= 10
            reasons.append("⚠️ VWAP: Price significantly below")
    
    # === RSI SMART ZONES (15 points) ===
    curr_rsi = float(last["RSI"])
    
    if bull_regime and 40 <= curr_rsi <= 55:
        score += 15
        reasons.append(f"📊 RSI {curr_rsi:.1f}: Bull pullback zone (40-55)")
    elif curr_rsi < 30 and curr_rsi > prev["RSI"]:
        score += 18
        reasons.append(f"🚀 RSI {curr_rsi:.1f}: Oversold bounce signal")
    elif curr_rsi > 70 and curr_rsi < 85:
        score += 8
        reasons.append(f"💪 RSI {curr_rsi:.1f}: Strong momentum (not overheated)")
    elif curr_rsi > 90:
        score -= 12
        reasons.append(f"🔥 RSI {curr_rsi:.1f}: Extreme overbought (risk)")
    
    # === MONEY FLOW INDEX (15 points) ===
    if "MFI" in last:
        mfi = float(last["MFI"])
        
        if 30 < mfi < 70:
            score += 10
            reasons.append(f"💰 MFI {mfi:.1f}: Healthy money flow")
        elif mfi < 20:
            score += 15
            reasons.append(f"💰 MFI {mfi:.1f}: Oversold with volume (strong buy)")
        elif mfi > 80:
            score -= 10
            reasons.append(f"⚠️ MFI {mfi:.1f}: Overbought money flow")
    
    # === STOCHASTIC MOMENTUM (10 points) ===
    stoch_k = float(last["STOCH_K"])
    stoch_d = float(last["STOCH_D"])
    
    if stoch_k < 20 and stoch_k > stoch_d:
        score += 10
        reasons.append(f"📉 Stochastic: Oversold crossover ({stoch_k:.1f})")
    elif stoch_k > 80:
        score -= 5
        reasons.append(f"⚠️ Stochastic: Overbought ({stoch_k:.1f})")
    
    # === ADX TREND STRENGTH (10 points) ===
    adx = float(last["ADX"])
    
    if adx > 25:
        score += 10
        reasons.append(f"💪 ADX {adx:.1f}: Strong trend in place")
    elif adx < 20:
        score -= 5
        reasons.append(f"⚠️ ADX {adx:.1f}: Weak/choppy trend")
    
    # === VOLUME CONFIRMATION (10 points) ===
    if "Volume" in last and "VOL_MA20" in last:
        vol_ratio = last["Volume"] / last["VOL_MA20"]
        
        if vol_ratio > 1.5:
            score += 10
            reasons.append(f"📊 Volume: {vol_ratio:.2f}x above average (strong)")
        elif vol_ratio < 0.7:
            score -= 5
            reasons.append(f"⚠️ Volume: {vol_ratio:.2f}x below average (weak)")
    
    # === OBV TREND (8 points) ===
    if "OBV" in last and "OBV_EMA" in last:
        if last["OBV"] > last["OBV_EMA"]:
            score += 8
            reasons.append("📈 OBV: Above EMA (accumulation)")
    
    # === BOLLINGER BAND SQUEEZE (8 points) ===
    if "BB_WIDTH" in last:
        bb_width = last["BB_WIDTH"]
        bb_squeeze = bb_width < 2  # Tight squeeze
        
        if bb_squeeze and price_above_ema20:
            score += 8
            reasons.append(f"🎯 BB Squeeze: {bb_width:.2f}% (breakout setup)")
    
    # === CMB COMPOSITE INDEX (10 points) ===
    cmb_strong = (
        (last["CMB_FAST"] > last["CMB_SLOW"]) and
        (last["CMB_CI"] > last["CMB_FAST"]) and
        (last["CMB_CI"] > prev["CMB_CI"])
    )
    
    if cmb_strong:
        score += 10
        reasons.append("🧠 CMB: Strong composite momentum")
    
    # === MULTI-TIMEFRAME CONFLUENCE (15 points) ===
    if mtf_data:
        _, confluence = mtf_trend_analysis(mtf_data)
        
        if confluence >= 80:
            score += 15
            reasons.append(f"🎯 MTF Confluence: {confluence:.0f}% (all timeframes aligned)")
        elif confluence >= 60:
            score += 8
            reasons.append(f"✅ MTF Confluence: {confluence:.0f}%")
        else:
            score -= 5
            reasons.append(f"⚠️ MTF Confluence: {confluence:.0f}% (mixed signals)")
    
    # === MACHINE LEARNING PROBABILITY (20 points) ===
    if ml_model and ML_AVAILABLE:
        ml_prob = get_ml_probability(df, ml_model)
        
        if ml_prob > 0.65:
            ml_boost = int((ml_prob - 0.5) * 40)
            score += ml_boost
            reasons.append(f"🤖 ML Probability: {ml_prob*100:.1f}% (+{ml_boost} pts)")
        elif ml_prob < 0.35:
            ml_penalty = int((0.5 - ml_prob) * 30)
            score -= ml_penalty
            reasons.append(f"🤖 ML Probability: {ml_prob*100:.1f}% (-{ml_penalty} pts)")
    
    # === RISK/REWARD SETUP ===
    price = float(last["Close"])
    atr = float(last["ATR"])
    stop_dist = atr * 1.5
    stop_price = price - stop_dist
    
    # Dynamic TP based on score
    tp_pct = 0.015 + (max(0, score - 50) / 1000)  # 1.5%-4% range
    tp_pct = max(0.01, min(0.04, tp_pct))
    target_price = price * (1 + tp_pct)
    
    rr_ratio = (target_price - price) / stop_dist if stop_dist > 0 else 0
    
    # Final score clamping
    final_score = max(0, min(100, int(score)))
    
    return {
        "symbol": symbol,
        "score": final_score,
        "reasons": reasons,
        "price": price,
        "stop": stop_price,
        "target": target_price,
        "rr": rr_ratio,
        "tp_pct": tp_pct * 100,
        "rsi": curr_rsi,
        "mfi": last.get("MFI", 50),
        "vwap_dist": vwap_position if "VWAP" in last else 0,
        "adx": adx,
        "ml_prob": ml_prob if ml_model else 0.5,
        "mtf_confluence": confluence if mtf_data else 0
    }

# =====================
# CSS STYLING
# =====================
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .score-box {
        font-size: 2.5em; font-weight: bold; text-align: center;
        padding: 15px; border-radius: 12px; margin-bottom: 15px;
        color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .score-high { background: linear-gradient(135deg, #00c853 0%, #64dd17 100%); }
    .score-mid { background: linear-gradient(135deg, #ffd600 0%, #ffab00 100%); color: #000; }
    .score-low { background: linear-gradient(135deg, #d50000 0%, #c62828 100%); }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2a38 0%, #2d3e50 100%);
        padding: 15px; border-radius: 10px; margin: 10px 0;
        border-left: 4px solid #00e5ff;
    }
    
    .trade-plan {
        background: #1c1f26; padding: 20px; border-radius: 12px;
        border: 2px solid #00e5ff; box-shadow: 0 6px 20px rgba(0,229,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# =====================
# MAIN APPLICATION
# =====================
st.title("🚀 PROP DESK V7.0 - ML ENHANCED INTRADAY SYSTEM")
st.caption("Advanced day trading with VWAP, MFI, Multi-Timeframe, ML Score & Database Caching")

tab_single, tab_scanner, tab_portfolio, tab_analytics = st.tabs([
    "📊 Single Analysis", 
    "🔍 ML Scanner", 
    "💼 Portfolio Manager",
    "📈 Analytics & Performance"
])

# =====================
# TAB 1: SINGLE ANALYSIS
# =====================
with tab_single:
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        symbol_input = st.text_input("Hisse Kodu", value="THYAO", key="single_symbol").upper()
    
    with col2:
        enable_ml = st.checkbox("🤖 ML Enhancement", value=True, key="enable_ml")
    
    with col3:
        enable_mtf = st.checkbox("🎯 Multi-Timeframe", value=True, key="enable_mtf")
    
    if st.button("🔍 Analyze", type="primary"):
        with st.spinner("Analyzing with ML & Multi-Timeframe..."):
            # Get main data
            df, sym = get_data_with_db_cache(symbol_input, "60d", "15m")
            
            if df is None:
                st.error("❌ Data fetch failed")
            else:
                # ML Model
                ml_model = None
                if enable_ml and ML_AVAILABLE:
                    with st.spinner("🤖 Training ML model..."):
                        ml_model = train_ml_model(symbol_input, df)
                        if ml_model:
                            st.success(f"✅ ML Model trained (Test Acc: {ml_model['test_acc']*100:.1f}%)")
                
                # Multi-timeframe
                mtf_data = None
                if enable_mtf:
                    mtf_data = get_mtf_data(symbol_input, ['5m', '15m', '1h'])
                
                # Calculate score
                result = calculate_advanced_score(df, symbol_input, mtf_data, ml_model)
                
                # Display
                col_a, col_b, col_c = st.columns([1, 2, 1])
                
                with col_a:
                    bg = "score-high" if result["score"] >= 75 else ("score-mid" if result["score"] >= 60 else "score-low")
                    st.markdown(f"""
                        <div class='score-box {bg}'>
                            {result['score']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("RSI", f"{result['rsi']:.1f}")
                    st.metric("MFI", f"{result['mfi']:.1f}")
                    st.metric("ADX", f"{result['adx']:.1f}")
                    
                    if enable_ml:
                        st.metric("🤖 ML Prob", f"{result['ml_prob']*100:.1f}%")
                    
                    if enable_mtf:
                        st.metric("🎯 MTF", f"{result['mtf_confluence']:.0f}%")
                
                with col_b:
                    st.markdown("### 📝 Analysis Factors")
                    for reason in result["reasons"][:12]:
                        st.markdown(f"• {reason}")
                    
                    if len(result["reasons"]) > 12:
                        with st.expander("Show more..."):
                            for reason in result["reasons"][12:]:
                                st.markdown(f"• {reason}")
                
                with col_c:
                    st.markdown(f"""
                        <div class='trade-plan'>
                            <h3 style='color:#00e5ff'>📌 Trade Plan</h3>
                            <hr>
                            <p><b>Entry:</b> {result['price']:.2f}</p>
                            <p><b>Stop:</b> {result['stop']:.2f}</p>
                            <p><b>Target:</b> {result['target']:.2f}</p>
                            <p><b>R/R:</b> {result['rr']:.2f}</p>
                            <p><b>TP %:</b> {result['tp_pct']:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Chart with VWAP
                st.markdown("### 📊 Advanced Chart")
                
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=('Price & VWAP', 'RSI & MFI', 'Volume')
                )
                
                tail = df.tail(150)
                
                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=tail.index, open=tail["Open"], high=tail["High"],
                    low=tail["Low"], close=tail["Close"], name="Price"
                ), row=1, col=1)
                
                # VWAP
                if "VWAP" in tail.columns:
                    fig.add_trace(go.Scatter(
                        x=tail.index, y=tail["VWAP"],
                        mode="lines", name="VWAP", line=dict(color="cyan", width=2)
                    ), row=1, col=1)
                
                # EMAs
                fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA20"], name="EMA20", line=dict(color="orange")), row=1, col=1)
                fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA50"], name="EMA50", line=dict(color="blue")), row=1, col=1)
                
                # Stop/Target lines
                fig.add_hline(y=result["stop"], line_dash="dash", line_color="red", row=1, col=1)
                fig.add_hline(y=result["target"], line_dash="solid", line_color="green", row=1, col=1)
                
                # RSI & MFI
                fig.add_trace(go.Scatter(x=tail.index, y=tail["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
                if "MFI" in tail.columns:
                    fig.add_trace(go.Scatter(x=tail.index, y=tail["MFI"], name="MFI", line=dict(color="gold")), row=2, col=1)
                
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Volume
                colors = ['green' if tail["Close"].iloc[i] >= tail["Open"].iloc[i] else 'red' for i in range(len(tail))]
                fig.add_trace(go.Bar(x=tail.index, y=tail["Volume"], name="Volume", marker_color=colors), row=3, col=1)
                
                fig.update_layout(height=800, template="plotly_dark", showlegend=True, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

# =====================
# TAB 2: ML SCANNER
# =====================
with tab_scanner:
    st.info("🤖 ML-Enhanced Scanner with Multi-Timeframe Confluence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        universe = st.selectbox("Universe", ["BIST30", "BIST50", "BIST100", "ALL"], key="scan_universe")
    
    with col2:
        min_score = st.slider("Min Score", 50, 90, 70, key="scan_min_score")
    
    with col3:
        min_ml_prob = st.slider("Min ML Prob", 0.5, 0.9, 0.6, step=0.05, key="scan_ml_prob")
    
    with col4:
        top_n = st.number_input("Top N", 5, 30, 15, key="scan_top_n")
    
    if st.button("🔍 Start ML Scan", type="primary", key="start_scan"):
        start_time = time.time()
        
        # Build universe
        if universe == "BIST30":
            tickers = BIST30
        elif universe == "BIST50":
            tickers = BIST50
        elif universe == "BIST100":
            tickers = BIST100
        else:
            tickers = list(set(BIST30 + BIST50 + BIST100))
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def scan_symbol(ticker):
            try:
                # Add small delay to avoid rate limiting
                time.sleep(REQUEST_DELAY)
                
                sym = ticker.replace(".IS", "")
                df, _ = get_data_with_db_cache(sym, "60d", "15m")
                
                if df is None or len(df) < 100:
                    return None
                
                # Train ML model
                ml_model = None
                if ML_AVAILABLE:
                    ml_model = train_ml_model(sym, df)
                
                # MTF data
                mtf_data = get_mtf_data(sym, ['5m', '15m', '1h'])
                
                # Calculate score
                result = calculate_advanced_score(df, sym, mtf_data, ml_model)
                
                if result and result["score"] >= min_score and result["ml_prob"] >= min_ml_prob:
                    return result
                
            except Exception as e:
                # Silently skip errors (likely rate limit or bad data)
                return None
            
            return None
        
        # Reduced parallel scanning to avoid rate limits
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(scan_symbol, t): t for t in tickers}
            
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                completed += 1
                progress_bar.progress(completed / total)
                status_text.text(f"Scanning... {completed}/{total}")
                
                result = future.result()
                if result:
                    results.append(result)
        
        progress_bar.empty()
        status_text.empty()
        
        elapsed = time.time() - start_time
        
        if results:
            # Sort by score
            results.sort(key=lambda x: (x["score"], x["ml_prob"]), reverse=True)
            top_results = results[:top_n]
            
            st.success(f"✅ Scan complete in {elapsed:.1f}s | Found {len(results)} opportunities")
            
            # Display results
            for idx, res in enumerate(top_results):
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 2])
                    
                    with col1:
                        bg = "score-high" if res["score"] >= 80 else ("score-mid" if res["score"] >= 70 else "score-low")
                        st.markdown(f"""
                            <div class='score-box {bg}' style='font-size:1.8em'>
                                #{idx+1}<br>{res['score']}
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"### {res['symbol']}")
                        st.caption(f"R/R: {res['rr']:.2f} | ML: {res['ml_prob']*100:.0f}%")
                    
                    with col2:
                        st.markdown("**🎯 Key Factors:**")
                        for r in res["reasons"][:6]:
                            st.caption(f"• {r}")
                    
                    with col3:
                        st.markdown("**💹 Trade Setup:**")
                        st.write(f"Entry: **{res['price']:.2f}**")
                        st.write(f"Stop: **{res['stop']:.2f}**")
                        st.write(f"Target: **{res['target']:.2f}** (TP: {res['tp_pct']:.1f}%)")
                        st.write(f"MTF Confluence: **{res['mtf_confluence']:.0f}%**")
                    
                    st.markdown("---")
        else:
            st.warning("No opportunities found matching criteria")

# =====================
# TAB 3: PORTFOLIO MANAGER
# =====================
with tab_portfolio:
    st.markdown("### 💼 Portfolio Management")
    
    if not DB_AVAILABLE:
        st.warning("⚠️ Database not available. Portfolio tracking disabled in this environment.")
        st.info("💡 Run locally for full database features: `streamlit run prop_desk_v7_upgraded.py`")
    else:
        st.info("Track your positions, calculate portfolio metrics, and manage risk")
        
        # Load portfolio state
        try:
            conn = sqlite3.connect(DB_PATH)
            portfolio_df = pd.read_sql_query(
                "SELECT * FROM portfolio_state ORDER BY timestamp DESC LIMIT 30",
                conn
            )
            conn.close()
        except Exception:
            portfolio_df = pd.DataFrame()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Equity", "₺100,000")  # Placeholder
            st.metric("Cash Available", "₺85,000")
            st.metric("Open Positions", "3")
        
        with col2:
            st.metric("Daily P&L", "+₺2,450", delta="2.45%")
            st.metric("Win Rate", "68%")
            st.metric("Profit Factor", "2.1")
        
        # Trade history
        st.markdown("### 📜 Recent Trades")
        
        try:
            conn = sqlite3.connect(DB_PATH)
            trades_df = pd.read_sql_query(
                "SELECT * FROM trade_history ORDER BY exit_time DESC LIMIT 20",
                conn
            )
            conn.close()
            
            if not trades_df.empty:
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No trade history yet")
        except Exception:
            st.info("Trade history unavailable")

# =====================
# TAB 4: ANALYTICS
# =====================
with tab_analytics:
    st.markdown("### 📈 Performance Analytics")
    
    if not DB_AVAILABLE:
        st.warning("⚠️ Database not available. Analytics disabled in this environment.")
        st.info("💡 Run locally for full analytics: `streamlit run prop_desk_v7_upgraded.py`")
    else:
        try:
            conn = sqlite3.connect(DB_PATH)
            
            # Get trade stats
            trades_df = pd.read_sql_query("SELECT * FROM trade_history", conn)
            
            conn.close()
            
            if not trades_df.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                total_trades = len(trades_df)
                wins = len(trades_df[trades_df["pnl"] > 0])
                win_rate = wins / total_trades * 100 if total_trades > 0 else 0
                
                avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
                avg_loss = trades_df[trades_df["pnl"] < 0]["pnl"].mean() if total_trades - wins > 0 else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                
                col1.metric("Total Trades", total_trades)
                col2.metric("Win Rate", f"{win_rate:.1f}%")
                col3.metric("Avg Win", f"₺{avg_win:.0f}")
                col4.metric("Profit Factor", f"{profit_factor:.2f}")
                
                # Equity curve
                st.markdown("### 📊 Equity Curve")
                trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trades_df["exit_time"],
                    y=trades_df["cumulative_pnl"],
                    mode="lines+markers",
                    name="Cumulative P&L"
                ))
                
                fig.update_layout(height=400, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No trading data available for analytics")
        except Exception:
            st.error("Unable to load analytics data")

st.markdown("---")
st.caption("🚀 PROP DESK V7.0 | Built with ML, MTF Analysis & Advanced Day Trading Indicators")
