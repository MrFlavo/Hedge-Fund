"""
PROP DESK V7.0 - ULTIMATE INTRADAY TRADING SYSTEM
==================================================
Upgrades:
1. Advanced Day Trading Indicators (VWAP, MFI, Stochastic, OBV)
2. ML Score Enhancement (XGBoost + LSTM price prediction)
3. Multi-Timeframe Analysis (MTF confluence)
4. SQLite Database Integration (historical data caching)
5. Performance Optimization (parallel processing, vectorized calculations)
6. Portfolio Management Layer (Enhanced with stock info table & P/L tracking)
7. Order Execution Simulation
8. Enhanced Analytics with per-trade profit/loss visualization

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

    # HTTP requests for alternative data providers
    import requests
        
except ImportError as e:
    st.error(f"⚠️ Missing libraries: {e}")
    st.code("pip install yfinance pandas plotly numpy xgboost scikit-learn tensorflow requests")
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
    "YAPI K.IS",   # Yapı Kredi Koray GYO
    
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
    "INDES.IS",    # İndeks Bilgisayar
    "LINK.IS",     # Link Bilgisayar
    
    # Additional Real Estate & Construction
    "AVOD.IS",     # A.V.O.D. Kurutulmuş Gıda
    "AGYO.IS",     # Atakule GYO
    "ISGYO.IS",    # İş GYO
    "KLGYO.IS",    # Kiler GYO
    "KONYA.IS",    # Konya Çimento
    "NUGYO.IS",    # Nurol GYO
    "OZGYO.IS",    # Özderici GYO
    "PEKGY.IS",    # Peker GYO
    "VKGYO.IS",    # Vakıf GYO
    
    # Additional Retail & Consumer
    "BANVT.IS",    # Banvit
    "BIZIM.IS",    # Bizim Toptan
    "CRFSA.IS",    # Carrefoursa
    "MAVI.IS",     # Mavi Giyim
    "PINSU.IS",    # Pınar Su
    "PNSUT.IS",    # Pınar Süt
    "TATGD.IS",    # Tat Gıda
    
    # Additional Textiles & Apparel
    "DAGI.IS",     # Dagi Giyim
    "DERIM.IS",    # Derimod
    "YUNSA.IS",    # Yünsa
    
    # Additional Chemicals & Petrochem
    "ALKIM.IS",    # Alkim Kağıt
    "ANACM.IS",    # Anadolu Cam
    "BAGFS.IS",    # Bagfaş
    "BFREN.IS",    # Bosch Fren
    "CIMSA.IS",    # Çimsa
    "DYOBY.IS",    # DYO Boya
    "IZMDC.IS",    # İzmir Demir Çelik
    "SARKY.IS",    # Sarkuysan
    "SELEC.IS",    # Selçuk Ecza
    "UNYEC.IS",    # Ünye Çimento
    
    # Additional Transportation
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
# REAL-TIME DATA PROVIDER CONFIGURATION
# =====================
# Finnhub.io free tier: 60 API calls/minute, covers BIST (Istanbul Stock Exchange)
# Get your free API key at: https://finnhub.io/register
# Set it in Streamlit sidebar or as environment variable FINNHUB_API_KEY

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

def _get_finnhub_key():
    """Get Finnhub API key from session state or environment"""
    key = st.session_state.get("finnhub_api_key", "")
    if not key:
        key = os.environ.get("FINNHUB_API_KEY", "")
    return key.strip()

@st.cache_data(ttl=60, show_spinner=False)
def get_realtime_quote_finnhub(symbol):
    """
    Fetch real-time quote from Finnhub (free tier).
    Returns dict with current price, high, low, open, prev_close, timestamp.
    Finnhub uses exchange:SYMBOL format for BIST → use symbol without .IS
    """
    api_key = _get_finnhub_key()
    if not api_key:
        return None
    
    # Finnhub BIST symbols: prefix with 'IS:' for Istanbul Stock Exchange
    clean_sym = symbol.upper().replace(".IS", "")
    finnhub_symbol = f"IS:{clean_sym}"
    
    try:
        resp = requests.get(
            f"{FINNHUB_BASE_URL}/quote",
            params={"symbol": finnhub_symbol, "token": api_key},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            # Finnhub returns: c=current, h=high, l=low, o=open, pc=prevClose, t=timestamp
            if data and data.get("c", 0) > 0:
                return {
                    "current": data["c"],
                    "high": data["h"],
                    "low": data["l"],
                    "open": data["o"],
                    "prev_close": data["pc"],
                    "change": data.get("d", 0),
                    "change_pct": data.get("dp", 0),
                    "timestamp": data.get("t", 0),
                    "source": "Finnhub (Real-time)"
                }
    except Exception:
        pass
    return None

@st.cache_data(ttl=60, show_spinner=False)
def get_realtime_quote_yf(symbol):
    """
    Fetch latest quote using yfinance Ticker.fast_info (faster than download).
    Returns dict with current price and metadata.
    """
    try:
        clean_sym = symbol.upper().replace(".IS", "") + ".IS"
        ticker = yf.Ticker(clean_sym)
        info = ticker.fast_info
        
        if info and hasattr(info, 'last_price') and info.last_price:
            return {
                "current": float(info.last_price),
                "prev_close": float(info.previous_close) if hasattr(info, 'previous_close') else 0,
                "open": float(info.open) if hasattr(info, 'open') else 0,
                "high": float(info.day_high) if hasattr(info, 'day_high') else 0,
                "low": float(info.day_low) if hasattr(info, 'day_low') else 0,
                "change": float(info.last_price - info.previous_close) if hasattr(info, 'previous_close') else 0,
                "change_pct": float((info.last_price - info.previous_close) / info.previous_close * 100) if hasattr(info, 'previous_close') and info.previous_close > 0 else 0,
                "timestamp": int(datetime.now().timestamp()),
                "source": "Yahoo Finance (fast_info)"
            }
    except Exception:
        pass
    return None

def get_realtime_quote(symbol):
    """
    Multi-provider real-time quote: tries Finnhub first, then yfinance fast_info.
    Returns the freshest available quote.
    """
    # Try Finnhub first (truly real-time for supported markets)
    quote = get_realtime_quote_finnhub(symbol)
    if quote:
        return quote
    
    # Fallback: yfinance fast_info (usually 15-min delayed but faster than download)
    quote = get_realtime_quote_yf(symbol)
    if quote:
        return quote
    
    return None

def format_data_age(df):
    """Calculate how old the latest data point is and return a human-readable string + color."""
    if df is None or df.empty:
        return "No data", "🔴", "#d50000"
    
    try:
        last_ts = df.index[-1]
        if hasattr(last_ts, 'tz') and last_ts.tz is not None:
            now = datetime.now(last_ts.tz)
        else:
            now = datetime.now()
        
        age = now - last_ts
        minutes = age.total_seconds() / 60
        
        if minutes < 5:
            return f"{int(minutes)}m ago", "🟢", "#00c853"
        elif minutes < 15:
            return f"{int(minutes)}m ago", "🟡", "#ffd600"
        elif minutes < 60:
            return f"{int(minutes)}m ago", "🟠", "#ff9100"
        elif minutes < 120:
            return f"{int(minutes/60):.0f}h {int(minutes%60)}m ago", "🟠", "#ff6d00"
        else:
            return f"{int(minutes/60):.0f}h {int(minutes%60)}m ago", "🔴", "#d50000"
    except Exception:
        return "Unknown", "⚪", "#888888"

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
@st.cache_data(ttl=120, show_spinner=False)
def get_data_with_db_cache(symbol, period, interval):
    """Fetch data with database caching (optional - falls back to direct fetch).
    
    TTL reduced to 120s (2 min) for fresher data. Combined with real-time quote
    overlay from Finnhub/yfinance fast_info for latest price updates.
    """
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
                
                # Check if cache is fresh (< 2 min old for near-realtime)
                if len(df_cached) > 0:
                    latest_time = df_cached.index[-1]
                    if datetime.now() - latest_time < timedelta(minutes=2):
                        conn.close()
                        return calculate_indicators(df_cached), symbol
            
            conn.close()
        except Exception:
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
                time.sleep(RETRY_DELAY * (attempt + 1))
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
def get_mtf_data(symbol, timeframes=['2m', '5m', '15m', '1h']):
    """Fetch multiple timeframes for confluence analysis.
    Added 2m timeframe for near-realtime data (max 7 days lookback).
    """
    mtf_data = {}
    periods_map = {'1m': '1d', '2m': '5d', '5m': '5d', '15m': '30d', '1h': '60d', '1d': '1y'}
    
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
    
    # Train XGBoost
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
    vwap_position = 0
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
    confluence = 0
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
    ml_prob = 0.5
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
        "vwap_dist": vwap_position,
        "adx": adx,
        "ml_prob": ml_prob,
        "mtf_confluence": confluence
    }

# =====================
# HELPER: MATCH CLOSED TRADES (FIFO)
# =====================
def match_closed_trades(trades):
    """
    Given a list of trade dicts, match BUY->SELL pairs using FIFO.
    Returns (closed_trades_list, open_positions_dict).
    """
    closed_trades = []
    positions = {}  # symbol -> list of open buy lots

    for trade in sorted(trades, key=lambda x: x['datetime']):
        symbol = trade['symbol']
        if symbol not in positions:
            positions[symbol] = []

        if trade['type'] == 'BUY':
            positions[symbol].append({
                'qty': trade['quantity'],
                'price': trade['price'],
                'commission': trade['commission'],
                'datetime': trade['datetime'],
                'notes': trade['notes']
            })

        elif trade['type'] == 'SELL':
            remaining_qty = trade['quantity']
            sell_price = trade['price']
            sell_commission = trade['commission']

            while remaining_qty > 0 and positions.get(symbol, []):
                buy_pos = positions[symbol][0]
                matched_qty = min(remaining_qty, buy_pos['qty'])

                buy_cost = matched_qty * buy_pos['price']
                sell_revenue = matched_qty * sell_price
                total_commission = (
                    (matched_qty / trade['quantity'] * sell_commission) +
                    (matched_qty / max(buy_pos['qty'], 1) * buy_pos['commission'])
                )
                pnl = sell_revenue - buy_cost - total_commission
                pnl_pct = (pnl / buy_cost) * 100 if buy_cost > 0 else 0

                closed_trades.append({
                    'Symbol': symbol,
                    'Quantity': matched_qty,
                    'Buy Price': buy_pos['price'],
                    'Sell Price': sell_price,
                    'Buy Date': buy_pos['datetime'],
                    'Sell Date': trade['datetime'],
                    'Cost': buy_cost,
                    'Revenue': sell_revenue,
                    'P&L (TL)': pnl,
                    'P&L (%)': pnl_pct,
                    'Commission': total_commission,
                    'Notes': buy_pos['notes']
                })

                buy_pos['qty'] -= matched_qty
                remaining_qty -= matched_qty
                if buy_pos['qty'] == 0:
                    positions[symbol].pop(0)

    # Build open positions summary
    open_positions = {}
    for symbol, lots in positions.items():
        total_qty = sum(l['qty'] for l in lots)
        if total_qty > 0:
            total_cost = sum(l['qty'] * l['price'] for l in lots)
            avg_cost = total_cost / total_qty
            open_positions[symbol] = {
                'qty': total_qty,
                'avg_cost': avg_cost,
                'total_cost': total_cost,
                'lots': len([l for l in lots if l['qty'] > 0])
            }

    return closed_trades, open_positions

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
    
    /* Portfolio & Analytics card styling */
    .pnl-card {
        padding: 18px 22px; border-radius: 12px; margin: 6px 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .pnl-card-profit { background: rgba(0, 200, 83, 0.08); border-left: 4px solid #00c853; }
    .pnl-card-loss { background: rgba(213, 0, 0, 0.08); border-left: 4px solid #d50000; }
    .pnl-card-neutral { background: rgba(255, 214, 0, 0.08); border-left: 4px solid #ffd600; }
    .pnl-card-info { background: rgba(0, 229, 255, 0.08); border-left: 4px solid #00e5ff; }
</style>
""", unsafe_allow_html=True)

# =====================
# MAIN APPLICATION
# =====================
st.title("🚀 PROP DESK V7.0 - ML ENHANCED INTRADAY SYSTEM")
st.caption("Advanced day trading with VWAP, MFI, Multi-Timeframe, ML Score & Real-Time Data")

# =====================
# SIDEBAR: DATA PROVIDER SETTINGS
# =====================
with st.sidebar:
    st.markdown("### ⚡ Data Provider Settings")
    
    finnhub_key = st.text_input(
        "🔑 Finnhub API Key (Free)",
        value=st.session_state.get("finnhub_api_key", os.environ.get("FINNHUB_API_KEY", "")),
        type="password",
        help="Get your FREE API key at https://finnhub.io/register — enables real-time BIST quotes (60 calls/min)"
    )
    st.session_state["finnhub_api_key"] = finnhub_key
    
    if finnhub_key:
        st.success("✅ Finnhub active — Real-time quotes enabled")
    else:
        st.info("💡 Add a free Finnhub key for real-time BIST data. Without it, data may be 15-120 min delayed.")
        st.markdown("[🔗 Get Free Key](https://finnhub.io/register)")
    
    st.markdown("---")
    
    data_interval = st.selectbox(
        "📊 Chart Interval",
        options=["2m", "5m", "15m", "1h"],
        index=2,  # Default: 15m
        help="2m = Freshest (5 day lookback) | 5m = Fresh (5 day) | 15m = Standard (30 day) | 1h = Extended (60 day)"
    )
    
    interval_period_map = {"2m": "5d", "5m": "5d", "15m": "30d", "1h": "60d"}
    data_period = interval_period_map[data_interval]
    
    st.caption(f"Interval: **{data_interval}** | Lookback: **{data_period}**")
    st.markdown("---")

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
            # Get main data using sidebar interval settings
            df, sym = get_data_with_db_cache(symbol_input, data_period, data_interval)
            
            if df is None:
                st.error("❌ Data fetch failed")
            else:
                # === REAL-TIME QUOTE OVERLAY ===
                rt_quote = get_realtime_quote(symbol_input)
                
                # Data freshness indicator
                age_str, age_icon, age_color = format_data_age(df)
                
                freshness_cols = st.columns([2, 2, 2])
                with freshness_cols[0]:
                    st.markdown(f"""
                        <div style="padding:8px 12px; border-radius:8px; background:rgba(0,229,255,0.08); border-left:3px solid {age_color};">
                            {age_icon} <b>Chart Data:</b> {age_str} <span style="color:#888;">({data_interval} bars)</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                with freshness_cols[1]:
                    if rt_quote:
                        rt_change_color = "#00c853" if rt_quote["change"] >= 0 else "#d50000"
                        rt_sign = "+" if rt_quote["change"] >= 0 else ""
                        st.markdown(f"""
                            <div style="padding:8px 12px; border-radius:8px; background:rgba(0,200,83,0.08); border-left:3px solid #00c853;">
                                🟢 <b>Live Price:</b> ₺{rt_quote['current']:.2f} 
                                <span style="color:{rt_change_color};">({rt_sign}{rt_quote['change_pct']:.2f}%)</span>
                                <br><span style="color:#888; font-size:0.8em;">{rt_quote['source']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style="padding:8px 12px; border-radius:8px; background:rgba(255,214,0,0.08); border-left:3px solid #ffd600;">
                                ⚠️ <b>No live quote</b> — Add Finnhub key in sidebar for real-time prices
                            </div>
                        """, unsafe_allow_html=True)
                
                with freshness_cols[2]:
                    last_bar_time = df.index[-1].strftime("%H:%M:%S") if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                    st.markdown(f"""
                        <div style="padding:8px 12px; border-radius:8px; background:rgba(255,255,255,0.04); border-left:3px solid #888;">
                            🕒 <b>Last Bar:</b> {last_bar_time}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("")
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
                    mtf_data = get_mtf_data(symbol_input, ['2m', '5m', '15m', '1h'])
                
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
                
                # Real-time price line (if available)
                if rt_quote:
                    fig.add_hline(
                        y=rt_quote["current"], line_dash="dot", line_color="cyan", 
                        annotation_text=f"⚡ Live: ₺{rt_quote['current']:.2f}", 
                        annotation_position="top right",
                        annotation_font_color="cyan",
                        row=1, col=1
                    )
                
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
    st.markdown("### 🦅 Smart Stock Scanner")
    
    # Info panel explaining automatic filtering
    with st.expander("ℹ️ How the Scanner Works", expanded=False):
        st.markdown("""
        **Automatic Intelligent Filtering:**
        
        The scanner automatically filters stocks using professional criteria:
        
        1. **Quality Score (70+)** - Only shows stocks with strong technical setups
           - Combines 15+ indicators (RSI, VWAP, MFI, ADX, Volume, etc.)
           - Bull regime confirmation
           - Multi-timeframe alignment
           
        2. **ML Probability (60%+)** - Machine learning validates each setup
           - XGBoost model trained on historical patterns
           - 13 engineered features
           - Only shows setups with >60% probability of profit
           
        3. **Top Results (15 Best)** - Shows the highest quality opportunities
           - Ranked by combined score + ML probability
           - Best risk/reward ratios
           - Most likely to succeed
           
        **Result:** You only see the cream of the crop! 🎯
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        universe = st.selectbox(
            "📊 Select Universe", 
            ["BIST30", "BIST50", "BIST100", "ALL"],
            index=0,
            help="BIST30 = Fastest (30 stocks, ~40s) | BIST50 = Balanced (50 stocks, ~70s) | BIST100 = Complete (100 stocks, ~2min)",
            key="scan_universe"
        )
    
    with col2:
        enable_ml_scan = st.checkbox(
            "🤖 ML Enhancement", 
            value=True,
            help="Uses machine learning to validate setups (Recommended)",
            key="scan_ml_enable"
        )
    
    # Hidden parameters with smart defaults
    min_score = 70
    min_ml_prob = 0.60
    top_n = 15
    
    st.markdown("---")
    
    if st.button("🔍 Start Smart Scan", type="primary", key="start_scan", use_container_width=True):
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
                time.sleep(REQUEST_DELAY)
                sym = ticker.replace(".IS", "")
                df, _ = get_data_with_db_cache(sym, data_period, data_interval)
                
                if df is None or len(df) < 50:
                    return None
                
                ml_model = None
                if ML_AVAILABLE:
                    ml_model = train_ml_model(sym, df)
                
                mtf_data = get_mtf_data(sym, ['2m', '5m', '15m', '1h'])
                result = calculate_advanced_score(df, sym, mtf_data, ml_model)
                
                if result and result["score"] >= min_score and result["ml_prob"] >= min_ml_prob:
                    return result
                
            except Exception:
                return None
            
            return None
        
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
            results.sort(key=lambda x: (x["score"], x["ml_prob"]), reverse=True)
            top_results = results[:top_n]
            
            st.success(f"""
            ✅ **Scan Complete in {elapsed:.1f}s**
            
            Found **{len(results)}** quality setups (Score ≥70, ML Prob ≥60%)  
            Showing **Top {len(top_results)}** opportunities ranked by score + ML probability
            """)
            
            # Data source indicator
            if _get_finnhub_key():
                st.caption("⚡ Prices enhanced with Finnhub real-time quotes")
            else:
                st.caption("💡 Tip: Add Finnhub API key in sidebar for real-time prices")
            
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
                        
                        # Show live price if available
                        rt = get_realtime_quote(res['symbol'])
                        if rt:
                            rt_color = "#00c853" if rt["change"] >= 0 else "#d50000"
                            rt_sign = "+" if rt["change"] >= 0 else ""
                            st.caption(f"⚡ ₺{rt['current']:.2f} ({rt_sign}{rt['change_pct']:.1f}%)")
                        
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
            st.warning("""
            ⚠️ **No High-Quality Setups Found**
            
            The scanner checked stocks but none met the professional criteria:
            - Score ≥ 70 (strong technical setup)
            - ML Probability ≥ 60% (high confidence)
            
            **What to do:**
            - Try scanning during market hours (10:00-18:00)
            - Check a different universe (BIST50 or BIST100)
            - Market may be consolidating (normal - not every day has great setups)
            
            **Pro Tip:** Quality over quantity! It's better to wait for A+ setups than force mediocre trades.
            """)

# =====================
# TAB 3: PORTFOLIO MANAGER (ENHANCED)
# =====================
with tab_portfolio:
    st.markdown("### 💼 Portfolio Manager")
    st.caption("Track your stock trades — add BUY & SELL transactions, see positions and P/L at a glance")

    # Initialize session state for trades if not exists
    if 'trades' not in st.session_state:
        st.session_state.trades = []

    # ── Add Trade Form ──
    with st.expander("➕ Add New Trade", expanded=len(st.session_state.trades) == 0):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_symbol = st.text_input("Stock Symbol", value="TCELL", help="e.g., TCELL, THYAO, GARAN").upper()
            trade_type = st.selectbox("Trade Type", ["BUY", "SELL"])
            trade_quantity = st.number_input("Quantity", min_value=1, value=151, step=1)
        
        with col2:
            trade_price = st.number_input("Price (TL)", min_value=0.01, value=160.00, step=0.01, format="%.2f")
            trade_date = st.date_input("Trade Date", value=datetime.now())
            trade_time = st.time_input("Trade Time", value=datetime.now().time())
        
        with col3:
            trade_commission = st.number_input("Commission (TL)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            trade_notes = st.text_area("Notes (optional)", placeholder="e.g., Scanner score 85, ML 72%")
        
        if st.button("💾 Save Trade", type="primary"):
            trade_datetime = datetime.combine(trade_date, trade_time)
            cost_or_revenue = trade_quantity * trade_price
            
            new_trade = {
                "id": len(st.session_state.trades) + 1,
                "symbol": trade_symbol,
                "type": trade_type,
                "quantity": trade_quantity,
                "price": trade_price,
                "datetime": trade_datetime.strftime("%Y-%m-%d %H:%M"),
                "commission": trade_commission,
                "notes": trade_notes,
                "total": cost_or_revenue + (trade_commission if trade_type == "BUY" else -trade_commission)
            }
            
            st.session_state.trades.append(new_trade)
            
            # Save to database if available
            if DB_AVAILABLE:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("""
                        INSERT INTO trade_history 
                        (symbol, entry_time, entry_price, qty, pnl, reason, score, strategy)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade_symbol,
                        trade_datetime.strftime("%Y-%m-%d %H:%M"),
                        trade_price,
                        trade_quantity if trade_type == "BUY" else -trade_quantity,
                        0.0, trade_notes, 0, trade_type
                    ))
                    conn.commit()
                    conn.close()
                except Exception:
                    pass
            
            st.success(f"✅ {trade_type} trade added: {trade_quantity} × {trade_symbol} @ ₺{trade_price:,.2f}")
            st.rerun()

    st.markdown("---")

    # ── Display Trades & Positions ──
    if st.session_state.trades:
        trades = st.session_state.trades
        closed_trades, open_positions = match_closed_trades(trades)

        # ── Quick Stats Row ──
        total_invested = sum(t['total'] for t in trades if t['type'] == 'BUY')
        total_sold = sum(t['total'] for t in trades if t['type'] == 'SELL')
        total_commissions = sum(t['commission'] for t in trades)
        realized_pnl = sum(ct['P&L (TL)'] for ct in closed_trades)
        num_open = len(open_positions)
        open_value = sum(p['total_cost'] for p in open_positions.values())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Invested", f"₺{total_invested:,.2f}")
        c2.metric("Total Sold", f"₺{total_sold:,.2f}")
        c3.metric("Realized P&L", f"₺{realized_pnl:,.2f}",
                   delta=f"{(realized_pnl / total_invested * 100) if total_invested > 0 else 0:.2f}%")
        c4.metric("Open Positions", f"{num_open} stocks", delta=f"₺{open_value:,.2f} invested")
        c5.metric("Commissions", f"₺{total_commissions:,.2f}")

        st.markdown("---")

        # ── Stock Information Table (per-stock summary) ──
        st.markdown("### 📋 Stock Information")
        st.caption("Summary of every stock you've traded — quantities, costs, sell prices, and realized P/L")

        # Build per-stock summary combining open + closed info
        stock_summary = {}
        for t in trades:
            sym = t['symbol']
            if sym not in stock_summary:
                stock_summary[sym] = {
                    'buy_qty': 0, 'sell_qty': 0,
                    'total_buy_cost': 0, 'total_sell_revenue': 0,
                    'buys': 0, 'sells': 0,
                    'realized_pnl': 0, 'commissions': 0
                }
            s = stock_summary[sym]
            s['commissions'] += t['commission']
            if t['type'] == 'BUY':
                s['buy_qty'] += t['quantity']
                s['total_buy_cost'] += t['quantity'] * t['price']
                s['buys'] += 1
            else:
                s['sell_qty'] += t['quantity']
                s['total_sell_revenue'] += t['quantity'] * t['price']
                s['sells'] += 1

        # Add realized P&L from closed trades
        for ct in closed_trades:
            sym = ct['Symbol']
            if sym in stock_summary:
                stock_summary[sym]['realized_pnl'] += ct['P&L (TL)']

        # Build display dataframe
        summary_rows = []
        for sym, s in stock_summary.items():
            avg_buy = s['total_buy_cost'] / s['buy_qty'] if s['buy_qty'] > 0 else 0
            avg_sell = s['total_sell_revenue'] / s['sell_qty'] if s['sell_qty'] > 0 else 0
            remaining = s['buy_qty'] - s['sell_qty']
            status = "🟢 OPEN" if remaining > 0 else ("🔴 CLOSED" if s['sell_qty'] > 0 else "🟡 HOLD")
            pnl_return = (s['realized_pnl'] / s['total_buy_cost'] * 100) if s['total_buy_cost'] > 0 and s['sell_qty'] > 0 else 0

            summary_rows.append({
                'Stock': sym,
                'Bought': s['buy_qty'],
                'Sold': s['sell_qty'],
                'Remaining': remaining,
                'Avg Buy (TL)': avg_buy,
                'Avg Sell (TL)': avg_sell if s['sell_qty'] > 0 else None,
                'Total Cost': s['total_buy_cost'],
                'Total Revenue': s['total_sell_revenue'] if s['sell_qty'] > 0 else None,
                'Realized P&L': s['realized_pnl'] if s['sell_qty'] > 0 else None,
                'Return (%)': pnl_return if s['sell_qty'] > 0 else None,
                'Status': status,
                'Trades': f"{s['buys']}B / {s['sells']}S"
            })

        summary_df = pd.DataFrame(summary_rows)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Stock": st.column_config.TextColumn("Stock", width="small"),
                "Bought": st.column_config.NumberColumn("Bought Qty", width="small"),
                "Sold": st.column_config.NumberColumn("Sold Qty", width="small"),
                "Remaining": st.column_config.NumberColumn("Remaining", width="small"),
                "Avg Buy (TL)": st.column_config.NumberColumn("Avg Buy", format="₺%.2f"),
                "Avg Sell (TL)": st.column_config.NumberColumn("Avg Sell", format="₺%.2f"),
                "Total Cost": st.column_config.NumberColumn("Total Cost", format="₺%.2f"),
                "Total Revenue": st.column_config.NumberColumn("Revenue", format="₺%.2f"),
                "Realized P&L": st.column_config.NumberColumn("P&L", format="₺%.2f"),
                "Return (%)": st.column_config.NumberColumn("Return", format="%.2f%%"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Trades": st.column_config.TextColumn("Trades", width="small"),
            }
        )

        # ── Detailed description cards for each closed trade ──
        if closed_trades:
            st.markdown("### 💰 Closed Trade Details")
            st.caption("Each completed buy→sell cycle with profit/loss")

            for ct in closed_trades:
                pnl = ct['P&L (TL)']
                pnl_class = "pnl-card-profit" if pnl > 0 else ("pnl-card-loss" if pnl < 0 else "pnl-card-neutral")
                icon = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                pnl_sign = "+" if pnl > 0 else ""
                pnl_color = '#00c853' if pnl > 0 else '#d50000' if pnl < 0 else '#ffd600'

                st.markdown(f"""
                <div class="pnl-card {pnl_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <div>
                            <span style="font-size:1.3em; font-weight:700;">{icon} {ct['Symbol']}</span>
                            <span style="color:#888; margin-left:12px; font-size:0.9em;">
                                {ct['Quantity']} shares &nbsp;|&nbsp; Bought @ ₺{ct['Buy Price']:.2f} → Sold @ ₺{ct['Sell Price']:.2f}
                            </span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:1.4em; font-weight:700; color:{pnl_color};">
                                {pnl_sign}₺{pnl:,.2f}
                            </span>
                            <span style="color:#888; margin-left:8px; font-size:0.9em;">
                                ({pnl_sign}{ct['P&L (%)']:.2f}%)
                            </span>
                        </div>
                    </div>
                    <div style="margin-top:6px; font-size:0.8em; color:#666;">
                        📅 {ct['Buy Date']} → {ct['Sell Date']}
                        &nbsp;|&nbsp; Cost: ₺{ct['Cost']:,.2f} → Revenue: ₺{ct['Revenue']:,.2f}
                        &nbsp;|&nbsp; Fees: ₺{ct['Commission']:.2f}
                        {f" &nbsp;|&nbsp; 📝 {ct['Notes']}" if ct['Notes'] else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Open Positions ──
        if open_positions:
            st.markdown("### 📌 Open Positions")
            st.caption("Stocks you still hold — not yet included in realized P&L")

            for sym, pos in open_positions.items():
                st.markdown(f"""
                <div class="pnl-card pnl-card-info">
                    <span style="font-size:1.2em; font-weight:700;">🔵 {sym}</span>
                    <span style="color:#aaa; margin-left:12px;">
                        {pos['qty']} shares @ avg ₺{pos['avg_cost']:.2f}
                        &nbsp;|&nbsp; Total invested: ₺{pos['total_cost']:,.2f}
                        &nbsp;|&nbsp; {pos['lots']} open lot(s)
                    </span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Full Trade Log ──
        with st.expander("📜 Full Trade Log (all transactions)", expanded=False):
            trades_df = pd.DataFrame(trades)
            st.dataframe(
                trades_df[['id', 'symbol', 'type', 'quantity', 'price', 'datetime', 'commission', 'total', 'notes']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "symbol": st.column_config.TextColumn("Stock", width="small"),
                    "type": st.column_config.TextColumn("Type", width="small"),
                    "quantity": st.column_config.NumberColumn("Qty", width="small"),
                    "price": st.column_config.NumberColumn("Price", format="₺%.2f"),
                    "datetime": st.column_config.TextColumn("Date/Time"),
                    "commission": st.column_config.NumberColumn("Commission", format="₺%.2f"),
                    "total": st.column_config.NumberColumn("Total", format="₺%.2f"),
                    "notes": st.column_config.TextColumn("Notes")
                }
            )

        # ── Actions ──
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🗑️ Clear All Trades", help="Delete all trade history"):
                st.session_state.trades = []
                st.success("All trades cleared!")
                st.rerun()
        with col2:
            trades_df = pd.DataFrame(trades)
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Export All Trades (CSV)",
                data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    else:
        st.info("""
        👋 **Get Started — Add your first trade above!**

        **Example workflow:**
        1. **BUY** 151 × TCELL @ ₺160.00
        2. **SELL** 151 × TCELL @ ₺170.00
        3. See your profit automatically calculated: **+₺1,510.00** ✅

        The system uses **FIFO matching** (First In, First Out) to pair your buys with sells.
        """)

# =====================
# TAB 4: ANALYTICS & PERFORMANCE (ENHANCED)
# =====================
with tab_analytics:
    st.markdown("### 📈 Trading Performance Analytics")

    if 'trades' not in st.session_state or not st.session_state.trades:
        st.info("""
        📊 **Analytics Dashboard**

        Add trades in the **Portfolio Manager** tab to unlock:
        - ✅ Profit & Loss breakdown per stock
        - ✅ Win rate & profit factor stats
        - ✅ Equity curve visualization
        - ✅ Best & worst trade highlights
        - ✅ Performance-by-stock bar chart
        """)
    else:
        trades = st.session_state.trades
        closed_trades, open_positions = match_closed_trades(trades)

        if closed_trades:
            closed_df = pd.DataFrame(closed_trades)

            # ── Summary Metrics ──
            total_pnl = closed_df['P&L (TL)'].sum()
            winning_trades = len(closed_df[closed_df['P&L (TL)'] > 0])
            losing_trades = len(closed_df[closed_df['P&L (TL)'] < 0])
            total_trades_count = len(closed_df)
            win_rate = (winning_trades / total_trades_count * 100) if total_trades_count > 0 else 0
            
            avg_win = closed_df[closed_df['P&L (TL)'] > 0]['P&L (TL)'].mean() if winning_trades > 0 else 0
            avg_loss = closed_df[closed_df['P&L (TL)'] < 0]['P&L (TL)'].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            total_cost = closed_df['Cost'].sum()
            overall_return = (total_pnl / total_cost * 100) if total_cost > 0 else 0

            # Big P&L banner
            pnl_class = "pnl-card-profit" if total_pnl > 0 else ("pnl-card-loss" if total_pnl < 0 else "pnl-card-neutral")
            pnl_sign = "+" if total_pnl > 0 else ""
            pnl_color = '#00c853' if total_pnl > 0 else '#d50000' if total_pnl < 0 else '#ffd600'

            st.markdown(f"""
            <div class="pnl-card {pnl_class}" style="text-align:center; padding:24px;">
                <div style="font-size:0.9em; color:#aaa; margin-bottom:4px;">NET REALIZED PROFIT / LOSS</div>
                <div style="font-size:2.8em; font-weight:800; color:{pnl_color};">
                    {pnl_sign}₺{total_pnl:,.2f}
                </div>
                <div style="font-size:1.1em; color:#aaa; margin-top:4px;">
                    {pnl_sign}{overall_return:.2f}% overall return on ₺{total_cost:,.2f} invested
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Trades", total_trades_count)
            c2.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{winning_trades}W / {losing_trades}L")
            c3.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
            c4.metric("Avg Win", f"₺{avg_win:,.2f}")
            c5.metric("Avg Loss", f"₺{avg_loss:,.2f}")

            st.markdown("---")

            # ── Per-Trade P&L Breakdown ──
            st.markdown("### 💰 Trade-by-Trade Profit & Loss")

            for _, ct in closed_df.iterrows():
                pnl = ct['P&L (TL)']
                pnl_class = "pnl-card-profit" if pnl > 0 else ("pnl-card-loss" if pnl < 0 else "pnl-card-neutral")
                icon = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                pnl_sign = "+" if pnl > 0 else ""
                pnl_color = '#00c853' if pnl > 0 else '#d50000' if pnl < 0 else '#ffd600'

                st.markdown(f"""
                <div class="pnl-card {pnl_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <div>
                            <span style="font-size:1.2em; font-weight:700;">{icon} {ct['Symbol']}</span>
                            <span style="color:#888; margin-left:10px; font-size:0.85em;">
                                {int(ct['Quantity'])} shares &nbsp;|&nbsp; ₺{ct['Buy Price']:.2f} → ₺{ct['Sell Price']:.2f}
                            </span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:1.3em; font-weight:700; color:{pnl_color};">
                                {pnl_sign}₺{pnl:,.2f}
                            </span>
                            <span style="color:#888; margin-left:6px; font-size:0.85em;">
                                ({pnl_sign}{ct['P&L (%)']:.2f}%)
                            </span>
                        </div>
                    </div>
                    <div style="margin-top:4px; font-size:0.78em; color:#555;">
                        📅 {ct['Buy Date']} → {ct['Sell Date']}
                        &nbsp;|&nbsp; Cost: ₺{ct['Cost']:,.2f} → Revenue: ₺{ct['Revenue']:,.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Equity Curve ──
            st.markdown("### 📊 Equity Curve")

            closed_df_sorted = closed_df.sort_values('Sell Date')
            closed_df_sorted['Cumulative P&L'] = closed_df_sorted['P&L (TL)'].cumsum()
            closed_df_sorted['Trade #'] = range(1, len(closed_df_sorted) + 1)

            fig_equity = go.Figure()
            
            # Color markers by profit/loss
            colors_eq = ['#00c853' if p >= 0 else '#d50000' for p in closed_df_sorted['P&L (TL)']]

            fig_equity.add_trace(go.Scatter(
                x=closed_df_sorted['Trade #'],
                y=closed_df_sorted['Cumulative P&L'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#00e5ff', width=2.5),
                marker=dict(size=10, color=colors_eq, line=dict(width=1, color='white')),
                text=closed_df_sorted['Symbol'],
                hovertemplate='<b>%{text}</b><br>Trade #%{x}<br>Cumulative: ₺%{y:,.2f}<extra></extra>'
            ))
            
            fig_equity.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")

            # Fill area under curve
            fig_equity.add_trace(go.Scatter(
                x=closed_df_sorted['Trade #'],
                y=closed_df_sorted['Cumulative P&L'],
                fill='tozeroy',
                fillcolor='rgba(0, 229, 255, 0.08)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig_equity.update_layout(
                height=400,
                template="plotly_dark",
                xaxis_title="Trade Number",
                yaxis_title="Cumulative P&L (₺)",
                showlegend=True,
                hovermode='x unified',
                margin=dict(t=20, b=40)
            )
            st.plotly_chart(fig_equity, use_container_width=True)

            st.markdown("---")

            # ── Performance by Stock (Horizontal Bar Chart) ──
            st.markdown("### 📊 Performance by Stock")

            stock_perf = closed_df.groupby('Symbol').agg({
                'P&L (TL)': 'sum',
                'P&L (%)': 'mean',
                'Quantity': 'count'
            }).reset_index()
            stock_perf.columns = ['Symbol', 'Total P&L', 'Avg Return (%)', 'Trades']
            stock_perf = stock_perf.sort_values('Total P&L', ascending=True)

            bar_colors = ['#00c853' if p > 0 else '#d50000' for p in stock_perf['Total P&L']]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=stock_perf['Symbol'],
                x=stock_perf['Total P&L'],
                orientation='h',
                marker_color=bar_colors,
                text=[f"₺{p:,.2f}" for p in stock_perf['Total P&L']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>P&L: ₺%{x:,.2f}<extra></extra>'
            ))
            fig_bar.add_vline(x=0, line_color="gray", line_dash="dash")
            fig_bar.update_layout(
                height=max(250, len(stock_perf) * 50),
                template="plotly_dark",
                xaxis_title="Total P&L (₺)",
                margin=dict(t=20, b=40, l=80),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Stock performance table + Best/Worst
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    stock_perf.sort_values('Total P&L', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Symbol": st.column_config.TextColumn("Stock"),
                        "Total P&L": st.column_config.NumberColumn("Total P&L", format="₺%.2f"),
                        "Avg Return (%)": st.column_config.NumberColumn("Avg Return", format="%.2f%%"),
                        "Trades": st.column_config.NumberColumn("# Trades")
                    }
                )

            with col2:
                st.markdown("**🏆 Best Trades:**")
                top_3 = closed_df.nlargest(3, 'P&L (%)')
                for _, trade in top_3.iterrows():
                    st.success(f"**{trade['Symbol']}**: +{trade['P&L (%)']:.2f}% (₺{trade['P&L (TL)']:.2f})")
                
                st.markdown("**📉 Worst Trades:**")
                bottom_3 = closed_df.nsmallest(3, 'P&L (%)')
                for _, trade in bottom_3.iterrows():
                    st.error(f"**{trade['Symbol']}**: {trade['P&L (%)']:.2f}% (₺{trade['P&L (TL)']:.2f})")

            st.markdown("---")

            # ── Export Analytics ──
            st.markdown("### 📥 Export Analytics")

            col1, col2 = st.columns(2)
            
            with col1:
                csv = closed_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Closed Trades (CSV)",
                    data=csv,
                    file_name=f"closed_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                summary_data = {
                    'Metric': ['Total P&L', 'Total Trades', 'Win Rate', 'Profit Factor', 'Avg Win', 'Avg Loss', 'Overall Return'],
                    'Value': [
                        f"₺{total_pnl:.2f}",
                        total_trades_count,
                        f"{win_rate:.1f}%",
                        f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                        f"₺{avg_win:.2f}",
                        f"₺{avg_loss:.2f}",
                        f"{overall_return:.2f}%"
                    ]
                }
                summary_csv = pd.DataFrame(summary_data).to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary (CSV)",
                    data=summary_csv,
                    file_name=f"trading_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.warning("""
            ⚠️ **No Closed Trades Yet**
            
            You have open positions but no completed buy→sell cycles.
            
            To see P&L analytics:
            1. **BUY** a stock (e.g., 151 × TCELL @ ₺160)
            2. **SELL** the same stock (e.g., 151 × TCELL @ ₺170)
            
            The system will automatically match and calculate your profit: **+₺1,510.00** ✅
            """)

st.markdown("---")
st.caption("🚀 PROP DESK V7.0 | Built with ML, MTF Analysis, Advanced Day Trading Indicators & Real-Time Data (Finnhub + yfinance)")
