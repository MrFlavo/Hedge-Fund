"""
PROP DESK V7.0 - ULTIMATE INTRADAY TRADING SYSTEM
==================================================
Optimized for BIST (Borsa Istanbul) intraday trading
Enhanced Portfolio Manager & Analytics tabs with stock info and P/L tracking
"""

import streamlit as st
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="PROP DESK V7.0 - ML ENHANCED INTRADAY", page_icon="🚀")

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
    try:
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
        st.warning("⚠️ ML libraries not available. Install: pip install xgboost scikit-learn")
    LSTM_AVAILABLE = False
except ImportError as e:
    st.error(f"⚠️ Missing libraries: {e}")
    st.code("pip install yfinance pandas plotly numpy xgboost scikit-learn")
    st.stop()

import tempfile, os
TEMP_DIR = tempfile.gettempdir()
DB_PATH = os.path.join(TEMP_DIR, "bist_trading_data.db")

def init_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS price_data (
            symbol TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, interval TEXT,
            PRIMARY KEY (symbol, timestamp, interval))""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS indicator_cache (
            symbol TEXT, timestamp TEXT, indicator_name TEXT, value REAL, interval TEXT,
            PRIMARY KEY (symbol, timestamp, indicator_name, interval))""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, entry_time TEXT, exit_time TEXT,
            entry_price REAL, exit_price REAL, qty INTEGER, pnl REAL, reason TEXT, score INTEGER, strategy TEXT)""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS portfolio_state (
            timestamp TEXT PRIMARY KEY, total_equity REAL, cash REAL, positions TEXT, daily_pnl REAL)""")
        conn.commit(); conn.close()
        return True
    except Exception as e:
        st.warning(f"⚠️ Database initialization failed: {e}.")
        return False

DB_AVAILABLE = init_database()

BIST30 = ["AKBNK.IS","GARAN.IS","ISCTR.IS","YKBNK.IS","ASELS.IS","EREGL.IS","FROTO.IS","TOASO.IS",
    "TUPRS.IS","SISE.IS","SASA.IS","KCHOL.IS","SAHOL.IS","EKGYO.IS","BIMAS.IS","MGROS.IS",
    "ULKER.IS","TCELL.IS","TTKOM.IS","THYAO.IS","PGSUS.IS","TAVHL.IS","PETKM.IS","GUBRF.IS",
    "ENKA.IS","KOZAL.IS","KRDMD.IS","AEFES.IS","ASTOR.IS","DSFAK.IS"]

BIST50 = list(set(BIST30 + ["HALKB.IS","VAKBN.IS","SKBNK.IS","ARCLK.IS","VESTL.IS","OTKAR.IS",
    "DOAS.IS","TTRAK.IS","ALARK.IS","AYGAZ.IS","TKFEN.IS","AKSEN.IS","AKENR.IS","ZOREN.IS",
    "GESAN.IS","LOGO.IS","KRONT.IS","SOKM.IS","CCOLA.IS","ADANA.IS","BUCIM.IS",
    "BJKAS.IS","GSRAY.IS","TSPOR.IS","FENER.IS"]))

BIST100 = list(set(BIST50 + ["ALBRK.IS","ICBCT.IS","QNBFB.IS","TSKB.IS","AKSA.IS","BRISA.IS",
    "BRSAN.IS","EGEEN.IS","GOLTS.IS","KARSN.IS","KLMSN.IS","NETAS.IS","SODA.IS","TRKCM.IS",
    "AKSUE.IS","AYEN.IS","CLEBI.IS","ENJSA.IS","GWIND.IS","HUNER.IS","ODAS.IS","PENTA.IS",
    "ARENA.IS","INDES.IS","LINK.IS","AVOD.IS","AGYO.IS","ISGYO.IS","KLGYO.IS","KONYA.IS",
    "NUGYO.IS","OZGYO.IS","PEKGY.IS","VKGYO.IS","BANVT.IS","BIZIM.IS","CRFSA.IS","MAVI.IS",
    "PINSU.IS","PNSUT.IS","TATGD.IS","DAGI.IS","DERIM.IS","YUNSA.IS","ALKIM.IS","ANACM.IS",
    "BAGFS.IS","BFREN.IS","CIMSA.IS","DYOBY.IS","IZMDC.IS","SARKY.IS","SELEC.IS","UNYEC.IS",
    "GSDHO.IS","RYGYO.IS","HEKTS.IS","OYAKC.IS","SMRTG.IS"]))

MAX_PARALLEL_WORKERS = 4; REQUEST_DELAY = 0.5; RETRY_DELAY = 2
# =====================
# OPTIMIZED INDICATOR CALCULATIONS
# =====================
def _ema_vectorized(series, length):
    return series.ewm(span=length, adjust=False).mean()

def _sma_vectorized(series, length):
    return series.rolling(length, min_periods=1).mean()

def _rsi_vectorized(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr_vectorized(high, low, close, length=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def _roc_vectorized(series, length):
    return (series / series.shift(length) - 1.0) * 100.0

def _stochastic_vectorized(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(d_period).mean()
    return k, d

def _mfi_vectorized(high, low, close, volume, length=14):
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
    obv = pd.Series(0.0, index=close.index)
    obv.iloc[0] = volume.iloc[0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]: obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]: obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else: obv.iloc[i] = obv.iloc[i-1]
    return obv

def _vwap_vectorized(high, low, close, volume):
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

def _adx_vectorized(high, low, close, length=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    tr = _atr_vectorized(high, low, close, length) * length
    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / tr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx, plus_di, minus_di

# =====================
# DATA FETCHING WITH CACHING
# =====================
@st.cache_data(ttl=900, show_spinner=False)
def get_data_with_db_cache(symbol, period, interval):
    symbol = symbol.upper().strip()
    if len(symbol) <= 5 and not symbol.endswith(".IS"): symbol += ".IS"
    if DB_AVAILABLE:
        try:
            conn = sqlite3.connect(DB_PATH)
            df_cached = pd.read_sql_query(
                "SELECT * FROM price_data WHERE symbol = ? AND interval = ? ORDER BY timestamp DESC LIMIT 1000",
                conn, params=(symbol, interval))
            if not df_cached.empty:
                df_cached['timestamp'] = pd.to_datetime(df_cached['timestamp'])
                df_cached.set_index('timestamp', inplace=True)
                if datetime.now() - df_cached.index[-1] < timedelta(minutes=15):
                    conn.close(); return calculate_indicators(df_cached), symbol
            conn.close()
        except Exception: pass
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df is not None and not df.empty: break
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1)); continue
            elif attempt == max_retries - 1: return None, symbol
    if df is None or df.empty: return None, symbol
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.title() for c in df.columns]
    if DB_AVAILABLE:
        try:
            conn = sqlite3.connect(DB_PATH)
            df_s = df.reset_index(); df_s['symbol'] = symbol; df_s['interval'] = interval
            df_s.rename(columns={'Date': 'timestamp', 'Datetime': 'timestamp'}, inplace=True)
            for _, row in df_s.iterrows():
                conn.execute("INSERT OR REPLACE INTO price_data VALUES (?,?,?,?,?,?,?,?)",
                    (row['symbol'], str(row['timestamp']), row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['interval']))
            conn.commit(); conn.close()
        except Exception: pass
    return calculate_indicators(df), symbol

def calculate_indicators(df):
    df["EMA9"] = _ema_vectorized(df["Close"], 9); df["EMA20"] = _ema_vectorized(df["Close"], 20)
    df["EMA50"] = _ema_vectorized(df["Close"], 50); df["EMA200"] = _ema_vectorized(df["Close"], 200)
    df["SMA20"] = _sma_vectorized(df["Close"], 20); df["SMA50"] = _sma_vectorized(df["Close"], 50)
    df["ATR"] = _atr_vectorized(df["High"], df["Low"], df["Close"], 14)
    df["RSI"] = _rsi_vectorized(df["Close"], 14); df["RSI3"] = _rsi_vectorized(df["Close"], 3)
    df["RSI14_ROC9"] = _roc_vectorized(df["RSI"], 9); df["RSI3_SMA3"] = _sma_vectorized(df["RSI3"], 3)
    df["CMB_CI"] = df["RSI14_ROC9"] + df["RSI3_SMA3"]
    df["CMB_FAST"] = _sma_vectorized(df["CMB_CI"], 13); df["CMB_SLOW"] = _sma_vectorized(df["CMB_CI"], 33)
    df["STOCH_K"], df["STOCH_D"] = _stochastic_vectorized(df["High"], df["Low"], df["Close"], 14, 3)
    if "Volume" in df.columns:
        df["VOL_MA20"] = _sma_vectorized(df["Volume"], 20); df["VOL_MA50"] = _sma_vectorized(df["Volume"], 50)
        df["MFI"] = _mfi_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], 14)
        df["OBV"] = _obv_vectorized(df["Close"], df["Volume"]); df["OBV_EMA"] = _ema_vectorized(df["OBV"], 20)
        df["VWAP"] = _vwap_vectorized(df["High"], df["Low"], df["Close"], df["Volume"])
    df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = _adx_vectorized(df["High"], df["Low"], df["Close"], 14)
    df["SWING_HIGH_20"] = df["High"].rolling(20).max(); df["SWING_LOW_20"] = df["Low"].rolling(20).min()
    df["SWING_HIGH_60"] = df["High"].rolling(60).max(); df["SWING_LOW_60"] = df["Low"].rolling(60).min()
    bb_std = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["SMA20"] + (2 * bb_std); df["BB_LOWER"] = df["SMA20"] - (2 * bb_std)
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["SMA20"] * 100
    return df.dropna()

def get_mtf_data(symbol, timeframes=['5m', '15m', '1h']):
    mtf_data = {}
    periods_map = {'5m': '5d', '15m': '30d', '1h': '60d', '1d': '1y'}
    for tf in timeframes:
        df, _ = get_data_with_db_cache(symbol, periods_map.get(tf, '60d'), tf)
        if df is not None and not df.empty: mtf_data[tf] = df
    return mtf_data

def mtf_trend_analysis(mtf_data):
    trends = {}
    for tf, df in mtf_data.items():
        if df is None or len(df) < 2: continue
        last = df.iloc[-1]
        trends[tf] = {
            "trend": "BULL" if last["Close"] > last["EMA50"] > last["EMA200"] else "BEAR",
            "adx": last["ADX"] if "ADX" in last else 0,
            "rsi": last["RSI"]
        }
    bull_count = sum(1 for t in trends.values() if t["trend"] == "BULL")
    total = len(trends)
    return trends, (bull_count / total * 100) if total > 0 else 0
# =====================
# MACHINE LEARNING
# =====================
def prepare_ml_features(df, lookback=20):
    if len(df) < lookback + 10: return None, None
    features, labels = [], []
    for i in range(lookback, len(df) - 5):
        row = df.iloc[i]; hist = df.iloc[i-lookback:i]
        features.append([
            row["RSI"], row["MFI"] if "MFI" in row else 50, row["STOCH_K"], row["STOCH_D"], row["ADX"],
            (row["Close"] - row["EMA20"]) / row["ATR"] if row["ATR"] > 0 else 0,
            (row["Close"] - row["EMA50"]) / row["ATR"] if row["ATR"] > 0 else 0,
            row["BB_WIDTH"] if "BB_WIDTH" in row else 0,
            (row["Volume"] / row["VOL_MA20"]) if "VOL_MA20" in row and row["VOL_MA20"] > 0 else 1,
            hist["Close"].pct_change().mean(), hist["Close"].pct_change().std(),
            row["CMB_CI"] if "CMB_CI" in row else 0,
            (row["OBV"] - row["OBV_EMA"]) if "OBV" in row else 0,
        ])
        future_max = df.iloc[i+1:i+6]["Close"].max()
        labels.append(1 if (future_max - row["Close"]) / row["Close"] > 0.01 else 0)
    return np.array(features), np.array(labels)

@st.cache_resource
def train_ml_model(symbol, df):
    if not ML_AVAILABLE: return None
    X, y = prepare_ml_features(df)
    if X is None or len(X) < 50: return None
    split = int(len(X) * 0.8)
    model = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
    model.fit(X[:split], y[:split])
    return {"model": model, "train_acc": model.score(X[:split], y[:split]),
            "test_acc": model.score(X[split:], y[split:]), "feature_importance": model.feature_importances_}

def get_ml_probability(df, model_data):
    if model_data is None or df is None or len(df) < 30: return 0.5
    last_row = df.iloc[-1]; hist = df.iloc[-20:]
    feature_vec = [[
        last_row["RSI"], last_row["MFI"] if "MFI" in last_row else 50,
        last_row["STOCH_K"], last_row["STOCH_D"], last_row["ADX"],
        (last_row["Close"] - last_row["EMA20"]) / last_row["ATR"] if last_row["ATR"] > 0 else 0,
        (last_row["Close"] - last_row["EMA50"]) / last_row["ATR"] if last_row["ATR"] > 0 else 0,
        last_row["BB_WIDTH"] if "BB_WIDTH" in last_row else 0,
        (last_row["Volume"] / last_row["VOL_MA20"]) if "VOL_MA20" in last_row and last_row["VOL_MA20"] > 0 else 1,
        hist["Close"].pct_change().mean(), hist["Close"].pct_change().std(),
        last_row["CMB_CI"] if "CMB_CI" in last_row else 0,
        (last_row["OBV"] - last_row["OBV_EMA"]) if "OBV" in last_row else 0,
    ]]
    return float(model_data["model"].predict_proba(feature_vec)[0][1])

# =====================
# SCORING ALGORITHM
# =====================
def calculate_advanced_score(df, symbol, mtf_data=None, ml_model=None):
    if df is None or len(df) < 50: return None
    last = df.iloc[-1]; prev = df.iloc[-2]; score = 50; reasons = []
    bull_regime = (last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"])
    price_above_ema50 = last["Close"] > last["EMA50"]
    price_above_ema20 = last["Close"] > last["EMA20"]
    if bull_regime: score += 15; reasons.append("✅ Bull Regime: EMA50 > EMA200")
    else: score -= 15; reasons.append("🔻 Bear/Neutral Regime")
    if price_above_ema50: score += 10; reasons.append("📈 Price above EMA50")
    vwap_position = 0
    if "VWAP" in last:
        vwap_position = (last["Close"] - last["VWAP"]) / last["VWAP"] * 100
        if vwap_position > 0.5: score += 20; reasons.append(f"💎 VWAP: +{vwap_position:.2f}%")
        elif vwap_position > 0: score += 10; reasons.append("💎 VWAP: Above")
        elif vwap_position < -1: score -= 10; reasons.append("⚠️ VWAP: Below")
    curr_rsi = float(last["RSI"])
    if bull_regime and 40 <= curr_rsi <= 55: score += 15; reasons.append(f"📊 RSI {curr_rsi:.1f}: Pullback zone")
    elif curr_rsi < 30 and curr_rsi > prev["RSI"]: score += 18; reasons.append(f"🚀 RSI {curr_rsi:.1f}: Oversold bounce")
    elif 70 < curr_rsi < 85: score += 8; reasons.append(f"💪 RSI {curr_rsi:.1f}: Momentum")
    elif curr_rsi > 90: score -= 12; reasons.append(f"🔥 RSI {curr_rsi:.1f}: Overbought")
    if "MFI" in last:
        mfi = float(last["MFI"])
        if 30 < mfi < 70: score += 10; reasons.append(f"💰 MFI {mfi:.1f}: Healthy")
        elif mfi < 20: score += 15; reasons.append(f"💰 MFI {mfi:.1f}: Oversold")
        elif mfi > 80: score -= 10; reasons.append(f"⚠️ MFI {mfi:.1f}: Overbought")
    stoch_k = float(last["STOCH_K"]); stoch_d = float(last["STOCH_D"])
    if stoch_k < 20 and stoch_k > stoch_d: score += 10; reasons.append(f"📉 Stoch: Oversold crossover")
    elif stoch_k > 80: score -= 5; reasons.append(f"⚠️ Stoch: Overbought")
    adx = float(last["ADX"])
    if adx > 25: score += 10; reasons.append(f"💪 ADX {adx:.1f}: Strong trend")
    elif adx < 20: score -= 5; reasons.append(f"⚠️ ADX {adx:.1f}: Weak trend")
    if "Volume" in last and "VOL_MA20" in last:
        vol_ratio = last["Volume"] / last["VOL_MA20"]
        if vol_ratio > 1.5: score += 10; reasons.append(f"📊 Vol: {vol_ratio:.2f}x above avg")
        elif vol_ratio < 0.7: score -= 5; reasons.append(f"⚠️ Vol: {vol_ratio:.2f}x below avg")
    if "OBV" in last and "OBV_EMA" in last and last["OBV"] > last["OBV_EMA"]:
        score += 8; reasons.append("📈 OBV: Accumulation")
    if "BB_WIDTH" in last and last["BB_WIDTH"] < 2 and price_above_ema20:
        score += 8; reasons.append(f"🎯 BB Squeeze: {last['BB_WIDTH']:.2f}%")
    if (last["CMB_FAST"] > last["CMB_SLOW"]) and (last["CMB_CI"] > last["CMB_FAST"]) and (last["CMB_CI"] > prev["CMB_CI"]):
        score += 10; reasons.append("🧠 CMB: Strong momentum")
    confluence = 0
    if mtf_data:
        _, confluence = mtf_trend_analysis(mtf_data)
        if confluence >= 80: score += 15; reasons.append(f"🎯 MTF: {confluence:.0f}% aligned")
        elif confluence >= 60: score += 8; reasons.append(f"✅ MTF: {confluence:.0f}%")
        else: score -= 5; reasons.append(f"⚠️ MTF: {confluence:.0f}% mixed")
    ml_prob = 0.5
    if ml_model and ML_AVAILABLE:
        ml_prob = get_ml_probability(df, ml_model)
        if ml_prob > 0.65: b = int((ml_prob-0.5)*40); score += b; reasons.append(f"🤖 ML: {ml_prob*100:.1f}% (+{b})")
        elif ml_prob < 0.35: p = int((0.5-ml_prob)*30); score -= p; reasons.append(f"🤖 ML: {ml_prob*100:.1f}% (-{p})")
    price = float(last["Close"]); atr = float(last["ATR"]); stop_dist = atr * 1.5
    tp_pct = max(0.01, min(0.04, 0.015 + (max(0, score - 50) / 1000)))
    return {
        "symbol": symbol, "score": max(0, min(100, int(score))), "reasons": reasons,
        "price": price, "stop": price - stop_dist, "target": price * (1 + tp_pct),
        "rr": (price * tp_pct) / stop_dist if stop_dist > 0 else 0,
        "tp_pct": tp_pct * 100, "rsi": curr_rsi, "mfi": last.get("MFI", 50),
        "vwap_dist": vwap_position, "adx": adx, "ml_prob": ml_prob, "mtf_confluence": confluence
    }

# =====================
# HELPER: MATCH CLOSED TRADES (FIFO)
# =====================
def match_closed_trades(trades):
    closed_trades = []; positions = {}
    for trade in sorted(trades, key=lambda x: x['datetime']):
        symbol = trade['symbol']
        if symbol not in positions: positions[symbol] = []
        if trade['type'] == 'BUY':
            positions[symbol].append({'qty': trade['quantity'], 'price': trade['price'],
                'commission': trade['commission'], 'datetime': trade['datetime'], 'notes': trade['notes']})
        elif trade['type'] == 'SELL':
            remaining_qty = trade['quantity']; sell_price = trade['price']; sell_commission = trade['commission']
            while remaining_qty > 0 and positions.get(symbol, []):
                buy_pos = positions[symbol][0]; matched_qty = min(remaining_qty, buy_pos['qty'])
                buy_cost = matched_qty * buy_pos['price']; sell_revenue = matched_qty * sell_price
                total_commission = (matched_qty / trade['quantity'] * sell_commission) + (matched_qty / max(buy_pos['qty'],1) * buy_pos['commission'])
                pnl = sell_revenue - buy_cost - total_commission
                closed_trades.append({
                    'Symbol': symbol, 'Quantity': matched_qty, 'Buy Price': buy_pos['price'],
                    'Sell Price': sell_price, 'Buy Date': buy_pos['datetime'], 'Sell Date': trade['datetime'],
                    'Cost': buy_cost, 'Revenue': sell_revenue, 'P&L (TL)': pnl,
                    'P&L (%)': (pnl / buy_cost * 100) if buy_cost > 0 else 0,
                    'Commission': total_commission, 'Notes': buy_pos['notes']
                })
                buy_pos['qty'] -= matched_qty; remaining_qty -= matched_qty
                if buy_pos['qty'] == 0: positions[symbol].pop(0)
    open_positions = {}
    for symbol, lots in positions.items():
        total_qty = sum(l['qty'] for l in lots)
        if total_qty > 0:
            total_cost = sum(l['qty'] * l['price'] for l in lots)
            open_positions[symbol] = {'qty': total_qty, 'avg_cost': total_cost / total_qty,
                'total_cost': total_cost, 'lots': len([l for l in lots if l['qty'] > 0])}
    return closed_trades, open_positions
# =====================
# CSS STYLING
# =====================
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .score-box { font-size: 2.5em; font-weight: bold; text-align: center; padding: 15px; border-radius: 12px; margin-bottom: 15px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
    .score-high { background: linear-gradient(135deg, #00c853 0%, #64dd17 100%); }
    .score-mid { background: linear-gradient(135deg, #ffd600 0%, #ffab00 100%); color: #000; }
    .score-low { background: linear-gradient(135deg, #d50000 0%, #c62828 100%); }
    .trade-plan { background: #1c1f26; padding: 20px; border-radius: 12px; border: 2px solid #00e5ff; box-shadow: 0 6px 20px rgba(0,229,255,0.2); }
    .pnl-card { padding: 18px 22px; border-radius: 12px; margin: 6px 0; border: 1px solid rgba(255,255,255,0.08); }
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
st.caption("Advanced day trading with VWAP, MFI, Multi-Timeframe, ML Score & Database Caching")

tab_single, tab_scanner, tab_portfolio, tab_analytics = st.tabs([
    "📊 Single Analysis", "🔍 ML Scanner", "💼 Portfolio Manager", "📈 Analytics & Performance"])

# =====================
# TAB 1: SINGLE ANALYSIS
# =====================
with tab_single:
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1: symbol_input = st.text_input("Hisse Kodu", value="THYAO", key="single_symbol").upper()
    with col2: enable_ml = st.checkbox("🤖 ML Enhancement", value=True, key="enable_ml")
    with col3: enable_mtf = st.checkbox("🎯 Multi-Timeframe", value=True, key="enable_mtf")
    if st.button("🔍 Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            df, sym = get_data_with_db_cache(symbol_input, "60d", "15m")
            if df is None: st.error("❌ Data fetch failed")
            else:
                ml_model = None
                if enable_ml and ML_AVAILABLE:
                    with st.spinner("🤖 Training ML model..."):
                        ml_model = train_ml_model(symbol_input, df)
                        if ml_model: st.success(f"✅ ML Model trained (Test Acc: {ml_model['test_acc']*100:.1f}%)")
                mtf_data = get_mtf_data(symbol_input, ['5m', '15m', '1h']) if enable_mtf else None
                result = calculate_advanced_score(df, symbol_input, mtf_data, ml_model)
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_a:
                    bg = "score-high" if result["score"] >= 75 else ("score-mid" if result["score"] >= 60 else "score-low")
                    st.markdown(f"<div class='score-box {bg}'>{result['score']}</div>", unsafe_allow_html=True)
                    st.metric("RSI", f"{result['rsi']:.1f}"); st.metric("MFI", f"{result['mfi']:.1f}")
                    st.metric("ADX", f"{result['adx']:.1f}")
                    if enable_ml: st.metric("🤖 ML Prob", f"{result['ml_prob']*100:.1f}%")
                    if enable_mtf: st.metric("🎯 MTF", f"{result['mtf_confluence']:.0f}%")
                with col_b:
                    st.markdown("### 📝 Analysis Factors")
                    for reason in result["reasons"][:12]: st.markdown(f"• {reason}")
                    if len(result["reasons"]) > 12:
                        with st.expander("Show more..."):
                            for reason in result["reasons"][12:]: st.markdown(f"• {reason}")
                with col_c:
                    st.markdown(f"""<div class='trade-plan'><h3 style='color:#00e5ff'>📌 Trade Plan</h3><hr>
                        <p><b>Entry:</b> {result['price']:.2f}</p><p><b>Stop:</b> {result['stop']:.2f}</p>
                        <p><b>Target:</b> {result['target']:.2f}</p><p><b>R/R:</b> {result['rr']:.2f}</p>
                        <p><b>TP %:</b> {result['tp_pct']:.2f}%</p></div>""", unsafe_allow_html=True)
                st.markdown("### 📊 Advanced Chart")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2], subplot_titles=('Price & VWAP', 'RSI & MFI', 'Volume'))
                tail = df.tail(150)
                fig.add_trace(go.Candlestick(x=tail.index, open=tail["Open"], high=tail["High"],
                    low=tail["Low"], close=tail["Close"], name="Price"), row=1, col=1)
                if "VWAP" in tail.columns:
                    fig.add_trace(go.Scatter(x=tail.index, y=tail["VWAP"], mode="lines", name="VWAP",
                        line=dict(color="cyan", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA20"], name="EMA20", line=dict(color="orange")), row=1, col=1)
                fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA50"], name="EMA50", line=dict(color="blue")), row=1, col=1)
                fig.add_hline(y=result["stop"], line_dash="dash", line_color="red", row=1, col=1)
                fig.add_hline(y=result["target"], line_dash="solid", line_color="green", row=1, col=1)
                fig.add_trace(go.Scatter(x=tail.index, y=tail["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
                if "MFI" in tail.columns:
                    fig.add_trace(go.Scatter(x=tail.index, y=tail["MFI"], name="MFI", line=dict(color="gold")), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                colors = ['green' if tail["Close"].iloc[i] >= tail["Open"].iloc[i] else 'red' for i in range(len(tail))]
                fig.add_trace(go.Bar(x=tail.index, y=tail["Volume"], name="Volume", marker_color=colors), row=3, col=1)
                fig.update_layout(height=800, template="plotly_dark", showlegend=True, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

# =====================
# TAB 2: ML SCANNER
# =====================
with tab_scanner:
    st.markdown("### 🦅 Smart Stock Scanner")
    with st.expander("ℹ️ How the Scanner Works", expanded=False):
        st.markdown("**Automatic Filtering:** Score ≥70, ML Prob ≥60%, Top 15 results.")
    col1, col2 = st.columns([1, 1])
    with col1: universe = st.selectbox("📊 Universe", ["BIST30","BIST50","BIST100","ALL"], index=0, key="scan_universe")
    with col2: enable_ml_scan = st.checkbox("🤖 ML Enhancement", value=True, key="scan_ml_enable")
    min_score = 70; min_ml_prob = 0.60; top_n = 15
    st.markdown("---")
    if st.button("🔍 Start Smart Scan", type="primary", key="start_scan", use_container_width=True):
        start_time = time.time()
        tickers = {"BIST30": BIST30, "BIST50": BIST50, "BIST100": BIST100}.get(universe, list(set(BIST30+BIST50+BIST100)))
        results = []; progress_bar = st.progress(0); status_text = st.empty()
        def scan_symbol(ticker):
            try:
                time.sleep(REQUEST_DELAY); sym = ticker.replace(".IS", "")
                df, _ = get_data_with_db_cache(sym, "60d", "15m")
                if df is None or len(df) < 100: return None
                ml_model = train_ml_model(sym, df) if ML_AVAILABLE else None
                mtf_data = get_mtf_data(sym, ['5m', '15m', '1h'])
                result = calculate_advanced_score(df, sym, mtf_data, ml_model)
                if result and result["score"] >= min_score and result["ml_prob"] >= min_ml_prob: return result
            except Exception: pass
            return None
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(scan_symbol, t): t for t in tickers}
            completed = 0
            for future in as_completed(futures):
                completed += 1; progress_bar.progress(completed / len(futures))
                status_text.text(f"Scanning... {completed}/{len(futures)}")
                result = future.result()
                if result: results.append(result)
        progress_bar.empty(); status_text.empty()
        if results:
            results.sort(key=lambda x: (x["score"], x["ml_prob"]), reverse=True)
            top_results = results[:top_n]
            st.success(f"✅ Scan done in {time.time()-start_time:.1f}s — {len(results)} setups, showing Top {len(top_results)}")
            for idx, res in enumerate(top_results):
                with st.container():
                    c1, c2, c3 = st.columns([1, 2, 2])
                    with c1:
                        bg = "score-high" if res["score"] >= 80 else ("score-mid" if res["score"] >= 70 else "score-low")
                        st.markdown(f"<div class='score-box {bg}' style='font-size:1.8em'>#{idx+1}<br>{res['score']}</div>", unsafe_allow_html=True)
                        st.markdown(f"### {res['symbol']}"); st.caption(f"R/R: {res['rr']:.2f} | ML: {res['ml_prob']*100:.0f}%")
                    with c2:
                        st.markdown("**🎯 Key Factors:**")
                        for r in res["reasons"][:6]: st.caption(f"• {r}")
                    with c3:
                        st.write(f"Entry: **{res['price']:.2f}** | Stop: **{res['stop']:.2f}** | Target: **{res['target']:.2f}** ({res['tp_pct']:.1f}%)")
                        st.write(f"MTF: **{res['mtf_confluence']:.0f}%**")
                    st.markdown("---")
        else: st.warning("⚠️ No high-quality setups found. Try during market hours or a larger universe.")
# =====================
# TAB 3: PORTFOLIO MANAGER (ENHANCED)
# =====================
with tab_portfolio:
    st.markdown("### 💼 Portfolio Manager")
    st.caption("Track your stock trades — add BUY & SELL transactions, see positions and P/L at a glance")

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
                "symbol": trade_symbol, "type": trade_type, "quantity": trade_quantity,
                "price": trade_price, "datetime": trade_datetime.strftime("%Y-%m-%d %H:%M"),
                "commission": trade_commission, "notes": trade_notes,
                "total": cost_or_revenue + (trade_commission if trade_type == "BUY" else -trade_commission)
            }
            st.session_state.trades.append(new_trade)
            if DB_AVAILABLE:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("INSERT INTO trade_history (symbol,entry_time,entry_price,qty,pnl,reason,score,strategy) VALUES (?,?,?,?,?,?,?,?)",
                        (trade_symbol, trade_datetime.strftime("%Y-%m-%d %H:%M"), trade_price,
                         trade_quantity if trade_type == "BUY" else -trade_quantity, 0.0, trade_notes, 0, trade_type))
                    conn.commit(); conn.close()
                except Exception: pass
            st.success(f"✅ {trade_type} trade added: {trade_quantity} × {trade_symbol} @ ₺{trade_price:,.2f}")
            st.rerun()

    st.markdown("---")

    if st.session_state.trades:
        trades = st.session_state.trades
        closed_trades, open_positions = match_closed_trades(trades)

        # ── Quick Stats Row ──
        total_invested = sum(t['total'] for t in trades if t['type'] == 'BUY')
        total_sold = sum(t['total'] for t in trades if t['type'] == 'SELL')
        total_commissions = sum(t['commission'] for t in trades)
        realized_pnl = sum(ct['P&L (TL)'] for ct in closed_trades)
        open_value = sum(p['total_cost'] for p in open_positions.values())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Invested", f"₺{total_invested:,.2f}")
        c2.metric("Total Sold", f"₺{total_sold:,.2f}")
        c3.metric("Realized P&L", f"₺{realized_pnl:,.2f}",
                   delta=f"{(realized_pnl / total_invested * 100) if total_invested > 0 else 0:.2f}%")
        c4.metric("Open Positions", f"{len(open_positions)} stocks", delta=f"₺{open_value:,.2f} invested")
        c5.metric("Commissions", f"₺{total_commissions:,.2f}")

        st.markdown("---")

        # ── Stock Information Table ──
        st.markdown("### 📋 Stock Information")
        st.caption("Summary of every stock you've traded — quantities, costs, sell prices, and realized P/L")

        stock_summary = {}
        for t in trades:
            sym = t['symbol']
            if sym not in stock_summary:
                stock_summary[sym] = {'buy_qty': 0, 'sell_qty': 0, 'total_buy_cost': 0,
                    'total_sell_revenue': 0, 'buys': 0, 'sells': 0, 'realized_pnl': 0, 'commissions': 0}
            s = stock_summary[sym]; s['commissions'] += t['commission']
            if t['type'] == 'BUY': s['buy_qty'] += t['quantity']; s['total_buy_cost'] += t['quantity'] * t['price']; s['buys'] += 1
            else: s['sell_qty'] += t['quantity']; s['total_sell_revenue'] += t['quantity'] * t['price']; s['sells'] += 1

        for ct in closed_trades:
            if ct['Symbol'] in stock_summary: stock_summary[ct['Symbol']]['realized_pnl'] += ct['P&L (TL)']

        summary_rows = []
        for sym, s in stock_summary.items():
            avg_buy = s['total_buy_cost'] / s['buy_qty'] if s['buy_qty'] > 0 else 0
            avg_sell = s['total_sell_revenue'] / s['sell_qty'] if s['sell_qty'] > 0 else 0
            remaining = s['buy_qty'] - s['sell_qty']
            status = "🟢 OPEN" if remaining > 0 else ("🔴 CLOSED" if s['sell_qty'] > 0 else "🟡 HOLD")
            pnl_return = (s['realized_pnl'] / s['total_buy_cost'] * 100) if s['total_buy_cost'] > 0 and s['sell_qty'] > 0 else 0
            summary_rows.append({
                'Stock': sym, 'Bought': s['buy_qty'], 'Sold': s['sell_qty'], 'Remaining': remaining,
                'Avg Buy (TL)': avg_buy, 'Avg Sell (TL)': avg_sell if s['sell_qty'] > 0 else None,
                'Total Cost': s['total_buy_cost'], 'Total Revenue': s['total_sell_revenue'] if s['sell_qty'] > 0 else None,
                'Realized P&L': s['realized_pnl'] if s['sell_qty'] > 0 else None,
                'Return (%)': pnl_return if s['sell_qty'] > 0 else None,
                'Status': status, 'Trades': f"{s['buys']}B / {s['sells']}S"
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True, column_config={
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
        })

        # ── Closed Trade Detail Cards ──
        if closed_trades:
            st.markdown("### 💰 Closed Trade Details")
            st.caption("Each completed buy→sell cycle with profit/loss")
            for ct in closed_trades:
                pnl = ct['P&L (TL)']
                pnl_class = "pnl-card-profit" if pnl > 0 else ("pnl-card-loss" if pnl < 0 else "pnl-card-neutral")
                icon = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                pnl_sign = "+" if pnl > 0 else ""
                st.markdown(f"""<div class="pnl-card {pnl_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <div><span style="font-size:1.3em; font-weight:700;">{icon} {ct['Symbol']}</span>
                            <span style="color:#888; margin-left:12px; font-size:0.9em;">
                            {ct['Quantity']} shares | Bought @ ₺{ct['Buy Price']:.2f} → Sold @ ₺{ct['Sell Price']:.2f}</span></div>
                        <div style="text-align:right;">
                            <span style="font-size:1.4em; font-weight:700; color:{'#00c853' if pnl > 0 else '#d50000' if pnl < 0 else '#ffd600'};">
                            {pnl_sign}₺{pnl:,.2f}</span>
                            <span style="color:#888; margin-left:8px; font-size:0.9em;">({pnl_sign}{ct['P&L (%)']:.2f}%)</span></div>
                    </div>
                    <div style="margin-top:6px; font-size:0.8em; color:#666;">
                        📅 {ct['Buy Date']} → {ct['Sell Date']} | Cost: ₺{ct['Cost']:,.2f} → Revenue: ₺{ct['Revenue']:,.2f} | Fees: ₺{ct['Commission']:.2f}
                        {f" | 📝 {ct['Notes']}" if ct['Notes'] else ""}</div></div>""", unsafe_allow_html=True)

        # ── Open Positions ──
        if open_positions:
            st.markdown("### 📌 Open Positions")
            st.caption("Stocks you still hold — not yet included in realized P&L")
            for sym, pos in open_positions.items():
                st.markdown(f"""<div class="pnl-card pnl-card-info">
                    <span style="font-size:1.2em; font-weight:700;">🔵 {sym}</span>
                    <span style="color:#aaa; margin-left:12px;">{pos['qty']} shares @ avg ₺{pos['avg_cost']:.2f}
                    | Total: ₺{pos['total_cost']:,.2f} | {pos['lots']} open lot(s)</span></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Full Trade Log ──
        with st.expander("📜 Full Trade Log (all transactions)", expanded=False):
            trades_df = pd.DataFrame(trades)
            st.dataframe(
                trades_df[['id', 'symbol', 'type', 'quantity', 'price', 'datetime', 'commission', 'total', 'notes']],
                use_container_width=True, hide_index=True, column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "symbol": st.column_config.TextColumn("Stock", width="small"),
                    "type": st.column_config.TextColumn("Type", width="small"),
                    "quantity": st.column_config.NumberColumn("Qty", width="small"),
                    "price": st.column_config.NumberColumn("Price", format="₺%.2f"),
                    "datetime": st.column_config.TextColumn("Date/Time"),
                    "commission": st.column_config.NumberColumn("Commission", format="₺%.2f"),
                    "total": st.column_config.NumberColumn("Total", format="₺%.2f"),
                    "notes": st.column_config.TextColumn("Notes")
                })

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🗑️ Clear All Trades"):
                st.session_state.trades = []; st.success("All trades cleared!"); st.rerun()
        with col2:
            csv = pd.DataFrame(trades).to_csv(index=False)
            st.download_button("📥 Export All Trades (CSV)", data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    else:
        st.info("👋 **Get Started — Add your first trade above!**\n\n"
            "**Example:** BUY 151 × TCELL @ ₺160, then SELL 151 × TCELL @ ₺170 → Profit: **+₺1,510** ✅\n\n"
            "Uses **FIFO matching** (First In, First Out) to pair buys with sells.")
# =====================
# TAB 4: ANALYTICS & PERFORMANCE (ENHANCED)
# =====================
with tab_analytics:
    st.markdown("### 📈 Trading Performance Analytics")

    if 'trades' not in st.session_state or not st.session_state.trades:
        st.info("📊 **Analytics Dashboard**\n\nAdd trades in the **Portfolio Manager** tab to unlock:\n"
            "- ✅ Profit & Loss breakdown per stock\n- ✅ Win rate & profit factor stats\n"
            "- ✅ Equity curve visualization\n- ✅ Best & worst trade highlights\n- ✅ Performance-by-stock bar chart")
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
            st.markdown(f"""<div class="pnl-card {pnl_class}" style="text-align:center; padding:24px;">
                <div style="font-size:0.9em; color:#aaa; margin-bottom:4px;">NET REALIZED PROFIT / LOSS</div>
                <div style="font-size:2.8em; font-weight:800; color:{pnl_color};">{pnl_sign}₺{total_pnl:,.2f}</div>
                <div style="font-size:1.1em; color:#aaa; margin-top:4px;">{pnl_sign}{overall_return:.2f}% overall return on ₺{total_cost:,.2f} invested</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Trades", total_trades_count)
            c2.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{winning_trades}W / {losing_trades}L")
            c3.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
            c4.metric("Avg Win", f"₺{avg_win:,.2f}")
            c5.metric("Avg Loss", f"₺{avg_loss:,.2f}")

            st.markdown("---")

            # ── Per-Trade P&L Cards ──
            st.markdown("### 💰 Trade-by-Trade Profit & Loss")
            for _, ct in closed_df.iterrows():
                pnl = ct['P&L (TL)']
                pnl_class = "pnl-card-profit" if pnl > 0 else ("pnl-card-loss" if pnl < 0 else "pnl-card-neutral")
                icon = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                pnl_sign = "+" if pnl > 0 else ""
                pnl_color = '#00c853' if pnl > 0 else '#d50000' if pnl < 0 else '#ffd600'
                st.markdown(f"""<div class="pnl-card {pnl_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <div><span style="font-size:1.2em; font-weight:700;">{icon} {ct['Symbol']}</span>
                            <span style="color:#888; margin-left:10px; font-size:0.85em;">
                            {int(ct['Quantity'])} shares | ₺{ct['Buy Price']:.2f} → ₺{ct['Sell Price']:.2f}</span></div>
                        <div style="text-align:right;"><span style="font-size:1.3em; font-weight:700; color:{pnl_color};">
                            {pnl_sign}₺{pnl:,.2f}</span>
                            <span style="color:#888; margin-left:6px; font-size:0.85em;">({pnl_sign}{ct['P&L (%)']:.2f}%)</span></div>
                    </div>
                    <div style="margin-top:4px; font-size:0.78em; color:#555;">
                        📅 {ct['Buy Date']} → {ct['Sell Date']} | Cost: ₺{ct['Cost']:,.2f} → Revenue: ₺{ct['Revenue']:,.2f}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Equity Curve ──
            st.markdown("### 📊 Equity Curve")
            closed_df_sorted = closed_df.sort_values('Sell Date')
            closed_df_sorted['Cumulative P&L'] = closed_df_sorted['P&L (TL)'].cumsum()
            closed_df_sorted['Trade #'] = range(1, len(closed_df_sorted) + 1)

            fig_equity = go.Figure()
            colors_eq = ['#00c853' if p >= 0 else '#d50000' for p in closed_df_sorted['P&L (TL)']]
            fig_equity.add_trace(go.Scatter(
                x=closed_df_sorted['Trade #'], y=closed_df_sorted['Cumulative P&L'],
                mode='lines+markers', name='Cumulative P&L',
                line=dict(color='#00e5ff', width=2.5),
                marker=dict(size=10, color=colors_eq, line=dict(width=1, color='white')),
                text=closed_df_sorted['Symbol'],
                hovertemplate='<b>%{text}</b><br>Trade #%{x}<br>Cumulative: ₺%{y:,.2f}<extra></extra>'))
            fig_equity.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
            fig_equity.add_trace(go.Scatter(
                x=closed_df_sorted['Trade #'], y=closed_df_sorted['Cumulative P&L'],
                fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.08)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_equity.update_layout(height=400, template="plotly_dark",
                xaxis_title="Trade Number", yaxis_title="Cumulative P&L (₺)",
                showlegend=True, hovermode='x unified', margin=dict(t=20, b=40))
            st.plotly_chart(fig_equity, use_container_width=True)

            st.markdown("---")

            # ── Performance by Stock (Horizontal Bar Chart) ──
            st.markdown("### 📊 Performance by Stock")
            stock_perf = closed_df.groupby('Symbol').agg({
                'P&L (TL)': 'sum', 'P&L (%)': 'mean', 'Quantity': 'count'
            }).reset_index()
            stock_perf.columns = ['Symbol', 'Total P&L', 'Avg Return (%)', 'Trades']
            stock_perf = stock_perf.sort_values('Total P&L', ascending=True)

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=stock_perf['Symbol'], x=stock_perf['Total P&L'], orientation='h',
                marker_color=['#00c853' if p > 0 else '#d50000' for p in stock_perf['Total P&L']],
                text=[f"₺{p:,.2f}" for p in stock_perf['Total P&L']], textposition='outside',
                hovertemplate='<b>%{y}</b><br>P&L: ₺%{x:,.2f}<extra></extra>'))
            fig_bar.add_vline(x=0, line_color="gray", line_dash="dash")
            fig_bar.update_layout(height=max(250, len(stock_perf) * 50), template="plotly_dark",
                xaxis_title="Total P&L (₺)", margin=dict(t=20, b=40, l=80), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(stock_perf.sort_values('Total P&L', ascending=False),
                    use_container_width=True, hide_index=True, column_config={
                        "Symbol": st.column_config.TextColumn("Stock"),
                        "Total P&L": st.column_config.NumberColumn("Total P&L", format="₺%.2f"),
                        "Avg Return (%)": st.column_config.NumberColumn("Avg Return", format="%.2f%%"),
                        "Trades": st.column_config.NumberColumn("# Trades")})
            with col2:
                st.markdown("**🏆 Best Trades:**")
                for _, t in closed_df.nlargest(3, 'P&L (%)').iterrows():
                    st.success(f"**{t['Symbol']}**: +{t['P&L (%)']:.2f}% (₺{t['P&L (TL)']:.2f})")
                st.markdown("**📉 Worst Trades:**")
                for _, t in closed_df.nsmallest(3, 'P&L (%)').iterrows():
                    st.error(f"**{t['Symbol']}**: {t['P&L (%)']:.2f}% (₺{t['P&L (TL)']:.2f})")

            st.markdown("---")
            st.markdown("### 📥 Export Analytics")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Closed Trades CSV", data=closed_df.to_csv(index=False),
                    file_name=f"closed_trades_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            with col2:
                summary_data = pd.DataFrame({
                    'Metric': ['Total P&L', 'Total Trades', 'Win Rate', 'Profit Factor', 'Avg Win', 'Avg Loss', 'Overall Return'],
                    'Value': [f"₺{total_pnl:.2f}", total_trades_count, f"{win_rate:.1f}%",
                        f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                        f"₺{avg_win:.2f}", f"₺{avg_loss:.2f}", f"{overall_return:.2f}%"]
                })
                st.download_button("📥 Summary CSV", data=summary_data.to_csv(index=False),
                    file_name=f"trading_summary_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        else:
            st.warning("⚠️ **No Closed Trades Yet**\n\nYou have open positions but no completed buy→sell cycles.\n\n"
                "To see P&L analytics:\n1. **BUY** a stock (e.g., 151 × TCELL @ ₺160)\n"
                "2. **SELL** the same stock (e.g., 151 × TCELL @ ₺170)\n\n"
                "Profit will be calculated automatically: **+₺1,510.00** ✅")

st.markdown("---")
st.caption("🚀 PROP DESK V7.0 | Built with ML, MTF Analysis & Advanced Day Trading Indicators")
