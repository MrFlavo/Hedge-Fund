
# =====================
# BIST EVREN SEÇİMİ
# =====================
BIST30 = [
    "AKBNK.IS","ARCLK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS","EREGL.IS",
    "FROTO.IS","GARAN.IS","GUBRF.IS","ISCTR.IS","KCHOL.IS","KOZAL.IS",
    "KRDMD.IS","PETKM.IS","PGSUS.IS","SAHOL.IS","SASA.IS","SISE.IS",
    "TAVHL.IS","TCELL.IS","THYAO.IS","TOASO.IS","TTKOM.IS","TUPRS.IS",
    "YKBNK.IS"
]

# Basit yaklaşım: BIST50 ve BIST100 = BIST30 + ek hisseler (manuel genişletilebilir)
BIST50 = list(set(BIST30 + [
    "ALARK.IS","ENKAI.IS","HEKTS.IS","ODAS.IS","OYAKC.IS","SMRTG.IS",
    "VESTL.IS","ZOREN.IS","KONTR.IS","CANTE.IS","ASTOR.IS","GWIND.IS"
]))

BIST100 = list(set(BIST50 + [
    "AKSA.IS","AKSEN.IS","BRSAN.IS","DOAS.IS","ENJSA.IS","GESAN.IS",
    "KONYA.IS","LOGO.IS","NTHOL.IS","OTKAR.IS","TKFEN.IS","ULKER.IS",
    "YATAS.IS","ALFAS.IS","CWENE.IS","EUPWR.IS","MIATK.IS","QUAGR.IS"
]))


import streamlit as st
import time

# --- SAYFA AYARLARI ---
st.set_page_config(layout="wide", page_title="PROP DESK V6.9 (INTRADAY FULLSCAN REALTIME)", page_icon="🦅")

# --- KÜTÜPHANE KONTROLÜ VE HATA YÖNETİMİ ---
try:
    import yfinance as yf
    import pandas as pd
    import plotly.graph_objects as go
    import numpy as np
except ImportError as e:
    st.error("⚠️ Kütüphane Eksik!")
    st.code(str(e))
    st.info("Gereken paketleri requirements.txt üzerinden kurun (aşağıdaki öneriyi uygulayın).")
    st.stop()

# =====================
# TA HESAPLAMALARI (pandas-ta YOK)
# =====================
def _ema(s, length: int):
    return s.ewm(span=length, adjust=False).mean()

def _sma(s, length: int):
    return s.rolling(length).mean()

def _roc(s, length: int):
    return (s / s.shift(length) - 1.0) * 100.0

def _rsi(close, length: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def _atr(high, low, close, length: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

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
    .score-mid  { background-color: #ffd600; color: black; } /* Sarı */
    .score-low  { background-color: #d50000; } /* Kırmızı */

    .metric-row {
        display: flex; justify-content: space-between;
        background-color: #262730; padding: 8px;
        border-radius: 5px; margin-top: 5px; font-size: 0.9em;
    }

    .reason-text { font-size: 0.85em; color: #b0b8c3; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# AUTO AYARLAR (KULLANICI AYARI YOK) - isterse koddan değişir
# =========================================================
DEFAULT_RISK_PCT = 1.0     # işlem başı risk (otomatik)
# --- GÜN İÇİ SEVİYELER (TP/SL) ---
TP_MIN_PCT = 1.0           # hedef minimum (%)
TP_MAX_PCT = 3.0           # hedef maksimum (%)
TP_BASE_PCT = 2.0          # backtest için varsayılan hedef (%)
SL_ATR_MULT = 1.2          # stop mesafesi için ATR katsayı (auto_stop_mult ile birlikte)
SL_MIN_PCT = 0.6           # stop minimum (%)
SL_MAX_PCT = 2.5           # stop maksimum (%)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
COMMISSION_BPS = 10        # 0.10% tek yön (giriş+çıkış ayrı uygulanır)
SLIPPAGE_BPS = 5           # 0.05% tek yön
RR_TARGET = 2.0            # TP = stop_dist * RR_TARGET
BACKTEST_BARS = 420        # arka planda backtest kaç bar üzerinde çalışsın
SWING_LOOKBACK = 60        # swing high/low hedef/direnç için
VOL_LOOKBACK = 20          # hacim ortalaması

# --- KONFİGÜRASYON (DEFAULT, GİZLİ) ---
# Kullanıcıya sidebar ayarları gösterilmez; tüm parametreler burada default çalışır.
capital = 100000  # TL

# Tarayıcı defaults
universe = "TÜMÜ (30+50+100)"
interval = "15m"
period = "60d"
min_score = 65
min_rr = 1.5
top_n = 15

# TP / SL defaults (gün içi)
tp_min = 1.0
tp_max = 3.0
tp_base = 2.0  # skor-dinamik hesap sonrası clamp uygulanır
sl_atr_mult = 1.2
sl_min = 0.6
sl_max = 2.5
max_hold_bars = 30


# BIST EVRENİ (TARAMA İÇİN)
BIST30_TICKERS = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI",
    "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD",
    "ODAS", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO",
    "TSKB", "TTKOM", "TUPRS", "YKBNK"
]
# --- ENDEKS LİSTELERİ (BIST30/50/100) ---
INDEX_URL = {
    "BIST30": "https://uzmanpara.milliyet.com.tr/canli-borsa/bist-30-hisseleri/",
    "BIST50": "https://uzmanpara.milliyet.com.tr/canli-borsa/bist-50-hisseleri/",
    "BIST100": "https://uzmanpara.milliyet.com.tr/canli-borsa/bist-100-hisseleri/",
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_index_tickers(index_name: str) -> list[str]:
    """BIST endeks bileşenlerini web'den çeker (1 saat cache).
    Hata olursa güvenli şekilde fallback döner.
    """
    url = INDEX_URL.get(index_name)
    if not url:
        return []
    try:
        tables = pd.read_html(url)
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("menkul" in c for c in cols) or any("sembol" in c for c in cols):
                sym_col = t.columns[0]
                syms = t[sym_col].astype(str).str.upper().str.strip().tolist()
                syms = [s.split()[0] for s in syms]  # olası ek metinleri kırp
                syms = [s + ".IS" if not s.endswith(".IS") else s for s in syms]
                syms = [s for s in syms if s and s[0].isalpha()]
                # unique preserve order
                seen=set()
                out=[]
                for s in syms:
                    if s not in seen:
                        out.append(s); seen.add(s)
                return out
    except Exception:
        return []

def build_universe(universe_choice: str) -> list[str]:
    """
    Evreni oluştur:
    - Önce web'den güncel endeks bileşenlerini çekmeyi dener.
    - Başarısız olursa dosya içindeki fallback listelerini kullanır (BIST30/BIST50/BIST100).
    """
    # 1) Web'den dene
    if universe_choice in ("BIST30", "BIST50", "BIST100"):
        tickers = fetch_index_tickers(universe_choice)
        if tickers:
            return tickers

    # 2) Fallback listeleri (dosya içi)
    fallback_map = {
        "BIST30": BIST30,
        "BIST50": BIST50,
        "BIST100": BIST100,
    }
    if universe_choice in fallback_map:
        return sorted(list(dict.fromkeys(fallback_map[universe_choice])))

    # 3) Tümü: web varsa web, yoksa fallback birleşim
    if universe_choice.startswith("TÜMÜ"):
        a = fetch_index_tickers("BIST30") or BIST30
        b = fetch_index_tickers("BIST50") or BIST50
        c = fetch_index_tickers("BIST100") or BIST100
        combined = []
        for lst in (a, b, c):
            for s in lst:
                if s not in combined:
                    combined.append(s)
        return combined

    # default
    return sorted(list(dict.fromkeys(BIST100)))



# --- MODÜL 1: VERİ ÇEKME + İNDİKATÖR (CACHE) ---
@st.cache_data(ttl=900, show_spinner=False)
def get_data(symbol: str, period: str, interval: str):
    symbol = symbol.upper().strip()
    if len(symbol) <= 5 and not symbol.endswith(".IS"):
        symbol += ".IS"

    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return None, symbol

    # MultiIndex düzeltmesi
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Daha güvenli kolon normalize
    df.columns = [c.title() for c in df.columns]

    # Core indicators
    df["EMA50"] = _ema(df["Close"], 50)
    df["EMA200"] = _ema(df["Close"], 200)
    df["ATR"] = _atr(df["High"], df["Low"], df["Close"], 14)
    df["RSI"] = _rsi(df["Close"], 14)

    # Constance Brown CMB Composite Index:
    # CI = ROC_9(RSI_14) + SMA_3(RSI_3)
    df["RSI3"] = _rsi(df["Close"], 3)
    df["RSI14_ROC9"] = _roc(df["RSI"], 9)
    df["RSI3_SMA3"] = _sma(df["RSI3"], 3)
    df["CMB_CI"] = df["RSI14_ROC9"] + df["RSI3_SMA3"]
    df["CMB_FAST"] = _sma(df["CMB_CI"], 13)
    df["CMB_SLOW"] = _sma(df["CMB_CI"], 33)

    # Volume helpers
    if "Volume" in df.columns:
        df["VOL_MA20"] = _sma(df["Volume"], VOL_LOOKBACK)

    # Rolling swing levels
    df["SWING_HIGH"] = df["High"].rolling(SWING_LOOKBACK).max()
    df["SWING_LOW"] = df["Low"].rolling(SWING_LOOKBACK).min()

    # Temizlik
    df = df.dropna().sort_index()
    return df, symbol


# --- MODÜL 2: OTOMATİK RİSK / STOP / SENTIMENT ---
def auto_stop_mult(atr: float, price: float) -> float:
    """Volatiliteye göre ATR çarpanını otomatik seç (daha stabil stop)."""
    if price <= 0 or atr <= 0:
        return 2.0
    atr_pct = atr / price
    if atr_pct < 0.015:
        return 1.5
    if atr_pct < 0.03:
        return 2.0
    return 2.5

@st.cache_data(ttl=30, show_spinner=False)
def get_realtime_price(symbol: str):
    """YFinance üzerinden olabildiğince 'anlık' (last) fiyatı al.
    - Önce fast_info (hızlı)
    - Olmazsa 1 dakikalık son veriden kapanışı al
    """
    try:
        sym = symbol.upper().strip()
        if len(sym) <= 5 and not sym.endswith(".IS"):
            sym += ".IS"

        t = yf.Ticker(sym)
        fi = getattr(t, "fast_info", None)
        if fi:
            for k in ("last_price", "lastPrice", "regularMarketPrice"):
                if k in fi and fi[k]:
                    return float(fi[k])

        # Fallback: son 1m bar kapanışı
        df = yf.download(sym, period="1d", interval="1m", progress=False)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


def sentiment_proxy(df) -> float:
    """
    Random değil: tamamen deterministik 'duyarlılık' proxy'si.
    momentum(20 bar) + EMA50'ye göre trend + relative volume -> tanh ile -1..+1
    """
    try:
        last = df.iloc[-1]
        mom = df["Close"].pct_change(20).iloc[-1]
        trend = (last["Close"] - last["EMA50"]) / (last["ATR"] + 1e-9)

        vol_ratio = 0.0
        if "Volume" in df.columns and "VOL_MA20" in df.columns and last["VOL_MA20"] > 0:
            vol_ratio = (last["Volume"] / last["VOL_MA20"]) - 1.0

        raw = 2.0 * mom + 0.15 * trend + 0.10 * vol_ratio
        return float(np.tanh(raw))
    except Exception:
        return 0.0


# --- MODÜL 3: ENTRY SİNYALİ (look-ahead azaltmak için kapanışta üretir) ---
def entry_signal(df, i: int) -> bool:
    if i < 1:
        return False

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    # Rejim (bull) + trend
    bull_regime = (row["Close"] > row["EMA200"]) and (row["EMA50"] > row["EMA200"])
    uptrend = row["Close"] > row["EMA50"]

    # Bull pullback zone (RSI 40-55) ve RSI yükseliyor
    pullback = bull_regime and uptrend and (40 <= row["RSI"] <= 55) and (row["RSI"] > prev["RSI"])

    # Dip dönüşü: RSI 30 yukarı kırılım + EMA50 üstü
    dip_reversal = uptrend and (prev["RSI"] < 30) and (row["RSI"] >= 30)

    # Breakout: swing high yakınında + hacim teyidi
    vol_ok = True
    if "Volume" in df.columns and "VOL_MA20" in df.columns:
        vol_ok = row["Volume"] > row["VOL_MA20"] * 1.2
    breakout = bull_regime and vol_ok and (row["Close"] >= row["SWING_HIGH"] * 0.995)

    # CMB Composite momentum confirm (fast>slow & CI yükseliyor)
    cmb_ok = (row["CMB_FAST"] > row["CMB_SLOW"]) and (row["CMB_CI"] > prev["CMB_CI"])

    return bool(cmb_ok and (pullback or dip_reversal or breakout))


# --- MODÜL 4: GERÇEKÇİ BACKTEST (AUTO) ---
def run_backtest(df, start_capital: float):
    """
    Hızlı ama daha gerçekçi:
    - Sinyal kapanışta -> giriş bir sonraki bar OPEN (look-ahead azaltır)
    - ATR stop (auto mult) + RR TP
    - Komisyon+slipaj
    - Tek pozisyon
    - Forced-exit (bitişte kapanır)
    """
    commission = COMMISSION_BPS / 10000.0
    slippage = SLIPPAGE_BPS / 10000.0

    d = df.tail(BACKTEST_BARS).copy()
    if len(d) < 250:
        return {
            "win_rate": 0.0, "profit_factor": np.nan, "max_dd": 0.0,
            "total_return": 0.0, "trades": 0, "trades_df": pd.DataFrame(),
            "equity": pd.Series(dtype=float), "dd": pd.Series(dtype=float)
        }

    cash = float(start_capital)
    in_pos = False
    pos = {}
    pending = None  # {"idx": i+1, "stop_dist": x}

    equity = []
    trades = []

    def position_value(row):
        if not in_pos:
            return 0.0
        return pos["qty"] * float(row["Close"])

    for i in range(1, len(d)):
        row = d.iloc[i]
        ts = d.index[i]

        # 1) Pending entry at OPEN
        if (pending is not None) and (pending["idx"] == i) and (not in_pos):
            entry_raw = float(row["Open"])
            entry = entry_raw * (1 + slippage)

            # STOP/TP seviyeleri: entry (OPEN) üzerinden yüzde clamp'li
            atr_sig = float(pending.get("atr", row.get("ATR", 0.0)))
            stop_mult_sig = float(pending.get("stop_mult", auto_stop_mult(float(row["ATR"]), float(row["Close"]))))
            raw_stop_dist = atr_sig * stop_mult_sig * float(sl_atr_mult)
            sl_pct = clamp(raw_stop_dist / entry if entry > 0 else 0.0, float(sl_min)/100.0, float(sl_max)/100.0)
            stop_dist = entry * sl_pct
            tp_pct = clamp(float(tp_base)/100.0, float(tp_min)/100.0, float(tp_max)/100.0)

            if stop_dist > 0 and entry > 0:
                risk_amt = cash * (DEFAULT_RISK_PCT / 100.0)
                qty_risk = int(risk_amt / stop_dist)
                qty_cash = int(cash / (entry * (1 + commission)))
                qty = max(0, min(qty_risk, qty_cash))

                if qty >= 1:
                    notional = qty * entry
                    fee = notional * commission
                    cash -= (notional + fee)

                    pos = {
                        "entry_time": ts,
                        "entry": entry,
                        "qty": qty,
                        "stop": entry - stop_dist,
                        "tp": entry * (1 + tp_pct),
                        "entry_fee": fee,
                        "stop_dist": stop_dist
                    }
                    in_pos = True

            pending = None

        # 2) Position management intrabar (konservatif: STOP öncelikli)
        if in_pos:
            stop_hit = float(row["Low"]) <= pos["stop"]
            tp_hit = float(row["High"]) >= pos["tp"]

            if stop_hit or tp_hit:
                if stop_hit:
                    exit_raw = float(row["Open"]) if float(row["Open"]) < pos["stop"] else pos["stop"]
                    reason = "STOP"
                else:
                    exit_raw = float(row["Open"]) if float(row["Open"]) > pos["tp"] else pos["tp"]
                    reason = "TP"

                exit_px = float(exit_raw) * (1 - slippage)
                notional = pos["qty"] * exit_px
                fee = notional * commission
                cash += (notional - fee)

                pnl = (exit_px - pos["entry"]) * pos["qty"] - pos["entry_fee"] - fee
                risk0 = pos["stop_dist"] * pos["qty"]
                r_mult = pnl / risk0 if risk0 > 0 else np.nan

                trades.append({
                    "EntryTime": pos["entry_time"], "ExitTime": ts, "Reason": reason,
                    "Entry": pos["entry"], "Exit": exit_px, "Qty": pos["qty"],
                    "PnL": pnl, "R": r_mult, "Fees": pos["entry_fee"] + fee
                })

                in_pos = False
                pos = {}

        # 3) Signal at CLOSE -> schedule entry next bar open
        if (not in_pos) and (pending is None) and (i < len(d) - 1):
            if entry_signal(d, i):
                stop_mult = auto_stop_mult(float(row["ATR"]), float(row["Close"]))
                stop_dist = float(row["ATR"]) * stop_mult
                if stop_dist > 0:
                    pending = {"idx": i + 1, "atr": float(row["ATR"]), "stop_mult": stop_mult}

        # 4) Equity mark-to-market
        eq = cash + position_value(row)
        equity.append(eq)

    # 5) Forced exit
    if in_pos:
        last = d.iloc[-1]
        ts = d.index[-1]
        exit_px = float(last["Close"]) * (1 - slippage)

        notional = pos["qty"] * exit_px
        fee = notional * commission
        cash += (notional - fee)

        pnl = (exit_px - pos["entry"]) * pos["qty"] - pos["entry_fee"] - fee
        risk0 = pos["stop_dist"] * pos["qty"]
        r_mult = pnl / risk0 if risk0 > 0 else np.nan

        trades.append({
            "EntryTime": pos["entry_time"], "ExitTime": ts, "Reason": "FORCED_EXIT",
            "Entry": pos["entry"], "Exit": exit_px, "Qty": pos["qty"],
            "PnL": pnl, "R": r_mult, "Fees": pos["entry_fee"] + fee
        })

        in_pos = False
        pos = {}
        if equity:
            equity[-1] = cash

    equity_s = pd.Series(equity, index=d.index[1:1+len(equity)])
    dd = equity_s / equity_s.cummax() - 1.0
    max_dd = float(dd.min() * 100.0) if len(dd) else 0.0
    total_ret = float((equity_s.iloc[-1] / start_capital - 1.0) * 100.0) if len(equity_s) else 0.0

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    win_rate = float((trades_df["PnL"] > 0).mean() * 100.0) if n else 0.0

    gross_profit = float(trades_df.loc[trades_df["PnL"] > 0, "PnL"].sum()) if n else 0.0
    gross_loss = float(trades_df.loc[trades_df["PnL"] < 0, "PnL"].sum()) if n else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else (np.inf if gross_profit > 0 else np.nan)

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "total_return": total_ret,
        "trades": n,
        "trades_df": trades_df,
        "equity": equity_s,
        "dd": dd
    }


# --- MODÜL 5: SMART LOGIC (PUAN + PLAN) ---
def calculate_smart_logic(df, symbol: str, cap: float, current_price=None):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 50
    reasons = []

    bull_regime = (last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"])
    uptrend = last["Close"] > last["EMA50"]

    if bull_regime:
        score += 20
        reasons.append("✅ Trend Rejimi: Bull (Close > EMA200 & EMA50 > EMA200)")
    else:
        score -= 20
        reasons.append("🔻 Trend Rejimi: Zayıf/Bear (EMA200 altı)")

    if uptrend:
        score += 10
        reasons.append("✅ Fiyat EMA50 üzerinde")
    else:
        score -= 10
        reasons.append("⚠️ Fiyat EMA50 altında")

    curr_rsi = float(last["RSI"])
    prev_rsi = float(prev["RSI"])

    # RSI range rules (daha akıllı değerlendirme)
    if bull_regime and 40 <= curr_rsi <= 55:
        score += 18
        reasons.append("📈 Bull Pullback Bölgesi (RSI 40-55)")
    elif bull_regime and curr_rsi >= 80:
        if curr_rsi >= 90:
            score -= 10
            reasons.append("🔥 RSI 90+: Aşırı ısınma (temkin)")
        else:
            score += 6
            reasons.append("💪 RSI 80-90: Güçlü momentum (trend devamı)")
    elif (not bull_regime) and curr_rsi >= 60:
        score -= 15
        reasons.append("🔻 Bear Rejiminde RSI 60+: Sıçrama/direnç bölgesi (risk)")

    # Dip dönüşü
    if uptrend and prev_rsi < 30 and curr_rsi >= 30:
        score += 22
        reasons.append("🚀 Dip Dönüşü: RSI 30'u yukarı kırdı")

    # CMB Composite Index momentum
    cmb_strong = (
        (last["CMB_FAST"] > last["CMB_SLOW"]) and
        (last["CMB_CI"] > last["CMB_FAST"]) and
        (last["CMB_CI"] > prev["CMB_CI"])
    )
    cmb_ok = (last["CMB_FAST"] > last["CMB_SLOW"]) and (last["CMB_CI"] > prev["CMB_CI"])
    if cmb_strong:
        score += 18
        reasons.append("🧠 CMB Composite Index: Güçlü yükseliş (CI > Fast > Slow)")
    elif cmb_ok:
        score += 10
        reasons.append("🧠 CMB Composite Index: Pozitif momentum")

    # Breakout + volume
    vol_ok = True
    if "Volume" in df.columns and "VOL_MA20" in df.columns:
        vol_ok = float(last["Volume"]) > float(last["VOL_MA20"]) * 1.2
    breakout = bull_regime and vol_ok and (last["Close"] >= last["SWING_HIGH"] * 0.995)
    if breakout:
        score += 10
        reasons.append("🚀 Breakout: Swing High yakınında + hacim teyidi")

    # Sentiment proxy (deterministik)
    sent = sentiment_proxy(df)
    if sent > 0.35:
        score += 6
        reasons.append("🟢 Momentum/Volüm pozitif (sentiment proxy)")
    elif sent < -0.35:
        score -= 6
        reasons.append("🔴 Momentum/Volüm negatif (sentiment proxy)")

    # --- Risk & Hedef (GÜN İÇİ: TP %1-%3 + ATR tabanlı STOP) ---
    price = float(current_price) if current_price is not None else float(last["Close"])  # taramada anlık/son fiyat, backtestte next open
    atr = float(last["ATR"])

    # STOP: ATR tabanlı + yüzde clamp
    stop_mult = auto_stop_mult(atr, price) * float(sl_atr_mult)
    raw_stop_dist = atr * stop_mult
    sl_pct = clamp(raw_stop_dist / price if price > 0 else 0.0, float(sl_min) / 100.0, float(sl_max) / 100.0)
    stop_dist = price * sl_pct
    stop_price = price - stop_dist

    # TP: skor tabanlı dinamik, ama %1-%3 aralığına kilitli
    tp_pct = 0.01 + (score - 50) * (0.02 / 30.0)  # 50->%1, 80->%3 yaklaşımı
    tp_pct = clamp(tp_pct, float(tp_min) / 100.0, float(tp_max) / 100.0)
    target_price = price * (1 + tp_pct)
    target_type = f"TP %{tp_pct*100:.1f} (min %{tp_min:.1f} / max %{tp_max:.1f})"

    rr_ratio = (target_price - price) / stop_dist if stop_dist > 0 else 0.0


    rr_ratio = (target_price - price) / stop_dist if stop_dist > 0 else 0.0
    if rr_ratio < 1.5:
        score -= 18
        reasons.append(f"⛔ R/R düşük ({rr_ratio:.2f})")

    # Lot (risk fixed) + kasa kontrolü
    risk_amt = cap * (DEFAULT_RISK_PCT / 100.0)
    lot = int(risk_amt / stop_dist) if stop_dist > 0 else 0

    commission = COMMISSION_BPS / 10000.0
    max_lot_cash = int(cap / (price * (1 + commission))) if price > 0 else 0
    final_lot = max(0, min(lot, max_lot_cash))

    potential_profit = (target_price - price) * final_lot
    risk_money = final_lot * stop_dist

    final_score = int(max(0, min(100, round(score))))

    # Backtest confirm (otomatik)
    bt = run_backtest(df, cap)
    if bt["trades"] >= 6:
        if bt["profit_factor"] > 1.2 and bt["total_return"] > 0:
            final_score = min(100, final_score + 8)
            reasons.append(f"🔙 Backtest Onayı: PF {bt['profit_factor']:.2f}, Getiri %{bt['total_return']:.1f}")
        elif bt["profit_factor"] < 1.0 and bt["total_return"] < 0:
            final_score = max(0, final_score - 8)
            reasons.append(f"🔙 Backtest Zayıf: PF {bt['profit_factor']:.2f}, Getiri %{bt['total_return']:.1f}")
    else:
        reasons.append("🔙 Backtest: Yeterli işlem yok (n<6)")

    return {
        "symbol": symbol,
        "price": price,
        "stop": stop_price,
        "target": target_price,
        "lot": final_lot,
        "potential_profit": potential_profit,
        "risk_money": risk_money,
        "score": final_score,
        "reasons": reasons,
        "rr": rr_ratio,
        "tp_pct": tp_pct*100,
        "sl_pct": sl_pct*100,
        "target_type": target_type,
        "rsi": curr_rsi,
        "stop_mult": stop_mult,
        "sent": sent,
        "backtest": bt
    }


# ==================
# --- ARAYÜZ ---
# ==================
tab_single, tab_hunter = st.tabs(["🛡️ TEKLİ ANALİZ", "🦅 AKILLI AVCI"])

# --- SEKME 1: TEKLİ ANALİZ ---
with tab_single:
    col_s, _ = st.columns([1, 3])
    with col_s:
        symbol_input = st.text_input("Hisse Kodu", value="THYAO").upper().strip()

    if symbol_input:
        with st.spinner("Veri + Smart RSI + CMB Composite hesaplanıyor..."):
            df, sym = get_data(symbol_input, period, interval)

        if df is None:
            st.error("Veri çekilemedi. Hisse kodunu kontrol edin.")
        else:
            st.caption(f"Veri kaynağı: {sym} | Periyot: {period} | Interval: {interval}")

            data = calculate_smart_logic(df, symbol_input, capital)

            # ÜST METRİKLER (PUAN KUTUSU)
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                bg = "score-high" if data["score"] >= 75 else ("score-mid" if data["score"] >= 50 else "score-low")
                st.markdown(f"<div class='score-box {bg}'>{data['score']}</div>", unsafe_allow_html=True)
                st.caption(f"RSI: {data['rsi']:.1f} | Stop: ATR x{data['stop_mult']:.1f}")
            with c2:
                st.markdown("#### 📝 Otomatik Rapor")
                for r in data["reasons"][:10]:
                    st.markdown(f"<span class='reason-text'>• {r}</span>", unsafe_allow_html=True)
                if len(data["reasons"]) > 10:
                    with st.expander("Diğer nedenleri göster"):
                        for r in data["reasons"][10:]:
                            st.markdown(f"<span class='reason-text'>• {r}</span>", unsafe_allow_html=True)
            with c3:
                st.metric("Potansiyel Kar", f"{data['potential_profit']:.0f} TL")
                st.metric("R/R Oranı", f"{data['rr']:.2f}")

            # İŞLEM KARTI VE GRAFİK
            col_l, col_r = st.columns([1, 2])

            with col_l:
                st.markdown(f"""
                <div class='trade-card'>
                    <h3 style="margin:0; color:#00e5ff">İŞLEM PLANI (AUTO)</h3>
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

                bt = data["backtest"]
                with st.expander("🔍 Backtest Özeti (otomatik, komisyon+slipaj dahil)"):
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Getiri", f"%{bt['total_return']:.1f}")
                    m2.metric("Max DD", f"%{bt['max_dd']:.1f}")
                    pf = bt["profit_factor"]
                    m3.metric("Profit Factor", "∞" if pf == np.inf else (f"{pf:.2f}" if np.isfinite(pf) else "n/a"))
                    m4.metric("Win Rate", f"%{bt['win_rate']:.0f} (n={bt['trades']})")

                    if bt["trades_df"] is not None and not bt["trades_df"].empty:
                        st.dataframe(bt["trades_df"].round(3), use_container_width=True, hide_index=True)

                    if bt["equity"] is not None and len(bt["equity"]) > 0:
                        st.line_chart(bt["equity"])

            with col_r:
                tail = df.tail(120)
                fig = go.Figure(data=[go.Candlestick(
                    x=tail.index,
                    open=tail["Open"],
                    high=tail["High"],
                    low=tail["Low"],
                    close=tail["Close"]
                )])
                fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA50"], mode="lines", name="EMA50"))
                fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA200"], mode="lines", name="EMA200"))
                fig.add_hline(y=data["stop"], line_dash="dash", line_color="red")
                fig.add_hline(y=data["target"], line_dash="solid", line_color="#00c853")
                fig.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=10, b=10),
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)

# --- SEKME 2: AVCI MODU ---
with tab_hunter:
    st.info("Tarayıcı; Bull rejimde (EMA200 üstü) Smart RSI (range rules) + CMB Composite momentum + hacim teyidi arar.")
    if st.button("TARAMAYI BAŞLAT", type="primary"):
        start = time.time()
        candidates = []
        bar = st.progress(0)

        
        tickers = build_universe(universe)

        # Eğer seçilen evren boş döndüyse yine de BIST30 ile devam edelim
        if not tickers:
            tickers = [t + ".IS" for t in BIST30_TICKERS]

        candidates = []
        bar = st.progress(0.0)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _scan_one(tkr: str):
            # tkr genelde 'AKBNK.IS' gibi gelir; calculate fonksiyonu sembol ister
            sym = tkr.replace(".IS", "")
            df_scan, _ = get_data(sym, period, interval)
            if df_scan is None or len(df_scan) <= 250:
                return None
            rt_price = get_realtime_price(sym)
            res = calculate_smart_logic(df_scan, sym, capital, current_price=rt_price)

            # Filtre: skor & rr, ya da özel setup
            is_special_setup = any(
                ("Dip Dönüşü" in r) or ("Breakout" in r) or ("Bull Pullback" in r)
                for r in res["reasons"]
            )

            if (res["score"] >= min_score and res["rr"] >= min_rr) or is_special_setup:
                return res
            return None

        max_workers = 10  # BIST100+ için iyi denge
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_scan_one, t) for t in tickers]
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                done += 1
                try:
                    r = fut.result()
                    if r:
                        candidates.append(r)
                except Exception:
                    pass
                bar.progress(done / total)

        bar.empty()
        elapsed = time.time() - start

        if candidates:
            candidates.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
            top_list = candidates[:top_n]

            st.success(f"Tarama tamamlandı ({elapsed:.1f} sn). En güçlü fırsatlar ({len(candidates)} adet eşleşme):")
            for idx, item in enumerate(top_list):
                with st.container():
                    c1, c2, c3 = st.columns([1, 2, 2])
                    with c1:
                        bg = "score-high" if item["score"] >= 75 else ("score-mid" if item["score"] >= 60 else "score-low")
                        st.markdown(
                            f"<div class='score-box {bg}' style='font-size:1.8em'>#{idx+1}<br>{item['score']}</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown(f"<h3 style='text-align:center'>{item['symbol']}</h3>", unsafe_allow_html=True)
                        st.caption(f"R/R: {item['rr']:.2f} | RSI: {item['rsi']:.1f}")
                    with c2:
                        st.markdown("**🔍 Neden?**")
                        for r in item["reasons"][:6]:
                            st.caption(r)
                    with c3:
                        st.markdown("**📊 Plan:**")
                        st.write(f"• Giriş: **{item['price']:.2f}**")
                        st.write(f"• Stop: **{item['stop']:.2f}**")
                        st.write(f"• Hedef: **{item['target']:.2f}**")
                        st.write(f"• Lot: **{item['lot']}**")

                        bt = item["backtest"]
                        pf = bt["profit_factor"]
                        pf_str = "∞" if pf == np.inf else (f"{pf:.2f}" if np.isfinite(pf) else "n/a")
                        st.write(f"• Backtest: **%{bt['total_return']:.1f}**, PF **{pf_str}**, DD **%{bt['max_dd']:.1f}**")

                    st.markdown("---")
        else:
            st.warning("Kriterlere uyan fırsat bulunamadı.")
