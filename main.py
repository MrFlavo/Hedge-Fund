"""
NEXUS TRADE — BIST Day Trading Terminal
==================================================
Real-time data powered exclusively by Finnhub API.
All Features:
1. Advanced Day Trading Indicators (VWAP, MFI, Stochastic, OBV, MACD, BB)
2. ML Score Enhancement (XGBoost with 22 features incl. candlestick patterns)
3. Multi-Timeframe Analysis (MTF confluence)
4. SQLite Database Integration (historical data caching)
5. Performance Optimization (parallel processing, vectorized calculations)
6. Portfolio Management Layer (Enhanced with stock info table & P/L tracking)
7. Order Execution Simulation
8. Enhanced Analytics with equity curve + drawdown chart
9. Candlestick Pattern Detection (Hammer, Engulfing, Morning Star, etc.)
10. Signal Grade System (A+ / A / B / C / D / F)
11. Real-Time Data (Finnhub API — single source of truth)
12. Watchlist Dashboard with live grades & prices
13. Support & Resistance levels + Pivot Points on chart
14. MACD / Bollinger Bands / Stochastic visible on chart
15. Sector Relative Strength scoring & comparison
16. Earnings / Financial Report Calendar (stock + sector peers)
17. Auto-Refresh (60s live reload toggle)

Optimized for BIST (Borsa Istanbul) active day trading
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
    page_title="NEXUS TRADE · BIST Day Trading Terminal",
    page_icon="◈"
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

    # HTTP requests for Finnhub API
    import requests
        
except ImportError as e:
    st.error(f"⚠️ Missing libraries: {e}")
    st.code("pip install yfinance pandas plotly numpy xgboost scikit-learn requests")
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
# All tickers — clean symbols without .IS suffix (Finnhub uses IS: prefix)

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
# SECTOR MAPPING & RELATIVE STRENGTH
# =====================
BIST_SECTORS = {
    "Bankacılık": ["AKBNK", "GARAN", "ISCTR", "YKBNK", "HALKB", "VAKBN", "SKBNK", "ALBRK", "QNBFB", "TSKB"],
    "Havacılık": ["THYAO", "PGSUS", "TAVHL", "CLEBI"],
    "Otomotiv": ["FROTO", "TOASO", "OTKAR", "DOAS", "TTRAK", "KARSN"],
    "Enerji": ["TUPRS", "PETKM", "AKSEN", "AKENR", "ZOREN", "ENJSA", "GWIND", "ODAS", "ASTOR"],
    "Holding": ["KCHOL", "SAHOL", "ALARK", "TKFEN", "ENKA", "GSDHO"],
    "Perakende": ["BIMAS", "MGROS", "SOKM", "BIZIM", "CRFSA", "MAVI"],
    "Telekom & Teknoloji": ["TCELL", "TTKOM", "LOGO", "KRONT", "ARENA", "INDES", "NETAS", "PENTA"],
    "Savunma": ["ASELS"],
    "Demir & Çelik": ["EREGL", "KRDMD", "IZMDC", "BRSAN", "SARKY"],
    "Cam & Seramik": ["SISE", "TRKCM", "ANACM"],
    "Kimya & Petrokimya": ["SASA", "SODA", "ALKIM", "BAGFS", "CIMSA", "GUBRF"],
    "Gıda & İçecek": ["AEFES", "ULKER", "CCOLA", "TATGD", "PNSUT", "PINSU", "BANVT"],
    "GYO": ["EKGYO", "ISGYO", "KLGYO", "AGYO", "NUGYO", "OZGYO", "PEKGY", "VKGYO"],
    "Madencilik": ["KOZAL"],
    "Spor Kulüpleri": ["BJKAS", "GSRAY", "TSPOR", "FENER"],
}

def get_sector_for_symbol(symbol):
    """Find which sector a symbol belongs to."""
    clean = symbol.upper().replace(".IS", "")
    for sector, members in BIST_SECTORS.items():
        if clean in members:
            return sector
    return None

def get_sector_peers(symbol, max_peers=5):
    """Get peer stocks in the same sector."""
    clean = symbol.upper().replace(".IS", "")
    sector = get_sector_for_symbol(symbol)
    if not sector:
        return [], sector
    peers = [s for s in BIST_SECTORS[sector] if s != clean][:max_peers]
    return peers, sector

@st.cache_data(ttl=300, show_spinner=False)
def calculate_sector_relative_strength(symbol, period="5d", interval="15m"):
    """
    Calculate how a stock performs vs its sector peers.
    Returns relative strength score (-100 to +100) and details.
    """
    peers, sector = get_sector_peers(symbol)
    if not peers or not sector:
        return None
    
    clean = symbol.upper().replace(".IS", "")
    
    try:
        # Get stock performance
        df_stock = yf.download(f"{clean}.IS", period=period, interval=interval, progress=False)
        if df_stock is None or len(df_stock) < 10:
            return None
        
        if isinstance(df_stock.columns, pd.MultiIndex):
            df_stock.columns = df_stock.columns.get_level_values(0)
        
        stock_return = (float(df_stock["Close"].iloc[-1]) - float(df_stock["Close"].iloc[0])) / float(df_stock["Close"].iloc[0]) * 100
        
        # Get peer performances
        peer_returns = []
        for peer in peers[:4]:  # Limit to 4 peers for speed
            try:
                df_peer = yf.download(f"{peer}.IS", period=period, interval=interval, progress=False)
                if df_peer is not None and len(df_peer) >= 10:
                    if isinstance(df_peer.columns, pd.MultiIndex):
                        df_peer.columns = df_peer.columns.get_level_values(0)
                    ret = (float(df_peer["Close"].iloc[-1]) - float(df_peer["Close"].iloc[0])) / float(df_peer["Close"].iloc[0]) * 100
                    peer_returns.append({"symbol": peer, "return": ret})
            except Exception:
                pass
            time.sleep(0.3)
        
        if not peer_returns:
            return None
        
        sector_avg = np.mean([p["return"] for p in peer_returns])
        relative_strength = stock_return - sector_avg
        
        # Normalize to -100 to +100 scale
        rs_score = max(-100, min(100, relative_strength * 10))
        
        return {
            "sector": sector,
            "stock_return": stock_return,
            "sector_avg": sector_avg,
            "relative_strength": relative_strength,
            "rs_score": rs_score,
            "peers": peer_returns
        }
    except Exception:
        return None

# =====================
# EARNINGS CALENDAR (Financial Report Dates)
# =====================
# Source: KAP (kap.org.tr) — Official BIST Public Disclosure Platform
# Compiled from: KAP, Midas, Investing.com TR, CNBC-E, Rota Borsa, Borsa Gündem
# Last updated: 18 February 2026
#
# MULTI-QUARTER STRUCTURE: Easy to add Q1 2026 dates when KAP announces them
# Deadlines Q4 2025: Non-consolidated = 2 Mar 2026, Consolidated = 11 Mar 2026
# Deadlines Q1 2026: Non-consolidated = ~15 May 2026, Consolidated = ~30 May 2026

BIST_EARNINGS = {
    # ═══════════════════════════════════════════
    # Q4 2025 (12-month annual) — Active Season
    # ═══════════════════════════════════════════
    "Q4_2025": {
        "period_label": "Q4 2025 (12M Yıllık)",
        "deadline_non_consolidated": "2026-03-02",
        "deadline_consolidated": "2026-03-11",
        "dates": {
            # ── January 2026 — Early reporters ──
            "TRGYO": "2026-01-26",  # Torunlar GYO
            "ARCLK": "2026-01-30",  # Arçelik
            
            # ── February 2026 — Banks & industrials ──
            "AKBNK": "2026-02-02",  # Akbank
            "GARAN": "2026-02-04",  # Garanti BBVA
            "AKSIGORTA": "2026-02-05",  # Aksigorta
            "YKBNK": "2026-02-05",  # Yapı Kredi
            "ISCTR": "2026-02-06",  # İş Bankası
            "TUPRS": "2026-02-06",  # Tüpraş
            "TSKB":  "2026-02-06",  # TSKB
            "FROTO": "2026-02-09",  # Ford Otosan
            "TOASO": "2026-02-09",  # Tofaş
            "OTKAR": "2026-02-09",  # Otokar
            "TTRAK": "2026-02-09",  # Türk Traktör
            "AYGAZ": "2026-02-09",  # Aygaz
            "KCHOL": "2026-02-11",  # Koç Holding
            "ALBRK": "2026-02-13",  # Albaraka Türk
            "SISE":  "2026-02-16",  # Şişecam
            "TAVHL": "2026-02-17",  # TAV Havalimanları
            "TATGD": "2026-02-17",  # Tat Gıda
            "PETKM": "2026-02-17",  # Petkim
            "MGROS": "2026-02-17",  # Migros (early estimate filing)
            "YUNSA": "2026-02-17",  # Yünsa
            "SELEC": "2026-02-17",  # Selçuk Ecza
            "AKSA":  "2026-02-19",  # Aksa Akrilik
            "ASELS": "2026-02-24",  # Aselsan
            "VESTL": "2026-02-25",  # Vestel
            "BRSAN": "2026-02-26",  # Borusan Mannesmann
            "DOAS":  "2026-02-27",  # Doğuş Otomotiv
            "TKFEN": "2026-02-27",  # Tekfen Holding
            
            # ── March 2026 — Bulk season ──
            "HALKB": "2026-03-02",  # Halkbank (non-consol deadline)
            "VAKBN": "2026-03-02",  # Vakıfbank
            "SKBNK": "2026-03-02",  # Şekerbank
            "QNBFB": "2026-03-02",  # QNB Finans
            "CCOLA": "2026-03-03",  # Coca-Cola İçecek
            "TTKOM": "2026-03-03",  # Türk Telekom
            "SAHOL": "2026-03-04",  # Sabancı Holding
            "ENKA":  "2026-03-04",  # ENKA İnşaat
            "TCELL": "2026-03-05",  # Turkcell
            "AEFES": "2026-03-05",  # Anadolu Efes
            "PGSUS": "2026-03-05",  # Pegasus
            "ULKER": "2026-03-06",  # Ülker
            "MAVI":  "2026-03-06",  # Mavi Giyim
            "ALARK": "2026-03-09",  # Alarko Holding
            "KOZAL": "2026-03-09",  # Koza Altın
            "ENJSA": "2026-03-09",  # Enerjisa
            
            # ── 11 March 2026 — Consolidated deadline (all remaining) ──
            "THYAO": "2026-03-11",  # THY (son tarih)
            "BIMAS": "2026-03-11",  # BİM
            "SOKM":  "2026-03-11",  # ŞOK Marketler
            "EREGL": "2026-03-11",  # Erdemir
            "KRDMD": "2026-03-11",  # Kardemir
            "SASA":  "2026-03-11",  # SASA Polyester
            "GUBRF": "2026-03-11",  # Gübre Fabrikaları
            "EKGYO": "2026-03-11",  # Emlak Konut GYO
            "AKSEN": "2026-03-11",  # Aksa Enerji
            "ZOREN": "2026-03-11",  # Zorlu Enerji
            "LOGO":  "2026-03-11",  # Logo Yazılım
            "KRONT": "2026-03-11",  # Kontrolmatik
            "ASTOR": "2026-03-11",  # Astor Enerji
            "GWIND": "2026-03-11",  # Galata Wind
            "SODA":  "2026-03-11",  # Soda Sanayii
            "TRKCM": "2026-03-11",  # Trakya Cam
            "ANACM": "2026-03-11",  # Anadolu Cam
            "CIMSA": "2026-03-11",  # Çimsa
            "ISGYO": "2026-03-11",  # İş GYO
            "DSFAK": "2026-03-11",  # Destek Finans
            "IZMDC": "2026-03-11",  # İzmir Demir Çelik
            "SARKY": "2026-03-11",  # Sarkuysan
            "BJKAS": "2026-03-11",  # Beşiktaş
            "GSRAY": "2026-03-11",  # Galatasaray
            "FENER": "2026-03-11",  # Fenerbahçe
            "TSPOR": "2026-03-11",  # Trabzonspor
        }
    },
    
    # ═══════════════════════════════════════════
    # Q1 2026 (3-month) — Dates TBD (will be announced on KAP ~April-May 2026)
    # ═══════════════════════════════════════════
    "Q1_2026": {
        "period_label": "Q1 2026 (3 Aylık)",
        "deadline_non_consolidated": "2026-05-15",
        "deadline_consolidated": "2026-05-30",
        "dates": {
            # Companies typically report in similar order each quarter.
            # Dates will be added here as KAP announcements come in.
            # Banks usually report first (mid-April), then industrials (May).
            # ── Placeholder based on historical patterns ──
            # (Uncomment and update as dates are confirmed on KAP)
            # "AKBNK": "2026-04-XX",
            # "GARAN": "2026-04-XX",
            # "YKBNK": "2026-04-XX",
            # ... etc
        }
    },
    
    # ═══════════════════════════════════════════
    # Q2 2026 (6-month) — Template for future use
    # ═══════════════════════════════════════════
    "Q2_2026": {
        "period_label": "Q2 2026 (6 Aylık)",
        "deadline_non_consolidated": "2026-08-14",
        "deadline_consolidated": "2026-08-31",
        "dates": {}
    },
}

def get_earnings_info(symbol):
    """
    Get earnings/bilanço date for a BIST stock.
    Scans ALL quarters and returns the NEXT relevant one.
    Priority: upcoming > today > most recent reported.
    """
    clean = symbol.upper().replace(".IS", "")
    now = datetime.now()
    
    best_match = None
    
    for quarter_key in ["Q4_2025", "Q1_2026", "Q2_2026"]:
        quarter = BIST_EARNINGS.get(quarter_key, {})
        dates = quarter.get("dates", {})
        period_label = quarter.get("period_label", quarter_key)
        deadline_con = quarter.get("deadline_consolidated", "")
        
        date_str = None
        is_deadline = False
        
        if clean in dates:
            date_str = dates[clean]
        elif deadline_con:
            # Stock exists in our BIST lists but no specific date → assign deadline
            all_known = set()
            for s_list in [BIST30, BIST50, BIST100]:
                for t in s_list:
                    all_known.add(t.replace(".IS", ""))
            if clean in all_known and quarter_key == "Q4_2025":
                date_str = deadline_con
                is_deadline = True
        
        if not date_str:
            continue
        
        try:
            earn_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        
        days_diff = (earn_date - now).days
        
        if days_diff < -1:
            status = "reported"
            label = f"Reported {date_str}"
        elif days_diff <= 0:
            status = "today"
            label = "⚡ REPORTING TODAY"
        else:
            status = "upcoming"
            label = f"{days_diff}d → {date_str}"
        
        entry = {
            "symbol": clean,
            "date": earn_date,
            "date_str": date_str,
            "days_until": days_diff,
            "status": status,
            "label": label,
            "period": period_label,
            "quarter": quarter_key,
            "is_deadline": is_deadline,
        }
        
        # Prefer upcoming over reported
        if status in ("upcoming", "today"):
            return entry  # Immediately return the next upcoming
        
        # Keep the most recent reported as fallback
        if best_match is None or earn_date > best_match["date"]:
            best_match = entry
    
    return best_match

def get_sector_earnings_calendar(symbol):
    """
    Get earnings dates for all stocks in the same sector.
    Returns sorted list + sector name.
    """
    peers, sector = get_sector_peers(symbol, max_peers=15)
    if not sector:
        return None, None
    
    clean = symbol.upper().replace(".IS", "")
    all_stocks = list(dict.fromkeys([clean] + peers))  # deduplicate, keep order
    
    earnings_list = []
    for stock in all_stocks:
        info = get_earnings_info(stock)
        if info:
            info["is_target"] = (stock == clean)
            earnings_list.append(info)
    
    earnings_list.sort(key=lambda x: x["date"])
    return earnings_list, sector

def get_full_earnings_calendar(quarter="Q4_2025"):
    """
    Get the full earnings calendar for a quarter.
    Returns list of all stocks with dates, sorted chronologically.
    """
    quarter_data = BIST_EARNINGS.get(quarter, {})
    dates = quarter_data.get("dates", {})
    period_label = quarter_data.get("period_label", quarter)
    now = datetime.now()
    
    result = []
    for sym, date_str in dates.items():
        try:
            earn_date = datetime.strptime(date_str, "%Y-%m-%d")
            days_diff = (earn_date - now).days
            status = "reported" if days_diff < -1 else ("today" if days_diff <= 0 else "upcoming")
            result.append({
                "symbol": sym, "date": earn_date, "date_str": date_str,
                "days_until": days_diff, "status": status, "period": period_label
            })
        except ValueError:
            pass
    
    result.sort(key=lambda x: x["date"])
    return result

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
FINNHUB_DEFAULT_KEY = "d6a5mc9r01qsjlb9qr60d6a5mc9r01qsjlb9qr6g"

def _get_finnhub_key():
    """Get Finnhub API key from session state, environment, or default"""
    key = st.session_state.get("finnhub_api_key", "")
    if not key:
        key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        key = FINNHUB_DEFAULT_KEY
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

def get_realtime_quote(symbol):
    """Real-time quote from Finnhub — single source of truth."""
    quote = get_realtime_quote_finnhub(symbol)
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


# (Old Finnhub candle + merge functions removed — now integrated into get_data_with_db_cache)

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

def _macd_vectorized(close, fast=12, slow=26, signal=9):
    """MACD Line, Signal Line, Histogram"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _support_resistance_levels(high, low, close, window=20, num_levels=3):
    """
    Detect Support and Resistance levels using pivot point clustering.
    Returns lists of support and resistance price levels.
    """
    supports = []
    resistances = []
    
    if len(close) < window * 2:
        return supports, resistances
    
    for i in range(window, len(close) - window):
        if low.iloc[i] == low.iloc[i-window:i+window+1].min():
            supports.append(float(low.iloc[i]))
        if high.iloc[i] == high.iloc[i-window:i+window+1].max():
            resistances.append(float(high.iloc[i]))
    
    def cluster_levels(levels, threshold_pct=0.005):
        if not levels:
            return []
        levels_sorted = sorted(levels)
        clusters = [[levels_sorted[0]]]
        for lvl in levels_sorted[1:]:
            if (lvl - clusters[-1][-1]) / clusters[-1][-1] < threshold_pct:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        cluster_means = [(np.mean(c), len(c)) for c in clusters]
        cluster_means.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in cluster_means[:num_levels]]
    
    return cluster_levels(supports), cluster_levels(resistances)

def _pivot_points(high_val, low_val, close_val):
    """Classic Pivot Points for intraday trading."""
    pp = (high_val + low_val + close_val) / 3
    r1 = 2 * pp - low_val
    s1 = 2 * pp - high_val
    r2 = pp + (high_val - low_val)
    s2 = pp - (high_val - low_val)
    r3 = high_val + 2 * (pp - low_val)
    s3 = low_val - 2 * (high_val - pp)
    return {"PP": pp, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}

# =====================
# ADVANCED DATA FETCHING — HYBRID: yfinance history + Finnhub real-time bridge
# =====================
@st.cache_data(ttl=60, show_spinner=False)
def get_data_with_db_cache(symbol, period="30d", interval="15m"):
    """Fetch OHLCV data: yfinance for historical bars, Finnhub quote for latest price.
    
    Architecture:
    - yfinance: reliable BIST historical OHLCV (may be 1-3h delayed)
    - Finnhub /quote: real-time price to create a fresh synthetic bar
    - Result: full chart history + up-to-date latest bar
    """
    symbol = symbol.upper().strip().replace(".IS", "")
    yf_symbol = f"{symbol}.IS"
    
    # Fetch historical data from yfinance
    max_retries = 3
    df = None
    last_error = ""
    
    for attempt in range(max_retries):
        try:
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
            
            if df is not None and not df.empty and len(df) > 10:
                break
            else:
                last_error = f"yfinance returned {len(df) if df is not None else 0} bars"
                df = None
                
        except Exception as e:
            last_error = f"yfinance error: {str(e)[:80]}"
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
    
    if df is None or df.empty:
        if "fetch_errors" not in st.session_state:
            st.session_state.fetch_errors = {}
        st.session_state.fetch_errors[symbol] = last_error
        return None, symbol
    
    # Fix MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.title() for c in df.columns]
    
    # === FINNHUB REAL-TIME BRIDGE ===
    # Check if yfinance data is stale, if so append a synthetic bar from Finnhub quote
    try:
        last_ts = df.index[-1]
        if hasattr(last_ts, 'tz') and last_ts.tz is not None:
            last_ts_naive = last_ts.tz_localize(None)
        else:
            last_ts_naive = last_ts
        
        try:
            import pytz
            turkey_tz = pytz.timezone("Europe/Istanbul")
            now_turkey = datetime.now(turkey_tz).replace(tzinfo=None)
        except ImportError:
            now_turkey = datetime.utcnow() + timedelta(hours=3)
        
        data_age_minutes = (now_turkey - last_ts_naive).total_seconds() / 60
        
        # Only bridge during BIST market hours (10:00 - 18:10 Turkey time)
        market_open = 10 <= now_turkey.hour < 18 or (now_turkey.hour == 18 and now_turkey.minute <= 10)
        
        if data_age_minutes > 20 and market_open:
            quote = get_realtime_quote_finnhub(symbol)
            if quote and quote.get("current", 0) > 0:
                # Create synthetic bar from live quote
                last_close = float(df["Close"].iloc[-1])
                q_price = quote["current"]
                q_high = max(quote.get("high", q_price), q_price)
                q_low = min(quote.get("low", q_price), q_price)
                q_open = quote.get("open", last_close)
                
                # Round to the current interval boundary
                interval_mins = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
                mins = interval_mins.get(interval, 15)
                bar_time = now_turkey.replace(second=0, microsecond=0)
                bar_time = bar_time.replace(minute=(bar_time.minute // mins) * mins)
                
                # Match timezone awareness of existing index
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    import pytz as pz
                    bar_time = pz.timezone("Europe/Istanbul").localize(bar_time)
                
                synthetic = pd.DataFrame({
                    "Open": [q_open],
                    "High": [q_high],
                    "Low": [q_low],
                    "Close": [q_price],
                    "Volume": [float(df["Volume"].iloc[-5:].mean())],  # Approximate volume
                }, index=pd.DatetimeIndex([bar_time], name=df.index.name))
                
                df = pd.concat([df, synthetic])
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
    except Exception:
        pass  # If bridge fails, continue with yfinance data
    
    # Save to database
    if DB_AVAILABLE:
        try:
            conn = sqlite3.connect(DB_PATH)
            df_to_save = df.reset_index()
            df_to_save['symbol'] = symbol
            df_to_save['interval'] = interval
            # Handle both Date and Datetime column names
            for col_name in ['Date', 'Datetime', 'index']:
                if col_name in df_to_save.columns:
                    df_to_save.rename(columns={col_name: 'timestamp'}, inplace=True)
                    break
            
            if 'timestamp' in df_to_save.columns:
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
    df["BB_MID"] = df["SMA20"]
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["SMA20"] * 100
    
    # MACD
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = _macd_vectorized(df["Close"])
    
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
# MACHINE LEARNING SCORE ENHANCEMENT (V2 — POOLED + RISK-ADJUSTED)
# =====================
# Key improvements:
# 1. Pool training data across BIST30 (12K+ samples vs 400)
# 2. Risk-adjusted labels (penalize drawdown during holding period)
# 3. Volume flow & unusual volume features (delta, acceleration, spikes)
# 4. Mean reversion detection (z-score, exhaustion, overextension)
# 5. 30 features (up from 22)

ML_FEATURE_NAMES = [
    # Oscillators (5)
    "rsi", "mfi", "stoch_k", "stoch_d", "adx",
    # Trend distance (3)
    "dist_ema20_atr", "dist_ema50_atr", "dist_vwap_atr",
    # Volatility (2)
    "bb_width", "atr_pct",
    # Volume flow (5) — NEW
    "vol_ratio_ma20", "vol_accel", "vol_delta_proxy", "obv_trend", "unusual_vol",
    # Momentum (4)
    "returns_5", "returns_10", "mean_return_20", "std_return_20",
    # Candle structure (5)
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "is_green", "engulfing",
    # Streak (1)
    "green_streak",
    # Mean reversion (4) — NEW
    "zscore_vwap", "bb_position", "consec_same_dir", "rsi_divergence",
    # MACD (1) — NEW
    "macd_hist_accel",
]

def _extract_features_from_row(df, i, lookback=20):
    """Extract feature vector for a single bar. Shared between training and inference."""
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    hist = df.iloc[max(0, i-lookback):i]
    
    if len(hist) < 5:
        return None
    
    atr = float(row.get("ATR", 0.001))
    if atr <= 0:
        atr = 0.001
    close = float(row["Close"])
    
    # Candle structure
    body = abs(row["Close"] - row["Open"])
    total_range = row["High"] - row["Low"] if row["High"] > row["Low"] else 0.001
    upper_wick = row["High"] - max(row["Close"], row["Open"])
    lower_wick = min(row["Close"], row["Open"]) - row["Low"]
    
    prev_body = abs(prev_row["Close"] - prev_row["Open"])
    is_green = 1 if row["Close"] > row["Open"] else 0
    engulfing = 1 if (body > prev_body * 1.1 and is_green == 1 and prev_row["Close"] < prev_row["Open"]) else (
        -1 if (body > prev_body * 1.1 and is_green == 0 and prev_row["Close"] > prev_row["Open"]) else 0
    )
    
    # Momentum
    r5 = (close - float(hist["Close"].iloc[-5])) / float(hist["Close"].iloc[-5]) * 100 if len(hist) >= 5 else 0
    r10 = (close - float(hist["Close"].iloc[-10])) / float(hist["Close"].iloc[-10]) * 100 if len(hist) >= 10 else 0
    pct = hist["Close"].pct_change().dropna()
    
    # Green streak
    recent_5 = df.iloc[max(0, i-4):i+1]
    green_streak = sum(1 for j in range(len(recent_5)) if recent_5["Close"].iloc[j] > recent_5["Open"].iloc[j])
    
    # === VOLUME FLOW FEATURES (NEW) ===
    vol = float(row.get("Volume", 0))
    vol_ma20 = float(row.get("VOL_MA20", 1))
    if vol_ma20 <= 0:
        vol_ma20 = 1
    vol_ratio = vol / vol_ma20
    
    # Volume acceleration (is volume increasing bar-over-bar?)
    vol_hist = hist["Volume"].tail(5) if "Volume" in hist else pd.Series([1]*5)
    vol_accel = 0
    if len(vol_hist) >= 3:
        vol_accel = float((vol_hist.iloc[-1] - vol_hist.iloc[-3]) / (vol_hist.iloc[-3] + 1))
    
    # Volume delta proxy (up-volume vs down-volume using close vs open)
    recent_bars = df.iloc[max(0, i-4):i+1]
    up_vol = sum(float(recent_bars["Volume"].iloc[j]) for j in range(len(recent_bars)) if recent_bars["Close"].iloc[j] > recent_bars["Open"].iloc[j])
    down_vol = sum(float(recent_bars["Volume"].iloc[j]) for j in range(len(recent_bars)) if recent_bars["Close"].iloc[j] <= recent_bars["Open"].iloc[j])
    total_vol = up_vol + down_vol
    vol_delta = (up_vol - down_vol) / total_vol if total_vol > 0 else 0
    
    # OBV trend
    obv = float(row.get("OBV", 0))
    obv_ema = float(row.get("OBV_EMA", 0))
    obv_trend = (obv - obv_ema) / (abs(obv_ema) + 1)
    
    # Unusual volume (z-score of current volume vs 20-bar history)
    if "Volume" in hist and len(hist) >= 10:
        vol_mean = float(hist["Volume"].mean())
        vol_std = float(hist["Volume"].std())
        unusual_vol = (vol - vol_mean) / (vol_std + 1) if vol_std > 0 else 0
    else:
        unusual_vol = 0
    
    # === MEAN REVERSION FEATURES (NEW) ===
    # Z-score of price from VWAP
    vwap = float(row.get("VWAP", close))
    zscore_vwap = (close - vwap) / atr
    
    # BB position (-1 = at lower band, 0 = at mid, +1 = at upper band)
    bb_upper = float(row.get("BB_UPPER", close + atr))
    bb_lower = float(row.get("BB_LOWER", close - atr))
    bb_range = bb_upper - bb_lower if bb_upper > bb_lower else 0.001
    bb_position = (close - bb_lower) / bb_range * 2 - 1  # -1 to +1
    
    # Consecutive same-direction bars (exhaustion detection)
    consec = 0
    direction = 1 if is_green else -1
    for j in range(i-1, max(i-10, 0), -1):
        bar_dir = 1 if df.iloc[j]["Close"] > df.iloc[j]["Open"] else -1
        if bar_dir == direction:
            consec += 1
        else:
            break
    consec_same_dir = consec * direction  # positive = green streak, negative = red streak
    
    # RSI divergence (price making new high but RSI is lower = bearish divergence)
    rsi_div = 0
    if len(hist) >= 10:
        price_higher = close > float(hist["Close"].iloc[-5])
        rsi_lower = float(row["RSI"]) < float(hist["RSI"].iloc[-5]) if "RSI" in hist else False
        price_lower = close < float(hist["Close"].iloc[-5])
        rsi_higher = float(row["RSI"]) > float(hist["RSI"].iloc[-5]) if "RSI" in hist else False
        if price_higher and rsi_lower:
            rsi_div = -1  # bearish divergence
        elif price_lower and rsi_higher:
            rsi_div = 1   # bullish divergence
    
    # MACD histogram acceleration
    macd_hist = float(row.get("MACD_HIST", 0))
    prev_macd_hist = float(prev_row.get("MACD_HIST", 0))
    macd_hist_accel = macd_hist - prev_macd_hist
    
    return [
        # Oscillators
        float(row.get("RSI", 50)), float(row.get("MFI", 50)),
        float(row.get("STOCH_K", 50)), float(row.get("STOCH_D", 50)),
        float(row.get("ADX", 20)),
        # Trend distance
        (close - float(row.get("EMA20", close))) / atr,
        (close - float(row.get("EMA50", close))) / atr,
        (close - vwap) / atr,
        # Volatility
        float(row.get("BB_WIDTH", 3)), atr / close * 100,
        # Volume flow
        vol_ratio, vol_accel, vol_delta, obv_trend, unusual_vol,
        # Momentum
        r5, r10, float(pct.mean()) if len(pct) > 0 else 0, float(pct.std()) if len(pct) > 0 else 0,
        # Candle structure
        body / total_range, upper_wick / total_range, lower_wick / total_range, is_green, engulfing,
        # Streak
        green_streak,
        # Mean reversion
        zscore_vwap, bb_position, consec_same_dir, rsi_div,
        # MACD
        macd_hist_accel,
    ]


def prepare_ml_features(df, lookback=20):
    """Prepare features with RISK-ADJUSTED labels.
    Label = 1 only if: price rises >1% AND max drawdown stays above -0.5% during holding.
    This eliminates false positives where price spikes then crashes.
    """
    if len(df) < lookback + 10:
        return None, None
    
    features = []
    labels = []
    
    for i in range(lookback, len(df) - 5):
        feat = _extract_features_from_row(df, i, lookback)
        if feat is None:
            continue
        
        features.append(feat)
        
        # === RISK-ADJUSTED LABEL ===
        entry_price = float(df.iloc[i]["Close"])
        future = df.iloc[i+1:i+6]["Close"]
        
        if len(future) == 0:
            labels.append(0)
            continue
        
        future_max = float(future.max())
        future_min = float(future.min())
        
        max_gain = (future_max - entry_price) / entry_price
        max_drawdown = (future_min - entry_price) / entry_price
        
        # Label = 1 ONLY if gain > 1% AND drawdown stays above -0.5%
        label = 1 if (max_gain > 0.01 and max_drawdown > -0.005) else 0
        labels.append(label)
    
    if not features:
        return None, None
    
    return np.array(features), np.array(labels)


@st.cache_resource(ttl=1800)
def train_ml_model_pooled(_symbols_tuple, interval="15m", period="30d"):
    """Train ML model POOLED across multiple stocks.
    
    Key improvement: Instead of training on 1 stock (~400 samples),
    we pool data from BIST30 (~12,000 samples). This dramatically
    reduces overfitting and improves generalization.
    """
    if not ML_AVAILABLE:
        return None
    
    all_X = []
    all_y = []
    
    symbols = list(_symbols_tuple)
    
    for sym in symbols:
        try:
            sym_clean = f"{sym}.IS" if not sym.endswith(".IS") else sym
            raw_df = yf.download(sym_clean, period=period, interval=interval, progress=False)
            
            if raw_df is None or raw_df.empty or len(raw_df) < 60:
                continue
            
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)
            raw_df.columns = [c.title() for c in raw_df.columns]
            
            df = calculate_indicators(raw_df.copy())
            
            if df is None or len(df) < 60:
                continue
            
            X, y = prepare_ml_features(df)
            if X is not None and len(X) > 20:
                all_X.append(X)
                all_y.append(y)
            
            time.sleep(0.3)
        except Exception:
            continue
    
    if not all_X:
        return None
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    if len(X) < 100:
        return None
    
    # Chronological split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train XGBoost with regularization to prevent overfitting
    model = xgb.XGBClassifier(
        max_depth=4,           # Reduced from 5 → less overfitting
        n_estimators=200,      # More trees but shallower
        learning_rate=0.05,    # Slower learning → better generalization
        min_child_weight=5,    # Minimum samples per leaf
        subsample=0.8,         # Row sampling
        colsample_bytree=0.8,  # Column sampling
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=1.0,        # L2 regularization
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=max(1, sum(y_train == 0) / max(1, sum(y_train == 1)))  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    # Calculate precision (what % of our "buy" signals are actually profitable)
    y_pred = model.predict(X_test)
    true_pos = sum((y_pred == 1) & (y_test == 1))
    false_pos = sum((y_pred == 1) & (y_test == 0))
    precision = true_pos / max(1, true_pos + false_pos)
    
    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "precision": precision,
        "samples": len(X),
        "feature_importance": model.feature_importances_,
        "feature_names": ML_FEATURE_NAMES,
    }


# Backward-compatible wrapper
@st.cache_resource
def train_ml_model(symbol, df):
    """Train pooled model using BIST30 universe for better generalization."""
    # Use top 15 liquid BIST30 stocks for pooled training
    pool_symbols = tuple([
        "THYAO", "GARAN", "AKBNK", "ISCTR", "ASELS", "EREGL", "FROTO",
        "TUPRS", "KCHOL", "SAHOL", "TCELL", "SISE", "BIMAS", "PGSUS", "YKBNK"
    ])
    return train_ml_model_pooled(pool_symbols)


def get_ml_probability(df, model_data):
    """Get ML probability for current bar — uses shared feature extraction."""
    if model_data is None or df is None or len(df) < 30:
        return 0.5
    
    feat = _extract_features_from_row(df, len(df) - 1, lookback=20)
    if feat is None:
        return 0.5
    
    try:
        proba = model_data["model"].predict_proba([feat])[0][1]
        return float(proba)
    except Exception:
        return 0.5


# =====================
# CONFIDENCE FILTER
# =====================
ML_CONFIDENCE_THRESHOLD = 0.58  # Minimum ML probability to show actionable signal

def get_signal_confidence(ml_prob, score, confluence):
    """
    Calculate overall signal confidence and filter low-quality signals.
    Returns (confidence_level, is_actionable, label)
    """
    if ml_prob >= 0.70 and score >= 75 and confluence >= 70:
        return ("HIGH", True, "🟢 HIGH CONFIDENCE")
    elif ml_prob >= 0.60 and score >= 65:
        return ("MEDIUM", True, "🟡 MEDIUM CONFIDENCE")
    elif ml_prob >= ML_CONFIDENCE_THRESHOLD and score >= 55:
        return ("LOW", True, "🟠 LOW CONFIDENCE")
    else:
        return ("SKIP", False, "⚪ NO EDGE — SKIP")


# =====================
# MEAN REVERSION DETECTOR
# =====================
def detect_mean_reversion(df):
    """
    Detect overextended moves likely to reverse.
    Returns dict with alerts and reversal probability.
    """
    if df is None or len(df) < 20:
        return None
    
    last = df.iloc[-1]
    close = float(last["Close"])
    atr = float(last.get("ATR", 0.001))
    vwap = float(last.get("VWAP", close))
    rsi = float(last.get("RSI", 50))
    bb_upper = float(last.get("BB_UPPER", close + atr))
    bb_lower = float(last.get("BB_LOWER", close - atr))
    stoch_k = float(last.get("STOCH_K", 50))
    
    alerts = []
    reversal_score = 0  # -100 (bearish reversal) to +100 (bullish reversal)
    
    # VWAP z-score overextension
    zscore = (close - vwap) / atr if atr > 0 else 0
    if zscore > 2.5:
        alerts.append(("⚠️ OVEREXTENDED ABOVE VWAP", f"Z-score: {zscore:.1f}σ — mean reversion likely"))
        reversal_score -= 30
    elif zscore < -2.5:
        alerts.append(("⚠️ OVEREXTENDED BELOW VWAP", f"Z-score: {zscore:.1f}σ — bounce likely"))
        reversal_score += 30
    
    # BB extreme
    if close > bb_upper:
        alerts.append(("📈 ABOVE UPPER BB", "Price broke above Bollinger — exhaustion risk"))
        reversal_score -= 20
    elif close < bb_lower:
        alerts.append(("📉 BELOW LOWER BB", "Price broke below Bollinger — bounce zone"))
        reversal_score += 20
    
    # RSI extremes with Stochastic confirmation
    if rsi > 80 and stoch_k > 85:
        alerts.append(("🔴 RSI + STOCH OVERBOUGHT", f"RSI: {rsi:.0f}, %K: {stoch_k:.0f} — selling pressure building"))
        reversal_score -= 25
    elif rsi < 20 and stoch_k < 15:
        alerts.append(("🟢 RSI + STOCH OVERSOLD", f"RSI: {rsi:.0f}, %K: {stoch_k:.0f} — buying opportunity"))
        reversal_score += 25
    
    # Consecutive direction exhaustion
    consec = 0
    direction = 1 if df.iloc[-1]["Close"] > df.iloc[-1]["Open"] else -1
    for j in range(len(df)-2, max(len(df)-10, 0), -1):
        bar_dir = 1 if df.iloc[j]["Close"] > df.iloc[j]["Open"] else -1
        if bar_dir == direction:
            consec += 1
        else:
            break
    
    if consec >= 5 and direction == 1:
        alerts.append(("🔥 5+ GREEN BARS", "Exhaustion likely — take partial profits"))
        reversal_score -= 15
    elif consec >= 5 and direction == -1:
        alerts.append(("❄️ 5+ RED BARS", "Capitulation likely — watch for reversal candle"))
        reversal_score += 15
    
    reversal_score = max(-100, min(100, reversal_score))
    
    return {
        "alerts": alerts,
        "reversal_score": reversal_score,
        "zscore_vwap": zscore,
        "bb_position": (close - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5,
        "consec_bars": consec * direction,
    }

# =====================
# CANDLESTICK PATTERN DETECTION
# =====================
def detect_candlestick_patterns(df, lookback=5):
    """
    Detect common candlestick patterns for day trading.
    Returns dict with detected patterns and their bullish/bearish signals.
    """
    if df is None or len(df) < lookback + 2:
        return {"patterns": [], "score_adjustment": 0}
    
    patterns = []
    score_adj = 0
    
    tail = df.iloc[-(lookback+1):]
    last = tail.iloc[-1]
    prev = tail.iloc[-2]
    
    body = abs(last["Close"] - last["Open"])
    upper_wick = last["High"] - max(last["Close"], last["Open"])
    lower_wick = min(last["Close"], last["Open"]) - last["Low"]
    total_range = last["High"] - last["Low"]
    
    prev_body = abs(prev["Close"] - prev["Open"])
    is_green = last["Close"] > last["Open"]
    is_prev_red = prev["Close"] < prev["Open"]
    is_prev_green = prev["Close"] > prev["Open"]
    
    atr = float(last["ATR"]) if "ATR" in last and last["ATR"] > 0 else total_range
    
    # === DOJI (indecision → reversal signal) ===
    if total_range > 0 and body / total_range < 0.1:
        patterns.append(("🕯️ Doji", "Indecision — potential reversal", 0))
    
    # === HAMMER (bullish reversal at bottom) ===
    if (total_range > 0 and 
        lower_wick > body * 2 and 
        upper_wick < body * 0.5 and
        last["RSI"] < 40):
        patterns.append(("🔨 Hammer", "Bullish reversal at support", 8))
        score_adj += 8
    
    # === INVERTED HAMMER / SHOOTING STAR ===
    if (total_range > 0 and 
        upper_wick > body * 2 and 
        lower_wick < body * 0.5):
        if last["RSI"] > 65:
            patterns.append(("💫 Shooting Star", "Bearish reversal at resistance", -8))
            score_adj -= 8
        elif last["RSI"] < 40:
            patterns.append(("💫 Inverted Hammer", "Potential bullish reversal", 5))
            score_adj += 5
    
    # === BULLISH ENGULFING ===
    if (is_prev_red and is_green and 
        last["Open"] <= prev["Close"] and 
        last["Close"] >= prev["Open"] and
        body > prev_body * 1.1):
        patterns.append(("🟢 Bullish Engulfing", "Strong reversal — buyers dominate", 12))
        score_adj += 12
    
    # === BEARISH ENGULFING ===
    if (is_prev_green and not is_green and 
        last["Open"] >= prev["Close"] and 
        last["Close"] <= prev["Open"] and
        body > prev_body * 1.1):
        patterns.append(("🔴 Bearish Engulfing", "Reversal — sellers dominate", -12))
        score_adj -= 12
    
    # === MORNING STAR (3-candle bullish reversal) ===
    if len(tail) >= 3:
        c3 = tail.iloc[-3]  # First: big red
        c2 = prev            # Second: small body (star)
        c1 = last             # Third: big green
        
        c3_red = c3["Close"] < c3["Open"]
        c3_big = abs(c3["Close"] - c3["Open"]) > atr * 0.5
        c2_small = abs(c2["Close"] - c2["Open"]) < atr * 0.2
        c1_green = c1["Close"] > c1["Open"]
        c1_big = abs(c1["Close"] - c1["Open"]) > atr * 0.5
        
        if c3_red and c3_big and c2_small and c1_green and c1_big:
            patterns.append(("⭐ Morning Star", "Strong 3-bar bullish reversal", 15))
            score_adj += 15
    
    # === EVENING STAR (3-candle bearish reversal) ===
    if len(tail) >= 3:
        c3 = tail.iloc[-3]
        c2 = prev
        c1 = last
        
        c3_green = c3["Close"] > c3["Open"]
        c3_big = abs(c3["Close"] - c3["Open"]) > atr * 0.5
        c2_small = abs(c2["Close"] - c2["Open"]) < atr * 0.2
        c1_red = c1["Close"] < c1["Open"]
        c1_big = abs(c1["Close"] - c1["Open"]) > atr * 0.5
        
        if c3_green and c3_big and c2_small and c1_red and c1_big:
            patterns.append(("🌙 Evening Star", "Strong 3-bar bearish reversal", -15))
            score_adj -= 15
    
    # === THREE WHITE SOLDIERS (strong uptrend confirmation) ===
    if len(tail) >= 3:
        last3 = tail.iloc[-3:]
        all_green = all(last3["Close"].iloc[i] > last3["Open"].iloc[i] for i in range(3))
        ascending = all(last3["Close"].iloc[i] > last3["Close"].iloc[i-1] for i in range(1, 3))
        decent_bodies = all(abs(last3["Close"].iloc[i] - last3["Open"].iloc[i]) > atr * 0.3 for i in range(3))
        
        if all_green and ascending and decent_bodies:
            patterns.append(("🎖️ Three White Soldiers", "Strong bullish continuation", 10))
            score_adj += 10
    
    # === MARUBOZU (full body, no wicks — strong conviction) ===
    if total_range > 0 and body / total_range > 0.85:
        if is_green:
            patterns.append(("🟩 Green Marubozu", "Strong bullish conviction (no wicks)", 7))
            score_adj += 7
        else:
            patterns.append(("🟥 Red Marubozu", "Strong bearish conviction", -7))
            score_adj -= 7
    
    return {
        "patterns": patterns,
        "score_adjustment": max(-25, min(25, score_adj))  # Cap adjustment
    }

# =====================
# SIGNAL GRADE SYSTEM
# =====================
def get_signal_grade(score, ml_prob, confluence, candle_patterns):
    """
    Convert raw score into a professional signal grade.
    A+ = Exceptional setup, A = Strong, B = Decent, C = Marginal, D = Weak, F = Avoid
    """
    # Bonus for strong candlestick patterns
    has_strong_pattern = any(abs(p[2]) >= 10 for p in candle_patterns) if candle_patterns else False
    bullish_patterns = sum(1 for p in candle_patterns if p[2] > 0) if candle_patterns else 0
    
    if score >= 85 and ml_prob >= 0.70 and confluence >= 80:
        grade, label, color, emoji = "A+", "EXCEPTIONAL", "#00e676", "💎"
    elif score >= 80 and ml_prob >= 0.65:
        grade, label, color, emoji = "A+", "EXCEPTIONAL", "#00e676", "💎"
    elif score >= 75 and ml_prob >= 0.60:
        grade, label, color, emoji = "A", "STRONG BUY", "#00c853", "🟢"
    elif score >= 70 and ml_prob >= 0.55:
        grade, label, color, emoji = "B+", "BUY", "#76ff03", "✅"
    elif score >= 65:
        grade, label, color, emoji = "B", "LEAN BUY", "#c6ff00", "📈"
    elif score >= 55:
        grade, label, color, emoji = "C", "NEUTRAL", "#ffd600", "⚖️"
    elif score >= 45:
        grade, label, color, emoji = "D", "WEAK", "#ff9100", "⚠️"
    else:
        grade, label, color, emoji = "F", "AVOID", "#d50000", "🚫"
    
    # Upgrade if strong bullish candle pattern confirms
    if has_strong_pattern and bullish_patterns > 0 and grade in ["B", "B+"]:
        grade, label, color, emoji = "A", "STRONG BUY (Pattern Confirmed)", "#00c853", "🟢"
    
    return {
        "grade": grade,
        "label": label,
        "color": color,
        "emoji": emoji
    }

# =====================
# ENHANCED SCORING ALGORITHM
# =====================
def calculate_advanced_score(df, symbol, mtf_data=None, ml_model=None, sector_rs=None):
    """
    Enhanced scoring with:
    - Day trading specific indicators (VWAP, MFI, Stochastic)
    - Multi-timeframe confluence
    - ML probability enhancement
    - Candlestick patterns
    - Sector relative strength
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
    
    # === CANDLESTICK PATTERN ANALYSIS (±25 points) ===
    candle_result = detect_candlestick_patterns(df)
    candle_patterns = candle_result["patterns"]
    score += candle_result["score_adjustment"]
    
    for pattern_name, pattern_desc, pattern_pts in candle_patterns:
        sign = "+" if pattern_pts >= 0 else ""
        reasons.append(f"{pattern_name} {pattern_desc} ({sign}{pattern_pts} pts)")
    
    # === SECTOR RELATIVE STRENGTH (±10 points) ===
    sector_data = None
    if sector_rs:
        sector_data = sector_rs
        rs = sector_rs.get("relative_strength", 0)
        if rs > 2:
            score += 10
            reasons.append(f"🏆 Sector RS: +{rs:.1f}% vs {sector_rs['sector']} avg (outperforming)")
        elif rs > 0.5:
            score += 5
            reasons.append(f"📊 Sector RS: +{rs:.1f}% vs {sector_rs['sector']} (slightly above)")
        elif rs < -2:
            score -= 8
            reasons.append(f"⚠️ Sector RS: {rs:.1f}% vs {sector_rs['sector']} (underperforming)")
    
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
    
    # Signal grade
    signal = get_signal_grade(final_score, ml_prob, confluence, candle_patterns)
    
    # Confidence filter
    confidence_level, is_actionable, confidence_label = get_signal_confidence(ml_prob, final_score, confluence)
    
    # Mean reversion detection
    mean_rev = detect_mean_reversion(df)
    
    # Adjust score if mean reversion detected against the trend
    if mean_rev and mean_rev["reversal_score"] != 0:
        rev_adj = int(mean_rev["reversal_score"] * 0.1)  # ±10 points max
        score += rev_adj
        if abs(rev_adj) >= 3:
            rev_dir = "bullish reversal" if rev_adj > 0 else "bearish reversal"
            reasons.append(f"🔄 Mean Reversion: {rev_dir} signal ({rev_adj:+d} pts)")
    
    final_score = max(0, min(100, int(score)))
    
    return {
        "symbol": symbol,
        "score": final_score,
        "signal": signal,
        "reasons": reasons,
        "candle_patterns": candle_patterns,
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
        "mtf_confluence": confluence,
        "sector_rs": sector_data,
        "confidence": confidence_level,
        "is_actionable": is_actionable,
        "confidence_label": confidence_label,
        "mean_reversion": mean_rev,
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
    /* ========================================
       NEXUS TRADE — CYBERPUNK TERMINAL THEME
       Bloomberg meets Tron. Dark glass. Neon pulse.
       ======================================== */
    
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
    
    /* === GLOBAL === */
    .stApp {
        background: #06080d;
        color: #e0e6ed;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Noise texture overlay */
    .stApp::before {
        content: '';
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: 
            radial-gradient(ellipse at 20% 50%, rgba(0,229,255,0.03) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(110,0,255,0.03) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(0,255,136,0.02) 0%, transparent 50%);
        pointer-events: none; z-index: 0;
    }
    
    /* === SCROLLBAR === */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0d14; }
    ::-webkit-scrollbar-thumb { background: #1a2332; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #00e5ff33; }
    
    /* === HEADERS === */
    h1, h2, h3, h4, h5 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080b12 0%, #0c1018 50%, #080b12 100%) !important;
        border-right: 1px solid rgba(0,229,255,0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* === TAB BAR === */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(10,15,25,0.8);
        border: 1px solid rgba(0,229,255,0.1);
        border-radius: 14px;
        padding: 4px;
        gap: 4px;
        backdrop-filter: blur(20px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78em;
        font-weight: 500;
        letter-spacing: 0.02em;
        padding: 8px 16px;
        color: #7a8a9e;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,229,255,0.12) 0%, rgba(110,0,255,0.08) 100%) !important;
        color: #00e5ff !important;
        border: 1px solid rgba(0,229,255,0.2);
        box-shadow: 0 0 20px rgba(0,229,255,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    
    /* === BUTTONS === */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.04em;
        border: 1px solid rgba(0,229,255,0.2);
        background: linear-gradient(135deg, rgba(0,229,255,0.08) 0%, rgba(0,229,255,0.02) 100%);
        color: #00e5ff;
        border-radius: 10px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        font-size: 0.8em;
    }
    .stButton > button:hover {
        border-color: #00e5ff;
        box-shadow: 0 0 25px rgba(0,229,255,0.15), 0 0 50px rgba(0,229,255,0.05);
        background: linear-gradient(135deg, rgba(0,229,255,0.15) 0%, rgba(0,229,255,0.05) 100%);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00e5ff 0%, #00b8d4 100%);
        color: #06080d;
        border: none;
        font-weight: 800;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 30px rgba(0,229,255,0.3), 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* === INPUTS === */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea > div > div > textarea {
        background: rgba(10,15,25,0.6) !important;
        border: 1px solid rgba(0,229,255,0.1) !important;
        border-radius: 10px !important;
        color: #e0e6ed !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85em !important;
        backdrop-filter: blur(10px);
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(0,229,255,0.4) !important;
        box-shadow: 0 0 15px rgba(0,229,255,0.08) !important;
    }
    
    /* === METRICS === */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(12,18,30,0.8) 0%, rgba(15,22,38,0.6) 100%);
        border: 1px solid rgba(0,229,255,0.06);
        border-radius: 12px;
        padding: 16px 18px;
        backdrop-filter: blur(15px);
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72em !important;
        color: #5a6a7e !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }
    
    /* === SCORE BOX === */
    .score-box {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8em; font-weight: 900; text-align: center;
        padding: 18px; border-radius: 16px; margin-bottom: 15px;
        color: white; position: relative; overflow: hidden;
        letter-spacing: -0.02em;
    }
    .score-box::before {
        content: ''; position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: conic-gradient(from 0deg, transparent 0%, rgba(255,255,255,0.05) 25%, transparent 50%);
        animation: score-shimmer 4s linear infinite;
    }
    @keyframes score-shimmer {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .score-high {
        background: linear-gradient(135deg, #00c853 0%, #00e676 50%, #69f0ae 100%);
        box-shadow: 0 4px 30px rgba(0,200,83,0.25);
    }
    .score-mid {
        background: linear-gradient(135deg, #ffa000 0%, #ffd600 50%, #ffea00 100%);
        color: #06080d;
        box-shadow: 0 4px 30px rgba(255,214,0,0.2);
    }
    .score-low {
        background: linear-gradient(135deg, #c62828 0%, #d50000 50%, #ff1744 100%);
        box-shadow: 0 4px 30px rgba(213,0,0,0.25);
    }
    
    /* === GLASS CARD (base) === */
    .glass-card {
        background: linear-gradient(145deg, rgba(12,18,30,0.7) 0%, rgba(18,26,42,0.5) 100%);
        border: 1px solid rgba(0,229,255,0.06);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(20px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(0,229,255,0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 0 20px rgba(0,229,255,0.03);
    }
    
    /* === METRIC CARD === */
    .metric-card {
        background: linear-gradient(145deg, rgba(12,18,30,0.8) 0%, rgba(18,26,42,0.5) 100%);
        padding: 16px 18px; border-radius: 14px; margin: 8px 0;
        border: 1px solid rgba(0,229,255,0.06);
        backdrop-filter: blur(15px);
        border-left: 3px solid #00e5ff;
    }
    
    /* === TRADE PLAN === */
    .trade-plan {
        background: linear-gradient(145deg, rgba(10,15,25,0.9) 0%, rgba(15,22,38,0.7) 100%);
        padding: 22px; border-radius: 16px;
        border: 1px solid rgba(0,229,255,0.15);
        box-shadow: 0 8px 32px rgba(0,229,255,0.05), inset 0 1px 0 rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
    }
    .trade-plan h3 { color: #00e5ff; font-family: 'JetBrains Mono', monospace; }
    .trade-plan p { font-family: 'JetBrains Mono', monospace; font-size: 0.9em; color: #c0cad8; }
    .trade-plan b { color: #ffffff; }
    
    /* === SIGNAL GRADE === */
    .signal-grade {
        text-align: center; padding: 14px 24px; border-radius: 20px;
        font-family: 'Outfit', sans-serif;
        font-size: 2em; font-weight: 900; letter-spacing: 3px;
        margin-bottom: 10px;
        position: relative;
    }
    .signal-label {
        text-align: center; font-size: 0.78em; font-weight: 600;
        letter-spacing: 2px; padding: 4px 0; margin-bottom: 14px;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
    }
    
    /* === CANDLE PATTERN CARDS === */
    .candle-card {
        padding: 10px 16px; border-radius: 10px; margin: 5px 0;
        background: rgba(255,255,255,0.02);
        border-left: 3px solid rgba(0,229,255,0.3);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82em;
        backdrop-filter: blur(10px);
    }
    .candle-card-bull { border-left-color: #00e676; background: rgba(0,230,118,0.04); }
    .candle-card-bear { border-left-color: #ff1744; background: rgba(255,23,68,0.04); }
    
    /* === SCANNER CARDS === */
    .scanner-card {
        background: linear-gradient(145deg, rgba(12,18,30,0.7) 0%, rgba(18,26,42,0.5) 100%);
        border: 1px solid rgba(0,229,255,0.06); border-radius: 16px;
        padding: 20px; margin: 8px 0;
        backdrop-filter: blur(15px);
        transition: all 0.3s ease;
    }
    .scanner-card:hover {
        border-color: rgba(0,229,255,0.15);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }
    .scanner-card-hot {
        border: 1px solid rgba(0,230,118,0.2);
        box-shadow: 0 0 30px rgba(0,230,118,0.05);
    }
    
    /* === P&L CARDS === */
    .pnl-card {
        padding: 18px 22px; border-radius: 14px; margin: 6px 0;
        border: 1px solid rgba(255,255,255,0.04);
        backdrop-filter: blur(15px);
        font-family: 'JetBrains Mono', monospace;
    }
    .pnl-card-profit { background: rgba(0,230,118,0.05); border-left: 3px solid #00e676; }
    .pnl-card-loss { background: rgba(255,23,68,0.05); border-left: 3px solid #ff1744; }
    .pnl-card-neutral { background: rgba(255,214,0,0.05); border-left: 3px solid #ffd600; }
    .pnl-card-info { background: rgba(0,229,255,0.05); border-left: 3px solid #00e5ff; }
    
    /* === FRESHNESS BADGE === */
    .freshness-badge {
        display: inline-block; padding: 6px 14px; border-radius: 24px;
        font-size: 0.75em; font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
    }
    
    /* === EXPANDER === */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85em !important;
        background: rgba(10,15,25,0.5) !important;
        border: 1px solid rgba(0,229,255,0.06) !important;
        border-radius: 10px !important;
    }
    
    /* === DATA FRAME === */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    
    /* === NEON PULSE ANIMATION (for live elements) === */
    @keyframes neon-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .neon-live { animation: neon-pulse 2s ease-in-out infinite; }
    
    /* === GRID BACKGROUND FOR HEADERS === */
    .grid-header {
        position: relative;
        padding: 24px 0 16px 0;
    }
    .grid-header::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: 
            linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        border-radius: 16px;
        pointer-events: none;
    }
    
    /* === DIVIDERS === */
    hr { border-color: rgba(0,229,255,0.06) !important; }
    
    /* === SUCCESS/WARNING/ERROR BOXES === */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 12px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85em !important;
    }
    
    /* === HIDE STREAMLIT BRANDING === */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: rgba(6,8,13,0.8); backdrop-filter: blur(20px); }
</style>
""", unsafe_allow_html=True)

# =====================
# MAIN APPLICATION
# =====================
st.markdown("""
<div class="grid-header">
    <div style="text-align:center;">
        <div style="display:inline-flex; align-items:center; gap:14px;">
            <svg width="42" height="42" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="logo_grad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#00e5ff;stop-opacity:1" />
                        <stop offset="50%" style="stop-color:#00b8d4;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#00e676;stop-opacity:1" />
                    </linearGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                        <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                </defs>
                <!-- Outer hexagon -->
                <polygon points="50,5 90,27.5 90,72.5 50,95 10,72.5 10,27.5" 
                         fill="none" stroke="url(#logo_grad)" stroke-width="2.5" filter="url(#glow)"/>
                <!-- Inner diamond -->
                <polygon points="50,22 72,50 50,78 28,50" 
                         fill="none" stroke="url(#logo_grad)" stroke-width="1.8" opacity="0.7"/>
                <!-- Center pulse dot -->
                <circle cx="50" cy="50" r="5" fill="#00e5ff" opacity="0.9">
                    <animate attributeName="r" values="4;7;4" dur="2s" repeatCount="indefinite"/>
                    <animate attributeName="opacity" values="0.9;0.5;0.9" dur="2s" repeatCount="indefinite"/>
                </circle>
                <!-- Cross lines -->
                <line x1="50" y1="30" x2="50" y2="70" stroke="#00e5ff" stroke-width="1" opacity="0.3"/>
                <line x1="30" y1="50" x2="70" y2="50" stroke="#00e5ff" stroke-width="1" opacity="0.3"/>
                <!-- N letter hint -->
                <path d="M38,62 L38,38 L62,62 L62,38" fill="none" stroke="url(#logo_grad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div>
                <div style="font-family:'Outfit',sans-serif; font-size:2.4em; font-weight:900; letter-spacing:-0.03em; 
                     background:linear-gradient(135deg, #00e5ff 0%, #00b8d4 30%, #ffffff 60%, #00e5ff 100%);
                     -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
                    NEXUS TRADE
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.62em; color:#3a5068; letter-spacing:0.3em; text-transform:uppercase; margin-top:-2px;">
                    BIST Day Trading Terminal · Finnhub + YF Hybrid
                </div>
            </div>
        </div>
        <div style="width:80px; height:2px; background:linear-gradient(90deg, transparent, #00e5ff, transparent); margin:10px auto 0;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================
# HARDCODED DATA SETTINGS (no sidebar clutter)
# =====================
data_interval = "15m"
data_period = "30d"

# =====================
# AUTO-REFRESH (60 seconds)
# =====================
REFRESH_INTERVAL = 60

# Sidebar: minimal cyberpunk terminal
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 12px;">
        <div style="font-family:'Outfit',sans-serif; font-size:1.3em; font-weight:800; letter-spacing:-0.02em;
             background:linear-gradient(135deg, #00e5ff, #00b8d4);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            NEXUS
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.6em; color:#2a3a4e; letter-spacing:0.2em; margin-top:2px;">
            TRADE TERMINAL
        </div>
        <div style="width:40px; height:1px; background:linear-gradient(90deg, transparent, #00e5ff40, transparent); margin:8px auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    auto_refresh = st.toggle("⏱ AUTO-REFRESH", value=False, key="auto_refresh_toggle",
                              help="Auto-reload every 60 seconds")
    
    if auto_refresh:
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.68em; color:#00e5ff; 
             background:rgba(0,229,255,0.04); padding:6px 10px; border-radius:8px; border:1px solid rgba(0,229,255,0.1);">
            <span class="neon-live">●</span> LIVE — {datetime.now().strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
        
        if "refresh_counter" not in st.session_state:
            st.session_state.refresh_counter = 0
        st.session_state.refresh_counter += 1
        time.sleep(0.1)
    
    st.markdown("""
    <div style="margin-top:16px; padding:12px; background:rgba(10,15,25,0.5); border-radius:10px; border:1px solid rgba(0,229,255,0.04);">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.65em; color:#3a5068; line-height:1.8;">
            <span style="color:#5a6a7e;">INTERVAL</span> <span style="color:#00e5ff;">15m</span><br>
            <span style="color:#5a6a7e;">LOOKBACK</span> <span style="color:#00e5ff;">30d</span><br>
            <span style="color:#5a6a7e;">DATA SRC</span> <span style="color:#00e676;">YF + Finnhub RT</span><br>
            <span style="color:#5a6a7e;">ML</span> <span style="color:#00e676;">XGBoost 22F</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Auto-refresh via meta tag
if auto_refresh:
    st.markdown(
        f'<meta http-equiv="refresh" content="{REFRESH_INTERVAL}">',
        unsafe_allow_html=True
    )

tab_watchlist, tab_single, tab_scanner, tab_portfolio, tab_analytics = st.tabs([
    "⚡ WATCHLIST",
    "◉ ANALYSIS", 
    "⌬ SCANNER", 
    "▣ PORTFOLIO",
    "◫ ANALYTICS"
])

# =====================
# TAB 0: WATCHLIST DASHBOARD (NEW)
# =====================
with tab_watchlist:
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span style="font-family:'Outfit',sans-serif; font-size:1.5em; font-weight:800; color:#ffffff;">Active Watchlist</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.7em; color:#3a5068; margin-left:12px; letter-spacing:0.15em;">LIVE GRADES & SIGNALS</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Default watchlist (user can customize)
    default_watchlist = "THYAO, GARAN, AKBNK, ASELS, TCELL, TUPRS, FROTO, SISE, KCHOL, EREGL"
    
    wl_input = st.text_input(
        "📋 Watchlist (comma-separated)", 
        value=st.session_state.get("watchlist_input", default_watchlist),
        key="wl_input_field",
        help="Enter stock codes separated by commas"
    )
    st.session_state["watchlist_input"] = wl_input
    
    wl_cols = st.columns([1, 1])
    with wl_cols[0]:
        wl_enable_ml = st.checkbox("🤖 ML Scores", value=True, key="wl_ml")
    with wl_cols[1]:
        wl_auto = st.checkbox("Auto-refresh on load", value=False, key="wl_auto", help="Scan automatically (slower initial load)")
    
    if st.button("⚡ Refresh Watchlist", type="primary", key="wl_refresh", use_container_width=True) or wl_auto:
        watchlist = [s.strip().upper().replace(".IS", "") for s in wl_input.split(",") if s.strip()]
        
        if watchlist:
            progress = st.progress(0)
            wl_results = []
            
            for idx, sym in enumerate(watchlist):
                progress.progress((idx + 1) / len(watchlist))
                try:
                    df, _ = get_data_with_db_cache(sym, data_period, data_interval)
                    if df is not None and len(df) >= 50:
                        ml_model = None
                        if wl_enable_ml and ML_AVAILABLE:
                            ml_model = train_ml_model(sym, df)
                        
                        result = calculate_advanced_score(df, sym, None, ml_model)
                        if result:
                            # Get live quote
                            rt = get_realtime_quote(sym)
                            result["rt_quote"] = rt
                            result["df"] = df
                            wl_results.append(result)
                except Exception:
                    pass
            
            progress.empty()
            
            if wl_results:
                # Sort by score
                wl_results.sort(key=lambda x: x["score"], reverse=True)
                
                # Summary row
                avg_score = np.mean([r["score"] for r in wl_results])
                top_grade = wl_results[0]["signal"]["grade"]
                bullish = sum(1 for r in wl_results if r["score"] >= 65)
                bearish = sum(1 for r in wl_results if r["score"] < 45)
                
                sum_cols = st.columns(4)
                sum_cols[0].metric("Avg Score", f"{avg_score:.0f}")
                sum_cols[1].metric("Top Signal", top_grade)
                sum_cols[2].metric("Bullish", f"{bullish}/{len(wl_results)}")
                sum_cols[3].metric("Bearish", f"{bearish}/{len(wl_results)}")
                
                st.markdown("---")
                
                # Display cards in 2 columns
                col_left, col_right = st.columns(2)
                
                for i, res in enumerate(wl_results):
                    target_col = col_left if i % 2 == 0 else col_right
                    sig = res["signal"]
                    rt = res.get("rt_quote")
                    
                    # Price display
                    if rt:
                        price_str = f"₺{rt['current']:.2f}"
                        chg = rt.get("change_pct", 0)
                        chg_color = "#00c853" if chg >= 0 else "#d50000"
                        chg_sign = "+" if chg >= 0 else ""
                        chg_str = f'<span style="color:{chg_color}">{chg_sign}{chg:.2f}%</span>'
                    else:
                        price_str = f"₺{res['price']:.2f}"
                        chg_str = ""
                    
                    # Candle patterns summary
                    candle_str = ""
                    if res.get("candle_patterns"):
                        candle_str = " ".join([p[0] for p in res["candle_patterns"][:2]])
                    
                    # Mini sparkline data (last 20 closes)
                    df_spark = res.get("df")
                    spark_trend = ""
                    if df_spark is not None and len(df_spark) >= 10:
                        recent = df_spark["Close"].tail(10).tolist()
                        if recent[-1] > recent[0]:
                            spark_trend = "📈"
                        else:
                            spark_trend = "📉"
                    
                    card_extra = "scanner-card-hot" if sig["grade"] in ["A+", "A"] else ""
                    
                    with target_col:
                        st.markdown(f"""
                        <div class="scanner-card {card_extra}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-size:1.4em; font-weight:800;">{res['symbol']}</span>
                                    <span style="margin-left:8px; font-size:1.1em;">{price_str}</span>
                                    {chg_str}
                                    <span style="margin-left:6px;">{spark_trend}</span>
                                </div>
                                <div style="text-align:right;">
                                    <div style="background:{sig['color']}22; color:{sig['color']}; padding:4px 14px; border-radius:20px; font-weight:800; font-size:1.1em; border:1px solid {sig['color']}44;">
                                        {sig['emoji']} {sig['grade']}
                                    </div>
                                </div>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-top:10px; font-size:0.85em; color:#aaa;">
                                <span>Score: <b style="color:white">{res['score']}</b></span>
                                <span>RSI: <b style="color:white">{res['rsi']:.0f}</b></span>
                                <span>ML: <b style="color:white">{res['ml_prob']*100:.0f}%</b></span>
                                <span>R/R: <b style="color:white">{res['rr']:.1f}</b></span>
                                <span style="color:{'#00e676' if res.get('confidence')=='HIGH' else ('#ffd600' if res.get('confidence')=='MEDIUM' else ('#ff9100' if res.get('confidence')=='LOW' else '#5a6a7e'))}; font-weight:700;">{res.get('confidence','—')}</span>
                            </div>
                            <div style="margin-top:6px; font-size:0.8em; color:#888;">
                                {sig['label']} {candle_str}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No data returned for watchlist symbols. Check if BIST market is open.")

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
                err = st.session_state.get("fetch_errors", {}).get(symbol_input.replace(".IS",""), "Unknown")
                st.error(f"❌ Data fetch failed for **{symbol_input}**")
                st.caption(f"Debug: {err}")
                st.info("💡 Check: Is the symbol correct? Is BIST market open? Try again in a few seconds.")
            else:
                # === REAL-TIME QUOTE OVERLAY ===
                rt_quote = get_realtime_quote(symbol_input)
                
                # Data freshness indicator
                age_str, age_icon, age_color = format_data_age(df)
                
                # Detect if Finnhub data was merged (last bar timestamp)
                last_bar_ts = df.index[-1]
                last_bar_str = str(last_bar_ts).split("+")[0]  # Remove tz info for display
                total_bars = len(df)
                
                freshness_cols = st.columns([2, 2, 2])
                with freshness_cols[0]:
                    st.markdown(f"""
                        <div style="padding:8px 12px; border-radius:8px; background:rgba(0,229,255,0.08); border-left:3px solid {age_color};">
                            {age_icon} <b>Last Bar:</b> {last_bar_str}
                            <br><span style="color:#888; font-size:0.78em;">{age_str} · {total_bars} bars · {data_interval}</span>
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
                                ⚠️ <b>No live quote</b> — Market may be closed
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
                sig = result["signal"]
                
                # === SIGNAL GRADE HEADER ===
                st.markdown(f"""
                    <div style="text-align:center; margin:10px 0 20px 0;">
                        <div class="signal-grade" style="background:{sig['color']}18; color:{sig['color']}; border:2px solid {sig['color']}44; display:inline-block;">
                            {sig['emoji']} {sig['grade']}
                        </div>
                        <div class="signal-label" style="color:{sig['color']};">{sig['label']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display
                col_a, col_b, col_c = st.columns([1, 2, 1])
                
                with col_a:
                    bg = "score-high" if result["score"] >= 75 else ("score-mid" if result["score"] >= 60 else "score-low")
                    st.markdown(f"""
                        <div class='score-box {bg}'>
                            {result['score']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence badge (NEW)
                    conf = result.get("confidence", "SKIP")
                    conf_label = result.get("confidence_label", "")
                    conf_colors = {"HIGH": "#00e676", "MEDIUM": "#ffd600", "LOW": "#ff9100", "SKIP": "#5a6a7e"}
                    actionable = result.get("is_actionable", False)
                    st.markdown(f"""
                        <div style="text-align:center; padding:6px 12px; border-radius:10px; margin:6px 0;
                             background:rgba({('0,230,118' if conf == 'HIGH' else ('255,214,0' if conf == 'MEDIUM' else ('255,145,0' if conf == 'LOW' else '90,106,126')))}, 0.1);
                             border:1px solid {conf_colors.get(conf, '#5a6a7e')}40;
                             font-family:'JetBrains Mono',monospace; font-size:0.72em; color:{conf_colors.get(conf, '#5a6a7e')};">
                            {conf_label}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("RSI", f"{result['rsi']:.1f}")
                    st.metric("MFI", f"{result['mfi']:.1f}")
                    st.metric("ADX", f"{result['adx']:.1f}")
                    
                    if enable_ml:
                        st.metric("🤖 ML Prob", f"{result['ml_prob']*100:.1f}%")
                        # Show model stats
                        if ml_model:
                            samples = ml_model.get("samples", "?")
                            test_acc = ml_model.get("test_acc", 0)
                            precision = ml_model.get("precision", 0)
                            st.markdown(f"""
                                <div style="font-family:'JetBrains Mono',monospace; font-size:0.62em; color:#3a5068; line-height:1.6;">
                                    Pooled: {samples} samples<br>
                                    Test Acc: {test_acc*100:.1f}%<br>
                                    Precision: {precision*100:.1f}%
                                </div>
                            """, unsafe_allow_html=True)
                    
                    if enable_mtf:
                        st.metric("🎯 MTF", f"{result['mtf_confluence']:.0f}%")
                
                with col_b:
                    # Mean reversion alerts (NEW)
                    mean_rev = result.get("mean_reversion")
                    if mean_rev and mean_rev.get("alerts"):
                        st.markdown("""
                        <div style="font-family:'Outfit',sans-serif; font-size:1.05em; font-weight:700; margin-bottom:6px;">
                            🔄 Mean Reversion Alerts
                        </div>
                        """, unsafe_allow_html=True)
                        for alert_title, alert_desc in mean_rev["alerts"]:
                            rev_color = "#ff6b6b" if mean_rev["reversal_score"] < 0 else "#69db7c"
                            st.markdown(f"""
                                <div style="font-family:'JetBrains Mono',monospace; font-size:0.78em; padding:6px 10px;
                                     margin:3px 0; border-radius:8px; background:rgba(255,255,255,0.02);
                                     border-left:3px solid {rev_color};">
                                    <b>{alert_title}</b><br>
                                    <span style="color:#7a8a9e;">{alert_desc}</span>
                                </div>
                            """, unsafe_allow_html=True)
                        st.markdown("")
                    
                    # Candlestick patterns section
                    if result.get("candle_patterns"):
                        st.markdown("### 🕯️ Candlestick Patterns")
                        for pname, pdesc, ppts in result["candle_patterns"]:
                            card_class = "candle-card-bull" if ppts > 0 else ("candle-card-bear" if ppts < 0 else "candle-card")
                            sign = "+" if ppts >= 0 else ""
                            st.markdown(f"""
                                <div class="candle-card {card_class}">
                                    <b>{pname}</b> — {pdesc} <span style="float:right; font-weight:700;">{sign}{ppts} pts</span>
                                </div>
                            """, unsafe_allow_html=True)
                        st.markdown("")
                    
                    st.markdown("### 📝 Analysis Factors")
                    # Separate candle pattern reasons from other reasons
                    non_candle_reasons = [r for r in result["reasons"] if not any(
                        cp[0] in r for cp in result.get("candle_patterns", [])
                    )]
                    for reason in non_candle_reasons[:12]:
                        st.markdown(f"• {reason}")
                    
                    if len(non_candle_reasons) > 12:
                        with st.expander("Show more..."):
                            for reason in non_candle_reasons[12:]:
                                st.markdown(f"• {reason}")
                
                with col_c:
                    st.markdown(f"""
                        <div class='trade-plan'>
                            <h3 style='color:#00e5ff'>📌 Trade Plan</h3>
                            <hr>
                            <p><b>Entry:</b> ₺{result['price']:.2f}</p>
                            <p><b>Stop:</b> ₺{result['stop']:.2f}</p>
                            <p><b>Target:</b> ₺{result['target']:.2f}</p>
                            <p><b>R/R:</b> {result['rr']:.2f}</p>
                            <p><b>TP %:</b> {result['tp_pct']:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Chart with VWAP + BB + MACD + Stochastic + S/R + Pivots
                st.markdown("### 📊 Advanced Chart")
                
                fig = make_subplots(
                    rows=5, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.40, 0.15, 0.15, 0.15, 0.15],
                    subplot_titles=('Price / VWAP / BB / S&R', 'RSI & MFI', 'MACD', 'Stochastic', 'Volume')
                )
                
                tail = df.tail(150)
                
                # Row 1: Candlesticks
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
                
                # Bollinger Bands (shaded)
                if "BB_UPPER" in tail.columns:
                    fig.add_trace(go.Scatter(
                        x=tail.index, y=tail["BB_UPPER"], name="BB Upper",
                        line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot")
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=tail.index, y=tail["BB_LOWER"], name="BB Lower",
                        line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                        fill='tonexty', fillcolor='rgba(173,216,230,0.06)'
                    ), row=1, col=1)
                
                # EMAs
                fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA20"], name="EMA20", line=dict(color="orange", width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA50"], name="EMA50", line=dict(color="blue", width=1)), row=1, col=1)
                
                # Support / Resistance levels
                supports, resistances = _support_resistance_levels(df["High"], df["Low"], df["Close"])
                for s_lvl in supports:
                    fig.add_hline(y=s_lvl, line_dash="dash", line_color="rgba(0,200,83,0.5)", 
                                  annotation_text=f"S ₺{s_lvl:.2f}", annotation_position="bottom left",
                                  annotation_font_size=9, annotation_font_color="rgba(0,200,83,0.7)", row=1, col=1)
                for r_lvl in resistances:
                    fig.add_hline(y=r_lvl, line_dash="dash", line_color="rgba(213,0,0,0.5)",
                                  annotation_text=f"R ₺{r_lvl:.2f}", annotation_position="top left",
                                  annotation_font_size=9, annotation_font_color="rgba(213,0,0,0.7)", row=1, col=1)
                
                # Pivot Points (from daily high/low/close)
                daily_high = float(df["High"].tail(96).max())  # ~1 day of 15m bars
                daily_low = float(df["Low"].tail(96).min())
                daily_close = float(df["Close"].iloc[-1])
                pivots = _pivot_points(daily_high, daily_low, daily_close)
                
                pivot_colors = {"PP": "#ffffff", "R1": "#ff6b6b", "R2": "#ff3333", "R3": "#cc0000",
                                "S1": "#69db7c", "S2": "#40c057", "S3": "#2b8a3e"}
                for pname, plevel in pivots.items():
                    if tail["Low"].min() * 0.98 < plevel < tail["High"].max() * 1.02:
                        fig.add_hline(y=plevel, line_dash="dot", line_color=pivot_colors.get(pname, "gray"),
                                      line_width=1, annotation_text=pname, annotation_position="right",
                                      annotation_font_size=8, annotation_font_color=pivot_colors.get(pname, "gray"),
                                      row=1, col=1)
                
                # Stop/Target lines
                fig.add_hline(y=result["stop"], line_dash="dash", line_color="red", 
                              annotation_text=f"Stop ₺{result['stop']:.2f}", annotation_font_size=10, row=1, col=1)
                fig.add_hline(y=result["target"], line_dash="solid", line_color="green",
                              annotation_text=f"TP ₺{result['target']:.2f}", annotation_font_size=10, row=1, col=1)
                
                # Live price line
                if rt_quote:
                    fig.add_hline(
                        y=rt_quote["current"], line_dash="dot", line_color="cyan", 
                        annotation_text=f"⚡ Live: ₺{rt_quote['current']:.2f}", 
                        annotation_position="top right", annotation_font_color="cyan",
                        row=1, col=1
                    )
                
                # Row 2: RSI & MFI
                fig.add_trace(go.Scatter(x=tail.index, y=tail["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
                if "MFI" in tail.columns:
                    fig.add_trace(go.Scatter(x=tail.index, y=tail["MFI"], name="MFI", line=dict(color="gold")), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Row 3: MACD
                if "MACD" in tail.columns:
                    fig.add_trace(go.Scatter(x=tail.index, y=tail["MACD"], name="MACD", line=dict(color="#00e5ff", width=1.5)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=tail.index, y=tail["MACD_SIGNAL"], name="Signal", line=dict(color="#ff6d00", width=1.5)), row=3, col=1)
                    macd_colors = ['#00c853' if v >= 0 else '#d50000' for v in tail["MACD_HIST"]]
                    fig.add_trace(go.Bar(x=tail.index, y=tail["MACD_HIST"], name="MACD Hist", marker_color=macd_colors), row=3, col=1)
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
                
                # Row 4: Stochastic
                fig.add_trace(go.Scatter(x=tail.index, y=tail["STOCH_K"], name="%K", line=dict(color="#00e5ff")), row=4, col=1)
                fig.add_trace(go.Scatter(x=tail.index, y=tail["STOCH_D"], name="%D", line=dict(color="#ff6d00")), row=4, col=1)
                fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
                
                # Row 5: Volume
                colors = ['green' if tail["Close"].iloc[i] >= tail["Open"].iloc[i] else 'red' for i in range(len(tail))]
                fig.add_trace(go.Bar(x=tail.index, y=tail["Volume"], name="Volume", marker_color=colors), row=5, col=1)
                if "VOL_MA20" in tail.columns:
                    fig.add_trace(go.Scatter(x=tail.index, y=tail["VOL_MA20"], name="Vol MA20", line=dict(color="yellow", width=1)), row=5, col=1)
                
                fig.update_layout(height=1100, template="plotly_dark", showlegend=True, 
                                  xaxis_rangeslider_visible=False, hovermode='x unified',
                                  margin=dict(t=30, b=30),
                                  paper_bgcolor='rgba(6,8,13,0)',
                                  plot_bgcolor='rgba(10,15,25,0.8)',
                                  font=dict(family='JetBrains Mono, monospace', size=10, color='#7a8a9e'),
                                  legend=dict(bgcolor='rgba(10,15,25,0.7)', bordercolor='rgba(0,229,255,0.1)', borderwidth=1))
                fig.update_xaxes(gridcolor='rgba(0,229,255,0.04)', zerolinecolor='rgba(0,229,255,0.06)')
                fig.update_yaxes(gridcolor='rgba(0,229,255,0.04)', zerolinecolor='rgba(0,229,255,0.06)')
                st.plotly_chart(fig, use_container_width=True)
                
                # === SECTOR RELATIVE STRENGTH ===
                st.markdown("### 💪 Sector Relative Strength")
                with st.spinner("Calculating sector strength..."):
                    sector_rs_data = calculate_sector_relative_strength(symbol_input)
                
                if sector_rs_data:
                    rs_cols = st.columns([1, 1, 2])
                    
                    with rs_cols[0]:
                        rs_color = "#00c853" if sector_rs_data["relative_strength"] > 0 else "#d50000"
                        rs_sign = "+" if sector_rs_data["relative_strength"] > 0 else ""
                        st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size:0.8em; color:#aaa;">📊 {sector_rs_data['sector']}</div>
                                <div style="font-size:1.8em; font-weight:800; color:{rs_color};">
                                    {rs_sign}{sector_rs_data['relative_strength']:.2f}%
                                </div>
                                <div style="font-size:0.8em; color:#aaa;">vs sector average</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with rs_cols[1]:
                        st.metric(f"{symbol_input} Return", f"{sector_rs_data['stock_return']:.2f}%")
                        st.metric(f"Sector Avg", f"{sector_rs_data['sector_avg']:.2f}%")
                    
                    with rs_cols[2]:
                        st.markdown("**Peer Comparison (5-day)**")
                        for peer in sector_rs_data["peers"]:
                            p_color = "#00c853" if peer["return"] > 0 else "#d50000"
                            p_sign = "+" if peer["return"] > 0 else ""
                            st.markdown(f"• **{peer['symbol']}**: <span style='color:{p_color}'>{p_sign}{peer['return']:.2f}%</span>", unsafe_allow_html=True)
                else:
                    st.caption("Sector data not available for this stock")
                
                # === EARNINGS / BILANÇO CALENDAR ===
                st.markdown("""
                <div style="margin-top:20px;">
                    <span style="font-family:'Outfit',sans-serif; font-size:1.3em; font-weight:700;">📅 Bilanço Takvimi</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.65em; color:#3a5068; margin-left:10px;">KAP · MULTI-QUARTER</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Show target stock earnings (next relevant quarter)
                target_earnings = get_earnings_info(symbol_input)
                if target_earnings:
                    e_color = "#00e5ff" if target_earnings["status"] == "upcoming" else (
                        "#ffd600" if target_earnings["status"] == "today" else "#5a6a7e"
                    )
                    e_icon = "⏳" if target_earnings["status"] == "upcoming" else (
                        "⚡" if target_earnings["status"] == "today" else "✅"
                    )
                    deadline_tag = ' <span style="color:#ff6b6b; font-size:0.7em;">[SON TARİH]</span>' if target_earnings.get("is_deadline") else ""
                    st.markdown(f"""
                        <div class="glass-card" style="border-left:3px solid {e_color}; margin:10px 0;">
                            <div style="font-family:'Outfit',sans-serif; font-size:1.1em; font-weight:700;">
                                {e_icon} {symbol_input} — {target_earnings['label']}{deadline_tag}
                            </div>
                            <div style="font-family:'JetBrains Mono',monospace; font-size:0.75em; color:#5a6a7e; margin-top:4px;">
                                {target_earnings['period']} · {target_earnings['quarter']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Show sector peers
                earnings_list, earn_sector = get_sector_earnings_calendar(symbol_input)
                
                if earnings_list:
                    upcoming = [e for e in earnings_list if e["status"] in ["upcoming", "today"]]
                    reported = [e for e in earnings_list if e["status"] == "reported"]
                    
                    earn_cols = st.columns(2)
                    
                    with earn_cols[0]:
                        st.markdown(f"""
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.8em; color:#00e5ff; margin-bottom:8px;">
                            ⏳ UPCOMING — {earn_sector}
                        </div>
                        """, unsafe_allow_html=True)
                        if upcoming:
                            for e in upcoming[:10]:
                                icon = "🔴" if e.get("is_target") else "○"
                                bold = "font-weight:800; color:#ffffff;" if e.get("is_target") else "color:#c0cad8;"
                                dl_tag = " ⚠️" if e.get("is_deadline") else ""
                                st.markdown(f"""
                                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.78em; padding:3px 0; {bold}">
                                        {icon} {e['symbol']} — {e['date_str']}{dl_tag}
                                        <span style="color:#ffd600;">({e['days_until']}d)</span>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.caption("All sector peers have reported ✅")
                    
                    with earn_cols[1]:
                        st.markdown(f"""
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.8em; color:#5a6a7e; margin-bottom:8px;">
                            ✅ REPORTED — {earn_sector}
                        </div>
                        """, unsafe_allow_html=True)
                        if reported:
                            for e in list(reversed(reported))[:10]:
                                icon = "🔴" if e.get("is_target") else "○"
                                st.markdown(f"""
                                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.78em; padding:3px 0; color:#5a6a7e;">
                                        {icon} {e['symbol']} — {e['date_str']}
                                    </div>
                                """, unsafe_allow_html=True)
                
                # ── Full Calendar Expander ──
                with st.expander("📋 Full Q4 2025 Bilanço Calendar (All Stocks)", expanded=False):
                    full_cal = get_full_earnings_calendar("Q4_2025")
                    if full_cal:
                        now_dt = datetime.now()
                        
                        # Group by week
                        upcoming_full = [e for e in full_cal if e["status"] in ("upcoming", "today")]
                        reported_full = [e for e in full_cal if e["status"] == "reported"]
                        
                        cal_c1, cal_c2 = st.columns(2)
                        
                        with cal_c1:
                            st.markdown(f"""
                            <div style="font-family:'JetBrains Mono',monospace; font-size:0.75em; color:#00e5ff; margin-bottom:6px;">
                                ⏳ UPCOMING ({len(upcoming_full)} stocks)
                            </div>
                            """, unsafe_allow_html=True)
                            # Group by date
                            from itertools import groupby
                            for date_key, group in groupby(upcoming_full, key=lambda x: x["date_str"]):
                                stocks = list(group)
                                names = ", ".join([s["symbol"] for s in stocks])
                                days = stocks[0]["days_until"]
                                st.markdown(f"""
                                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72em; padding:2px 0; color:#c0cad8;">
                                        <span style="color:#ffd600; min-width:40px; display:inline-block;">{days}d</span>
                                        <span style="color:#5a6a7e;">{date_key}</span> → 
                                        <b>{names}</b>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        with cal_c2:
                            st.markdown(f"""
                            <div style="font-family:'JetBrains Mono',monospace; font-size:0.75em; color:#5a6a7e; margin-bottom:6px;">
                                ✅ REPORTED ({len(reported_full)} stocks)
                            </div>
                            """, unsafe_allow_html=True)
                            for date_key, group in groupby(reversed(reported_full), key=lambda x: x["date_str"]):
                                stocks = list(group)
                                names = ", ".join([s["symbol"] for s in stocks])
                                st.markdown(f"""
                                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72em; padding:2px 0; color:#5a6a7e;">
                                        <span style="color:#3a5068;">{date_key}</span> → {names}
                                    </div>
                                """, unsafe_allow_html=True)
                
                # ── Q1 2026 Status ──
                q1_data = BIST_EARNINGS.get("Q1_2026", {})
                q1_dates = q1_data.get("dates", {})
                q1_deadline = q1_data.get("deadline_consolidated", "TBD")
                
                st.markdown(f"""
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.7em; color:#3a5068; margin-top:12px;
                     padding:10px 14px; border:1px solid rgba(0,229,255,0.06); border-radius:10px; 
                     background:rgba(10,15,25,0.5);">
                    <div style="margin-bottom:4px;">
                        <span style="color:#00e5ff;">Q4 2025</span> Son Tarih: 
                        Konsol. olmayan → <b>2 Mar</b> | Konsolide → <b>11 Mar 2026</b>
                    </div>
                    <div>
                        <span style="color:#ffd600;">Q1 2026</span> Son Tarih: 
                        Konsol. olmayan → <b>~15 May</b> | Konsolide → <b>~30 May 2026</b>
                        {'· <span style="color:#00e676;">' + str(len(q1_dates)) + ' tarih açıklandı</span>' if q1_dates else '· <span style="color:#5a6a7e;">Tarihler henüz açıklanmadı (KAP\'tan Nisan\'da bekleniyor)</span>'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

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
                    # Apply confidence filter — only return actionable signals
                    if result.get("is_actionable", True):
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
            
            st.caption("⚡ Real-time prices via Finnhub")
            
            # Display results with signal grades
            for idx, res in enumerate(top_results):
                sig = res.get("signal", {"grade": "?", "label": "", "color": "#888", "emoji": ""})
                card_extra = "scanner-card-hot" if sig["grade"] in ["A+", "A"] else ""
                
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 2])
                    
                    with col1:
                        # Signal grade badge
                        st.markdown(f"""
                            <div style="text-align:center; margin-bottom:8px;">
                                <div style="background:{sig['color']}18; color:{sig['color']}; padding:8px 16px; border-radius:14px; font-weight:900; font-size:1.6em; border:2px solid {sig['color']}44; display:inline-block;">
                                    {sig['emoji']} {sig['grade']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        bg = "score-high" if res["score"] >= 80 else ("score-mid" if res["score"] >= 70 else "score-low")
                        st.markdown(f"""
                            <div class='score-box {bg}' style='font-size:1.5em'>
                                #{idx+1} — {res['score']}
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
                        
                        # Confidence badge
                        conf = res.get('confidence', 'SKIP')
                        conf_colors = {"HIGH": "#00e676", "MEDIUM": "#ffd600", "LOW": "#ff9100", "SKIP": "#5a6a7e"}
                        st.markdown(f"""
                            <div style="font-family:'JetBrains Mono',monospace; font-size:0.7em; color:{conf_colors.get(conf,'#5a6a7e')}; font-weight:700;">
                                {res.get('confidence_label', '')}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Show candle patterns first if any
                        candle_pats = res.get("candle_patterns", [])
                        if candle_pats:
                            st.markdown("**🕯️ Patterns:**")
                            for pname, pdesc, ppts in candle_pats[:3]:
                                st.caption(f"{pname} {pdesc}")
                        
                        st.markdown("**🎯 Key Factors:**")
                        non_candle = [r for r in res["reasons"] if not any(
                            cp[0] in r for cp in candle_pats
                        )]
                        for r in non_candle[:5]:
                            st.caption(f"• {r}")
                    
                    with col3:
                        st.markdown(f"**💹 Trade Setup — {sig['label']}**")
                        st.write(f"Entry: **₺{res['price']:.2f}**")
                        st.write(f"Stop: **₺{res['stop']:.2f}**")
                        st.write(f"Target: **₺{res['target']:.2f}** (TP: {res['tp_pct']:.1f}%)")
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

            # ── Equity Curve (Enhanced with time axis + drawdown) ──
            st.markdown("### 📊 Equity Curve")

            closed_df_sorted = closed_df.sort_values('Sell Date')
            closed_df_sorted['Cumulative P&L'] = closed_df_sorted['P&L (TL)'].cumsum()
            closed_df_sorted['Trade #'] = range(1, len(closed_df_sorted) + 1)
            
            # Calculate drawdown
            cumulative = closed_df_sorted['Cumulative P&L']
            running_max = cumulative.cummax()
            drawdown = cumulative - running_max

            fig_equity = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.08, row_heights=[0.7, 0.3],
                subplot_titles=('Cumulative P&L', 'Drawdown')
            )
            
            # Color markers by profit/loss
            colors_eq = ['#00c853' if p >= 0 else '#d50000' for p in closed_df_sorted['P&L (TL)']]

            # P&L Line
            fig_equity.add_trace(go.Scatter(
                x=closed_df_sorted['Sell Date'],
                y=closed_df_sorted['Cumulative P&L'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#00e5ff', width=2.5),
                marker=dict(size=8, color=colors_eq, line=dict(width=1, color='white')),
                text=closed_df_sorted['Symbol'],
                hovertemplate='<b>%{text}</b><br>%{x}<br>Cumulative: ₺%{y:,.2f}<extra></extra>'
            ), row=1, col=1)
            
            fig_equity.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even", row=1, col=1)

            # Fill area under curve
            fig_equity.add_trace(go.Scatter(
                x=closed_df_sorted['Sell Date'],
                y=closed_df_sorted['Cumulative P&L'],
                fill='tozeroy',
                fillcolor='rgba(0, 229, 255, 0.08)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            
            # Drawdown area
            fig_equity.add_trace(go.Scatter(
                x=closed_df_sorted['Sell Date'],
                y=drawdown,
                fill='tozeroy',
                fillcolor='rgba(213, 0, 0, 0.15)',
                line=dict(color='#d50000', width=1.5),
                name='Drawdown',
                hovertemplate='Drawdown: ₺%{y:,.2f}<extra></extra>'
            ), row=2, col=1)
            fig_equity.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

            fig_equity.update_layout(
                height=550, template="plotly_dark",
                showlegend=True, hovermode='x unified',
                margin=dict(t=30, b=40)
            )
            fig_equity.update_yaxes(title_text="Cumulative P&L (₺)", row=1, col=1)
            fig_equity.update_yaxes(title_text="Drawdown (₺)", row=2, col=1)
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Max drawdown stat
            max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0
            st.markdown(f"""
                <div class="pnl-card pnl-card-info" style="text-align:center;">
                    📉 <b>Max Drawdown:</b> ₺{max_dd:,.2f} | 
                    🏔️ <b>Peak P&L:</b> ₺{float(running_max.max()):,.2f} |
                    📊 <b>Recovery:</b> {'✅ Recovered' if float(cumulative.iloc[-1]) >= float(running_max.max()) * 0.95 else '⏳ In drawdown'}
                </div>
            """, unsafe_allow_html=True)

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

st.markdown("""
<div style="text-align:center; margin-top:40px; padding:20px 0;">
    <div style="width:60px; height:1px; background:linear-gradient(90deg, transparent, #00e5ff20, transparent); margin:0 auto 12px;"></div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.6em; color:#2a3a4e; letter-spacing:0.2em; text-transform:uppercase;">
        NEXUS TRADE — ML · MACD · BB · STOCH · S/R · PIVOTS · SECTOR RS · EARNINGS
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.55em; color:#1a2a3e; letter-spacing:0.15em; margin-top:4px;">
        POWERED BY YFINANCE + FINNHUB REAL-TIME · XGBOOST POOLED ML · BIST OPTIMIZED
    </div>
</div>
""", unsafe_allow_html=True)
