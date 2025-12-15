import os
import sys
import json
import requests
import datetime
import warnings
import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import io, sys
from contextlib import redirect_stdout, redirect_stderr

import matplotlib
matplotlib.use("Agg")   # use non-interactive backend — prevents Tk/Tcl GUI issues
import matplotlib.pyplot as plt


def capture_output(func, *args, **kwargs):
    """Capture all printed terminal output from a function."""
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            func(*args, **kwargs)
    except Exception as e:
        buffer.write(f"\n[ERROR] {str(e)}\n")
    return buffer.getvalue()



try:
    from config_db import log_to_db
except Exception:
    log_to_db=None

warnings.filterwarnings("ignore")
logging.getLogger("autogen").setLevel(logging.ERROR)
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)

try:
    import autogen
except Exception:
    autogen = None

# ----------------------
# PARSE AI ANALYSIS FOR CLEAN FORMATTING
# ----------------------
def parse_ai_analysis(ai_text: str) -> dict:
    """Parse AI response into structured sections"""
    sections = {
        'rating': '',
        'highlights': [],
        'financial_strength': '',
        'growth_prospects': '',
        'valuation': '',
        'risks': [],
        'technical_outlook': '',
        'recommendations': {}
    }
    
    lines = ai_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove markdown formatting
        line = line.replace('**', '').replace('*', '')
        
        upper_line = line.upper()
        
        if 'INVESTMENT RATING' in upper_line:
            current_section = 'rating'
            continue
        elif 'KEY HIGHLIGHTS' in upper_line:
            current_section = 'highlights'
            continue
        elif 'FINANCIAL STRENGTH' in upper_line:
            current_section = 'financial_strength'
            continue
        elif 'GROWTH PROSPECTS' in upper_line:
            current_section = 'growth_prospects'
            continue
        elif 'VALUATION ASSESSMENT' in upper_line or 'VALUATION' in upper_line:
            current_section = 'valuation'
            continue
        elif 'KEY RISKS' in upper_line or 'RISK' in upper_line:
            current_section = 'risks'
            continue
        elif 'TECHNICAL OUTLOOK' in upper_line:
            current_section = 'technical_outlook'
            continue
        elif 'RECOMMENDATION SUMMARY' in upper_line:
            current_section = 'recommendations'
            continue
        
        # Process content
        if current_section == 'rating':
            sections['rating'] = line
        elif current_section == 'highlights':
            if line.startswith('-') or line.startswith('•'):
                sections['highlights'].append(line[1:].strip())
        elif current_section == 'financial_strength':
            sections['financial_strength'] += line + ' '
        elif current_section == 'growth_prospects':
            sections['growth_prospects'] += line + ' '
        elif current_section == 'valuation':
            sections['valuation'] += line + ' '
        elif current_section == 'risks':
            if line.startswith('-') or line.startswith('•'):
                sections['risks'].append(line[1:].strip())
        elif current_section == 'technical_outlook':
            sections['technical_outlook'] += line + ' '
        elif current_section == 'recommendations':
            if 'long-term' in line.lower():
                sections['recommendations']['long_term'] = line.split(':', 1)[-1].strip()
            elif 'short-term' in line.lower():
                sections['recommendations']['short_term'] = line.split(':', 1)[-1].strip()
    
    return sections

# ----------------------
# TRADING JOURNAL ANALYSIS
# ----------------------
def analyze_trading_journal(csv_path: str, symbol_filter: Optional[str] = None):
    """Analyze personal trading journal for the given symbol"""
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as e:
        print(f"⚠️  Warning: could not read trading journal: {e}")
        return None

    if symbol_filter:
        if "symbol" in df.columns:
            df = df[df["symbol"].str.upper() == symbol_filter.upper()]

    if df.empty:
        return None

    # Calculate P&L if not present
    if "pnl" not in df.columns and {"entry", "exit", "qty"}.issubset(df.columns):
        df["pnl"] = (df["exit"] - df["entry"]) * df["qty"]

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    total_trades = len(df)
    win_rate = len(wins) / total_trades if total_trades > 0 else None
    avg_pnl = df["pnl"].mean()
    net = df["pnl"].sum()
    avg_win = wins["pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0.0
    expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if win_rate is not None else None

    return {
        "total_trades": int(total_trades),
        "win_rate_pct": float(win_rate * 100) if win_rate is not None else None,
        "avg_pnl": float(avg_pnl) if pd.notna(avg_pnl) else None,
        "net_pnl": float(net),
        "avg_win": float(avg_win) if pd.notna(avg_win) else None,
        "avg_loss": float(avg_loss) if pd.notna(avg_loss) else None,
        "expectancy": float(expectancy) if expectancy is not None else None,
    }

# ----------------------
# CONFIG / API KEY LOAD
# ----------------------
CONFIG_PATH = "config_api_keys"

try:
    if os.path.exists(CONFIG_PATH):
        path = CONFIG_PATH
    elif os.path.exists(CONFIG_PATH + ".json"):
        path = CONFIG_PATH + ".json"
    else:
        raise FileNotFoundError(f"{CONFIG_PATH} not found")
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
except Exception as e:
    raise RuntimeError(f"ERROR: Unable to read {CONFIG_PATH}: {e}")

MISTRAL_API_KEY = cfg.get("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

if not MISTRAL_API_KEY or str(MISTRAL_API_KEY).strip() == "":
    raise ValueError("ERROR: MISTRAL_API_KEY missing in config_api_keys")

# ----------------------
# SETUP AUTOGEN LLM CONFIG
# ----------------------
def setup_llm_config_mistral(api_key: str, model: str = "mistral-small-latest") -> dict:
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is required.")
    return {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
                "base_url": "https://api.mistral.ai/v1",
                "api_type": "openai"
            }
        ],
        "timeout": 120,
        "temperature": 0.2,
        "max_tokens": 4000
    }

# ----------------------
# UTILITIES
# ----------------------
def rsi(series: pd.Series, window: int = 14) -> float:
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(window - 1), adjust=False).mean()
    ma_down = down.ewm(com=(window - 1), adjust=False).mean()
    rs = ma_up / ma_down
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])

def safe_get(d: dict, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

# ----------------------
# DATA FETCHING
# ----------------------
def fetch_ticker_data(symbol: str, period: str = "1y"):
    ticker = yf.Ticker(symbol)
    try:
        hist = ticker.history(period=period, auto_adjust=False)
    except Exception:
        hist = pd.DataFrame()

    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    try:
        fin = ticker.financials
        bal = ticker.balance_sheet
        cash = ticker.cashflow
    except Exception:
        fin, bal, cash = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        news = ticker.news if hasattr(ticker, "news") else []
    except Exception:
        news = []

    return dict(hist=hist, info=info, financials=fin, balance=bal, cash=cash, news=news)

# ----------------------
# METRICS COMPUTATION
# ----------------------
def compute_metrics(data: dict) -> dict:
    hist = data.get("hist", pd.DataFrame())
    info = data.get("info", {})
    metrics = {}

    if isinstance(hist, pd.DataFrame) and not hist.empty:
        last = hist.iloc[-1]
        metrics["current_price"] = float(last.get("Close", float("nan")))
        metrics["open"] = float(last.get("Open", float("nan")))
        metrics["high"] = float(last.get("High", float("nan")))
        metrics["low"] = float(last.get("Low", float("nan")))
        
        prev = hist.iloc[-2] if len(hist) >= 2 else last
        metrics["prev_close"] = float(prev.get("Close", metrics["current_price"]))
        metrics["daily_change"] = ((metrics["current_price"] - metrics["prev_close"]) / metrics["prev_close"] * 100) if metrics["prev_close"] else 0
    else:
        metrics["current_price"] = None
        metrics["daily_change"] = 0

    # Fundamentals
    metrics["market_cap"] = safe_get(info, "marketCap")
    metrics["trailingPE"] = safe_get(info, "trailingPE")
    metrics["forwardPE"] = safe_get(info, "forwardPE")
    metrics["trailingEps"] = safe_get(info, "trailingEps")
    metrics["returnOnEquity"] = safe_get(info, "returnOnEquity")
    metrics["grossMargins"] = safe_get(info, "grossMargins")
    metrics["currentRatio"] = safe_get(info, "currentRatio")
    metrics["quickRatio"] = safe_get(info, "quickRatio")
    metrics["debtToEquity"] = safe_get(info, "debtToEquity")
    metrics["beta"] = safe_get(info, "beta")
    metrics["dividend_yield"] = safe_get(info, "dividendYield")
    metrics["payout_ratio"] = safe_get(info, "payoutRatio")
    
    # Company info - need to get symbol from somewhere, use info or create parameter
    metrics["company_name"] = safe_get(info, "longName", default="Unknown")
    metrics["sector"] = safe_get(info, "sector", default="N/A")
    metrics["industry"] = safe_get(info, "industry", default="N/A")

    # Balance sheet
    bal = data.get("balance", pd.DataFrame())
    if isinstance(bal, pd.DataFrame) and not bal.empty:
        try:
            latest = bal.columns[0]
            metrics["total_assets"] = float(bal.loc["Total Assets", latest]) if "Total Assets" in bal.index else None
            metrics["total_liab"] = float(bal.loc["Total Liab", latest]) if "Total Liab" in bal.index else None
        except Exception:
            metrics["total_assets"] = None
            metrics["total_liab"] = None

    # Technicals
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        close = hist["Close"]
        metrics["sma50"] = float(close.rolling(window=50).mean().iloc[-1]) if len(close) >= 50 else None
        metrics["sma200"] = float(close.rolling(window=200).mean().iloc[-1]) if len(close) >= 200 else None
        try:
            metrics["rsi14"] = rsi(close)
        except Exception:
            metrics["rsi14"] = None
        
        # Volume analysis
        if "Volume" in hist.columns:
            metrics["avg_volume"] = float(hist["Volume"].tail(20).mean())
            metrics["current_volume"] = float(hist["Volume"].iloc[-1])

    # Headlines
    news_list = data.get("news", []) or []
    headlines = []
    for n in news_list[:6]:
        if isinstance(n, dict) and "title" in n:
            headlines.append(n["title"])
        elif isinstance(n, str):
            headlines.append(n)
    metrics["headlines"] = headlines

    return metrics

# ----------------------
# CHART GENERATION
# ----------------------
def generate_financial_charts(symbol: str, data: dict, metrics: dict, output_dir: str = "charts"):
    os.makedirs(output_dir, exist_ok=True)
    hist = data.get("hist", pd.DataFrame())
    fin = data.get("financials", pd.DataFrame())
    bal = data.get("balance", pd.DataFrame())

    # 1) Financial Health Score
    score = 0
    if metrics.get("currentRatio") and metrics["currentRatio"] > 1: score += 25
    if metrics.get("grossMargins") and metrics["grossMargins"] > 0.4: score += 25
    if metrics.get("returnOnEquity") and metrics["returnOnEquity"] > 0.15: score += 25
    if metrics.get("debtToEquity") is not None and metrics["debtToEquity"] < 150: score += 25

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ['red' if score < 40 else 'orange' if score < 70 else 'green']
    ax.barh(["Financial Health"], [score], color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (0-100)")
    ax.set_title(f"{symbol} Financial Health Score: {score}/100")
    for i, v in enumerate([score]):
        ax.text(v + 2, i, f"{v}", va='center', fontweight='bold')
    plt.tight_layout()
    fh_path = os.path.join(output_dir, f"{symbol}_financial_health.png")
    plt.savefig(fh_path)
    plt.close()

    # 2) Price chart with SMA
    price_path = None
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        plt.figure(figsize=(12, 5))
        plt.plot(hist.index, hist["Close"], label="Close", linewidth=2)
        if len(hist["Close"]) >= 50:
            plt.plot(hist.index, hist["Close"].rolling(50).mean(), label="SMA50", linestyle='--')
        if len(hist["Close"]) >= 200:
            plt.plot(hist.index, hist["Close"].rolling(200).mean(), label="SMA200", linestyle='--')
        plt.title(f"{symbol} Price Chart with Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        price_path = os.path.join(output_dir, f"{symbol}_price_sma.png")
        plt.savefig(price_path)
        plt.close()

    # 3) Revenue & Net Income trend
    rev_path = None
    try:
        if isinstance(fin, pd.DataFrame) and not fin.empty and "Total Revenue" in fin.index and "Net Income" in fin.index:
            rev = fin.loc["Total Revenue"].head(8)[::-1]
            profit = fin.loc["Net Income"].head(8)[::-1]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            x = range(len(rev))
            ax.plot(x, rev.values / 1e9, marker='o', label="Revenue", linewidth=2)
            ax.plot(x, profit.values / 1e9, marker='s', label="Net Income", linewidth=2)
            ax.set_xticks(x)
            ax.set_xticklabels([str(d)[:10] for d in rev.index])
            ax.set_title(f"{symbol} Revenue & Net Income Trend (Billions)")
            ax.set_ylabel("Amount (Billions $)")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            rev_path = os.path.join(output_dir, f"{symbol}_revenue_trend.png")
            plt.savefig(rev_path)
            plt.close()
    except Exception:
        pass

    # 4) Valuation Ratios Comparison
    radar_path = None
    try:
        # Normalize metrics for radar chart
        pe_norm = min(metrics.get("trailingPE", 20) / 30, 1) if metrics.get("trailingPE") else 0.5
        roe_norm = min(metrics.get("returnOnEquity", 0.1) / 0.3, 1) if metrics.get("returnOnEquity") else 0
        margin_norm = min(metrics.get("grossMargins", 0.3) / 0.5, 1) if metrics.get("grossMargins") else 0
        cr_norm = min(metrics.get("currentRatio", 1) / 2, 1) if metrics.get("currentRatio") else 0
        
        company = [pe_norm, roe_norm, margin_norm, cr_norm]
        labels = ["P/E Ratio", "ROE", "Gross Margin", "Liquidity"]
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        company += company[:1]
        angles += angles[:1]
        
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, company, 'o-', linewidth=2, label="Company", color='blue')
        ax.fill(angles, company, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title(f"{symbol} Financial Metrics Radar", pad=20)
        ax.legend(loc="upper right")
        ax.grid(True)
        plt.tight_layout()
        radar_path = os.path.join(output_dir, f"{symbol}_metrics_radar.png")
        plt.savefig(radar_path)
        plt.close()
    except Exception:
        pass

    # 5) Balance Sheet Breakdown
    bs_path = None
    try:
        if isinstance(bal, pd.DataFrame) and not bal.empty:
            latest = bal.columns[0]
            assets = bal.loc["Total Assets", latest] if "Total Assets" in bal.index else None
            liab = None
            for key in ["Total Liab", "Total Liabilities Net Minority Interest"]:
                if key in bal.index:
                    liab = bal.loc[key, latest]
                    break
            if assets is not None and liab is not None:
                equity = float(assets) - float(liab)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(["Assets", "Liabilities", "Equity"], 
                              [float(assets)/1e9, float(liab)/1e9, float(equity)/1e9],
                              color=['green', 'red', 'blue'])
                ax.set_title(f"{symbol} Balance Sheet Breakdown (Billions)")
                ax.set_ylabel("Amount (Billions $)")
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:.1f}B', ha='center', va='bottom')
                
                plt.tight_layout()
                bs_path = os.path.join(output_dir, f"{symbol}_balance_sheet.png")
                plt.savefig(bs_path)
                plt.close()
    except Exception:
        pass

    saved = {
        "financial_health": fh_path,
        "price_sma": price_path,
        "revenue_net": rev_path,
        "metrics_radar": radar_path,
        "balance_sheet": bs_path,
    }
    return saved

# ----------------------
# BUILD ANALYSIS PROMPT
# ----------------------
def build_analysis_prompt(symbol: str, metrics: dict) -> str:
    prompt = f"""Analyze {symbol} and provide a CONCISE professional assessment following this EXACT structure:

INVESTMENT RATING
Provide: Buy/Hold/Sell with 1-sentence rationale

KEY HIGHLIGHTS
List 3-4 most important bullet points about the company's current state

FINANCIAL STRENGTH
Assess: Strong/Moderate/Weak with 2-3 supporting bullets

GROWTH PROSPECTS
Assess: High/Moderate/Low with 2-3 supporting bullets

VALUATION ASSESSMENT
State if: Undervalued/Fairly Valued/Overvalued with brief reasoning

KEY RISKS
List top 3-4 risk factors

TECHNICAL OUTLOOK
Short-term view: Bullish/Neutral/Bearish with 1-2 bullets

RECOMMENDATION SUMMARY
For Long-term Investors: [1 sentence]
For Short-term Traders: [1 sentence]

Available Data:
- Company: {metrics.get('company_name')}
- Sector: {metrics.get('sector')} | Industry: {metrics.get('industry')}
- Current Price: ${metrics.get('current_price'):.2f}
- Daily Change: {metrics.get('daily_change', 0):+.2f}%
- Market Cap: ${metrics.get('market_cap', 0):,.0f}
- P/E (Trailing): {metrics.get('trailingPE', 'N/A')}
- P/E (Forward): {metrics.get('forwardPE', 'N/A')}
- EPS: {metrics.get('trailingEps', 'N/A')}
- ROE: {metrics.get('returnOnEquity', 'N/A')}
- Gross Margin: {metrics.get('grossMargins', 'N/A')}
- Current Ratio: {metrics.get('currentRatio', 'N/A')}
- Debt-to-Equity: {metrics.get('debtToEquity', 'N/A')}
- Beta: {metrics.get('beta', 'N/A')}
- RSI(14): {metrics.get('rsi14', 'N/A')}
- SMA50: ${metrics.get('sma50' , 0):.2f}
- SMA200: ${metrics.get('sma200', 0):.2f}

Keep each section brief and actionable. Use ONLY the data provided above."""
    
    return prompt

# ----------------------
# CALL AUTOGEN
# ----------------------
def call_autogen_analysis(llm_config: dict, prompt: str) -> str:
    """
    For Mistral, AutoGen adds a 'name' field to messages which causes
    422 errors ('extra_forbidden' on messages[*].user.name).
    To avoid this, we bypass AutoGen here and call Mistral directly.
    """
    print("🤖 Running AI analysis via Mistral direct API (no AutoGen)...")
    return call_mistral_direct(prompt)


def call_mistral_direct(prompt: str, temperature: float = 0.2):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    try:
        r = requests.post(MISTRAL_API_URL, json=payload, headers=headers, timeout=30)
    except Exception as e:
        return f"ERROR: {e}"

    if r.status_code != 200:
        return f"ERROR: {r.status_code}: {r.text}"

    try:
        out = r.json()
        return out["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"

# ----------------------
# PROFESSIONAL REPORT PRINTING
# ----------------------
def print_professional_report(symbol: str, metrics: dict, ai_analysis: str, saved_charts: dict, journal_summary: Optional[dict] = None):
    now = datetime.datetime.now()
    date_str = now.strftime("%d %b %Y")
    time_str = now.strftime("%H:%M")
    
    # Header
    print("\n" + "="*70)
    print("📈 EQUITY RESEARCH REPORT")
    #print("="*70)
    # print(f"📅 Date: {date_str} | ⏰ Time: {time_str}")
    # print(f"📊 Company: {metrics.get('company_name', symbol)}")
    # print(f"🏢 Sector: {metrics.get('sector', 'N/A')} | Industry: {metrics.get('industry', 'N/A')}")
    # print(f"🔖 Ticker: {symbol}")
    # print(f"✅ Data Source: yfinance API")
    # print("="*70)

    
    # Executive Summary Box
    # print("\n" + "┌" + "─"*68 + "┐")
    # print("│" + " "*20 + "EXECUTIVE SUMMARY" + " "*31 + "│")
    # print("├" + "─"*68 + "┤")
    # print(f"│ Current Price: ${metrics.get('current_price', 0):>10.2f}  │  Daily Change: {metrics.get('daily_change', 0):>6.2f}%  │")
    # print(f"│ Market Cap:    ${metrics.get('market_cap', 0)/1e9:>10.1f}B │  P/E Ratio:    {str(metrics.get('trailingPE', 'N/A')):>8}  │")
    
    # Determine recommendation
    analysis = parse_ai_analysis(ai_analysis)
    rec_text = "HOLD"
    if analysis['rating']:
        if 'Buy' in analysis['rating'] or 'BUY' in analysis['rating']:
            rec_text = "BUY ⬆"
        elif 'Sell' in analysis['rating'] or 'SELL' in analysis['rating']:
            rec_text = "SELL ⬇"
        else:
            rec_text = "HOLD ➡"



    print("\n📌 EXECUTIVE SUMMARY")
    print("-" * 78)
    print(f"Current Price: ${metrics.get('current_price', 0):.2f}")
    print(f"Daily Change: {metrics.get('daily_change', 0):.2f}%")
    print(f"Market Cap: ${metrics.get('market_cap', 0)/1e9:.1f}B")
    print(f"P/E Ratio: {metrics.get('trailingPE', 'N/A')}")
    print(f"Recommendation: {rec_text}")
    # print("-" * 78)
    
    # print(f"│ Recommendation: {rec_text:>12}  │  Data Period:  {' 1 Year':>8}  │")
    # print("└" + "─"*68 + "┘")
    
    # SECTION 1: MARKET DATA
    print("\n" + "━"*70)
    print("SECTION 1: CURRENT MARKET DATA")
    print("-" * 78)
    print("━"*70)
    
    print("\n1.1 Price Information")
    print("   " + "─"*50)
    print(f"   Current Price:        ${metrics.get('current_price', 0):.2f}")
    print(f"   Previous Close:       ${metrics.get('prev_close', 0):.2f}")
    print(f"   Day's Range:          ${metrics.get('low', 0):.2f} - ${metrics.get('high', 0):.2f}")
    print(f"   Daily Change:         {metrics.get('daily_change', 0):+.2f}%")
    
    intraday_range = (metrics.get('high', 0) - metrics.get('low', 0))
    print(f"   Intraday Volatility:  ${intraday_range:.2f}")
    
    print("\n1.2 Technical Indicators")
    print("   " + "─"*50)
    rsi = metrics.get('rsi14', 50)
    rsi_status = "Overbought ⚠" if rsi > 70 else "Oversold 📉" if rsi < 30 else "Neutral ➡"
    print(f"   RSI (14-day):         {rsi:.1f} - {rsi_status}")
    print(f"   50-Day MA:            ${metrics.get('sma50', 0):.2f}")
    print(f"   200-Day MA:           ${metrics.get('sma200', 0):.2f}")
    
    if metrics.get('current_price') and metrics.get('sma50'):
        vs_sma50 = ((metrics['current_price'] - metrics['sma50']) / metrics['sma50'] * 100)
        trend = "Bullish 📈" if vs_sma50 > 0 else "Bearish 📉"
        print(f"   Price vs 50-Day MA:  {vs_sma50:+.2f}% ({trend})")
    
    # SECTION 2: FUNDAMENTAL ANALYSIS
    print("\n" + "━"*70)
    print("SECTION 2: FUNDAMENTAL ANALYSIS")
    print("-" * 78)
    print("━"*70)
    
    print("\n2.1 Valuation Metrics")
    print("   " + "─"*50)
    print(f"   Market Capitalization: ${metrics.get('market_cap', 0):,.0f}")
    print(f"   P/E Ratio (Trailing):  {metrics.get('trailingPE', 'N/A')}")
    print(f"   P/E Ratio (Forward):   {metrics.get('forwardPE', 'N/A')}")
    print(f"   EPS (Trailing):        {metrics.get('trailingEps', 'N/A')}")
    print(f"   Beta:                  {metrics.get('beta', 'N/A')}")
    
    if metrics.get('dividend_yield'):
        print(f"   Dividend Yield:        {metrics['dividend_yield']*100:.2f}%")
    
    print("\n2.2 Profitability Metrics")
    print("   " + "─"*50)
    
    roe = metrics.get('returnOnEquity')
    if roe:
        roe_pct = roe * 100
        roe_rating = "Excellent ⭐" if roe_pct > 15 else "Good ✓" if roe_pct > 10 else "Weak ⚠"
        print(f"   Return on Equity:      {roe_pct:.2f}% ({roe_rating})")
    else:
        print(f"   Return on Equity:      N/A")
    
    margin = metrics.get('grossMargins')
    if margin:
        margin_pct = margin * 100
        margin_rating = "Strong ⭐" if margin_pct > 40 else "Moderate ✓" if margin_pct > 25 else "Weak ⚠"
        print(f"   Gross Margin:          {margin_pct:.2f}% ({margin_rating})")
    else:
        print(f"   Gross Margin:          N/A")
    
    print("\n2.3 Financial Health")
    print("   " + "─"*50)
    
    cr = metrics.get('currentRatio')
    if cr:
        cr_rating = "Healthy ✓" if cr > 1.5 else "Adequate ➡" if cr > 1 else "Weak ⚠"
        print(f"   Current Ratio:         {cr:.2f} ({cr_rating})")
    else:
        print(f"   Current Ratio:         N/A")
    
    qr = metrics.get('quickRatio')
    if qr:
        print(f"   Quick Ratio:           {qr:.2f}")
    
    de = metrics.get('debtToEquity')
    if de:
        de_rating = "Low Risk ✓" if de < 50 else "Moderate ➡" if de < 150 else "High Risk ⚠"
        print(f"   Debt-to-Equity:        {de:.1f} ({de_rating})")
    else:
        print(f"   Debt-to-Equity:        N/A")
    
    # Calculate financial health score
    health_score = 0
    health_items = 0
    if cr and cr > 1: health_score += 25; health_items += 1
    if margin and margin > 0.4: health_score += 25; health_items += 1
    if roe and roe > 0.15: health_score += 25; health_items += 1
    if de and de < 150: health_score += 25; health_items += 1
    
    if health_items > 0:
        health_rating = "Strong 🟢" if health_score >= 75 else "Moderate 🟡" if health_score >= 50 else "Weak 🔴"
        print(f"\n   Overall Health Score:  {health_score}/100 ({health_rating})")
    
    # SECTION 3: INVESTMENT ANALYSIS
    print("\n" + "━"*70)
    print("SECTION 3: INVESTMENT ANALYSIS")
    print("-" * 78)
    print("━"*70)
    
    # AI ANALYSIS - PROFESSIONALLY FORMATTED
    print("\n📌 Investment Analysis & Recommendation:")
    print("━"*70)
    
    # Parse AI analysis
    analysis = parse_ai_analysis(ai_analysis)
    
    # Investment Rating
    if analysis['rating']:
        rating_text = analysis['rating']
        if 'Buy' in rating_text or 'BUY' in rating_text:
            rating_emoji = "🟢"
        elif 'Sell' in rating_text or 'SELL' in rating_text:
            rating_emoji = "🔴"
        else:
            rating_emoji = "🟡"
        
        print(f"   {rating_emoji} Investment Rating:")
        print(f"   {analysis['rating']}")
    
    # Key Highlights
    if analysis['highlights']:
        print("\n   💬 Key Highlights:")
        for highlight in analysis['highlights'][:4]:
            print(f"      • {highlight}")
    
    print()
    
    # Financial Strength
    if analysis['financial_strength']:
        strength_text = analysis['financial_strength'].strip()
        print("   📊 Financial Strength:")
        print(f"   {strength_text}")
        print()
    
    # Growth Prospects
    if analysis['growth_prospects']:
        growth_text = analysis['growth_prospects'].strip()
        print("   📈 Growth Prospects:")
        print(f"   {growth_text}")
        print()
    
    # Valuation Assessment
    if analysis['valuation']:
        valuation_text = analysis['valuation'].strip()
        print("   💰 Valuation Assessment:")
        print(f"   {valuation_text}")
        print()
    
    # Key Risks
    if analysis['risks']:
        print("   ⚠️  Key Risk Factors:")
        for risk in analysis['risks'][:4]:
            print(f"      • {risk}")
        print()
    
    # Technical Outlook
    if analysis['technical_outlook']:
        tech_text = analysis['technical_outlook'].strip()
        print("   📉 Technical Outlook:")
        print(f"   {tech_text}")
        print()
    
    # Recommendations
    if analysis['recommendations']:
        print("   🎯 Action Recommendations:")
        if analysis['recommendations'].get('long_term'):
            print(f"      Long-term Investors: {analysis['recommendations']['long_term']}")
        if analysis['recommendations'].get('short_term'):
            print(f"      Short-term Traders: {analysis['recommendations']['short_term']}")
    
    # NEWS & SENTIMENT
    headlines = metrics.get('headlines', [])
    if headlines:
        print("\n📌 Recent News Headlines:")
        print("━"*70)
        for i, headline in enumerate(headlines[:5], 1):
            print(f"  {i}. {headline}")
        print()
        print("   💬 Sentiment Impact:")
        print("      • Monitor news for material developments")
        print("      • Consider sentiment shifts in trading decisions")
    
    # TRADING JOURNAL ANALYSIS (if provided)
    if journal_summary:
        print("\n📌 Personal Trading Performance:")
        print("━"*70)
        print(f"   Total Trades: {journal_summary['total_trades']}")
        print(f"   Win Rate: {journal_summary['win_rate_pct']:.1f}%")
        print(f"   Net P&L: ${journal_summary['net_pnl']:.2f}")
        print(f"   Average Win: ${journal_summary['avg_win']:.2f}")
        print(f"   Average Loss: ${journal_summary['avg_loss']:.2f}")
        if journal_summary['expectancy']:
            print(f"   Expectancy: ${journal_summary['expectancy']:.2f}")
        print()
        print("   💬 Performance Analysis:")
        if journal_summary['win_rate_pct'] > 50:
            print("      • ✅ Positive win rate indicates good trade selection")
        else:
            print("      • ⚠️  Win rate below 50% - review entry criteria")
        
        if journal_summary['net_pnl'] > 0:
            print("      • ✅ Net profitable trading on this symbol")
        else:
            print("      • ⚠️  Net losses - consider strategy adjustment")
        
        if journal_summary['expectancy'] and journal_summary['expectancy'] > 0:
            print("      • ✅ Positive expectancy supports continued trading")
        elif journal_summary['expectancy']:
            print("      • ⚠️  Negative expectancy - strategy needs revision")
        
        print("      • Review risk management and position sizing")
        print("      • Consider journaling emotions and market conditions")
    
    # CHARTS GENERATED
    print("\n📌 Visual Analysis Charts:")
    print("-" * 78)
    print("━"*70)
    print("\nFinancial Health Score:")
    print("  • Overall health rating based on key metrics")
    print("  • Combines liquidity, profitability, and leverage")
    print(f"  → Chart: {saved_charts.get('financial_health', 'N/A')}")
    
    if saved_charts.get('price_sma'):
        print("\nPrice & Moving Averages:")
        print("  • Historical price action with trend indicators")
        print("  • SMA50 and SMA200 for trend identification")
        print(f"  → Chart: {saved_charts['price_sma']}")
    
    if saved_charts.get('revenue_net'):
        print("\nRevenue & Profitability Trend:")
        print("  • Multi-period revenue and net income growth")
        print("  • Identifies business momentum")
        print(f"  → Chart: {saved_charts['revenue_net']}")
    
    if saved_charts.get('metrics_radar'):
        print("\nFinancial Metrics Radar:")
        print("  • Normalized comparison across key ratios")
        print("  • Visual snapshot of financial position")
        print(f"  → Chart: {saved_charts['metrics_radar']}")
    
    if saved_charts.get('balance_sheet'):
        print("\nBalance Sheet Structure:")
        print("  • Assets, Liabilities, and Equity breakdown")
        print("  • Shows capital structure and leverage")
        print(f"  → Chart: {saved_charts['balance_sheet']}")

    print("="*70)
    print("✨ Financial analysis completed successfully.")
    print("="*70)

# ----------------------
# MAIN FUNCTION
# ----------------------
def main():
    print("\n" + "="*70)
    print("📈 PROFESSIONAL FINANCIAL REPORT GENERATOR")
    print("="*70)
    print()
    
    # Check AutoGen availability
    if autogen:
        print("✓ AutoGen framework detected - using AI agent analysis")
    else:
        print("⚠️  AutoGen not installed - using direct API mode")
        print("   Install with: pip install pyautogen\n")
    
    # Get user input
    symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    if not symbol:
        print("❌ No symbol provided. Exiting.")
        return
    
    period = input("History period (e.g., 6mo, 1y, 2y) [default: 1y]: ").strip() or "1y"
    
    # Optional trading journal
    journal_path = input("Optional: Trading journal CSV path (press Enter to skip): ").strip() or None
    
    # Fetch data
    print(f"\n🔍 Fetching comprehensive data for {symbol}...")
    try:
        data = fetch_ticker_data(symbol, period)
    except Exception as e:
        print(f"❌ Data fetch error: {e}")
        return
    
    print("📊 Computing financial metrics...")
    try:
        metrics = compute_metrics(data)
    except Exception as e:
        print(f"❌ Metrics computation error: {e}")
        return
    
    # Validate we have price data
    if not metrics.get('current_price'):
        print(f"❌ No price data available for {symbol}. Check symbol and try again.")
        return
    
    # Analyze trading journal if provided
    journal_summary = None
    if journal_path:
        print("📋 Analyzing trading journal...")
        try:
            journal_summary = analyze_trading_journal(journal_path, symbol_filter=symbol)
            if journal_summary:
                print(f"✅ Found {journal_summary['total_trades']} trades for {symbol}")
            else:
                print(f"⚠️  No trades found for {symbol} in journal")
        except Exception as e:
            print(f"⚠️  Journal analysis error: {e}")
            journal_summary = None
    
    # Generate charts
    print("📈 Generating visual analysis charts...")
    try:
        saved_charts = generate_financial_charts(symbol, data, metrics)
        print(f"✅ Charts saved to ./charts/\n")
    except Exception as e:
        print(f"⚠️  Chart generation error: {e}")
        saved_charts = {}
    
    # Setup LLM config
    try:
        llm_config = setup_llm_config_mistral(api_key=MISTRAL_API_KEY, model=MISTRAL_MODEL)
    except Exception as e:
        print(f"❌ LLM config error: {e}")
        return
    
    # Build analysis prompt
    prompt = build_analysis_prompt(symbol, metrics)
    
        # Call AI analysis
    print("🤖 Running AI-powered financial analysis...")
    try:
        ai_analysis = call_autogen_analysis(llm_config, prompt)
    except Exception as e:
        print(f"⚠️  AI analysis error: {e}")
        print("Attempting fallback to direct API call...")
        try:
            ai_analysis = call_mistral_direct(prompt)
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            ai_analysis = "AI analysis unavailable. Please review metrics manually."

    # -------------------------------------------------
    # ✔ RUN REPORT AND CAPTURE ENTIRE TERMINAL OUTPUT
    # -------------------------------------------------
    captured_output = capture_output(
        print_professional_report,
        symbol, metrics, ai_analysis, saved_charts, journal_summary
    )

    # Print to terminal
    print(captured_output)

    # -------------------------------------------------
    # ✔ SAVE FULL OUTPUT TO DATABASE
    # -------------------------------------------------
    if log_to_db:
        try:
            log_to_db(
                agent_name="financial_report",
                ticker=symbol,
                params={"period": period},
                output=captured_output
            )
            print("\n💾 Successfully saved to database.\n")
        except Exception as e:
            print(f"\n⚠️ Failed to log to database: {e}\n")




if __name__ == "__main__":
    main()  