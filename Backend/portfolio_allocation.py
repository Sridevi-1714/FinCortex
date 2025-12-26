import os
import json
import warnings
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from textwrap import dedent
import io
from contextlib import redirect_stdout, redirect_stderr

import matplotlib
matplotlib.use("Agg")   # use non-interactive backend — prevents Tk/Tcl GUI issues
import matplotlib.pyplot as plt


def capture_output(func, *args, **kwargs):
    """Capture all printed output for database logging."""
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            func(*args, **kwargs)
    except Exception as e:
        buffer.write(f"\n[ERROR] {e}\n")
    return buffer.getvalue()


try:
    from config_db import log_to_db
except Exception:
    log_to_db=None

# suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("autogen").setLevel(logging.ERROR)

import requests

# yfinance for stock data fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not available. Install with: pip install yfinance")

# autogen
try:
    import autogen
except Exception:
    autogen = None

# -----------------------------
# Helper: load config_api_keys
# -----------------------------
def load_config_keys(path: str = "config_api_keys") -> Dict:
    """Load JSON keys from config_api_keys"""
    if os.path.exists(path):
        p = path
    elif os.path.exists(path + ".json"):
        p = path + ".json"
    else:
        print(f"⚠️  Config file '{path}' not found.")
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"⚠️  Error loading config file: {e}")
        return {}

# -----------------------------
# LLM config for OpenRouter
# -----------------------------
def setup_llm_config_openrouter(api_key: str, model: str = "meta-llama/llama-3.1-70b-instruct") -> Dict:
    """Setup autogen-compatible config for OpenRouter"""
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required.")
    return {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "api_type": "openai"
            }
        ],
        "timeout": 120,
        "temperature": 0.3,
        "max_tokens": 1200
    }

# -----------------------------
# Data retriever with API priority
# -----------------------------
class PortfolioDataRetriever:
    def __init__(self, api_keys: Dict):
        self.finnhub_key = api_keys.get("FINNHUB_API_KEY", "").strip()
        self.fmp_key = api_keys.get("FMP_API_KEY", "").strip()
    
    def get_stock_quote_finnhub(self, ticker: str) -> Dict:
        """Get current price from Finnhub"""
        if not self.finnhub_key:
            return None
        
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                return {
                    "current_price": data.get("c", 0),
                    "previous_close": data.get("pc", 0),
                    "high": data.get("h", 0),
                    "low": data.get("l", 0)
                }
        except Exception as e:
            print(f"⚠️  Finnhub quote error for {ticker}: {e}")
        
        return None
    
    def get_company_profile_fmp(self, ticker: str) -> Dict:
        """Get company profile from FMP"""
        if not self.fmp_key:
            return None
        
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={self.fmp_key}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    profile = data[0]
                    return {
                        "name": profile.get("companyName", ticker),
                        "sector": profile.get("sector", "N/A"),
                        "industry": profile.get("industry", "N/A"),
                        "market_cap": profile.get("mktCap", 0),
                        "beta": profile.get("beta", 1.0),
                        "pe_ratio": profile.get("price", 0) / profile.get("eps", 1) if profile.get("eps") else "N/A"
                    }
        except Exception as e:
            print(f"⚠️  FMP profile error for {ticker}: {e}")
        
        return None
    
    def get_financial_ratios_fmp(self, ticker: str) -> Dict:
        """Get financial ratios from FMP"""
        if not self.fmp_key:
            return None
        
        try:
            url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={self.fmp_key}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    ratios = data[0]
                    return {
                        "debt_to_equity": ratios.get("debtEquityRatio", "N/A"),
                        "roe": ratios.get("returnOnEquity", "N/A"),
                        "current_ratio": ratios.get("currentRatio", "N/A")
                    }
        except Exception as e:
            print(f"⚠️  FMP ratios error for {ticker}: {e}")
        
        return None
    
    def get_stock_data_yfinance(self, ticker: str) -> Dict:
        """Get stock data from yfinance (fallback)"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            hist = stock.history(period="1y")
            
            # Calculate annual return
            if not hist.empty and len(hist) > 1:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                annual_return = ((end_price - start_price) / start_price) * 100
            else:
                annual_return = 0
            
            # Calculate volatility (standard deviation of returns)
            if not hist.empty:
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * 100 * (252 ** 0.5)  # Annualized
            else:
                volatility = 20
            
            return {
                "ticker": ticker,
                "name": info.get("longName", ticker),
                "price": info.get("currentPrice", 0),
                "sector": info.get("sector", "N/A"),
                "beta": info.get("beta", 1.0),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "annual_return": round(annual_return, 2),
                "volatility": round(volatility, 2),
                "debt_to_equity": info.get("debtToEquity", "N/A"),
                "roe": info.get("returnOnEquity", "N/A"),
                "source": "yfinance"
            }
        except Exception as e:
            print(f"⚠️  yfinance error for {ticker}: {e}")
            return None
    
    def get_stock_data(self, ticker: str) -> Dict:
        """Get comprehensive stock data - API first, then yfinance fallback"""
        
        # Initialize result
        result = {
            "ticker": ticker,
            "name": ticker,
            "price": 0,
            "sector": "N/A",
            "beta": 1.0,
            "pe_ratio": "N/A",
            "market_cap": 0,
            "annual_return": 0,
            "volatility": 20,
            "debt_to_equity": "N/A",
            "roe": "N/A",
            "source": "none"
        }
        
        # Try API sources first
        quote = self.get_stock_quote_finnhub(ticker)
        profile = self.get_company_profile_fmp(ticker)
        ratios = self.get_financial_ratios_fmp(ticker)
        
        # If we got API data, use it
        if quote or profile:
            result["source"] = "API"
            
            if quote:
                result["price"] = quote.get("current_price", 0)
                print(f"✓ {ticker}: Price from Finnhub: ${result['price']:.2f}")
            
            if profile:
                result["name"] = profile.get("name", ticker)
                result["sector"] = profile.get("sector", "N/A")
                result["beta"] = profile.get("beta", 1.0)
                result["pe_ratio"] = profile.get("pe_ratio", "N/A")
                result["market_cap"] = profile.get("market_cap", 0)
                print(f"✓ {ticker}: Profile from FMP")
            
            if ratios:
                result["debt_to_equity"] = ratios.get("debt_to_equity", "N/A")
                result["roe"] = ratios.get("roe", "N/A")
            
            # Calculate returns/volatility from yfinance if available
            if YFINANCE_AVAILABLE:
                try:
                    hist = yf.Ticker(ticker).history(period="1y")
                    if not hist.empty and len(hist) > 1:
                        start_price = hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]
                        result["annual_return"] = round(((end_price - start_price) / start_price) * 100, 2)
                        
                        returns = hist['Close'].pct_change().dropna()
                        result["volatility"] = round(returns.std() * 100 * (252 ** 0.5), 2)
                        print(f"✓ {ticker}: Historical data from yfinance")
                except Exception:
                    pass
            
            return result
        
        # Fallback to yfinance
        print(f"⚠️  {ticker}: API data unavailable, falling back to yfinance...")
        yf_data = self.get_stock_data_yfinance(ticker)
        
        if yf_data:
            return yf_data
        
        # Ultimate fallback - return minimal data
        print(f"❌ {ticker}: No data available from any source")
        return result

# -----------------------------
# Portfolio Allocation Agent
# -----------------------------
class PortfolioAllocationAgent:
    def __init__(self, llm_config: Dict, data_retriever: PortfolioDataRetriever):
        if autogen is None:
            raise RuntimeError("autogen is required. Install it first.")
        
        self.llm_config = llm_config
        self.data_retriever = data_retriever
        
        self.assistant = autogen.AssistantAgent(
            name="Portfolio_Manager",
            llm_config=llm_config,
            system_message=dedent("""
                You are an expert portfolio manager. Create optimal portfolio allocations based on:
                - Risk tolerance (Conservative/Moderate/Aggressive)
                - Stock fundamentals and metrics
                - Diversification principles
                - Modern portfolio theory
                
                Always provide structured, actionable recommendations.
            """)
        )
        
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )
    
    def create_allocation(self, tickers: List[str], investment: float, risk_profile: str) -> Dict:
        """Create portfolio allocation"""
        
        print(f"\n📊 Fetching data for {len(tickers)} stocks...")
        print("=" * 60)
        
        # Get data for all stocks
        stock_data = []
        for ticker in tickers:
            data = self.data_retriever.get_stock_data(ticker)
            if data["price"] > 0:
                stock_data.append(data)
        
        print("=" * 60)
        
        if not stock_data:
            raise ValueError("No valid stock data available")
        
        print(f"\n✓ Successfully fetched data for {len(stock_data)}/{len(tickers)} stocks\n")
        
        # Build prompt
        stocks_summary = "\n".join([
            f"- {s['ticker']}: ${s['price']:.2f}, Beta: {s['beta']}, Volatility: {s['volatility']}%, Return: {s['annual_return']}%, Sector: {s['sector']}, D/E: {s['debt_to_equity']}, ROE: {s['roe']}"
            for s in stock_data
        ])
        
        prompt = dedent(f"""
        Create a portfolio allocation for:
        
        Investment Amount: ${investment:,.0f}
        Risk Profile: {risk_profile}
        
        Available Stocks:
        {stocks_summary}
        
        Provide allocation in EXACT format:
        
        ALLOCATION:
        [TICKER]: $[AMOUNT] ([PERCENTAGE]%) - [Risk Level, Description]
        ... (for each stock)
        Cash: $[AMOUNT] ([PERCENTAGE]%) - No Risk, Buffer
        
        SECTOR_BREAKDOWN:
        [Sector]: $[AMOUNT] ([PERCENTAGE]%)
        ... (for each sector)
        
        METRICS:
        Expected Return: [X.X]% annually
        Portfolio Risk: [X.X]% volatility
        Sharpe Ratio: [X.XX]
        Diversification Score: [X.X]/10
        Portfolio Beta: [X.X]
        Maximum Drawdown: -[XX]%
        Value at Risk (95%): $[amount]
        
        REASONING:
        ✅ [TICKER] ([XX]%): [Short reason]
        ... (for each stock)
        
        REBALANCING:
        - Review: [Quarterly/Monthly]
        - Trigger: [X]% drift
        - Next Review: [Date]
        
        PROJECTION:
        1 Year: $[amount] ([+X]% return)
        5 Years: $[amount]
        10 Years: $[amount]
        
        Guidelines:
        - Conservative: 50-60% stocks, 30-35% cash
        - Moderate: 60-70% stocks, 20-25% cash
        - Aggressive: 75-85% stocks, 10-15% cash
        - Diversify across sectors
        - Include cash buffer
        - Consider debt levels and ROE
        """).strip()
        
        print("🤖 Running AI allocation analysis...")
        
        self.user_proxy.initiate_chat(
            self.assistant,
            message=prompt,
            clear_history=True
        )
        
        response = self.user_proxy.last_message()
        if not response:
            raise RuntimeError("LLM did not return response")
        
        content = response.get("content", "").strip()
        
        return {
            "investment": investment,
            "risk_profile": risk_profile,
            "stocks": stock_data,
            "allocation": content,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def format_output(self, result: Dict):
        """Format portfolio allocation output"""
        
        ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        date_str = ist_now.strftime("%d %b %Y")
        time_str = ist_now.strftime("%H:%M IST")
        
        allocation = result['allocation']
        
        # Parse allocation sections
        lines = allocation.split('\n')
        
        allocations = []
        sectors = []
        metrics = {}
        reasoning = []
        rebalancing = []
        projections = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.upper().startswith("ALLOCATION:"):
                current_section = "allocation"
            elif line.upper().startswith("SECTOR"):
                current_section = "sector"
            elif line.upper().startswith("METRICS:"):
                current_section = "metrics"
            elif line.upper().startswith("REASONING:"):
                current_section = "reasoning"
            elif line.upper().startswith("REBALANCING:"):
                current_section = "rebalancing"
            elif line.upper().startswith("PROJECTION:"):
                current_section = "projection"
            elif current_section == "allocation" and (":" in line or "$" in line):
                allocations.append(line)
            elif current_section == "sector" and (":" in line or "$" in line):
                sectors.append(line)
            elif current_section == "metrics" and ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    metrics[parts[0].strip()] = parts[1].strip()
            elif current_section == "reasoning":
                reasoning.append(line)
            elif current_section == "rebalancing":
                rebalancing.append(line)
            elif current_section == "projection":
                projections.append(line)
        
        # Print formatted output
        # <-- CHANGED: use exact REPORT header that run_query_once expects
        print("\n📊 PORTFOLIO ALLOCATION REPORT")
        print("=" * 70)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        # print(f"📅 Date: {date_str} | ⏰ Time: {time_str}")
        print(f"💰 Investment: ${result['investment']:,.0f} ")
        
        # Show data sources with color coding
        sources = set(s.get('source', 'unknown') for s in result['stocks'])
        source_emoji = "✅" if "API" in sources else "⚠️"
        # print(f"{source_emoji} Data Sources: {', '.join(sources)}")
        print("=" * 70)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        print("\n💼 Recommended Allocation:")
        print("-" * 70)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("📊 By Stock:")
        for i, alloc in enumerate(allocations):
            if i == len(allocations) - 1:
                print(f"  └─ {alloc}")
            else:
                print(f"  ├─ {alloc}")
        
        if sectors:
            print("\n🎯 By Sector:")
            for i, sector in enumerate(sectors):
                if i == len(sectors) - 1:
                    print(f"  └─ {sector}")
                else:
                    print(f"  ├─ {sector}")
        
        if metrics:
            print("\n📈 Portfolio Metrics:")
            print("-" * 70)
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for key, value in metrics.items():
                # Add checkmarks and warnings for metrics
                indicator = "  "
                if "Sharpe" in key:
                    try:
                        val = float(''.join(c for c in value.split()[0] if c.isdigit() or c == '.' or c == '-'))
                        if val > 1.5: indicator = "✅"
                        elif val > 1.0: indicator = "👍"
                        elif val < 0.5: indicator = "⚠️"
                    except:
                        pass
                elif "Diversification" in key:
                    try:
                        val = float(''.join(c for c in value.split("/")[0] if c.isdigit() or c == '.'))
                        if val >= 7: indicator = "✅"
                        elif val >= 5: indicator = "👍"
                        else: indicator = "⚠️"
                    except:
                        pass
                elif "Beta" in key:
                    try:
                        val = float(''.join(c for c in value.split()[0] if c.isdigit() or c == '.' or c == '-'))
                        if 0.8 <= val <= 1.2: indicator = "✅"
                        elif val > 1.5: indicator = "⚠️"
                    except:
                        pass
                
                print(f"{indicator} {key}: {value}")
        
        print("\n🎲 Risk Analysis:")
        print("-" * 70)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if "Portfolio Beta" in metrics:
            beta_val = metrics['Portfolio Beta']
            print(f"📊 Portfolio Beta: {beta_val}")
            print("   💬 Meaning: How your portfolio moves compared to the market")
            print("      • Beta = 1.0 → Moves with market")
            print("      • Beta > 1.0 → More volatile than market")
            print("      • Beta < 1.0 → Less volatile than market")
        
        if "Maximum Drawdown" in metrics:
            dd_val = metrics['Maximum Drawdown']
            print(f"\n📉 Maximum Drawdown: {dd_val}")
            print("   💬 Meaning: Largest peak-to-trough decline in portfolio value")
            print("      This represents your worst-case historical loss scenario")
        
        if "Value at Risk (95%)" in metrics or "Value at Risk" in metrics:
            var_key = "Value at Risk (95%)" if "Value at Risk (95%)" in metrics else "Value at Risk"
            var_val = metrics[var_key]
            print(f"\n⚠️  Value at Risk (95%): {var_val}")
            print("   💬 Meaning: Maximum expected loss in bad scenarios")
            print("      There's a 5% chance you could lose this much in a year")
        
        if "Expected Return" in metrics:
            ret_val = metrics["Expected Return"]
            print(f"\n📈 Expected Return: {ret_val}")
            print("   💬 Meaning: Average annual return you can expect")
            print("      Based on historical data, not guaranteed")
        
        if reasoning:
            print("\n💡 Allocation Reasoning:")
            print("-" * 70)
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for reason in reasoning:
                print(f"  {reason}")
        
        if rebalancing:
            print("\n🔄 Rebalancing Plan:")
            print("-" * 70)
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for rb in rebalancing:
                print(f"  {rb}")
            print("\n  💬 Why Rebalance?")
            print("     As stocks perform differently, your allocation drifts from target.")
            print("     Rebalancing brings it back, controlling risk and enforcing discipline.")
            print("     It's like trimming winners and buying losers at better prices.")
        
        if projections:
            print("\n📊 Performance Projection:")
            print("-" * 70)
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for proj in projections:
                print(f"  {proj}")
            print("\n  💬 About These Numbers:")
            print("     • Based on historical returns and compound growth")
            print("     • Assumes consistent market conditions (rarely happens!)")
            print("     • Actual results WILL differ—markets are unpredictable")
            print("     • Use as rough guidelines, not guarantees")
        
        # Quick action items
        print("\n✅ Next Steps:")
        print("-" * 70)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  1. Review the allocation and ensure you're comfortable with the risk")
        print("  2. Consider adding more stocks for better diversification")
        print("  3. Set up automatic rebalancing alerts")
        print("  4. Document your investment thesis and review quarterly")
        
        # Risk warnings based on portfolio
        if len(result['stocks']) == 1:
            print("\n")
            # print("\n⚠️  Important Warnings:")
            # print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            # print("-" * 70)
            # print("  🚨 SINGLE STOCK RISK: Your portfolio has only 1 stock!")
            # print("     • Zero diversification = Maximum company-specific risk")
            # print("     • Consider adding 4-5 more stocks from different sectors")
            # print("     • A diversified portfolio reduces volatility significantly")
        
        # <-- CHANGED: add an explicit end marker that run_query_once can also detect if needed
        print("\n")
        print("=" * 70)
        print("📊 END OF PORTFOLIO ALLOCATION REPORT")
        print("=" * 70 + "\n")
        
        # Enhanced data source summary
        # print("📋 Data Source Summary:")
        # print("-" * 70)
        # print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        # for stock in result['stocks']:
        #     source = stock.get('source', 'unknown')
        #     source_emoji = "✅" if source == "API" else "⚠️" if source == "yfinance" else "❌"
        #     print(f"{source_emoji} {stock['ticker']}: {source} | Price: ${stock['price']:.2f}")
        # print()


# -----------------------------
# Main
# -----------------------------
def main():
    print("🔧 Loading configuration...")
    api_keys = load_config_keys("config_api_keys")
    
    openrouter_key = api_keys.get("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in config_api_keys!")
        print("Get your free key at: https://openrouter.ai/keys")
        return
    
    # Setup OpenRouter LLM
    llm_config = setup_llm_config_openrouter(
        api_key=openrouter_key,
        model="meta-llama/llama-3.1-70b-instruct"  # Free on OpenRouter
    )
    
    data_retriever = PortfolioDataRetriever(api_keys)
    allocator = PortfolioAllocationAgent(llm_config, data_retriever)
    
    # User inputs
    print("\n💼 PORTFOLIO ALLOCATION TOOL\n")
    
    # Get stock tickers
    tickers_input = input("Enter stock tickers (comma-separated, e.g., AAPL,MSFT,TSLA,NVDA): ").strip().upper()
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    
    if not tickers:
        print("❌ No tickers provided!")
        return
    
    # Get investment amount
    investment_input = input("Enter total investment amount (e.g., 50000): ").strip()
    try:
        investment = float(investment_input.replace(",", ""))
        if investment <= 0:
            raise ValueError
    except ValueError:
        print("❌ Invalid investment amount!")
        return
    
    # Get risk profile
    print("\nRisk Profiles:")
    print("1. Conservative (Low risk, stable returns)")
    print("2. Moderate (Balanced risk/reward)")
    print("3. Aggressive (High risk, high potential returns)")
    
    risk_input = input("\nSelect risk profile (1-3 or Conservative/Moderate/Aggressive): ").strip()
    
    if risk_input == "1" or risk_input.lower().startswith("con"):
        risk_profile = "Conservative"
    elif risk_input == "2" or risk_input.lower().startswith("mod"):
        risk_profile = "Moderate"
    elif risk_input == "3" or risk_input.lower().startswith("agg"):
        risk_profile = "Aggressive"
    else:
        risk_profile = "Moderate"
    
        print(f"\n🚀 Creating portfolio allocation for {len(tickers)} stocks...\n")
    
    try:
        # Run allocation (may print inside)
        result = allocator.create_allocation(tickers, investment, risk_profile)

        # ---- CAPTURE FULL OUTPUT of the pretty formatter ----
        captured_text = capture_output(
            allocator.format_output,
            result
        )

        # show same output in terminal
        print(captured_text)

        # ==============================
        # SAVE TO DATABASE (SUPABASE)
        # ==============================
        try:
            if log_to_db:
                log_to_db(
                    agent_name="portfolio_allocation",
                    ticker=",".join(tickers),
                    params={
                        "investment": investment,
                        "risk_profile": risk_profile
                    },
                    output=captured_text  # <--- save the full terminal output
                )
                print("\n💾 Successfully saved portfolio allocation to database.\n")
            else:
                print("\n⚠️ config_db not found — skipping DB logging.\n")
        except Exception as db_error:
            print(f"\n⚠️ Warning: Failed to log to database: {db_error}\n")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()