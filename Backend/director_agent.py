import os
import re
import json
import warnings
import logging
import traceback
import contextlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from textwrap import dedent

import requests
import yfinance as yf

import io
import sys
import glob

# Suppress noisy warnings & autogen logs
warnings.filterwarnings("ignore")
logging.getLogger("autogen").setLevel(logging.ERROR)

try:
    import autogen
except Exception:
    autogen = None

# Optional DB logger provided by user in config_db.py
try:
    from config_db import log_to_db
except Exception:
    log_to_db = None


# ---------------------------
# Utility: capture stdout/stderr and tracebacks
# ---------------------------
def capture_output(func, *args, **kwargs) -> str:
    """
    Run func(*args, **kwargs) and capture stdout/stderr into a string.
    If func raises, capture the traceback and return it (does not re-raise).
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                ret = func(*args, **kwargs)
                # If the function returned something, print a small repr (helps debugging)
                if ret is not None:
                    try:
                        print(f"\n<RETURN_VALUE> {repr(ret)}")
                    except Exception:
                        print("\n<RETURN_VALUE> (unprintable)")
            except Exception:
                traceback.print_exc(file=buf)
    finally:
        contents = buf.getvalue()
        buf.close()
    return contents


# ---------------------------
# Config loader
# ---------------------------
def load_config_keys(path: str = "config_api_keys") -> Dict:
    """Load API keys from config file. Accepts 'config_api_keys' or 'config_api_keys.json'."""
    p = None
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
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"⚠️  Error loading config: {e}")
        return {}


# ---------------------------
# LLM setup for director
# ---------------------------
def setup_director_llm_config(api_keys: Dict) -> Dict:
    """Setup a default LLM config for the Director (Groq preferred, else Ollama)."""
    groq_key = api_keys.get("GROQ_API_KEY", "").strip()
    if groq_key:
        return {
            "config_list": [{
                "model": "llama-3.3-70b-versatile",
                "api_key": groq_key,
                "base_url": "https://api.groq.com/openai/v1",
                "api_type": "openai"
            }],
            "timeout": 60,
            "temperature": 0.3,
            "max_tokens": 400
        }

    ollama_key = api_keys.get("OLLAMA_API_KEY", "").strip()
    if ollama_key:
        ollama_base = api_keys.get("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
        ollama_model = api_keys.get("OLLAMA_MODEL", "llama3.1").strip()
        return {
            "config_list": [{
                "model": ollama_model,
                "api_key": ollama_key,
                "base_url": ollama_base,
                "api_type": "openai"
            }],
            "timeout": 60,
            "temperature": 0.3,
            "max_tokens": 400
        }

    raise ValueError("No valid API key found. Please add GROQ_API_KEY or OLLAMA_API_KEY to config_api_keys")


# ---------------------------
# Entity extraction
# ---------------------------
class EntityExtractor:
    COMPANY_MAP = {
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
        "amazon": "AMZN", "tesla": "TSLA", "meta": "META", "facebook": "META",
        "nvidia": "NVDA", "netflix": "NFLX", "disney": "DIS", "walmart": "WMT",
        "jpmorgan": "JPM", "visa": "V", "mastercard": "MA", "paypal": "PYPL",
        "intel": "INTC", "amd": "AMD", "oracle": "ORCL", "salesforce": "CRM",
        "adobe": "ADBE", "cisco": "CSCO", "ibm": "IBM", "qualcomm": "QCOM",
        "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS"
    }

    @staticmethod
    def extract_tickers(text: str) -> Optional[List[str]]:
        if not text:
            return None
        tickers: List[str] = []
        text_lower = text.lower()

        # explicit symbol matches (uppercase tokens)
        ticker_pattern = r'\b([A-Z]{1,8}(?:\.[A-Z]{1,4})?)\b'
        matches = re.findall(ticker_pattern, text)
        for m in matches:
            if any(ch.isalpha() for ch in m):
                if m not in tickers:
                    tickers.append(m)

        # company name map
        for company, ticker in EntityExtractor.COMPANY_MAP.items():
            if company in text_lower and ticker not in tickers:
                tickers.append(ticker)

        # comma-separated fallback
        if ',' in text:
            parts = text.split(',')
            for part in parts:
                match = re.search(ticker_pattern, part.strip())
                if match:
                    sym = match.group(1)
                    if sym not in tickers:
                        tickers.append(sym)
        return tickers if tickers else None

    @staticmethod
    def extract_ticker(text: str) -> Optional[str]:
        tickers = EntityExtractor.extract_tickers(text)
        return tickers[0] if tickers else None

    @staticmethod
    def extract_amount(text: str) -> Optional[float]:
        if not text:
            return None

        text = text.lower().replace(",", "").strip()

    # explicit k / m handling
        km_match = re.search(r'\b(\d+(?:\.\d+)?)\s*(k|m)\b', text)
        if km_match:
            value = float(km_match.group(1))
            unit = km_match.group(2)
            return value * (1000 if unit == "k" else 1_000_000)

    # plain number (NO scaling)
        num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
        if num_match:
            return float(num_match.group(1))

        return None


    @staticmethod
    def extract_risk_level(text: str) -> Optional[str]:
        if not text:
            return None
        t = text.lower()
        if any(word in t for word in ["low risk", "conservative", "safe", "low"]):
            return "Conservative"
        if any(word in t for word in ["high risk", "aggressive", "risky", "high"]):
            return "Aggressive"
        if any(word in t for word in ["medium risk", "moderate", "balanced", "medium"]):
            return "Moderate"
        return None


# ---------------------------
# Ticker lookup helpers
# ---------------------------
_TICKER_STOPWORDS = {
    "market", "for", "the", "a", "an", "of", "price", "report", "analysis",
    "risk", "strategy", "portfolio", "allocate", "allocation", "stock",
    "stocks", "company", "give", "me", "create", "make", "do", "on", "in",
    "and", "with", "to", "my"
}


def _build_lookup_terms(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9\.\-]+", text.lower())
    candidates = [t for t in tokens if t not in _TICKER_STOPWORDS]
    terms: List[str] = []
    if candidates:
        terms.append(candidates[-1])
        phrase = " ".join(candidates)
        if phrase and phrase not in terms:
            terms.append(phrase)
    if text and text.lower() not in terms:
        terms.append(text.lower())
    return terms


def _lookup_ticker_via_apis(term: str, api_keys: Dict) -> Optional[str]:
    term = term.strip()
    if not term:
        return None

    finnhub_key = api_keys.get("FINNHUB_API_KEY", "").strip()
    if finnhub_key:
        try:
            resp = requests.get("https://finnhub.io/api/v1/search",
                                params={"q": term, "token": finnhub_key},
                                timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("result", []):
                    symbol = item.get("symbol")
                    if symbol:
                        return symbol.upper()
        except Exception:
            pass

    # Yahoo Finance search
    try:
        resp = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                            params={"q": term, "quotesCount": 1, "newsCount": 0},
                            timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            quotes = data.get("quotes", [])
            if quotes:
                sym = quotes[0].get("symbol")
                if sym:
                    return sym.upper()
    except Exception:
        pass

    # yfinance direct check
    try:
        t = yf.Ticker(term)
        finfo = getattr(t, "fast_info", None)
        if finfo and getattr(finfo, "last_price", None):
            return term.upper()
        info = t.info if hasattr(t, "info") else {}
        if info.get("regularMarketPrice"):
            return term.upper()
    except Exception:
        pass

    return None


def resolve_ticker_from_text(user_query: str, api_keys: Dict) -> Optional[str]:
    for term in _build_lookup_terms(user_query):
        sym = _lookup_ticker_via_apis(term, api_keys)
        if sym:
            return sym
    return None


def resolve_multiple_tickers_from_text(user_query: str, api_keys: Dict) -> Optional[List[str]]:
    parts = re.split(r",|&|/|\band\b", user_query, flags=re.IGNORECASE)
    symbols: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        sym = resolve_ticker_from_text(part, api_keys)
        if sym and sym not in symbols:
            symbols.append(sym)
    return symbols if symbols else None


# ---------------------------
# DirectorAgent: intent extraction & routing
# ---------------------------
class DirectorAgent:
    AGENT_CAPABILITIES = {
        "market_forecaster": {
            "keywords": ["market forecast", "market", "forecast", "predict", "price target", "stock prediction", "market outlook", "trend", "future price"],
            "required": ["ticker"],
            "optional": ["period", "horizon"],
            "defaults": {"period": "1y", "horizon": 14},
            "description": "Predicts stock price movements and market trends"
        },
        "financial_report": {
            "keywords": ["financial report", "analysis", "company analysis", "fundamentals", "balance sheet", "income statement", "comprehensive", "detailed analysis", "report"],
            "required": ["ticker"],
            "optional": ["period", "journal_path"],
            "defaults": {"period": "1y"},
            "description": "Provides comprehensive financial analysis and company reports"
        },
        "risk_assessment": {
            "keywords": ["risk", "volatility", "risk assessment", "how risky", "risk analysis", "safety", "dangerous", "safe investment"],
            "required": ["ticker"],
            "optional": [],
            "defaults": {},
            "description": "Analyzes investment risks and volatility"
        },
        "portfolio_allocation": {
            "keywords": ["portfolio", "allocation", "diversify", "distribute", "asset allocation", "balance", "invest in multiple"],
            "required": ["tickers", "amount"],
            "optional": ["risk_level"],
            "defaults": {"risk_level": "Moderate"},
            "description": "Suggests optimal portfolio distribution across assets"
        },
        "trade_strategy": {
            "keywords": ["trade", "strategy", "entry", "exit", "stop loss", "buy", "sell", "trading plan", "position"],
            "required": ["ticker", "amount"],
            "optional": ["risk_level"],
            "defaults": {"risk_level": "medium"},
            "description": "Creates detailed entry/exit trading strategies"
        }
    }

    def __init__(self, llm_config: Dict, api_keys: Optional[Dict] = None):
        if autogen is None:
            raise RuntimeError("autogen library required. Install or provide alternate flow.")
        self.llm_config = llm_config
        self.api_keys = api_keys or {}
        self.conversation_history = []
        # Setup autogen assistant & user_proxy (used only for fallback intent detection)
        self.assistant = autogen.AssistantAgent(
            name="Director",
            llm_config=llm_config,
            system_message=dedent("""
                You are a financial assistant director. Your job is to:
                1. Understand what financial service the user needs
                2. Identify which agent should handle it: market_forecaster, financial_report, risk_assessment, portfolio_allocation, or trade_strategy
                3. Extract key entities: company/ticker, investment amount, risk level
                4. Be conversational and helpful

                Respond in this JSON format:
                {
                    "agent": "agent_name",
                    "confidence": 0.9,
                    "entities": {
                        "ticker": "AAPL",
                        "amount": 10000,
                        "risk_level": "medium"
                    },
                    "missing": ["field1", "field2"],
                    "clarification_needed": true/false
                }
            """)
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )

    def analyze_intent(self, user_query: str) -> Dict:
        print("\n🤔 Analyzing your request...")
        entities = {
            "ticker": EntityExtractor.extract_ticker(user_query),
            "tickers": EntityExtractor.extract_tickers(user_query),
            "amount": EntityExtractor.extract_amount(user_query),
            "risk_level": EntityExtractor.extract_risk_level(user_query)
        }
        # try API resolution for ticker(s)
        if not entities["ticker"]:
            sym = resolve_ticker_from_text(user_query, self.api_keys)
            if sym:
                entities["ticker"] = sym
        tickers_list = entities.get("tickers") or []
        if entities["ticker"] and entities["ticker"] not in tickers_list:
            tickers_list.append(entities["ticker"])
        if (len(tickers_list) < 2 and re.search(r"\band\b|,|&", user_query, re.IGNORECASE)):
            multi_syms = resolve_multiple_tickers_from_text(user_query, self.api_keys)
            if multi_syms and len(multi_syms) >= 1:
                tickers_list = multi_syms
                entities["ticker"] = multi_syms[0]
        entities["tickers"] = tickers_list if tickers_list else None

        # rule-based scoring
        scores: Dict[str, int] = {}
        query_lower = user_query.lower()
        for agent_name, config in self.AGENT_CAPABILITIES.items():
            score = sum(1 for kw in config["keywords"] if kw in query_lower)
            if score > 0:
                scores[agent_name] = score

        if scores:
            best_agent = max(scores, key=scores.get)
            confidence = scores[best_agent] / max(1, len(self.AGENT_CAPABILITIES[best_agent]["keywords"]))
        else:
            best_agent, confidence = self._llm_intent_detection(user_query)

        required_fields = self.AGENT_CAPABILITIES[best_agent]["required"]
        missing = []
        for field in required_fields:
            if field == "tickers":
                if not entities.get("tickers") or len(entities.get("tickers", [])) < 2:
                    missing.append(field)
            elif not entities.get(field):
                missing.append(field)

        return {
            "agent": best_agent,
            "confidence": confidence,
            "entities": entities,
            "missing": missing,
            "clarification_needed": len(missing) > 0,
            "scores": scores
        }

    def _llm_intent_detection(self, user_query: str) -> Tuple[str, float]:
        """Fallback LLM intent detection (simple). Safe fallback if autogen fails."""
        prompt = f"""
        Analyze this financial query and determine which service is needed:

        Query: "{user_query}"

        Services:
        1. market_forecaster - price predictions, forecasts
        2. financial_report - company analysis, fundamentals
        3. risk_assessment - risk analysis, volatility
        4. portfolio_allocation - portfolio distribution
        5. trade_strategy - trading plans, entry/exit

        Respond with just: service_name confidence
        Example: market_forecaster 0.9
        """
        try:
            self.user_proxy.initiate_chat(self.assistant, message=prompt, clear_history=True)
            response = self.user_proxy.last_message().get("content", "").strip()
            parts = response.split()
            if len(parts) >= 2:
                return parts[0], float(parts[1])
        except Exception:
            pass
        return "financial_report", 0.5

    def get_clarification(self, intent: Dict) -> str:
        missing = intent["missing"]
        agent = intent["agent"]
        questions = []
        if "ticker" in missing:
            questions.append("Which company or stock ticker would you like to analyze? (e.g., AAPL, TSLA, MSFT)")
        if "tickers" in missing:
            questions.append("Which stocks would you like in your portfolio? (e.g., AAPL,MSFT,GOOGL,TSLA)")
        if "amount" in missing:
            if agent == "portfolio_allocation":
                questions.append("What is your total investment amount? (e.g., $50,000)")
            else:
                questions.append("How much are you planning to invest? (e.g., $10,000)")
        if "risk_level" in missing and agent in ["portfolio_allocation", "trade_strategy"]:
            questions.append("What is your risk tolerance? (Conservative/Moderate/Aggressive)")
        return "\n".join(f"❓ {q}" for q in questions)

    def apply_defaults(self, agent: str, entities: Dict) -> Dict:
        defaults = self.AGENT_CAPABILITIES[agent]["defaults"]
        completed = entities.copy()
        for key, default_value in defaults.items():
            if not completed.get(key):
                completed[key] = default_value
        return completed

    def route_to_agent(self, intent: Dict) -> Dict:
        agent = intent["agent"]
        entities = self.apply_defaults(agent, intent["entities"])
        print(f"\n✅ Routing to: {agent.replace('_', ' ').title()}")
        print(f"📊 Parameters: {entities}")
        return {
            "agent": agent,
            "parameters": entities,
            "description": self.AGENT_CAPABILITIES[agent]["description"]
        }


# ---------------------------
# FinRobotDirector: orchestrator
# ---------------------------
class FinRobotDirector:
    def __init__(self, api_keys: Dict):
        self.api_keys = api_keys
        llm_config = setup_director_llm_config(api_keys)
        self.director = DirectorAgent(llm_config, api_keys=api_keys)
        self.session_data = {}
        self.current_request: Dict = {}

    def _clean_charts_for_ticker(self, ticker: str):
        if not ticker:
            return

        base = os.path.dirname(os.path.abspath(__file__))
        charts_dir = os.path.join(base, "charts")

        if not os.path.exists(charts_dir):
            return

        for f in glob.glob(os.path.join(charts_dir, f"*{ticker}*.png")):
            try:
                os.remove(f)
            except Exception as e:
                print(f"⚠️ Could not remove chart: {f} → {e}")


    # DB logging helper (optional)
    def _log_db(self, phase: str, payload: Dict):
        if not log_to_db:
            return
        try:
            service = payload.get("service") or payload.get("agent")
            ticker = (
                payload.get("entities", {}).get("ticker")
                or payload.get("parameters", {}).get("ticker")
                or service
            )
            params = payload.get("parameters") or payload.get("entities") or {}
            output = payload.get("output") or {}
            log_to_db(
                agent_name="director_agent",
                ticker=ticker,
                params=params,
                output={"phase": phase, **output, "raw": payload},
            )
        except Exception as e:
            print(f"⚠️ DB Logging Failed (phase={phase}): {e}")

    def process_query(self, user_query: str) -> Dict:
        """
        Process a user query:
         - If user query doesn't clearly request a service (no service keywords),
           return status 'choose_service' with a list of available services so the
           frontend can show a popup and let the user pick.
         - Otherwise run normal intent detection & routing.
        """
        print("\n" + "=" * 60)
        print("🎯 FINROBOT DIRECTOR - PROCESSING REQUEST")
        print("=" * 60)

        # quick check: did user mention a service? (used for web/headless flows)
        # build keyword set from agent capabilities (lowercased)
        service_keywords = set()
        for cfg in DirectorAgent.AGENT_CAPABILITIES.values():
            for kw in cfg["keywords"]:
                service_keywords.add(kw.lower())

        query_lower = (user_query or "").lower()
        has_service_keyword = any(kw in query_lower for kw in service_keywords)

        # If no service keywords found, return a structured choose_service response
        # so web UI can show a popup. CLI interactive_session already handles this.
        if not has_service_keyword:
            # Build compact service list the frontend can render
            services = []
            for name, cfg in DirectorAgent.AGENT_CAPABILITIES.items():
                services.append({
                    "id": name,
                    "title": name.replace("_", " ").title(),
                    "description": cfg.get("description", ""),
                    "required": cfg.get("required", []),
                    "optional": cfg.get("optional", []),
                })

            # Log and return a response indicating selection needed
            self._log_db("choose_service_prompt", {
                "user_query": user_query,
                "available_services": [s["id"] for s in services]
            })

            return {
                "status": "choose_service",
                "message": "Which service would you like for this request?",
                "services": services
            }

        # Otherwise proceed as before
        intent = self.director.analyze_intent(user_query)

        self._log_db("intent_detection", {
            "user_query": user_query,
            "agent": intent.get("agent"),
            "confidence": intent.get("confidence"),
            "entities": intent.get("entities"),
            "missing": intent.get("missing"),
            "scores": intent.get("scores")
        })

        if intent["clarification_needed"]:
            clarification = self.director.get_clarification(intent)
            self._log_db("clarification_needed", {
                "user_query": user_query,
                "agent": intent.get("agent"),
                "missing": intent.get("missing"),
                "entities": intent.get("entities")
            })
            return {
                "status": "needs_clarification",
                "agent": intent["agent"],
                "message": clarification,
                "current_entities": intent["entities"],
                "missing": intent["missing"]
            }

        routing = self.director.route_to_agent(intent)
        self._log_db("routing", {
            "user_query": user_query,
            "routing": routing,
            "confidence": intent.get("confidence"),
            "entities": intent.get("entities")
        })
        return {"status": "ready", "routing": routing}


    def execute_agent(self, agent_name: str, parameters: Dict):
        print(f"\n🚀 Executing {agent_name}...")
        self._log_db("execution_started", {"agent": agent_name, "parameters": parameters})
        self.current_request = {
            "agent": agent_name,
            "parameters": parameters,
            "started_at": datetime.utcnow().isoformat()
        }
        try:
            # Clean charts for this run so glob only returns charts from this run
            ticker_for_cleanup = (parameters.get("ticker") or (parameters.get("tickers")[0] if parameters.get("tickers") else ""))
            if ticker_for_cleanup:
                self._clean_charts_for_ticker(ticker_for_cleanup)

            # Capture full agent run output (including tracebacks)
            full_output = capture_output(self._run_agent_internal, agent_name, parameters)
            final_output = self._extract_final_block(full_output)

            # Print clean final block to console
            if final_output:
                print("\n" + final_output + "\n")
            else:
                # If extraction failed, print everything captured for debugging
                print("\n<RAW AGENT OUTPUT>\n")
                print(full_output)

            self._log_db("execution_completed", {
                "agent": agent_name,
                "parameters": parameters,
                "output": {"status": "success", "final_output": final_output}
            })

            self.current_request["completed_at"] = datetime.utcnow().isoformat()
            self.current_request["final_output"] = final_output
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            self._log_db("execution_error", {
                "agent": agent_name,
                "parameters": parameters,
                "output": {"status": "error", "error": error_msg, "traceback": tb}
            })
            print(f"\n❌ Error while executing {agent_name}: {error_msg}")
            print(tb)
            self.current_request["completed_at"] = datetime.utcnow().isoformat()
            self.current_request["error"] = error_msg
            self.current_request["traceback"] = tb



    def _extract_final_block(self, full_output: str) -> str:
        """
        More robust extraction of the final formatted block.
        Priority:
         1) Known report headers (REPORT_KEYS)
         2) Last large separator (====, ###) or emoji headings
         3) Last N non-empty lines as a final fallback
        """
        if not full_output:
            return ""

        REPORT_KEYS = [
            "📈 MARKET FORECAST REPORT", "📈 EQUITY RESEARCH REPORT",
            "🛡  COMPREHENSIVE RISK ASSESSMENT REPORT", "📊 PORTFOLIO ALLOCATION REPORT",
            "📈 TRADE STRATEGY REPORT",
        ]

        # 1) Try explicit report headers
        for key in REPORT_KEYS:
            if key in full_output:
                try:
                    return key + full_output.split(key, 1)[1]
                except Exception:
                    pass

        # 2) Find last likely start (====, ### or emoji headings)
        lines = full_output.splitlines()
        final_start = None
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            if s.startswith("====") or s.startswith("###") or s.startswith("📈") or s.startswith("📉") or s.startswith("📊"):
                final_start = i
        if final_start is not None:
            return "\n".join(lines[final_start:]).strip()

        # 3) Fallback: return last N non-empty lines
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return ""
        return "\n".join(non_empty[-200:]).strip()

    def _run_agent_internal(self, agent_name: str, parameters: Dict):
        """Dispatch to the proper agent runner"""
        if agent_name == "market_forecaster":
            self._run_market_forecaster(parameters)
        elif agent_name == "financial_report":
            self._run_financial_report(parameters)
        elif agent_name == "risk_assessment":
            self._run_risk_assessment(parameters)
        elif agent_name == "portfolio_allocation":
            self._run_portfolio_allocation(parameters)
        elif agent_name == "trade_strategy":
            self._run_trade_strategy(parameters)
        else:
            print(f"❌ Unknown agent: {agent_name}")
            raise ValueError(f"Unknown agent: {agent_name}")

    # ---------------------------
    # Agent runners (real implementations)
    # Note: these import from your submodules and keep their original flows.
    # ---------------------------
    def _run_market_forecaster(self, params: Dict):
        try:
            from market_forecaster import (
                MarketForecasterAgent, FinancialDataRetriever,
                setup_llm_config_groq, headlines_sentiment_percent
            )

            groq_key = self.api_keys.get("GROQ_API_KEY", "")
            llm_config = setup_llm_config_groq(groq_key)
            data_retriever = FinancialDataRetriever(self.api_keys)
            forecaster = MarketForecasterAgent(llm_config, data_retriever)

            ticker = params["ticker"]
            print(f"\n📊 Analyzing {ticker}...")
            print("⏳ Gathering market data and running AI analysis...\n")

            result = forecaster.analyze(ticker)

            # sentiment
            try:
                news_items = data_retriever.get_company_news(ticker, days=7)
                computed_sentiment = headlines_sentiment_percent(news_items)
            except Exception:
                computed_sentiment = 50

            if "SENTIMENT:" not in result.get("analysis", "").upper():
                result["analysis"] = result.get("analysis", "") + f"\nSENTIMENT: {computed_sentiment}% Positive"

            forecaster.format_output(result)

            # DB summary
            self._log_db("agent_result", {
                "agent": "market_forecaster",
                "parameters": params,
                "output": {"summary": (result.get("summary") or result.get("analysis", "")[:200])}
            })

        except ImportError as e:
            print("❌ market_forecaster.py missing or import error:", e)
            self._log_db("agent_error", {"agent": "market_forecaster", "parameters": params, "output": {"error": str(e)}})
        except Exception as e:
            print("❌ Error running Market Forecaster:", e)
            traceback.print_exc()
            self._log_db("agent_error", {"agent": "market_forecaster", "parameters": params, "output": {"error": str(e)}})

    def _run_financial_report(self, params: Dict):
        try:
            from financial_report import (
                setup_llm_config_mistral, fetch_ticker_data, compute_metrics,
                generate_financial_charts, build_analysis_prompt,
                call_autogen_analysis, print_professional_report,
                analyze_trading_journal,
            )

            mistral_key = self.api_keys.get("MISTRAL_API_KEY", "")
            if not mistral_key:
                print("❌ MISTRAL_API_KEY required for financial reports")
                self._log_db("agent_error", {"agent": "financial_report", "parameters": params, "output": {"error": "Missing MISTRAL_API_KEY"}})
                return

            ticker = params["ticker"]
            period = params.get("period", "1y")
            journal_path = params.get("journal_path")

            print(f"\n📄 Generating comprehensive financial report for {ticker}...")
            print("⏳ Fetching data and running analysis...\n")

            data = fetch_ticker_data(ticker, period)
            metrics = compute_metrics(data)
            if not metrics.get("current_price"):
                print(f"❌ No price data available for {ticker}.")
                self._log_db("agent_error", {"agent": "financial_report", "parameters": params, "output": {"error": "No price data"}})
                return

            journal_summary = None
            if journal_path:
                try:
                    journal_summary = analyze_trading_journal(journal_path, symbol_filter=ticker)
                    if journal_summary:
                        print(f"✅ Found {journal_summary.get('total_trades', 0)} trades for {ticker} in journal")
                    else:
                        print("⚠️ No usable trades found for this symbol in journal")
                except Exception as e:
                    print("⚠️ Journal analysis error:", e)
                    journal_summary = None

            saved_charts = generate_financial_charts(ticker, data, metrics)
            print(f"✅ Charts saved to ./charts/\n")

            llm_config = setup_llm_config_mistral(api_key=mistral_key)
            prompt = build_analysis_prompt(ticker, metrics)
            print("🤖 Running AI-powered financial analysis...")
            ai_analysis = call_autogen_analysis(llm_config, prompt)
            print_professional_report(ticker, metrics, ai_analysis, saved_charts, journal_summary)

            self._log_db("agent_result", {"agent": "financial_report", "parameters": params, "output": {"summary": (ai_analysis or "")[:200]}})

        except ImportError as e:
            print("❌ financial_report.py missing or import error:", e)
            self._log_db("agent_error", {"agent": "financial_report", "parameters": params, "output": {"error": str(e)}})
        except Exception as e:
            print("❌ Error running Financial Report:", e)
            traceback.print_exc()
            self._log_db("agent_error", {"agent": "financial_report", "parameters": params, "output": {"error": str(e)}})

    def _run_risk_assessment(self, params: Dict):
        try:
            from risk_assessment import (
                setup_llm_config_hf, fetch_stock_data_and_news, build_prompt,
                call_autogen_risk_analysis, parse_llm_output, display_enhanced_risk_report,
                convert_label_to_val, plot_risk_gauge, plot_risk_pie, plot_volatility
            )

            # allow either HF_API_KEY (with or without 'hf_' prefix) OR AutoGen fallback
            hf_key = (self.api_keys.get("HF_API_KEY") or "").strip()

            if not hf_key:
    # If autogen library is available, use AutoGen multi-agent flow;
    # otherwise, require HF key and abort.
                if autogen:
                    print("⚠️ HF_API_KEY not found — using AutoGen multi-agent fallback for risk assessment.")
                else:
                    print("❌ Valid HF_API_KEY required for risk assessment (or install AutoGen).")
                    self._log_db("agent_error", {
                    "agent": "risk_assessment",
                    "parameters": params,
                    "output": {"error": "Missing HF_API_KEY and AutoGen not installed"}
                    })
                    return
            else:
    # Normalize HF keys that may or may not include the 'hf_' prefix.
    # downstream code (risk_assessment.py) expects an HF key; keep as-is.
    # no further action needed here
                pass


            ticker = params["ticker"]
            print(f"\n⚖️  Analyzing investment risk for {ticker}...")
            print("⏳ Gathering data and running risk analysis...\n")

            data = fetch_stock_data_and_news(ticker)
            ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
            llm_config = setup_llm_config_hf(api_key=hf_key)
            prompt = build_prompt(data, ticker)
            ok, result = call_autogen_risk_analysis(llm_config, prompt)
            if not ok:
                print("❌ LLM Error:", result)
                self._log_db("agent_error", {"agent": "risk_assessment", "parameters": params, "output": {"error": result}})
                return

            parsed = parse_llm_output(result)
            display_enhanced_risk_report(ticker, data, parsed, ist)

            score = parsed.get("score")
            if score is None:
                print("\n⚠️ Could not parse risk score from AI response.")
                self._log_db("agent_error", {"agent": "risk_assessment", "parameters": params, "output": {"error": "No risk score parsed", "raw": result[:200]}})
                return

            # Generate visual analytics
            m_val = convert_label_to_val(parsed.get("market"))
            f_val = convert_label_to_val(parsed.get("financial"))
            b_val = convert_label_to_val(parsed.get("business"))

            plot_risk_gauge(score, ticker)
            plot_risk_pie(m_val, f_val, b_val, ticker)
            plot_volatility(ticker)



            print("\n✅ All charts saved to ./charts/ directory")
            print("=" * 70 + "\n")

            self._log_db("agent_result", {"agent": "risk_assessment", "parameters": params, "output": {"score": parsed.get("score"), "summary": parsed.get("risk_meaning", "")}})

        except ImportError as e:
            print("❌ risk_assessment.py missing or import error:", e)
            self._log_db("agent_error", {"agent": "risk_assessment", "parameters": params, "output": {"error": str(e)}})
        except Exception as e:
            print("❌ Error running Risk Assessment:", e)
            traceback.print_exc()
            self._log_db("agent_error", {"agent": "risk_assessment", "parameters": params, "output": {"error": str(e)}})

    def _run_portfolio_allocation(self, params: Dict):
        try:
            from portfolio_allocation import (
                setup_llm_config_openrouter, PortfolioDataRetriever,
                PortfolioAllocationAgent
            )

            openrouter_key = self.api_keys.get("OPENROUTER_API_KEY", "")
            if not openrouter_key:
                print("❌ OPENROUTER_API_KEY required for portfolio allocation")
                self._log_db("agent_error", {"agent": "portfolio_allocation", "parameters": params, "output": {"error": "Missing OPENROUTER_API_KEY"}})
                return

            tickers = params.get("tickers", [])
            amount = params.get("amount", 0)
            risk_level = params.get("risk_level", "Moderate")

            if not tickers or len(tickers) < 2:
                print("❌ Portfolio allocation requires at least 2 stocks")
                self._log_db("agent_error", {"agent": "portfolio_allocation", "parameters": params, "output": {"error": "Insufficient tickers"}})
                return
            if amount <= 0:
                print("❌ Invalid investment amount")
                self._log_db("agent_error", {"agent": "portfolio_allocation", "parameters": params, "output": {"error": "Invalid amount"}})
                return

            print(f"\n📊 Creating portfolio allocation for: {', '.join(tickers)}")
            print(f"   Investment: ${amount:,.0f}")
            print(f"   Risk Profile: {risk_level}")
            print("⏳ Analyzing stocks and optimizing allocation...\n")

            #llm_config = setup_llm_config_openrouter(api_key=openrouter_key, model="meta-llama/llama-3.1-70b-instruct")
            llm_config = setup_llm_config_openrouter(
            api_key=openrouter_key,
            model="meta-llama/llama-3.1-70b-instruct"  # must exist on OpenRouter
            )
            print("⚡ DEBUG MODEL:", llm_config["config_list"][0]["model"])

            data_retriever = PortfolioDataRetriever(self.api_keys)
            allocator = PortfolioAllocationAgent(llm_config, data_retriever)
            result = allocator.create_allocation(tickers, amount, risk_level)
            allocator.format_output(result)

            self._log_db("agent_result", {"agent": "portfolio_allocation", "parameters": params, "output": {"allocation_preview": (result.get("allocation") or "")[:300]}})

        except ImportError as e:
            print("❌ portfolio_allocation.py missing or import error:", e)
            self._log_db("agent_error", {"agent": "portfolio_allocation", "parameters": params, "output": {"error": str(e)}})
        except Exception as e:
            print("❌ Error running Portfolio Allocation:", e)
            traceback.print_exc()
            self._log_db("agent_error", {"agent": "portfolio_allocation", "parameters": params, "output": {"error": str(e)}})

    def _run_trade_strategy(self, params: Dict):
        try:
            from trade_strategy import (
                fetch_ohlc, build_prompt, call_autogen, parse_enhanced_strategy,
                ensure_numbers, ensure_charts_dir, entry_exit_chart, risk_reward_chart,
                get_company_info, print_professional_report
            )

            groq_key = self.api_keys.get("GROQ_API_KEY", "").strip()
            ollama_key = self.api_keys.get("OLLAMA_API_KEY", "").strip()

            if groq_key:
                llm_config = {"config_list": [{"model": "llama-3.3-70b-versatile", "api_key": groq_key, "base_url": "https://api.groq.com/openai/v1", "api_type": "openai"}], "timeout": 120, "temperature": 0.2, "max_tokens": 800}
            elif ollama_key:
                ollama_base = self.api_keys.get("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
                ollama_model = self.api_keys.get("OLLAMA_MODEL", "llama3.1").strip()
                llm_config = {"config_list": [{"model": ollama_model, "api_key": ollama_key, "base_url": ollama_base, "api_type": "openai"}], "timeout": 120, "temperature": 0.2, "max_tokens": 800}
            else:
                print("❌ Need either GROQ_API_KEY or OLLAMA_API_KEY for trade strategy")
                self._log_db("agent_error", {"agent": "trade_strategy", "parameters": params, "output": {"error": "Missing LLM key (GROQ or OLLAMA)"}})
                return

            ticker = params["ticker"]
            amount = params.get("amount", 0)
            risk = params.get("risk_level", "medium")

            print(f"\n💼 Creating trade strategy for {ticker}")
            print(f"   Investment: ${amount:,.0f}")
            print(f"   Risk Level: {risk.capitalize()}")
            print("⏳ Fetching market data and generating strategy...\n")

            df = fetch_ohlc(ticker)
            company_info = get_company_info(ticker)
            prompt = build_prompt(ticker, amount, risk, df, company_info)
            raw = call_autogen(llm_config, prompt)

            parsed = parse_enhanced_strategy(raw)
            parsed = ensure_numbers(parsed, amount, df)

            charts_dir = ensure_charts_dir()
            charts = {
                "entry_exit": entry_exit_chart(df, parsed, ticker, charts_dir),
                "risk_reward": risk_reward_chart(parsed, ticker, charts_dir),
            }

            print_professional_report(ticker, amount, risk, parsed, charts, df, company_info, raw)
            self._log_db("agent_result", {"agent": "trade_strategy", "parameters": params, "output": {"entry_price": parsed.get("entry", {}).get("buy_price"), "targets": [parsed.get("exit", {}).get("t1"), parsed.get("exit", {}).get("t2")]}})

        except ImportError as e:
            print("❌ trade_strategy.py missing or import error:", e)
            self._log_db("agent_error", {"agent": "trade_strategy", "parameters": params, "output": {"error": str(e)}})
        except Exception as e:
            print("❌ Error running Trade Strategy:", e)
            traceback.print_exc()
            self._log_db("agent_error", {"agent": "trade_strategy", "parameters": params, "output": {"error": str(e)}})


# ---------------------------
# Interactive CLI
# ---------------------------
def interactive_session():
    print("\n" + "=" * 60)
    print("🤖 FINROBOT SMART DIRECTOR")
    print("=" * 60)
    print("\nAvailable Services:")
    print("  1. 📈 Market Forecaster - Stock price predictions")
    print("  2. 📄 Financial Report - Company analysis")
    print("  3. ⚖️  Risk Assessment - Investment risk analysis")
    print("  4. 📊 Portfolio Allocation - Asset distribution")
    print("  5. 💼 Trade Strategy - Entry/exit planning")
    print("=" * 60)

    api_keys = load_config_keys()
    if not api_keys:
        print("\n❌ No API keys found. Please create config_api_keys.json")
        return

    director = FinRobotDirector(api_keys)

    # Pre-build service keyword set for quick checks
    service_keywords = []
    for cfg in DirectorAgent.AGENT_CAPABILITIES.values():
        service_keywords.extend(cfg["keywords"])
    service_keywords = list(set(service_keywords))

    while True:
        print("\n")
        user_input = input("💬 Your request (or 'quit' to exit): ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Thank you for using FinRobot!")
            break
        if not user_input:
            continue

        lower = user_input.lower()
        quick_ticker = EntityExtractor.extract_ticker(user_input)
        has_for = bool(re.search(r"\bfor\b", lower))
        explicit_ticker_pattern = bool(re.search(r"\b[A-Z]{2,8}(?:\.[A-Z]{1,4})?\b", user_input))

        if not quick_ticker and not has_for and not explicit_ticker_pattern:
            print("\n⚠️  Your request doesn't clearly mention any company or stock symbol.")
            print("   Example: 'market forecast for Apple' or 'trade strategy for TSLA with 10000'\n")
            continue

        has_service_keyword = any(kw in lower for kw in service_keywords)
        if not has_service_keyword:
            print("\nIt looks like you didn't specify which type of analysis you want.")
            print("Please choose a service (1-5 or name):")
            print("  1. Market Forecaster")
            print("  2. Financial Report")
            print("  3. Risk Assessment")
            print("  4. Portfolio Allocation")
            print("  5. Trade Strategy")
            choice = input("\n🔢 Select a service (1-5 or name): ").strip().lower()
            if choice in ["1", "market", "forecaster", "market forecaster"]:
                user_input = f"market forecast for {user_input}"
            elif choice in ["2", "report", "financial", "financial report"]:
                user_input = f"financial report for {user_input}"
            elif choice in ["3", "risk", "risk assessment"]:
                user_input = f"risk assessment for {user_input}"
            elif choice in ["4", "portfolio", "allocation", "portfolio allocation"]:
                user_input = f"portfolio allocation for {user_input}"
            elif choice in ["5", "trade", "strategy", "trade strategy"]:
                user_input = f"trade strategy for {user_input}"
            else:
                print("\n⚠️  Invalid selection. Please type your request again more clearly.")
                continue

        result = director.process_query(user_input)
        if result["status"] == "needs_clarification":
            print("\n" + result["message"])
            completed_entities = result["current_entities"].copy()
            for missing_field in result["missing"]:
                if missing_field == "ticker":
                    ticker = input("\n🔤 Stock Ticker: ").strip().upper()
                    completed_entities["ticker"] = ticker
                elif missing_field == "tickers":
                    tickers_input = input("\n🔤 Stock Tickers (comma-separated, e.g., AAPL,MSFT): ").strip().upper()
                    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
                    if len(tickers) < 2:
                        print("⚠️  Portfolio allocation requires at least 2 stocks. Please try again.")
                        continue
                    completed_entities["tickers"] = tickers
                elif missing_field == "amount":
                    amount_str = input("\n💰 Investment Amount ($): ").strip()
                    try:
                        amount = float(amount_str.replace("$", "").replace(",", ""))
                        if amount <= 0:
                            raise ValueError
                        completed_entities["amount"] = amount
                    except Exception:
                        print("⚠️  Invalid amount. Using default: $10,000")
                        completed_entities["amount"] = 10000
                elif missing_field == "risk_level":
                    print("\nRisk Profiles:")
                    print("  1. Conservative")
                    print("  2. Moderate")
                    print("  3. Aggressive")
                    risk = input("\n⚖️  Risk Level (1-3 or name): ").strip()
                    if risk == "1" or risk.lower().startswith("con"):
                        completed_entities["risk_level"] = "Conservative"
                    elif risk == "2" or risk.lower().startswith("mod"):
                        completed_entities["risk_level"] = "Moderate"
                    elif risk == "3" or risk.lower().startswith("agg"):
                        completed_entities["risk_level"] = "Aggressive"
                    else:
                        completed_entities["risk_level"] = "Moderate"

            if result["agent"] == "financial_report":
                journal_path = input("\n📎 Optional: Trading journal CSV path for this stock (press Enter to skip): ").strip()
                if journal_path:
                    completed_entities["journal_path"] = journal_path

            completed_entities = director.director.apply_defaults(result["agent"], completed_entities)
            print("\n" + "=" * 60)
            print("✅ All required information collected!")
            print(f"📋 Final Parameters: {completed_entities}")
            print("=" * 60)

            director._log_db("clarification_completed", {"user_query": user_input, "agent": result["agent"], "final_parameters": completed_entities})
            confirm = input("\n✅ Proceed? (yes/no): ").strip().lower()
            if confirm in ["yes", "y", ""]:
                director.execute_agent(result["agent"], completed_entities)
            else:
                print("\n⏹️  Cancelled. No analysis was run.")
        else:
            routing = result["routing"]
            if routing["agent"] == "financial_report":
                journal_path = input("\n📎 Optional: Trading journal CSV path for this stock (press Enter to skip): ").strip()
                if journal_path:
                    routing["parameters"]["journal_path"] = journal_path
            print(f"\n📝 {routing['description']}")
            print(f"📋 Parameters: {routing['parameters']}")
            confirm = input("\n✅ Proceed? (yes/no): ").strip().lower()
            if confirm in ["yes", "y", ""]:
                director._log_db("user_confirmed_execution", {"user_query": user_input, "routing": routing})
                director.execute_agent(routing["agent"], routing["parameters"])
            else:
                print("\n⏹️  Cancelled. No analysis was run.")


# ---------------------------
# Headless helper: run once and return cleaned output + chart URLs
# ---------------------------
def clean_report(text: str) -> str:
    remove_keywords = [
        "TERMINATING RUN", "(to Director)", "(to User)", "User (to",
        "Director (to", "Auto-reply", "Analyzing your request", "Executing",
        "Routing to:", "AssistantAgent", "UserProxy", "Please wait"
    ]
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(key in stripped for key in remove_keywords):
            continue
        if all(ch in "━─│┌└═" for ch in stripped) and len(stripped) > 3:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


SECTION_TITLES = [
    "📌 Market Prediction:", "📌 Current Market Status:", "📌 Price Target Analysis:",
    "📌 Positive Developments:", "📌 Risk Factors & Concerns:", "📌 Technical Indicators:",
    "📌 Fundamental Health:", "📌 Sentiment Analysis:", "📌 Recommended Action Plan:",
]
DIV = "-" * 78


def auto_format_sections(text: str) -> str:
    for title in SECTION_TITLES:
        if title in text:
            text = text.replace(title, f"{title}\n{DIV}")
    return text


def run_query_once(user_query: str) -> dict:
    """
    Safe one-shot execution for the web API.
    Prevents KeyError('routing') and handles clarification gracefully.
    """
    api_keys = load_config_keys()
    if not api_keys:
        return {"status": "error", "message": "Missing API keys", "charts": []}

    director = FinRobotDirector(api_keys)
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        result = director.process_query(user_query)

        # 1️⃣ If clarification required → return immediately
        if result.get("status") == "needs_clarification":
            return {
                "status": "needs_clarification",
                "message": result.get("message", "More information required."),
                "charts": []
            }

        # 2️⃣ If routing missing → do not crash
        routing = result.get("routing")
        if not routing:
            return {
                "status": "error",
                "message": "I need more details to determine the correct financial service.",
                "charts": []
            }

        # 3️⃣ Execute agent safely ⚠️ THIS WAS MISSING THE EXECUTION!
        try:
            director.execute_agent(routing["agent"], routing["parameters"])
        except Exception as e:
            return {
                "status": "error",
                "message": f"Agent execution failed: {str(e)}",
                "charts": []
            }

    # 4️⃣ Capture clean output
    raw = buffer.getvalue()

    REPORT_KEYS = [
        "🛡  COMPREHENSIVE RISK ASSESSMENT REPORT",  # ⚠️ ADD THIS LINE
        "📈 MARKET FORECAST REPORT",
        "📈 EQUITY RESEARCH REPORT",
        "📉 RISK ASSESSMENT REPORT",
        "📊 PORTFOLIO ALLOCATION REPORT",
        "📈 TRADE STRATEGY REPORT",
    ]

    clean_output = None
    for key in REPORT_KEYS:
        if key in raw:
            clean_output = key + raw.split(key, 1)[1]
            break

    if not clean_output:
        # Fallback: return last meaningful section
        lines = raw.splitlines()
        non_empty = [l for l in lines if l.strip()]
        if non_empty:
            clean_output = "\n".join(non_empty[-200:])
        else:
            clean_output = raw

    clean_output = clean_report(clean_output)
    clean_output = auto_format_sections(clean_output).strip()

    # 5️⃣ Return charts
    ticker = routing["parameters"].get("ticker", "").upper()
    base = os.path.dirname(os.path.abspath(__file__))
    charts_dir = os.path.join(base, "charts")

    chart_files = (
    glob.glob(os.path.join(charts_dir, f"*{ticker}*.png"))
    if ticker else []
    )

    chart_urls = [f"/charts/{os.path.basename(c)}" for c in chart_files]

    return {
        "status": "ok",
        "message": clean_output,
        "charts": chart_urls
    }




# ---------------------------
# Entry point
# ---------------------------
def main():
    try:
        interactive_session()
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
