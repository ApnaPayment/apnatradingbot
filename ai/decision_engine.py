"""
AI Decision Engine
Uses the Anthropic SDK (with prompt caching) to reason over trade signals,
portfolio state, and market context before execution.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
import anthropic
from dotenv import load_dotenv
from core.risk_manager import TradeSignal

# Ensure .env is loaded regardless of working directory
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)

# System prompt cached across all calls — must be ≥1024 tokens to qualify for caching.
_SYSTEM_PROMPT = """
You are an expert Indian stock market trader, quantitative analyst, and risk manager with
20+ years of experience trading NSE/BSE equities, derivatives, and ETFs. You have deep
knowledge of:

MARKET MECHANICS
- NSE/BSE trading sessions, circuit breakers, SEBI regulations
- F&O expiry effects (every last Thursday of the month), rollover activity
- Operator activity patterns, bulk deals, institutional flows
- India VIX interpretation and its effect on option premiums and directional bias

TECHNICAL ANALYSIS
- Price action: support/resistance, candlestick patterns, trend structure
- Indicators: EMA crossovers, RSI divergence, MACD, Bollinger Bands, VWAP
- Volume analysis: delivery volume vs intraday volume, volume breakouts
- ATR-based stop loss and position sizing

RISK MANAGEMENT
- Position sizing using Kelly criterion and fixed-fractional methods
- Portfolio correlation and sector concentration
- Max drawdown controls, daily loss limits
- SEBI algo trading compliance — orders must pass exchange risk checks

MARKET CONTEXT
- RBI policy effect on banking and financial stocks
- FII/DII flows from NSDL/CDSL data
- Quarterly earnings season patterns
- Global cues: US markets, crude oil, USD/INR exchange rate

DECISION FRAMEWORK
When evaluating a trade signal:
1. Verify the technical setup is clean and unambiguous
2. Check portfolio context — avoid correlated positions, sector concentration
3. Assess macro headwinds/tailwinds relevant to the stock
4. Confirm risk:reward is acceptable (minimum 1:2)
5. Consider liquidity — can the quantity be filled without slippage?
6. Flag any upcoming events (results, ex-dividend, bonus record date) that could spike volatility

TOOL USE — STRICT RULES:
You have only 2 tools: get_live_quote and get_recent_candles.
- Call get_live_quote ONLY to check if the current price has moved significantly from the signal entry.
- Call get_recent_candles ONLY if the trend direction is unclear from the signal context already provided.
- Do NOT call any tool if the signal reasoning already contains the information you need.
- Portfolio data, P&L, sector, open positions are already provided in the prompt — do NOT try to fetch them via tools (those tools don't exist).
- Typically 0 or 1 tool calls are sufficient. 2 tool calls (quote + candles) is the maximum needed.

RESPONSE FORMAT — CRITICAL:
Your FINAL response (after any tool calls) MUST be ONLY a valid JSON object.
No preamble, no explanation text, no markdown fences, no ```json wrapper.
Start your response with { and end with }. Nothing else.
Example of correct final response:
{"approved": true, "confidence_adjustment": 0.05, "reasoning": "...", "suggested_stop_loss": 100.0, "suggested_target": 110.0, "concerns": []}

Never hallucinate price levels or corporate events. If uncertain, say so in your concerns.
"""


class AIDecisionEngine:
    """
    Uses Claude (via Anthropic SDK) with prompt caching for cost efficiency.
    The system prompt is cached — only the trade-specific user message is billed per call.

    Phase 3: evaluate_signal() uses Claude tool use — Claude can call live data
    lookups (live quote, recent candles, open positions, today's P&L) before
    deciding to approve or reject a trade.
    """

    MODEL = "claude-sonnet-4-6"
    # Max tool-call rounds per evaluation to prevent infinite loops
    MAX_TOOL_ROUNDS = 5

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set. AI decisions will be disabled.")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self._tool_executor = None   # Injected by AlgoTrader after init

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def set_tool_executor(self, executor) -> None:
        """Inject a ToolExecutor so evaluate_signal() can call live data tools."""
        self._tool_executor = executor

    def evaluate_signal(self, signal: TradeSignal, portfolio: dict,
                        market_context: dict = None,
                        news_sentiment: dict = None) -> dict:
        """
        Evaluate a trade signal using Claude tool use (Phase 3).

        Claude can call up to MAX_TOOL_ROUNDS live data tools before deciding:
          - get_live_quote       → verify current price / spread
          - get_recent_candles   → check trend direction and support/resistance
          - get_open_positions   → check sector concentration
          - get_today_pnl        → check daily loss budget
          - get_symbol_sector    → detect sector overlap

        Returns:
        {
            "approved":             bool,
            "confidence_adjustment": float,   # -0.2 to +0.2
            "reasoning":            str,
            "suggested_stop_loss":  float | None,
            "suggested_target":     float | None,
            "concerns":             list[str],
            "tools_used":           list[str],   # which tools Claude called
        }
        """
        if not self.client:
            return self._passthrough(signal)

        prompt = self._build_evaluation_prompt(signal, portfolio, market_context, news_sentiment)

        # Use tool-use mode when a ToolExecutor is available
        if self._tool_executor is not None:
            try:
                return self._evaluate_with_tools(prompt, signal)
            except Exception as e:
                logger.error(f"Tool-use evaluation failed, falling back to plain: {e}")
                # Fall through to plain call

        try:
            text = self._call_claude(prompt)
            return self._parse_evaluation(text, signal)
        except Exception as e:
            logger.error(f"AI evaluation failed: {e}")
            return self._reject_on_failure(signal, str(e))

    def _evaluate_with_tools(self, prompt: str, signal: TradeSignal) -> dict:
        """
        Agentic evaluation loop:
        1. Send evaluation prompt + tool schemas to Claude
        2. If Claude calls tools → execute them → feed results back
        3. Repeat until Claude returns a final text response (no more tool calls)
        4. Parse the final JSON from Claude's text

        Claude's final message MUST be JSON in the same format as the plain eval.
        """
        messages = [{"role": "user", "content": prompt}]
        tools_used: list[str] = []

        for round_num in range(self.MAX_TOOL_ROUNDS):
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=2048,
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=self._tool_executor.tool_schemas(),
                messages=messages,
            )

            # Collect any text content blocks for later
            text_blocks = [
                b.text for b in response.content
                if hasattr(b, "text") and b.text
            ]

            # If no tool calls → Claude is done; parse the JSON from text
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                final_text = "\n".join(text_blocks)
                result = self._parse_evaluation(final_text, signal)
                result["tools_used"] = tools_used
                return result

            # Execute every tool Claude requested
            tool_results = []
            for block in tool_use_blocks:
                tool_name  = block.name
                tool_input = block.input
                tools_used.append(tool_name)
                logger.info(
                    f"Claude called tool '{tool_name}' "
                    f"with {json.dumps(tool_input)[:80]}"
                )
                result_data = self._tool_executor.run(tool_name, tool_input)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result_data),
                })

            # Append Claude's response + tool results to the conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})

        # Exhausted rounds — do a final plain call with a nudge
        logger.warning(
            f"Tool-use hit max rounds ({self.MAX_TOOL_ROUNDS}). "
            "Asking Claude to conclude."
        )
        messages.append({
            "role": "user",
            "content": (
                "You have used the maximum number of tool calls. "
                "Based on everything you have seen, respond NOW with your final "
                "JSON evaluation (approved, confidence_adjustment, reasoning, "
                "suggested_stop_loss, suggested_target, concerns)."
            )
        })
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        )
        text_blocks = [
            b.text for b in response.content if hasattr(b, "text") and b.text
        ]
        result = self._parse_evaluation("\n".join(text_blocks), signal)
        result["tools_used"] = tools_used
        return result

    def get_portfolio_advice(self, portfolio: dict, market_context: dict = None) -> dict:
        """
        End-of-day portfolio advice.
        Returns a structured dict with keys:
          portfolio_health, key_risks, tomorrow_focus, summary
        (never raw JSON string — always parsed)
        """
        if not self.client:
            return {}

        prompt = self._build_portfolio_prompt(portfolio, market_context)
        try:
            result = self._call_claude_json(prompt)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Portfolio advice failed: {e}")
            return {}

    def explain_trade(self, signal: TradeSignal, outcome: dict = None) -> str:
        """Plain-English trade explanation for Telegram / dashboard."""
        if not self.client:
            return signal.reasoning

        prompt = self._build_explanation_prompt(signal, outcome)
        try:
            return self._call_claude(prompt)
        except Exception as e:
            logger.error(f"Trade explanation failed: {e}")
            return signal.reasoning

    def score_news_sentiment(self, symbol: str, announcements: list[str]) -> dict:
        """
        Score recent corporate announcements for a symbol.

        Returns:
        {
            "score":    float,    # -1.0 (very bearish) to +1.0 (very bullish)
            "label":   "positive" | "negative" | "neutral",
            "reason":  str,       # one sentence
            "flags":   list[str], # e.g. ["earnings_beat", "dividend", "rights_issue"]
        }
        """
        if not self.client or not announcements:
            return {"score": 0.0, "label": "neutral", "reason": "No news", "flags": []}

        ann_text = "\n".join(f"  - {a}" for a in announcements[:5])
        prompt = f"""Score the sentiment of these NSE corporate announcements for {symbol}.

RECENT ANNOUNCEMENTS:
{ann_text}

Consider:
- Earnings: beat/miss/in-line with estimates?
- Dividends, buybacks, splits: generally positive
- Rights issue, QIP, FPO: dilutive, often negative short-term
- Regulatory action, SEBI notices: negative
- Board changes, acquisitions: depends on context
- Results date announcements: neutral (just a date)

Respond in JSON only (no markdown):
{{
    "score":  -1.0 to 1.0,
    "label":  "positive" | "negative" | "neutral",
    "reason": "one sentence summary",
    "flags":  ["tag1", "tag2"]
}}"""

        try:
            result = self._call_claude_json(prompt)
            result.setdefault("score",  0.0)
            result.setdefault("label",  "neutral")
            result.setdefault("reason", "")
            result.setdefault("flags",  [])
            # Clamp score
            result["score"] = max(-1.0, min(1.0, float(result["score"])))
            return result
        except Exception as e:
            logger.error(f"Sentiment scoring failed for {symbol}: {e}")
            return {"score": 0.0, "label": "neutral", "reason": "AI error", "flags": []}

    def morning_market_brief(self, headlines: list = None,
                              vix: float = None,
                              fii_net: float = None,
                              dii_net: float = None) -> dict:
        """
        Pre-market AI brief run at 8:45 AM before the first trading cycle.

        Returns:
        {
            "avoid_trading": bool,       # True = skip all trades today
            "caution_level": "low" | "medium" | "high",
            "reason": str,               # 1–2 sentence explanation
            "suggestion": str,           # What to do / watch
            "bias": "bullish" | "bearish" | "neutral",
        }
        """
        if not self.client:
            return {
                "avoid_trading": False,
                "caution_level": "low",
                "reason": "AI not configured",
                "suggestion": "Proceed normally",
                "bias": "neutral",
            }

        headlines_text = "\n".join(f"- {h}" for h in (headlines or [])) or "None available"
        fii_str  = f"₹{fii_net:,.0f} cr" if fii_net is not None else "N/A"
        dii_str  = f"₹{dii_net:,.0f} cr" if dii_net is not None else "N/A"
        vix_str  = f"{vix:.1f}" if vix is not None else "N/A"

        prompt = f"""You are conducting a pre-market analysis for Indian equity trading (NSE).
Today's date: {__import__('datetime').date.today().strftime('%d %b %Y')}

PRE-MARKET DATA:
- India VIX: {vix_str}
- FII net buy/sell (prev day): {fii_str}
- DII net buy/sell (prev day): {dii_str}

RECENT HEADLINES:
{headlines_text}

Based on this data, should the automated trading bot proceed normally today?

Respond in JSON only (no markdown):
{{
    "avoid_trading": true | false,
    "caution_level": "low" | "medium" | "high",
    "reason": "1-2 sentence explanation of the market environment",
    "suggestion": "What the bot should do or watch out for today",
    "bias": "bullish" | "bearish" | "neutral"
}}

Only set avoid_trading=true for genuinely extreme conditions: VIX > 25, major geopolitical shock,
RBI emergency action, market-wide circuit breaker risk. Normal volatility is not a reason to avoid."""

        try:
            result = self._call_claude_json(prompt)
            result.setdefault("avoid_trading", False)
            result.setdefault("caution_level", "low")
            result.setdefault("reason", "")
            result.setdefault("suggestion", "Proceed normally")
            result.setdefault("bias", "neutral")
            return result
        except Exception as e:
            logger.error(f"Morning brief failed: {e}")
            return {
                "avoid_trading": False,
                "caution_level": "low",
                "reason": f"AI brief unavailable: {e}",
                "suggestion": "Proceed normally",
                "bias": "neutral",
            }

    def detect_market_regime(self, nifty_data: dict, vix: float = None) -> dict:
        """
        Classify market regime: trending_up | trending_down | ranging |
        high_volatility | breakout
        """
        if not self.client:
            return {"regime": "unknown", "confidence": 0,
                    "suggestion": "Set ANTHROPIC_API_KEY", "risk_level": "medium"}

        prompt = f"""Analyze the current Indian stock market and classify the regime.

Nifty 50:
- Current price: {nifty_data.get('ltp', 'N/A')}
- Change today:  {nifty_data.get('change_pct', 'N/A')}%
- 52-week high:  {nifty_data.get('52w_high', 'N/A')}
- India VIX:     {vix or 'N/A'}

Respond in JSON only (no markdown):
{{
    "regime": "trending_up" | "trending_down" | "ranging" | "high_volatility" | "breakout",
    "confidence": 0.0-1.0,
    "suggestion": "one sentence trading suggestion",
    "risk_level": "low" | "medium" | "high"
}}"""

        try:
            return self._call_claude_json(prompt)
        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            return {"regime": "unknown", "confidence": 0,
                    "suggestion": "API error", "risk_level": "medium"}

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_evaluation_prompt(self, signal: TradeSignal, portfolio: dict,
                                  market_context: dict = None,
                                  news_sentiment: dict = None) -> str:
        _default_news = {"score": 0, "label": "neutral", "reason": "No news fetched"}
        news_sentiment = news_sentiment or _default_news
        # ── Basic portfolio snapshot ──────────────────────────────────────────
        basic = {
            "open_positions":  list(portfolio.get("open_positions", {}).keys()),
            "position_count":  portfolio.get("position_count", 0),
            "daily_pnl":       portfolio.get("daily_pnl", 0),
            "capital_at_risk": portfolio.get("capital_at_risk", 0),
        }

        # ── Rich portfolio context (Phase 5) ─────────────────────────────────
        rich = (market_context or {}).get("portfolio", {}) if market_context else {}
        portfolio_section = ""
        if rich:
            sec_alloc = rich.get("sector_allocation", {})
            sec_lines = "\n".join(
                f"    {sec}: {d['pct']:.0f}%  ({', '.join(d['symbols'])})"
                for sec, d in sec_alloc.items()
            ) or "    (empty portfolio)"
            corr_pairs = rich.get("high_correlation_pairs", [])
            corr_lines = "\n".join(
                f"    {p['a']} ↔ {p['b']}  corr={p['corr']:.2f}"
                for p in corr_pairs
            ) or "    none above 0.65"
            conc_warns = "\n".join(
                f"  ⚠ {w}" for w in rich.get("concentration_warnings", [])
            ) or "  none"
            same_sec   = rich.get("same_sector_already_held", [])
            new_sector = rich.get("incoming_signal_sector", "Unknown")

            portfolio_section = f"""
PORTFOLIO HEALTH:
- Open positions ({rich.get('open_position_count', 0)}): {rich.get('open_symbols', [])}
- Today P&L: ₹{rich.get('daily_pnl', 0):,.0f}  |  Loss budget used: {rich.get('loss_budget_used_pct', 0):.0f}%  |  Remaining: ₹{rich.get('loss_budget_remaining', 0):,.0f}
- Recent win rate (last 10): {rich.get('recent_win_rate', 0.5):.0%}
- Win streak: {rich.get('current_win_streak', 0)}  |  Lose streak: {rich.get('current_lose_streak', 0)}

SECTOR ALLOCATION:
{sec_lines}
  → Incoming signal sector: {new_sector}
  → Already held in same sector: {same_sec if same_sec else 'none'}

CONCENTRATION WARNINGS:
{conc_warns}

CORRELATION (pairs ≥0.65):
{corr_lines}"""
        else:
            portfolio_section = f"\nCURRENT PORTFOLIO:\n{json.dumps(basic, indent=2)}"

        # Build trade-type-aware risk:reward display
        _is_short_option = (
            getattr(signal, "action", "BUY") == "SELL"
            and getattr(signal, "product", "CNC") == "NRML"
        )
        if _is_short_option:
            # Short option: profit = credit collected when premium DECAYS
            # premium received = signal.price; target = buy back cheap; stop = buy back at 2×
            _premium   = signal.price
            _profit    = _premium - signal.target   # e.g. 176.40 - 52.92 = 123.48 (70% of premium)
            _risk      = signal.stop_loss - _premium # e.g. 352.80 - 176.40 = 176.40 (100% of premium)
            _rr_str    = (
                f"Credit received: ₹{_premium:.2f}/unit. "
                f"Max profit if premium decays to ₹{signal.target:.2f}: ₹{_profit:.2f}/unit ({_profit/_premium*100:.0f}% of credit). "
                f"Max loss if premium rises to ₹{signal.stop_loss:.2f}: ₹{_risk:.2f}/unit ({_risk/_premium*100:.0f}% of credit). "
                f"This is a PREMIUM SELLING strategy — profit comes from TIME DECAY (theta), not directional movement. "
                f"Win rate is typically 60–70%+ for premium selling; a 2:1 R:R requirement does NOT apply. "
                f"The position has a defined stop loss (not naked/unlimited risk)."
            )
        else:
            _rr = abs(signal.target - signal.price) / max(abs(signal.price - signal.stop_loss), 0.01)
            _rr_str = (
                f"{_rr:.1f}:1 "
                f"{'✓' if _rr >= 2 else '✗ below 2:1'}"
            )

        return f"""You are evaluating a trade signal as a portfolio manager — not just a signal approver.
Your job is to decide whether this trade FITS the current portfolio, not just whether the signal is technically valid.

TRADE SIGNAL:
- Symbol:     {signal.symbol}
- Action:     {signal.action}  {'← OPTION WRITING (short option, collect premium upfront)' if _is_short_option else ''}
- Entry:      ₹{signal.price}  {'(premium received/collected)' if _is_short_option else ''}
- Stop Loss:  ₹{signal.stop_loss}  {'(buy back if premium RISES to this level — defined max loss, NOT naked/unlimited)' if _is_short_option else ''}
- Target:     ₹{signal.target}  {'(buy back when premium FALLS to this level — profit from decay)' if _is_short_option else ''}
- Risk/Reward: {_rr_str}
- Strategy:   {signal.strategy}
- Confidence: {signal.confidence:.0%}
- Reasoning:  {signal.reasoning}
{portfolio_section}

MARKET CONTEXT:
{json.dumps({k: v for k, v in (market_context or {}).items() if k != 'portfolio'}, indent=2)}

NEWS SENTIMENT FOR {signal.symbol}:
{json.dumps(news_sentiment, indent=2)}

ML ENSEMBLE PREDICTION:
{self._format_ml_prediction((market_context or {}).get("ml_prediction"))}

POSITION SIZING (Kelly):
{self._format_kelly((market_context or {}).get("kelly"))}

MULTI-TIMEFRAME ALIGNMENT:
{self._format_mtf((market_context or {}).get("mtf"))}

F&O EXPIRY & ECONOMIC CALENDAR:
{self._format_calendar((market_context or {}).get("calendar"))}

EVALUATION CHECKLIST (address each):
1. Technical setup — is the signal clean and timely? Does it align with the multi-timeframe picture?
2. Portfolio fit — does adding this position increase correlation or sector concentration?
3. Risk budget — given today's P&L and loss budget remaining, is this appropriate?
4. News/macro — any near-term event risk for this stock?
5. Lose streak caution — if on a losing streak, apply extra scrutiny.
6. ML prediction — if ML shows high-confidence caution, weight that appropriately.
7. Calendar risk — expiry day/week and upcoming events often cause sharp reversals; adjust accordingly.
{"8. Option writing specific: evaluate whether the underlying will stay BELOW (for CE) or ABOVE (for PE) the strike by expiry. The stop loss is defined — this is NOT a naked short. Do NOT require 2:1 R:R for short options." if _is_short_option else ""}

Tools available: get_live_quote (verify price hasn't moved), get_recent_candles (verify trend if unclear).
Only call a tool if that specific data is NOT already provided above. Portfolio/P&L/sector data is already in this prompt — do not call tools for it.

Respond in JSON only (no markdown):
{{
    "approved": true | false,
    "confidence_adjustment": -0.2 to 0.2,
    "reasoning": "2-3 sentence explanation addressing portfolio fit specifically",
    "suggested_stop_loss": price or null,
    "suggested_target": price or null,
    "concerns": ["concern 1", "concern 2"]
}}"""

    @staticmethod
    def _format_kelly(kelly: dict | None) -> str:
        if not kelly or kelly.get("source") == "fallback":
            return "Fallback sizing (insufficient trade history for Kelly — need 15+ completed trades)"
        src = kelly.get("source", "?")
        return (
            f"Multiplier={kelly['multiplier']:.2f}  "
            f"(raw Kelly={kelly['kelly_raw']:.2f} × ¼-Kelly fraction)  "
            f"win_prob={kelly['win_prob']:.0%}  payoff={kelly['payoff']:.2f}  "
            f"n_trades={kelly['n_trades']}  source={src}"
        )

    @staticmethod
    def _format_mtf(mtf: dict | None) -> str:
        if not mtf:
            return "Not available (insufficient OHLCV history)"
        emoji = {"aligned": "🟢", "partial": "🟡", "against": "🔴"}.get(mtf.get("verdict", ""), "⚪")
        lines = [
            f"{emoji} Verdict: {mtf.get('verdict','?').upper()}  "
            f"score={mtf.get('score', 0):.2f}  "
            f"conf_adj={mtf.get('confidence_adjustment', 0):+.2f}",
        ]
        for label, key in [("5-min (base)", "tf5"), ("15-min", "tf15"), ("30-min", "tf30")]:
            tf = mtf.get(key, {})
            aligned = tf.get("aligned", "?")
            reason  = tf.get("reason", "")
            lines.append(f"  {label}: {aligned}  — {reason}")
        return "\n".join(lines)

    @staticmethod
    def _format_calendar(cal: dict | None) -> str:
        if not cal:
            return "Not available"
        dte = cal.get("days_to_expiry")
        dte_str = "TODAY" if dte == 0 else f"in {dte} day(s)"
        lines = [
            f"Expiry {dte_str}  ({cal.get('next_expiry', '')})",
            f"Calendar risk level: {cal.get('risk_level', 'normal').upper()}  "
            f"| Size multiplier applied: {cal.get('size_multiplier', 1.0):.0%}",
        ]
        for ev in cal.get("economic_events_14d", []):
            d = ev.get("days_away", "?")
            lines.append(f"⚡ {ev.get('label', '')} — in {d} day(s) [{ev.get('impact','').upper()}]")
        for note in cal.get("trading_notes", []):
            lines.append(f"⚠ {note}")
        return "\n".join(lines)

    @staticmethod
    def _format_ml_prediction(ml_pred: dict | None) -> str:
        if not ml_pred:
            return "Not available (model not yet trained — need ≥30 labelled decisions)"
        rec_emoji = {"proceed": "🟢", "caution": "🟡", "skip": "🔴"}.get(
            ml_pred.get("recommendation", ""), "⚪"
        )
        auc_str = f"  AUC={ml_pred['auc']:.3f}" if ml_pred.get("auc") else ""
        return (
            f"{rec_emoji} Win probability: {ml_pred['win_prob']:.0%}  "
            f"(confidence: {ml_pred['ml_confidence']},  "
            f"recommendation: {ml_pred['recommendation']},  "
            f"trained on {ml_pred['n_samples']} samples{auc_str})"
        )

    def _build_portfolio_prompt(self, portfolio: dict, market_context: dict = None) -> str:
        return f"""Analyze this Indian stock market portfolio and give end-of-day advice.

PORTFOLIO STATE:
{json.dumps(portfolio, indent=2)}

MARKET CONTEXT:
{json.dumps(market_context or {}, indent=2)}

Respond in JSON only (no markdown, no preamble). Use this exact structure:
{{
  "portfolio_health": "One sentence — POSITIVE / NEUTRAL / NEGATIVE and why",
  "key_risks": ["Risk 1 (max 12 words)", "Risk 2", "Risk 3"],
  "tomorrow_focus": ["Action 1 (max 12 words)", "Action 2", "Action 3"],
  "summary": "One sentence closing summary — what the bot should do tomorrow"
}}

Rules:
- Each list has 2-4 items maximum
- Use plain English, no JSON keys visible to user
- Be specific to Indian markets (NSE, FII flows, RBI, VIX)
- If portfolio is empty, give generic next-session preparation advice"""

    def _build_explanation_prompt(self, signal: TradeSignal, outcome: dict = None) -> str:
        base = (
            f"Explain this trade in simple English for a retail investor.\n\n"
            f"TRADE: {signal.action} {signal.symbol} at ₹{signal.price}\n"
            f"Strategy: {signal.strategy}\n"
            f"Technical reasoning: {signal.reasoning}\n"
            f"Stop Loss: ₹{signal.stop_loss} | Target: ₹{signal.target}\n"
        )
        if outcome:
            base += (
                f"\nOUTCOME:\n"
                f"- Exit price: ₹{outcome.get('exit_price', 'N/A')}\n"
                f"- P&L: ₹{outcome.get('pnl', 'N/A')}\n"
                f"- Exit reason: {outcome.get('exit_reason', 'N/A')}\n"
                f"Explain what happened and what we can learn.\n"
            )
        else:
            base += "\nExplain why this trade was taken and what we're expecting.\n"
        base += "\nKeep it under 100 words. Simple language, no jargon."
        return base

    # ─────────────────────────────────────────────────────────────────────────
    # Anthropic SDK call (system prompt cached)
    # ─────────────────────────────────────────────────────────────────────────

    def _call_claude_json(self, user_prompt: str) -> dict:
        """Call Claude and parse the JSON response, stripping markdown fences if present."""
        raw = self._call_claude(user_prompt).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def _call_claude(self, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_evaluation(self, text: str, signal: TradeSignal) -> dict:
        """
        Extract JSON from Claude's response. Handles three formats:
        1. Pure JSON: {...}
        2. Fence at start: ```json\n{...}\n```
        3. Text then fence anywhere: "Analysed.\n```json\n{...}\n```"
        """
        import re as _re
        raw = text.strip()

        # Strategy 1: try parsing the whole response directly
        try:
            result = json.loads(raw)
            result.setdefault("approved", False)
            result.setdefault("confidence_adjustment", 0.0)
            result.setdefault("reasoning", "AI evaluation complete")
            result.setdefault("suggested_stop_loss", signal.stop_loss)
            result.setdefault("suggested_target", signal.target)
            result.setdefault("concerns", [])
            return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract first ```json ... ``` block anywhere in the text
        fence_match = _re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, _re.DOTALL)
        if fence_match:
            try:
                result = json.loads(fence_match.group(1))
                result.setdefault("approved", False)
                result.setdefault("confidence_adjustment", 0.0)
                result.setdefault("reasoning", "AI evaluation complete")
                result.setdefault("suggested_stop_loss", signal.stop_loss)
                result.setdefault("suggested_target", signal.target)
                result.setdefault("concerns", [])
                return result
            except json.JSONDecodeError:
                pass

        # Strategy 3: extract first bare {...} JSON object in the text
        brace_match = _re.search(r'\{[^{}]*"approved"[^{}]*\}', raw, _re.DOTALL)
        if brace_match:
            try:
                result = json.loads(brace_match.group(0))
                result.setdefault("approved", False)
                result.setdefault("confidence_adjustment", 0.0)
                result.setdefault("reasoning", "AI evaluation complete")
                result.setdefault("suggested_stop_loss", signal.stop_loss)
                result.setdefault("suggested_target", signal.target)
                result.setdefault("concerns", [])
                return result
            except json.JSONDecodeError:
                pass

        # All strategies failed — extract reasoning from plain text and fail closed
        logger.warning(f"AI returned unparseable response for {signal.symbol} — failing closed.")
        result = {
            "approved": False,
            "confidence_adjustment": 0.0,
            "reasoning": raw[:400],   # use the actual AI text as reasoning so it's visible
            "concerns": ["AI response format unrecognised — trade rejected for safety"],
        }

        result.setdefault("approved", False)  # fail-closed: reject if field missing
        result.setdefault("confidence_adjustment", 0.0)
        result.setdefault("reasoning", "AI evaluation complete")
        result.setdefault("suggested_stop_loss", signal.stop_loss)
        result.setdefault("suggested_target", signal.target)
        result.setdefault("concerns", [])
        return result

    def _passthrough(self, signal: TradeSignal) -> dict:
        """Only called when ANTHROPIC_API_KEY is not configured at all (self.client is None)."""
        return {
            "approved": True,
            "confidence_adjustment": 0.0,
            "reasoning": "AI not configured — passing signal through as-is",
            "suggested_stop_loss": signal.stop_loss,
            "suggested_target": signal.target,
            "concerns": ["ANTHROPIC_API_KEY not set — configure key to enable AI veto"],
        }

    def _reject_on_failure(self, signal: TradeSignal, reason: str) -> dict:
        """
        Called when AI IS configured but the API call failed (529 overload, timeout, etc.).
        Rejects the signal for safety — never auto-approve when the veto layer is down.
        The signal will be retried naturally in the next 5-minute cycle.
        """
        logger.warning(
            f"AI temporarily unavailable ({reason[:80]}) — "
            f"REJECTING {signal.symbol} for safety. Will retry next cycle."
        )
        return {
            "approved": False,
            "confidence_adjustment": 0.0,
            "reasoning": (
                f"AI veto layer temporarily unavailable ({reason[:120]}). "
                "Signal rejected for safety — will be re-evaluated next cycle."
            ),
            "suggested_stop_loss": signal.stop_loss,
            "suggested_target": signal.target,
            "concerns": [f"AI service error: {reason[:120]}"],
        }
