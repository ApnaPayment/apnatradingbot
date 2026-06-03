"""
Phase 8 — F&O Expiry & Economic Calendar
Provides expiry proximity, RBI MPC dates, budget dates, and earnings windows
so the bot can reduce risk and inform AI reasoning around high-volatility events.

NSE/BSE expiry rules (as of 2025):
  - NIFTY       : weekly every Tuesday (NSE changed from Thursday, Jan 2025)
  - FINNIFTY    : weekly every Tuesday
  - MIDCPNIFTY  : weekly every Tuesday
  - BANKNIFTY   : monthly last Wednesday (NSE changed from Thursday, 2024)
  - SENSEX      : weekly every Thursday (BSE)

The DEFAULT expiry used for generic risk calculations (is_expiry_week, etc.)
is NIFTY (Tuesday). For underlying-specific DTE use days_to_fo_expiry(underlying).

All dates are in IST.  No external API needed — logic is computed or hardcoded
from official NSE / RBI calendars updated annually.
"""

from datetime import date, datetime, timedelta
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Per-underlying expiry configuration
# Weekday numbers: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday
# ─────────────────────────────────────────────────────────────────────────────
_FO_EXPIRY_WEEKDAY: dict[str, int] = {
    "NIFTY":      1,   # Tuesday (NSE, changed Jan 2025)
    "FINNIFTY":   1,   # Tuesday
    "MIDCPNIFTY": 1,   # Tuesday
    "BANKNIFTY":  2,   # Wednesday (NSE, changed 2024)
    "SENSEX":     3,   # Thursday (BSE weekly)
}

# Underlyings with monthly-only options (no weekly contracts in Kotak scrip master)
_MONTHLY_ONLY_FO: set[str] = {"BANKNIFTY"}

# Default underlying for generic expiry functions (is_expiry_week, days_to_expiry, etc.)
_DEFAULT_UNDERLYING = "NIFTY"

# ─────────────────────────────────────────────────────────────────────────────
# RBI Monetary Policy Committee meeting dates (decision day)
# Source: RBI calendar. Update each April for the new fiscal year.
# ─────────────────────────────────────────────────────────────────────────────
RBI_MPC_DATES: list[date] = [
    # FY 2025-26
    date(2025, 4, 9),
    date(2025, 6, 6),
    date(2025, 8, 6),
    date(2025, 10, 8),
    date(2025, 12, 5),
    date(2026, 2, 6),
    # FY 2026-27
    date(2026, 4, 7),
    date(2026, 6, 5),
    date(2026, 8, 5),
    date(2026, 10, 7),
    date(2026, 12, 4),
    date(2027, 2, 5),
]

# Union Budget presentation (usually last day of January or first day of February)
BUDGET_DATES: list[date] = [
    date(2025, 2, 1),
    date(2026, 2, 1),   # placeholder — confirm when announced
]

# ─────────────────────────────────────────────────────────────────────────────
# Expiry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of `weekday` (0=Mon…6=Sun) in a given month."""
    import calendar as _cal
    last_day = date(year, month, _cal.monthrange(year, month)[1])
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of a given month (kept for backward compat)."""
    return _last_weekday_of_month(year, month, 3)


def _next_weekday(from_date: date, weekday: int) -> date:
    """Return the next date on or after from_date that falls on `weekday` (0=Mon…6=Sun)."""
    days_ahead = (weekday - from_date.weekday()) % 7
    return from_date + timedelta(days=days_ahead)


def _next_thursday(from_date: date) -> date:
    """Return the next Thursday on or after from_date (kept for backward compat)."""
    return _next_weekday(from_date, 3)


def get_fo_expiry(underlying: str, from_date: Optional[date] = None,
                  min_days: int = 0) -> Optional[date]:
    """
    Return the nearest F&O expiry for a specific underlying that is at least
    `min_days` calendar days away.

    This is the single source of truth for expiry dates — used by both the
    calendar risk context AND the options strategy, so they always agree.

    Args:
        underlying: "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"
        from_date:  Reference date (defaults to today).
        min_days:   Minimum days until expiry (0 = include today).
    """
    today   = from_date or date.today()
    weekday = _FO_EXPIRY_WEEKDAY.get(underlying.upper(), 1)  # default Tuesday

    if underlying.upper() in _MONTHLY_ONLY_FO:
        # Monthly: last <weekday> of each upcoming month
        for month_offset in range(4):
            yr  = today.year + (today.month + month_offset - 1) // 12
            mon = (today.month + month_offset - 1) % 12 + 1
            candidate = _last_weekday_of_month(yr, mon, weekday)
            if (candidate - today).days >= min_days:
                return candidate
        return None
    else:
        # Weekly: every <weekday>
        base = _next_weekday(today, weekday)
        while True:
            if (base - today).days >= min_days:
                return base
            base += timedelta(days=7)


def days_to_fo_expiry(underlying: str, from_date: Optional[date] = None,
                      min_days: int = 0) -> Optional[int]:
    """
    Days until the nearest F&O expiry for a specific underlying.
    Returns None if no expiry found (should never happen in practice).
    """
    exp = get_fo_expiry(underlying, from_date, min_days)
    if exp is None:
        return None
    today = from_date or date.today()
    return (exp - today).days


def get_upcoming_expiries(from_date: Optional[date] = None, count: int = 6,
                          underlying: str = _DEFAULT_UNDERLYING) -> list[dict]:
    """
    Return the next `count` weekly expiry dates for the given underlying,
    together with a flag indicating whether each is also a monthly expiry.

    Defaults to NIFTY (Tuesday expiry).  Previously always used Thursday —
    updated to use per-underlying weekday so AI context is accurate.
    """
    today   = from_date or date.today()
    weekday = _FO_EXPIRY_WEEKDAY.get(underlying.upper(), 1)
    results = []
    d = _next_weekday(today, weekday)
    while len(results) < count:
        last_of_month = _last_weekday_of_month(d.year, d.month, weekday)
        monthly = (last_of_month == d)
        results.append({
            "date":      d,
            "days_away": (d - today).days,
            "monthly":   monthly,
            "label":     f"{'Monthly' if monthly else 'Weekly'} expiry {d.strftime('%d %b %Y')}",
        })
        d = _next_weekday(d + timedelta(days=1), weekday)
    return results


def get_next_expiry(from_date: Optional[date] = None,
                    underlying: str = _DEFAULT_UNDERLYING) -> dict:
    """Return the single nearest upcoming expiry for the given underlying."""
    return get_upcoming_expiries(from_date, count=1, underlying=underlying)[0]


def days_to_expiry(from_date: Optional[date] = None,
                   underlying: str = _DEFAULT_UNDERLYING) -> int:
    """Calendar days until the nearest expiry for the given underlying (default: NIFTY/Tuesday)."""
    return get_next_expiry(from_date, underlying)["days_away"]


def is_expiry_day(on_date: Optional[date] = None,
                  underlying: str = _DEFAULT_UNDERLYING) -> bool:
    return days_to_expiry(on_date, underlying) == 0


def is_expiry_week(on_date: Optional[date] = None,
                   underlying: str = _DEFAULT_UNDERLYING) -> bool:
    """True when we are within 2 calendar days of expiry (Mon/Tue for NIFTY)."""
    return days_to_expiry(on_date, underlying) <= 2


def is_monthly_expiry_week(on_date: Optional[date] = None,
                            underlying: str = _DEFAULT_UNDERLYING) -> bool:
    """True when the nearest expiry is a monthly expiry and it's this week."""
    exp = get_next_expiry(on_date, underlying)
    return exp["monthly"] and exp["days_away"] <= 2


# ─────────────────────────────────────────────────────────────────────────────
# Economic event helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_upcoming_economic_events(from_date: Optional[date] = None,
                                  days_ahead: int = 14) -> list[dict]:
    """
    Return known high-impact economic events in the next `days_ahead` days.
    Includes RBI MPC decisions and Union Budget.
    """
    today  = from_date or date.today()
    cutoff = today + timedelta(days=days_ahead)
    events = []

    for d in RBI_MPC_DATES:
        if today <= d <= cutoff:
            events.append({
                "date":      d,
                "days_away": (d - today).days,
                "type":      "RBI_MPC",
                "label":     f"RBI MPC decision ({d.strftime('%d %b %Y')})",
                "impact":    "high",
            })

    for d in BUDGET_DATES:
        if today <= d <= cutoff:
            events.append({
                "date":      d,
                "days_away": (d - today).days,
                "type":      "BUDGET",
                "label":     f"Union Budget ({d.strftime('%d %b %Y')})",
                "impact":    "very_high",
            })

    events.sort(key=lambda e: e["date"])
    return events


def days_to_next_rbi(from_date: Optional[date] = None) -> Optional[int]:
    """Days until the next RBI MPC decision, or None if beyond 90 days."""
    today = from_date or date.today()
    future = [d for d in RBI_MPC_DATES if d >= today]
    if not future:
        return None
    return (min(future) - today).days


# ─────────────────────────────────────────────────────────────────────────────
# Composite context (used by main.py + AI prompt)
# ─────────────────────────────────────────────────────────────────────────────

def get_calendar_context(from_date: Optional[date] = None) -> dict:
    """
    Single call that returns everything the bot needs to make calendar-aware
    trading decisions.

    days_to_expiry and expiry_day/week now use NIFTY (Tuesday) as default,
    matching the actual expiry weekday for the most-traded underlying.

    Also includes per_underlying_dte dict so the AI knows the correct DTE for
    each options underlying (NIFTY Tue ≠ BANKNIFTY Wed ≠ SENSEX Thu).

    Returns:
        {
          "expiry_day": bool,
          "expiry_week": bool,
          "monthly_expiry_week": bool,
          "days_to_expiry": int,             # NIFTY (Tuesday) expiry
          "next_expiry": {"date", "days_away", "monthly", "label"},
          "upcoming_expiries": [...],
          "per_underlying_dte": {            # per-underlying DTE for options sizing
              "NIFTY": int, "FINNIFTY": int, "MIDCPNIFTY": int,
              "BANKNIFTY": int, "SENSEX": int
          },
          "economic_events_14d": [...],
          "days_to_rbi": int | None,
          "risk_level": "normal" | "elevated" | "high",
          "trading_notes": [str, ...],
        }
    """
    today     = from_date or date.today()
    # Use NIFTY (Tuesday) as the canonical expiry for generic risk signals
    dte       = days_to_expiry(today, "NIFTY")
    exp       = get_next_expiry(today, "NIFTY")
    events_14 = get_upcoming_economic_events(today, days_ahead=14)
    d_rbi     = days_to_next_rbi(today)

    notes: list[str] = []
    risk_level = "normal"

    # Expiry-day specific notes
    if dte == 0:
        notes.append("TODAY IS NIFTY EXPIRY DAY — extreme intraday volatility expected; avoid new entries after 14:00")
        risk_level = "high"
    elif dte == 1:
        notes.append("NIFTY expiry tomorrow — theta decay accelerating; avoid mean-reversion shorts on indices")
        risk_level = "elevated"
    elif dte <= 2:
        notes.append("NIFTY expiry week — heightened volatility in index options; prefer momentum over mean-reversion")
        if risk_level == "normal":
            risk_level = "elevated"

    if exp["monthly"] and dte <= 5:
        notes.append("Monthly expiry approaching — expect larger-than-usual index moves and rollover activity")
        risk_level = "elevated"

    # Economic events
    for ev in events_14:
        if ev["days_away"] == 0:
            notes.append(f"HIGH IMPACT EVENT TODAY: {ev['label']} — consider sitting out until outcome known")
            risk_level = "high"
        elif ev["days_away"] <= 2:
            notes.append(f"High-impact event in {ev['days_away']}d: {ev['label']} — reduce position sizes")
            risk_level = "elevated" if risk_level != "high" else "high"
        elif ev["days_away"] <= 7:
            notes.append(f"Watch: {ev['label']} in {ev['days_away']} days")

    if d_rbi is not None and d_rbi <= 3:
        notes.append(f"RBI MPC decision in {d_rbi} day(s) — rate-sensitive sectors (banks, NBFCs, real estate) may gap")

    # Per-underlying DTE — what the options strategy actually uses, fed to AI context
    per_dte = {u: days_to_fo_expiry(u, today) for u in _FO_EXPIRY_WEEKDAY}

    return {
        "expiry_day":          dte == 0,
        "expiry_week":         dte <= 2,
        "monthly_expiry_week": exp["monthly"] and dte <= 2,
        "days_to_expiry":      dte,
        "next_expiry":         {**exp, "date": exp["date"].isoformat()},
        "upcoming_expiries":   [
            {**e, "date": e["date"].isoformat()}
            for e in get_upcoming_expiries(today, count=4, underlying="NIFTY")
        ],
        "per_underlying_dte":  per_dte,
        "economic_events_14d": [
            {**e, "date": e["date"].isoformat()}
            for e in events_14
        ],
        "days_to_rbi":         d_rbi,
        "risk_level":          risk_level,
        "trading_notes":       notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Risk sizing multipliers
# ─────────────────────────────────────────────────────────────────────────────

def get_calendar_size_multiplier(calendar_ctx: Optional[dict] = None) -> float:
    """
    Return a position-size multiplier (0.25–1.0) based on calendar risk.
    Applied on top of the VIX-based multiplier in RiskManager.
    """
    ctx = calendar_ctx or get_calendar_context()
    level = ctx.get("risk_level", "normal")
    if level == "high":
        return 0.40    # expiry day or event day — very small size
    if level == "elevated":
        return 0.65    # expiry week or event approaching
    return 1.0
