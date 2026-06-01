"""
Phase 8 — F&O Expiry & Economic Calendar
Provides expiry proximity, RBI MPC dates, budget dates, and earnings windows
so the bot can reduce risk and inform AI reasoning around high-volatility events.

NSE expiry rules:
  - Weekly: every Thursday (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
  - Monthly: last Thursday of each calendar month

All dates are in IST.  No external API needed — logic is computed or hardcoded
from official NSE / RBI calendars updated annually.
"""

from datetime import date, datetime, timedelta
from typing import Optional

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

def _last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of a given month."""
    # Find last day of month
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    # Walk back to Thursday (weekday 3)
    offset = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=offset)


def _next_thursday(from_date: date) -> date:
    """Return the next Thursday on or after from_date."""
    days_ahead = (3 - from_date.weekday()) % 7   # 3 = Thursday
    return from_date + timedelta(days=days_ahead)


def get_upcoming_expiries(from_date: Optional[date] = None, count: int = 6) -> list[dict]:
    """
    Return the next `count` weekly expiry Thursdays together with a flag
    indicating whether each is also a monthly expiry.
    """
    today = from_date or date.today()
    results = []
    d = _next_thursday(today)
    while len(results) < count:
        monthly = _last_thursday_of_month(d.year, d.month) == d
        results.append({
            "date":     d,
            "days_away": (d - today).days,
            "monthly":  monthly,
            "label":    f"{'Monthly' if monthly else 'Weekly'} expiry {d.strftime('%d %b %Y')}",
        })
        d = _next_thursday(d + timedelta(days=1))
    return results


def get_next_expiry(from_date: Optional[date] = None) -> dict:
    """Return the single nearest upcoming weekly expiry."""
    return get_upcoming_expiries(from_date, count=1)[0]


def days_to_expiry(from_date: Optional[date] = None) -> int:
    """Calendar days until the nearest weekly expiry."""
    return get_next_expiry(from_date)["days_away"]


def is_expiry_day(on_date: Optional[date] = None) -> bool:
    return days_to_expiry(on_date) == 0


def is_expiry_week(on_date: Optional[date] = None) -> bool:
    """True when we are within 2 calendar days of expiry (Wed–Thu)."""
    return days_to_expiry(on_date) <= 2


def is_monthly_expiry_week(on_date: Optional[date] = None) -> bool:
    """True when the nearest expiry is a monthly expiry and it's this week."""
    exp = get_next_expiry(on_date)
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

    Returns:
        {
          "expiry_day": bool,
          "expiry_week": bool,
          "monthly_expiry_week": bool,
          "days_to_expiry": int,
          "next_expiry": {"date", "days_away", "monthly", "label"},
          "upcoming_expiries": [...],
          "economic_events_14d": [...],
          "days_to_rbi": int | None,
          "risk_level": "normal" | "elevated" | "high",
          "trading_notes": [str, ...],
        }
    """
    today = from_date or date.today()
    dte   = days_to_expiry(today)
    exp   = get_next_expiry(today)
    events_14 = get_upcoming_economic_events(today, days_ahead=14)
    d_rbi = days_to_next_rbi(today)

    notes: list[str] = []
    risk_level = "normal"

    # Expiry-day specific notes
    if dte == 0:
        notes.append("TODAY IS EXPIRY DAY — extreme intraday volatility expected; avoid new entries after 14:00")
        risk_level = "high"
    elif dte == 1:
        notes.append("Expiry tomorrow — theta decay accelerating; avoid mean-reversion shorts on indices")
        risk_level = "elevated"
    elif dte <= 2:
        notes.append("Expiry week — heightened volatility in index options; prefer momentum over mean-reversion")
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

    return {
        "expiry_day":          dte == 0,
        "expiry_week":         dte <= 2,
        "monthly_expiry_week": exp["monthly"] and dte <= 2,
        "days_to_expiry":      dte,
        "next_expiry":         {**exp, "date": exp["date"].isoformat()},
        "upcoming_expiries":   [
            {**e, "date": e["date"].isoformat()}
            for e in get_upcoming_expiries(today, count=4)
        ],
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
