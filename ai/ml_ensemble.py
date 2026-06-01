"""
Phase 7 — ML Ensemble
Trains a GradientBoostingClassifier on completed AI decisions (from the
feedback loop DB) to predict the probability that a new signal will win.

Features (all available at signal time, no look-ahead):
  signal_conf, action, strategy, hour_of_day, day_of_week,
  vix, ai_confidence_adj, regime_encoded

Target: outcome_correct (1 = profitable trade, 0 = loss)

The model is retrained weekly (called from job_weekly_optimisation) once
≥30 labelled examples exist.  Before that it returns None so the bot falls
back to pure AI/rule-based logic.

Model is persisted to data/ml_model.pkl so restarts don't lose the weights.
"""

import json
import logging
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH   = Path(__file__).parent.parent / "data" / "market_data.db"
MODEL_PATH = Path(__file__).parent.parent / "data" / "ml_model.pkl"
MIN_SAMPLES = 30   # minimum labelled decisions before we train


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGY_MAP = {"momentum": 0, "mean_reversion": 1, "options": 2}
_ACTION_MAP   = {"BUY": 1, "SELL": 0}
_REGIME_MAP   = {
    "trending_up": 2, "trending": 2,
    "trending_down": 0,
    "ranging": 1, "sideways": 1,
    "volatile": -1, "unknown": 1,
}


def _extract_features(
    signal_conf: float,
    action: str,
    strategy: str,
    decided_at: str,
    vix: Optional[float],
    ai_confidence_adj: float,
    regime: str = "unknown",
) -> list[float]:
    """Convert raw decision fields into a fixed-length feature vector."""
    try:
        dt = datetime.fromisoformat(decided_at)
        hour = dt.hour
        dow  = dt.weekday()   # 0=Mon … 4=Fri
    except Exception:
        hour, dow = 10, 0

    return [
        float(signal_conf),
        float(_ACTION_MAP.get(action, 0)),
        float(_STRATEGY_MAP.get(strategy, 1)),
        float(hour),
        float(dow),
        float(vix) if vix is not None else 15.0,   # median VIX as default
        float(ai_confidence_adj),
        float(_REGIME_MAP.get(regime, 1)),
    ]


FEATURE_NAMES = [
    "signal_conf", "action", "strategy",
    "hour", "dow", "vix", "ai_conf_adj", "regime",
]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Pull all completed decisions (outcome_correct IS NOT NULL) from DB.
    Returns (X, y) arrays.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """SELECT signal_conf, action, strategy, decided_at,
                          ai_confidence_adj, outcome_correct
                   FROM ai_decisions
                   WHERE outcome_correct IS NOT NULL AND approved = 1""",
            ).fetchall()
    except Exception as e:
        logger.warning(f"ML: DB read failed: {e}")
        return np.array([]), np.array([])

    if not rows:
        return np.array([]), np.array([])

    X, y = [], []
    for r in rows:
        conf, action, strategy, decided_at, ai_adj, correct = r
        fv = _extract_features(
            conf or 0.5, action or "BUY", strategy or "momentum",
            decided_at or "", None, ai_adj or 0.0,
        )
        X.append(fv)
        y.append(int(correct))

    return np.array(X, dtype=float), np.array(y, dtype=int)


def train_model() -> Optional[object]:
    """
    Train (or retrain) the GradientBoostingClassifier.
    Saves the model to MODEL_PATH and returns it.
    Returns None if not enough data.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X, y = _load_training_data()
    if len(X) < MIN_SAMPLES:
        logger.info(
            f"ML: only {len(X)} labelled decisions — need {MIN_SAMPLES} to train. Skipping."
        )
        return None

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    logger.info(f"ML: training on {len(X)} samples ({n_pos} wins, {n_neg} losses)")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )),
    ])
    model.fit(X, y)

    # Cross-validated ROC-AUC
    try:
        if len(X) >= 50:
            scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
            auc = scores.mean()
            logger.info(f"ML: 5-fold CV ROC-AUC = {auc:.3f} ± {scores.std():.3f}")
        else:
            auc = None
    except Exception:
        auc = None

    # Save model + metadata
    payload = {
        "model":      model,
        "trained_at": datetime.now().isoformat(),
        "n_samples":  len(X),
        "auc":        auc,
        "feature_names": FEATURE_NAMES,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"ML: model saved to {MODEL_PATH}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

_cached_model: Optional[dict] = None


def _load_model() -> Optional[dict]:
    """Load model from disk (once per process, then cached in memory)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            _cached_model = pickle.load(f)
        logger.info(
            f"ML: loaded model trained at {_cached_model['trained_at']}"
            f" on {_cached_model['n_samples']} samples"
        )
        return _cached_model
    except Exception as e:
        logger.warning(f"ML: failed to load model: {e}")
        return None


def reload_model():
    """Force a fresh load from disk (call after retraining)."""
    global _cached_model
    _cached_model = None
    return _load_model()


def predict_win_probability(
    signal_conf: float,
    action: str,
    strategy: str,
    vix: Optional[float] = None,
    ai_confidence_adj: float = 0.0,
    regime: str = "unknown",
) -> Optional[dict]:
    """
    Return win-probability prediction for a new signal.

    Returns:
        {
          "win_prob": 0.68,          # P(win | features)
          "ml_confidence": "high",   # high/medium/low based on distance from 0.5
          "recommendation": "proceed" | "caution" | "skip",
          "trained_at": "...",
          "n_samples": 42,
        }
    or None if no model is available yet.
    """
    payload = _load_model()
    if payload is None:
        return None

    model = payload["model"]
    fv = _extract_features(
        signal_conf, action, strategy,
        datetime.now().isoformat(), vix, ai_confidence_adj, regime,
    )
    try:
        proba = model.predict_proba([fv])[0]
        win_prob = float(proba[1])   # class 1 = win
    except Exception as e:
        logger.warning(f"ML predict failed: {e}")
        return None

    # Confidence = how far from the decision boundary (0.5)
    distance = abs(win_prob - 0.5)
    if distance >= 0.20:
        ml_confidence = "high"
    elif distance >= 0.10:
        ml_confidence = "medium"
    else:
        ml_confidence = "low"

    # Recommendation
    if win_prob >= 0.65:
        recommendation = "proceed"
    elif win_prob >= 0.45:
        recommendation = "caution"
    else:
        recommendation = "skip"

    return {
        "win_prob":       round(win_prob, 3),
        "ml_confidence":  ml_confidence,
        "recommendation": recommendation,
        "trained_at":     payload["trained_at"],
        "n_samples":      payload["n_samples"],
        "auc":            payload.get("auc"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance (for dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance() -> Optional[dict]:
    """Return feature importances from the trained model (for dashboard display)."""
    payload = _load_model()
    if payload is None:
        return None
    try:
        clf = payload["model"].named_steps["clf"]
        importances = clf.feature_importances_
        return {
            "features":  FEATURE_NAMES,
            "importances": [round(float(v), 4) for v in importances],
            "trained_at": payload["trained_at"],
            "n_samples":  payload["n_samples"],
            "auc":        payload.get("auc"),
        }
    except Exception:
        return None


def get_model_stats() -> dict:
    """Return a summary dict for the Settings dashboard panel."""
    payload = _load_model()
    if payload is None:
        X, _ = _load_training_data()
        return {
            "status":      "untrained",
            "reason":      f"{len(X)}/{MIN_SAMPLES} labelled decisions collected",
            "trained_at":  None,
            "n_samples":   len(X),
            "auc":         None,
        }
    fi = get_feature_importance()
    top = sorted(
        zip(FEATURE_NAMES, fi["importances"]) if fi else [],
        key=lambda x: x[1], reverse=True
    )[:3]
    return {
        "status":    "ready",
        "trained_at": payload["trained_at"],
        "n_samples":  payload["n_samples"],
        "auc":        payload.get("auc"),
        "top_features": top,
    }
