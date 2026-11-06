"""Out-of-sample trading decision guardrail.

The project's baseline backtest always carries the least-bad in-sample threshold
forward, even if every eligible threshold lost money in training. This module
makes a precommitted cash/no-trade alternative explicit without changing the
original threshold sweep or historical outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.backtest import COST_MODELS, MIN_TRAIN_TRADES, TEST_START, trades


def select_with_cash_option(
    sweep: pd.DataFrame, *, min_trades: int = MIN_TRAIN_TRADES
) -> dict:
    """Choose a training threshold only when net training P&L beats holding cash.

    Selection uses *only* training rows at executable bid/ask prices with fees.
    When no candidate qualifies or the best result is <= 0, select no trade.
    Test rows must never influence the decision.
    """
    required = {"period", "cost_model", "theta", "n_trades", "total_pnl_per_100"}
    missing = required.difference(sweep.columns)
    if missing:
        raise ValueError(f"Missing sweep columns: {sorted(missing)}")
    if min_trades <= 0:
        raise ValueError("min_trades must be positive")

    eligible = sweep.loc[
        (sweep["period"] == "train")
        & (sweep["cost_model"] == "bid/ask + fees")
        & (sweep["n_trades"] >= min_trades)
        & np.isfinite(sweep["total_pnl_per_100"])
    ]
    if eligible.empty:
        return {"decision": "no_trade", "theta": None, "train_pnl_per_100": 0.0,
                "reason": "no_eligible_training_threshold"}

    best = eligible.sort_values(
        ["total_pnl_per_100", "theta"], ascending=[False, True]
    ).iloc[0]
    pnl = float(best["total_pnl_per_100"])
    if pnl <= 0:
        return {"decision": "no_trade", "theta": None, "train_pnl_per_100": pnl,
                "reason": "training_did_not_beat_cash"}
    return {"decision": "trade", "theta": float(best["theta"]),
            "train_pnl_per_100": pnl, "reason": "positive_training_net_pnl"}


def evaluate_cash_option(
    forecast: pd.DataFrame, selection: dict, *, test_start=TEST_START
) -> dict:
    """Apply the frozen training decision to held-out data only.

    A no-trade result has zero positions and zero dollars of P&L; per-trade
    returns are undefined rather than misleadingly shown as 0%.
    """
    if selection["decision"] not in {"trade", "no_trade"}:
        raise ValueError("Unknown decision")
    test = forecast.loc[pd.to_datetime(forecast["month"]) >= pd.Timestamp(test_start)]
    if selection["decision"] == "no_trade":
        return {"decision": "no_trade", "theta": None, "test_trades": 0,
                "test_total_pnl_per_100": 0.0, "mean_pnl_cents": np.nan}

    theta = selection.get("theta")
    if theta is None or not np.isfinite(theta):
        raise ValueError("A trade decision requires a finite theta")
    positions = trades(test, float(theta), "bid/ask + fees")
    return {"decision": "trade", "theta": float(theta),
            "test_trades": int(len(positions)),
            "test_total_pnl_per_100": float(positions["pnl"].sum() * 100)
            if not positions.empty else 0.0,
            "mean_pnl_cents": float(positions["pnl"].mean() * 100)
            if not positions.empty else np.nan}
