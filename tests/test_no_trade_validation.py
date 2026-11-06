"""Regression tests for the cash/no-trade benchmark."""
import pandas as pd
import pytest

from src.analysis.no_trade_validation import select_with_cash_option, evaluate_cash_option


def sweep_rows(train_pnl=-10.0):
    return pd.DataFrame([
        {"period": "train", "cost_model": "bid/ask + fees", "theta": 0.05,
         "n_trades": 40, "total_pnl_per_100": train_pnl},
        {"period": "train", "cost_model": "bid/ask + fees", "theta": 0.10,
         "n_trades": 22, "total_pnl_per_100": -2.0},
        {"period": "train", "cost_model": "bid/ask + fees", "theta": 0.20,
         "n_trades": 2, "total_pnl_per_100": 100.0},
        {"period": "test", "cost_model": "bid/ask + fees", "theta": 0.05,
         "n_trades": 80, "total_pnl_per_100": 9999.0},
    ])


def test_all_losing_training_thresholds_select_no_trade():
    result = select_with_cash_option(sweep_rows())
    assert result["decision"] == "no_trade" and result["theta"] is None


def test_positive_eligible_training_threshold_selects_trade():
    result = select_with_cash_option(sweep_rows(train_pnl=3.0))
    assert result["decision"] == "trade" and result["theta"] == pytest.approx(0.05)


def test_test_performance_cannot_influence_training_selection():
    rows = sweep_rows()
    rows.loc[rows.period == "test", "total_pnl_per_100"] = 1_000_000
    assert select_with_cash_option(rows)["decision"] == "no_trade"


def test_ineligible_high_return_does_not_override_cash():
    rows = sweep_rows()
    rows.loc[rows.theta == 0.20, "total_pnl_per_100"] = 1_000_000
    assert select_with_cash_option(rows)["decision"] == "no_trade"


def test_no_eligible_training_thresholds_select_no_trade():
    rows = sweep_rows()
    rows.loc[rows.period == "train", "n_trades"] = 1
    assert select_with_cash_option(rows)["reason"] == "no_eligible_training_threshold"


def test_cash_position_has_zero_pnl_and_no_undefined_return_claim():
    forecast = pd.DataFrame({"month": ["2025-06-01"], "p_fed": [0.9]})
    out = evaluate_cash_option(forecast, select_with_cash_option(sweep_rows()))
    assert out["test_trades"] == 0 and out["test_total_pnl_per_100"] == 0.0
    assert pd.isna(out["mean_pnl_cents"])


def test_training_decision_requires_columns_and_valid_minimum():
    with pytest.raises(ValueError, match="Missing"):
        select_with_cash_option(pd.DataFrame())
    with pytest.raises(ValueError, match="positive"):
        select_with_cash_option(sweep_rows(), min_trades=0)


def test_zero_training_pnl_does_not_trigger_a_trade():
    rows = sweep_rows()
    rows.loc[rows.theta == 0.05, "total_pnl_per_100"] = 0
    assert select_with_cash_option(rows)["decision"] == "no_trade"
