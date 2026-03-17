"""Tests for the correlation-aware portfolio heat engine."""
import pytest

from src.quantbuild.execution.portfolio_heat import (
    PortfolioHeatEngine,
    _get_correlation,
)


@pytest.fixture
def engine():
    return PortfolioHeatEngine({
        "max_portfolio_heat_pct": 6.0,
        "max_instrument_heat_pct": 3.0,
        "max_correlated_exposure": 2,
        "max_same_direction": 4,
    })


def test_empty_engine(engine):
    assert engine.naive_heat == 0.0
    assert engine.effective_heat == 0.0
    assert engine.open_positions == []


def test_single_position(engine):
    engine.add_position("XAUUSD", "LONG", 1.5, "metals")
    assert engine.naive_heat == 1.5
    assert engine.effective_heat == 1.5
    assert len(engine.open_positions) == 1


def test_uncorrelated_positions_reduce_effective_heat(engine):
    engine.add_position("XAUUSD", "LONG", 1.5, "metals")
    engine.add_position("USDJPY", "LONG", 1.5, "fx_major")
    # Near-zero correlation -> effective heat < naive heat
    assert engine.naive_heat == 3.0
    assert engine.effective_heat < engine.naive_heat


def test_instrument_heat(engine):
    engine.add_position("XAUUSD", "LONG", 1.5, "metals")
    engine.add_position("XAUUSD", "SHORT", 1.0, "metals")
    assert engine.instrument_heat("XAUUSD") == 2.5
    assert engine.instrument_heat("GBPUSD") == 0.0


def test_can_open_respects_max_heat(engine):
    # Fill up to near limit — same instrument to avoid diversification benefit
    engine.add_position("XAUUSD", "LONG", 2.9, "metals")
    engine.add_position("XAUUSD", "LONG", 2.9, "metals")
    ok, reason = engine.can_open("XAUUSD", "LONG", 1.0, "metals")
    assert not ok


def test_can_open_respects_instrument_limit(engine):
    engine.add_position("XAUUSD", "LONG", 2.5, "metals")
    ok, reason = engine.can_open("XAUUSD", "SHORT", 1.0, "metals")
    assert not ok
    assert "instrument_heat" in reason


def test_remove_position(engine):
    engine.add_position("XAUUSD", "LONG", 1.5, "metals")
    assert engine.naive_heat == 1.5
    removed = engine.remove_position("XAUUSD")
    assert removed is True
    assert engine.naive_heat == 0.0


def test_clear(engine):
    engine.add_position("XAUUSD", "LONG", 1.5, "metals")
    engine.add_position("GBPUSD", "SHORT", 1.0, "fx_major")
    engine.clear()
    assert engine.naive_heat == 0.0
    assert len(engine.open_positions) == 0


def test_get_status(engine):
    engine.add_position("XAUUSD", "LONG", 1.5, "metals")
    status = engine.get_status()
    assert status["positions"] == 1
    assert status["naive_heat"] == 1.5
    assert "XAUUSD" in status["per_instrument"]


def test_correlation_lookup():
    assert _get_correlation("XAUUSD", "XAUUSD") == 1.0
    assert _get_correlation("XAUUSD", "GBPUSD") == _get_correlation("GBPUSD", "XAUUSD")
    assert abs(_get_correlation("XAUUSD", "GBPUSD")) < 0.1
