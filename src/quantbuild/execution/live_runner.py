"""Live runner — main loop for paper/live trading with news + regime integration."""
import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.quantbuild.config import load_config
from src.quantbuild.data.sessions import session_from_timestamp, ENTRY_SESSIONS
from src.quantbuild.execution.broker_oanda import OandaBroker
from src.quantbuild.execution.order_manager import OrderManager
from src.quantbuild.execution.position_monitor import PositionMonitor
from src.quantbuild.io.parquet_loader import load_parquet
from src.quantbuild.strategies.sqe_xauusd import get_sqe_default_config, _compute_modules_once, run_sqe_conditions
from src.quantbuild.strategy_modules.regime.detector import RegimeDetector, REGIME_EXPANSION

logger = logging.getLogger(__name__)


class LiveRunner:
    """Main live/paper trading loop with regime detection and news integration."""

    def __init__(self, cfg: Dict[str, Any], dry_run: bool = True):
        self.cfg = cfg
        self.dry_run = dry_run
        self._running = False

        self.broker = OandaBroker(
            account_id=cfg.get("broker", {}).get("account_id", ""),
            token=cfg.get("broker", {}).get("token", ""),
            environment=cfg.get("broker", {}).get("environment", "practice"),
            instrument=cfg.get("broker", {}).get("instrument", "XAU_USD"),
        )
        self.order_manager = OrderManager(broker=self.broker if not dry_run else None, config=cfg.get("order_management"))
        self.position_monitor = PositionMonitor(cfg)

        self._regime_detector = RegimeDetector(config=cfg.get("regime", {}))
        self._current_regime: Optional[str] = None
        self._news_regime_override: Optional[str] = None

        self._news_poller = None
        self._news_gate = None
        self._sentiment_engine = None
        self._counter_news = None
        self._relevance_filter = None
        self._news_history = None

        if cfg.get("news", {}).get("enabled", False):
            self._setup_news_layer()

    def _setup_news_layer(self):
        try:
            from src.quantbuild.news.poller import NewsPoller
            from src.quantbuild.news.relevance_filter import RelevanceFilter
            from src.quantbuild.news.gold_classifier import GoldEventClassifier
            from src.quantbuild.news.sentiment import HybridSentiment
            from src.quantbuild.news.counter_news import CounterNewsDetector
            from src.quantbuild.news.history import NewsHistory
            from src.quantbuild.strategy_modules.news_gate import NewsGate

            self._news_poller = NewsPoller(self.cfg)
            self._news_poller.setup()
            self._relevance_filter = RelevanceFilter(self.cfg)
            self._gold_classifier = GoldEventClassifier(self.cfg)
            self._sentiment_engine = HybridSentiment(self.cfg)
            self._counter_news = CounterNewsDetector(self.cfg)
            self._news_gate = NewsGate(self.cfg)
            self._news_history = NewsHistory()
            logger.info("News layer initialized: %d sources", self._news_poller.source_count)
        except Exception as e:
            logger.warning("News layer setup failed: %s", e)

    def get_effective_regime(self) -> Optional[str]:
        """Return the effective regime, with news override taking priority."""
        if self._news_regime_override:
            return self._news_regime_override
        return self._current_regime

    def run(self):
        """Main loop: connect, check signals, manage positions."""
        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info("Starting LiveRunner in %s mode", mode)

        if not self.dry_run:
            if not self.broker.connect():
                logger.error("Cannot connect to broker. Exiting.")
                return

        self.order_manager.load_state()
        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)

        check_interval = 60
        news_interval = self.cfg.get("news", {}).get("poll_interval_seconds", 30)
        last_news_poll = datetime.min

        try:
            while self._running:
                now = datetime.now(timezone.utc)

                # Poll news + update regime override
                if self._news_poller and (now - last_news_poll).total_seconds() >= news_interval:
                    self._poll_news()
                    self._update_news_regime_override()
                    last_news_poll = now

                session = session_from_timestamp(now, mode=self.cfg.get("backtest", {}).get("session_mode", "killzone"))
                if session in ENTRY_SESSIONS:
                    self._check_signals(now)

                self._monitor_positions()
                time.sleep(check_interval)

        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _poll_news(self):
        if not self._news_poller:
            return
        try:
            events = self._news_poller.poll()
            if self._relevance_filter:
                events = self._relevance_filter.filter_batch(events)

            for event in events:
                classification = self._gold_classifier.classify(event) if hasattr(self, '_gold_classifier') else None
                sentiment = self._sentiment_engine.analyze(event) if self._sentiment_engine else None

                if self._news_gate and sentiment:
                    self._news_gate.add_news_event(event, sentiment)

                if self._news_history and sentiment:
                    self._news_history.add_event(event, sentiment)

                if self._counter_news and sentiment:
                    positions = self.position_monitor.all_positions
                    if positions:
                        affected = self._counter_news.check_against_positions(event, positions)
                        for hit in affected:
                            if hit["action"] == "exit":
                                self.position_monitor.invalidate_thesis(hit["trade_id"], hit["reason"])

                if classification:
                    logger.info("News: [%s/%s] %s | sentiment: %s",
                                classification.niche, classification.event_type,
                                event.headline[:60], sentiment.direction if sentiment else "?")
        except Exception as e:
            logger.error("News poll error: %s", e)

    def _update_news_regime_override(self):
        """Use news state to override technical regime when appropriate."""
        if not self._news_gate:
            self._news_regime_override = None
            return

        # High-impact event coming -> override to EXPANSION
        now = datetime.now(timezone.utc)
        for evt in self._news_gate._scheduled_events:
            evt_time = evt["time"]
            if evt_time.tzinfo is None:
                evt_time = evt_time.replace(tzinfo=timezone.utc)
            if evt["name"] in self._news_gate._high_impact_events:
                window_start = evt_time - timedelta(minutes=60)
                window_end = evt_time + timedelta(minutes=30)
                if window_start <= now <= window_end:
                    self._news_regime_override = REGIME_EXPANSION
                    logger.info("News regime override -> EXPANSION (high-impact: %s)", evt["name"])
                    return

        # Strong directional sentiment -> keep or confirm TREND
        summary = self._news_gate.get_current_sentiment_summary()
        if summary["event_count"] >= 3 and abs(summary["avg_impact"]) > 0.5:
            self._news_regime_override = None
            logger.debug("Strong news sentiment (%.2f) - technical regime stands", summary["avg_impact"])
            return

        self._news_regime_override = None

    def _check_signals(self, now: datetime):
        """Check for entry signals (stub for live wiring)."""
        regime = self.get_effective_regime()
        regime_profiles = self.cfg.get("regime_profiles", {})
        if regime and regime_profiles.get(regime, {}).get("skip", False):
            logger.debug("Regime %s -> skip (no entries)", regime)
            return

        if self._news_gate:
            for direction in ("LONG", "SHORT"):
                gate_result = self._news_gate.check_gate(now, direction)
                if not gate_result["allowed"]:
                    logger.debug("NewsGate blocks %s: %s", direction, gate_result["reason"])

        logger.debug("Checking signals at %s (regime=%s)", now.strftime("%H:%M"), regime)

    def _monitor_positions(self):
        for pos in self.position_monitor.all_positions:
            if not pos.thesis_valid:
                logger.warning("Position %s thesis invalid -- should close", pos.trade_id)

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received")
        self._running = False

    def _shutdown(self):
        logger.info("Shutting down LiveRunner...")
        self.order_manager.save_state()
        if self._news_history:
            self._news_history.save_to_parquet()
            self._news_history.save_latest_json()
            logger.info("News history saved (%d events)", self._news_history.event_count)
        if self.broker.is_connected:
            self.broker.disconnect()
        logger.info("LiveRunner stopped.")
