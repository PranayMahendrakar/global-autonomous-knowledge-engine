"""
Global Autonomous Knowledge Engine (GAKE)
==========================================
Main entry point for the autonomous knowledge discovery and learning system.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

from gake.config import EngineConfig
from gake.discovery.agent_manager import AgentManager
from gake.extraction.knowledge_extractor import KnowledgeExtractor
from gake.graph.knowledge_graph import KnowledgeGraph
from gake.validation.knowledge_validator import KnowledgeValidator
from gake.learning.continuous_learner import ContinuousLearner
from gake.query.query_engine import QueryEngine
from gake.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class GlobalAutonomousKnowledgeEngine:
    """
    Core engine orchestrating autonomous knowledge discovery,
    extraction, validation, and continuous learning.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.running = False
        self.start_time = None

        self.graph = KnowledgeGraph(config.graph)
        self.agent_manager = AgentManager(config.discovery)
        self.extractor = KnowledgeExtractor(config.extraction)
        self.validator = KnowledgeValidator(config.validation, self.graph)
        self.learner = ContinuousLearner(config.learning, self.graph)
        self.query_engine = QueryEngine(self.graph)

        logger.info("Global Autonomous Knowledge Engine initialized")

    async def start(self):
        """Start the autonomous knowledge engine."""
        self.running = True
        self.start_time = datetime.utcnow()
        logger.info(f"Engine started at {self.start_time.isoformat()}")

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)

        try:
            await asyncio.gather(
                self._discovery_loop(),
                self._learning_loop(),
                self._maintenance_loop(),
            )
        except asyncio.CancelledError:
            logger.info("Engine tasks cancelled")
        finally:
            await self.stop()

    async def _discovery_loop(self):
        """Continuously discover and ingest new knowledge."""
        logger.info("Discovery loop started")
        while self.running:
            try:
                sources = await self.agent_manager.discover_sources()
                logger.info(f"Discovered {len(sources)} new sources")

                for source in sources:
                    raw_knowledge = await self.extractor.extract(source)
                    validated = await self.validator.validate(raw_knowledge)
                    if validated:
                        await self.graph.integrate(validated)
                        logger.info(f"Integrated knowledge from: {source.url}")

                await asyncio.sleep(self.config.discovery.interval_seconds)

            except Exception as e:
                logger.error(f"Discovery loop error: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _learning_loop(self):
        """Continuously update and evolve knowledge representations."""
        while self.running:
            try:
                await self.learner.run_learning_cycle()
                await asyncio.sleep(self.config.learning.interval_seconds)
            except Exception as e:
                logger.error(f"Learning loop error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _maintenance_loop(self):
        """Perform periodic graph maintenance and optimization."""
        while self.running:
            try:
                await self.graph.optimize()
                await self.validator.run_consistency_check()
                await asyncio.sleep(self.config.maintenance_interval_seconds)
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}", exc_info=True)
                await asyncio.sleep(300)

    def _shutdown(self):
        self.running = False

    async def stop(self):
        await self.graph.save()
        logger.info("Engine stopped")

    def query(self, question: str) -> dict:
        return self.query_engine.answer(question)

    def get_stats(self) -> dict:
        return {
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            if self.start_time else 0,
            "graph_stats": self.graph.get_stats(),
            "discovery_stats": self.agent_manager.get_stats(),
            "learning_stats": self.learner.get_stats(),
        }


async def main():
    setup_logger()
    config = EngineConfig.from_yaml(Path("config/engine_config.yaml"))
    engine = GlobalAutonomousKnowledgeEngine(config)
    await engine.start()


if __name__ == "__main__":
    asyncio.run(main())
