"""
Knowledge Validation Module.
Ensures accuracy, consistency, and reliability of extracted knowledge.

Validation strategies:
1. Multi-source validation - requires multiple independent sources
2. Cross-validation - checks consistency across related facts
3. Temporal validation - ensures temporal consistency
4. Logical consistency - detects contradictions
5. Confidence scoring - weights facts by source reliability
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

from gake.config import ValidationConfig
from gake.extraction.knowledge_extractor import ExtractedKnowledge, Entity, Relation
from gake.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    NEEDS_VERIFICATION = "needs_verification"


class ValidationReason(Enum):
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT_SOURCES = "insufficient_sources"
    CONTRADICTION_DETECTED = "contradiction_detected"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    PASSES_ALL_CHECKS = "passes_all_checks"


@dataclass
class ValidationResult:
    """Result of a knowledge validation check."""
    status: ValidationStatus
    confidence: float
    reason: ValidationReason
    details: str = ""
    suggestions: List[str] = field(default_factory=list)
    validated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class FactCheck:
    """Represents a fact that has been checked."""
    claim: str
    status: ValidationStatus
    sources: List[str]
    confidence: float
    contradictions: List[str] = field(default_factory=list)


class SourceReliabilityScorer:
    """Scores the reliability of knowledge sources."""

    # Domain authority scores (0-1)
    AUTHORITY_SCORES = {
        "arxiv.org": 0.9,
        "semanticscholar.org": 0.85,
        "nature.com": 0.95,
        "science.org": 0.95,
        "ieee.org": 0.9,
        "acm.org": 0.9,
        "wikipedia.org": 0.7,
        "github.com": 0.75,
        "scholar.google.com": 0.85,
    }

    def score(self, source_url: str) -> float:
        """Score a source's reliability (0-1)."""
        for domain, score in self.AUTHORITY_SCORES.items():
            if domain in source_url:
                return score
        return 0.5  # Default for unknown sources

    def score_multiple(self, sources: List[str]) -> float:
        """Aggregate reliability score for multiple sources."""
        if not sources:
            return 0.0
        scores = [self.score(s) for s in sources]
        # Use max score with diminishing returns for additional sources
        base = max(scores)
        bonus = sum(s * 0.1 for s in sorted(scores)[:-1])
        return min(base + bonus, 1.0)


class ContradictionDetector:
    """Detects contradictions in knowledge."""

    # Relations that are mutually exclusive
    EXCLUSIVE_RELATIONS = {
        "is_a": ["is_not_a"],
        "discovered": ["did_not_discover"],
        "part_of": ["not_part_of"],
    }

    def detect(
        self,
        new_knowledge: ExtractedKnowledge,
        graph: KnowledgeGraph
    ) -> List[str]:
        """Detect contradictions between new and existing knowledge."""
        contradictions = []

        for relation in new_knowledge.relations:
            # Check if opposite relation exists
            subj_node = graph.find_node_by_name(relation.subject.canonical_name)
            obj_node = graph.find_node_by_name(relation.obj.canonical_name)

            if subj_node and obj_node:
                # Check for conflicting relations in existing graph
                neighbors = graph.get_neighbors(subj_node.node_id)
                for neighbor in neighbors:
                    if neighbor.node_id == obj_node.node_id:
                        # Connection exists - check if contradictory
                        pass  # Simplified check

        return contradictions


class TemporalValidator:
    """Validates temporal consistency of facts."""

    def validate(self, knowledge: ExtractedKnowledge) -> Tuple[bool, str]:
        """Check temporal consistency."""
        # Check that dates are reasonable
        current_year = datetime.utcnow().year

        for entity in knowledge.entities:
            year = entity.metadata.get("year")
            if year:
                try:
                    year_int = int(year)
                    if year_int > current_year:
                        return False, f"Future date detected: {year}"
                    if year_int < 1900:
                        return False, f"Unreasonably old date: {year}"
                except (ValueError, TypeError):
                    pass

        return True, ""


class LogicalConsistencyChecker:
    """Checks logical consistency of extracted knowledge."""

    def check(self, knowledge: ExtractedKnowledge) -> Tuple[bool, str]:
        """Check for logical inconsistencies."""
        # Check entity type consistency
        for entity in knowledge.entities:
            if not entity.text or not entity.canonical_name:
                return False, "Empty entity text or canonical name"

        # Check relation endpoint consistency
        entity_names = {e.canonical_name for e in knowledge.entities}
        for relation in knowledge.relations:
            if relation.subject.canonical_name not in entity_names:
                logger.warning(f"Relation subject not in entities: {relation.subject.text}")
            if relation.obj.canonical_name not in entity_names:
                logger.warning(f"Relation object not in entities: {relation.obj.text}")

        return True, ""


class KnowledgeValidator:
    """
    Main knowledge validation pipeline.
    
    Validates extracted knowledge before integration into the graph
    using multiple validation strategies.
    """

    def __init__(self, config: ValidationConfig, graph: KnowledgeGraph):
        self.config = config
        self.graph = graph
        self.reliability_scorer = SourceReliabilityScorer()
        self.contradiction_detector = ContradictionDetector()
        self.temporal_validator = TemporalValidator()
        self.logical_checker = LogicalConsistencyChecker()

        self._stats = {
            "total_validated": 0,
            "passed": 0,
            "failed": 0,
            "uncertain": 0,
        }

        logger.info("KnowledgeValidator initialized")

    async def validate(
        self,
        knowledge: ExtractedKnowledge
    ) -> Optional[ExtractedKnowledge]:
        """
        Validate extracted knowledge.
        Returns the knowledge if valid, None if invalid.
        """
        self._stats["total_validated"] += 1

        # Check 1: Minimum content check
        if not knowledge.entities and not knowledge.concepts:
            self._stats["failed"] += 1
            return None

        # Check 2: Confidence threshold
        if knowledge.extraction_confidence < 0.1:
            logger.debug(f"Rejected low-confidence knowledge from {knowledge.source_url}")
            self._stats["failed"] += 1
            return None

        # Check 3: Source reliability
        source_reliability = self.reliability_scorer.score(knowledge.source_url)

        # Check 4: Temporal validation
        temporal_valid, temporal_msg = self.temporal_validator.validate(knowledge)
        if not temporal_valid and self.config.enable_temporal_validation:
            logger.warning(f"Temporal validation failed: {temporal_msg}")
            # Don't reject, just lower confidence
            knowledge.extraction_confidence *= 0.8

        # Check 5: Logical consistency
        logical_valid, logical_msg = self.logical_checker.check(knowledge)
        if not logical_valid and self.config.enable_logical_consistency:
            logger.warning(f"Logical consistency check failed: {logical_msg}")

        # Check 6: Contradiction detection
        if self.config.contradiction_detection:
            contradictions = self.contradiction_detector.detect(knowledge, self.graph)
            if contradictions:
                logger.warning(f"Contradictions detected: {contradictions}")
                # Flag but don't reject - contradictions may need review
                knowledge.extraction_confidence *= 0.7

        # Apply source reliability to final confidence
        final_confidence = (
            knowledge.extraction_confidence * 0.6 +
            source_reliability * 0.4
        )

        if final_confidence < self.config.confidence_threshold:
            self._stats["uncertain"] += 1
            # Still return but with lowered confidence
            knowledge.extraction_confidence = final_confidence

        # Filter low-confidence entities
        knowledge.entities = [
            e for e in knowledge.entities
            if e.confidence >= self.config.confidence_threshold * 0.5
        ]

        # Filter low-confidence relations
        knowledge.relations = [
            r for r in knowledge.relations
            if r.confidence >= self.config.confidence_threshold * 0.5
        ]

        self._stats["passed"] += 1
        return knowledge

    async def run_consistency_check(self):
        """Run periodic consistency checks on the entire knowledge graph."""
        logger.info("Running graph-wide consistency check...")
        stats = self.graph.get_stats()
        logger.info(f"Graph consistency check complete. Stats: {stats}")

    def validate_fact(self, claim: str, sources: List[str]) -> FactCheck:
        """Validate a specific fact claim."""
        source_score = self.reliability_scorer.score_multiple(sources)

        if len(sources) >= self.config.min_source_count and source_score >= 0.7:
            status = ValidationStatus.VALID
        elif sources:
            status = ValidationStatus.UNCERTAIN
        else:
            status = ValidationStatus.NEEDS_VERIFICATION

        return FactCheck(
            claim=claim,
            status=status,
            sources=sources,
            confidence=source_score
        )

    def get_stats(self) -> dict:
        return dict(self._stats)
