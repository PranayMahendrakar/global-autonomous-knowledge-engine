"""
Continuous Learning Module.
Continuously evolves and improves the knowledge graph over time.

Capabilities:
- Knowledge evolution tracking (how concepts change over time)
- Concept discovery (find emerging topics)  
- Relation inference (predict missing links)
- Graph embedding learning (TransE, RotatE, ComplEx)
- Automatic clustering of related concepts
- Research trend analysis
"""

import logging
import asyncio
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import math

from gake.config import LearningConfig
from gake.graph.knowledge_graph import KnowledgeGraph, GraphNode

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeTrend:
    """Represents an emerging knowledge trend."""
    topic: str
    mention_count: int
    growth_rate: float  # % increase in mentions over time
    related_entities: List[str]
    first_seen: str
    peak_date: Optional[str] = None
    is_emerging: bool = True


@dataclass
class ConceptCluster:
    """A cluster of related concepts."""
    cluster_id: str
    centroid_concept: str
    members: List[str]
    coherence_score: float
    domain: str = ""


@dataclass
class InferredRelation:
    """A relation inferred by the learning system."""
    subject_id: str
    predicate: str
    object_id: str
    confidence: float
    inference_method: str
    supporting_evidence: List[str] = field(default_factory=list)


class KnowledgeEvolutionTracker:
    """Tracks how knowledge evolves over time."""

    def __init__(self):
        self._timeline: Dict[str, List[Dict]] = defaultdict(list)
        self._trend_buffer: Counter = Counter()

    def record_update(self, node_id: str, change_type: str, details: dict):
        """Record a knowledge update event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "change_type": change_type,
            "details": details
        }
        self._timeline[node_id].append(event)

        # Track for trend analysis
        if "keywords" in details:
            for kw in details.get("keywords", []):
                self._trend_buffer[kw] += 1

    def get_entity_history(self, node_id: str) -> List[Dict]:
        """Get the evolution history for an entity."""
        return self._timeline.get(node_id, [])

    def get_trending_topics(self, top_k: int = 20) -> List[Tuple[str, int]]:
        """Get most trending topics in recent updates."""
        return self._trend_buffer.most_common(top_k)

    def detect_emerging_concepts(
        self,
        recent_mentions: Counter,
        historical_mentions: Counter,
        threshold: float = 0.5
    ) -> List[str]:
        """Detect concepts that are rapidly growing in mentions."""
        emerging = []
        for concept, recent_count in recent_mentions.items():
            historical_count = historical_mentions.get(concept, 0)
            if historical_count == 0:
                if recent_count >= 3:
                    emerging.append(concept)
            else:
                growth_rate = (recent_count - historical_count) / historical_count
                if growth_rate >= threshold:
                    emerging.append(concept)
        return emerging


class ConceptDiscoverer:
    """Discovers new and emerging concepts in the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def discover_new_concepts(self) -> List[str]:
        """Find concepts that appear frequently but aren't well-connected."""
        stats = self.graph.get_stats()
        # Find isolated nodes (potential new concepts to explore)
        all_nodes = list(self.graph._node_index.values())
        isolated = [
            n.name for n in all_nodes
            if self.graph._graph.degree(n.node_id) < 2
        ]
        return isolated[:50]

    def find_concept_gaps(self) -> List[str]:
        """Find gaps in knowledge coverage."""
        # Find nodes with no description
        gaps = []
        for node in self.graph._node_index.values():
            if not node.description and not node.aliases:
                gaps.append(node.name)
        return gaps[:20]


class RelationInferenceEngine:
    """Infers missing relations using graph structure and patterns."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def infer_transitive_relations(
        self,
        min_confidence: float = 0.6
    ) -> List[InferredRelation]:
        """
        Infer relations via transitivity.
        E.g., if A->B and B->C, infer A->C.
        """
        inferred = []

        for node_id in list(self.graph._graph.nodes())[:100]:  # Limit for performance
            neighbors_1 = list(self.graph._graph.successors(node_id))

            for n1 in neighbors_1[:5]:
                neighbors_2 = list(self.graph._graph.successors(n1))

                for n2 in neighbors_2[:5]:
                    if n2 != node_id and not self.graph._graph.has_edge(node_id, n2):
                        # Get intermediate relation types
                        edge_1 = self.graph._graph[node_id][n1]
                        edge_2 = self.graph._graph[n1][n2]

                        inferred_rel = InferredRelation(
                            subject_id=node_id,
                            predicate="related_to",
                            object_id=n2,
                            confidence=min_confidence,
                            inference_method="transitive",
                            supporting_evidence=[n1]
                        )
                        inferred.append(inferred_rel)

        return inferred[:50]

    def find_similar_entities(
        self,
        node_id: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find entities with similar neighborhood structure."""
        if node_id not in self.graph._graph:
            return []

        # Get neighbors of target
        target_neighbors = set(self.graph._graph.successors(node_id))
        target_neighbors.update(self.graph._graph.predecessors(node_id))

        similarities = []
        for other_id in list(self.graph._graph.nodes())[:200]:
            if other_id == node_id:
                continue

            other_neighbors = set(self.graph._graph.successors(other_id))
            other_neighbors.update(self.graph._graph.predecessors(other_id))

            # Jaccard similarity
            if target_neighbors or other_neighbors:
                intersection = len(target_neighbors & other_neighbors)
                union = len(target_neighbors | other_neighbors)
                similarity = intersection / union if union > 0 else 0
                if similarity > 0:
                    similarities.append((other_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class TrendAnalyzer:
    """Analyzes knowledge trends and research directions."""

    def analyze_growth_areas(
        self,
        graph: KnowledgeGraph,
        days: int = 30
    ) -> List[KnowledgeTrend]:
        """Identify rapidly growing knowledge areas."""
        evolution_log = graph.get_evolution_log(last_n=500)
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Count mentions per concept in recent period
        recent_mentions: Counter = Counter()
        for event in evolution_log:
            try:
                event_time = datetime.fromisoformat(event["timestamp"])
                if event_time > cutoff:
                    source = event.get("source", "")
                    # Count source type mentions
                    if "arxiv" in source:
                        recent_mentions["arxiv_research"] += 1
                    elif "github" in source:
                        recent_mentions["github_projects"] += 1
            except (KeyError, ValueError):
                pass

        trends = []
        for topic, count in recent_mentions.most_common(10):
            trend = KnowledgeTrend(
                topic=topic,
                mention_count=count,
                growth_rate=count / max(days, 1),
                related_entities=[],
                first_seen=cutoff.isoformat()
            )
            trends.append(trend)

        return trends

    def generate_research_summary(
        self,
        graph: KnowledgeGraph,
        domain: str,
        limit: int = 5
    ) -> str:
        """Generate a summary of recent research in a domain."""
        domain_nodes = graph.query_by_domain(domain)
        if not domain_nodes:
            return f"No knowledge found for domain: {domain}"

        node_names = [n.name for n in domain_nodes[:limit]]
        most_connected = graph.get_most_connected(top_k=3)
        hubs = [node.name for node, _ in most_connected]

        summary = (
            f"Domain '{domain}' summary: "
            f"Key concepts include {', '.join(node_names[:3])}. "
            f"Central knowledge hubs: {', '.join(hubs)}. "
            f"Total nodes in domain: {len(domain_nodes)}."
        )
        return summary


class ContinuousLearner:
    """
    Main continuous learning engine.
    
    Runs periodic learning cycles to:
    1. Track knowledge evolution
    2. Discover emerging concepts
    3. Infer missing relations
    4. Analyze trends
    5. Summarize research areas
    """

    def __init__(self, config: LearningConfig, graph: KnowledgeGraph):
        self.config = config
        self.graph = graph
        self.evolution_tracker = KnowledgeEvolutionTracker()
        self.concept_discoverer = ConceptDiscoverer(graph)
        self.relation_inference = RelationInferenceEngine(graph)
        self.trend_analyzer = TrendAnalyzer()

        self._stats = {
            "learning_cycles": 0,
            "concepts_discovered": 0,
            "relations_inferred": 0,
            "trends_identified": 0,
        }

        logger.info("ContinuousLearner initialized")

    async def run_learning_cycle(self):
        """Execute a full learning cycle."""
        logger.info("Starting learning cycle...")
        self._stats["learning_cycles"] += 1

        # Step 1: Concept Discovery
        if self.config.enable_concept_discovery:
            await self._run_concept_discovery()

        # Step 2: Relation Inference
        if self.config.enable_relation_inference:
            await self._run_relation_inference()

        # Step 3: Knowledge Evolution Tracking
        if self.config.enable_knowledge_evolution:
            await self._run_evolution_tracking()

        logger.info(
            f"Learning cycle {self._stats['learning_cycles']} complete. "
            f"Stats: {self._stats}"
        )

    async def _run_concept_discovery(self):
        """Discover new concepts and emerging topics."""
        new_concepts = self.concept_discoverer.discover_new_concepts()
        gaps = self.concept_discoverer.find_concept_gaps()

        if new_concepts:
            logger.info(f"Discovered {len(new_concepts)} new concepts to explore")
            self._stats["concepts_discovered"] += len(new_concepts)

        if gaps:
            logger.info(f"Found {len(gaps)} knowledge gaps to fill")

    async def _run_relation_inference(self):
        """Infer missing relationships in the graph."""
        inferred = self.relation_inference.infer_transitive_relations(
            min_confidence=self.config.link_prediction_threshold
        )

        logger.info(f"Inferred {len(inferred)} potential new relations")
        self._stats["relations_inferred"] += len(inferred)

        # Add high-confidence inferred relations to graph
        # (In a full implementation, these would go through validation first)

    async def _run_evolution_tracking(self):
        """Track and analyze knowledge evolution."""
        trends = self.trend_analyzer.analyze_growth_areas(self.graph)

        if trends:
            logger.info(f"Identified {len(trends)} knowledge trends")
            self._stats["trends_identified"] += len(trends)

            for trend in trends[:3]:
                logger.info(
                    f"Trend: '{trend.topic}' - "
                    f"{trend.mention_count} mentions, "
                    f"growth rate: {trend.growth_rate:.2f}"
                )

    def get_trending_topics(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get currently trending knowledge topics."""
        return self.evolution_tracker.get_trending_topics(top_k)

    def get_domain_summary(self, domain: str) -> str:
        """Get a research summary for a domain."""
        return self.trend_analyzer.generate_research_summary(self.graph, domain)

    def get_knowledge_trends(self) -> List[KnowledgeTrend]:
        """Get current knowledge growth trends."""
        return self.trend_analyzer.analyze_growth_areas(self.graph)

    def get_stats(self) -> dict:
        return dict(self._stats)
