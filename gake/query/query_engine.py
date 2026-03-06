"""
Query Engine - Natural Language Interface to the Knowledge Graph.

Answers complex questions like:
- "What new ML techniques appeared this year?"
- "What research papers connect LLMs and robotics?"
- "What are the most important concepts in quantum computing?"
- "How is Einstein related to modern AI?"
- "What is the evolution of transformer architectures?"

Query types:
- Entity lookup
- Relationship queries
- Path finding
- Domain exploration
- Trend analysis
- Cross-domain reasoning
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from gake.graph.knowledge_graph import KnowledgeGraph, GraphNode
from gake.extraction.knowledge_extractor import EntityType, RelationType

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a knowledge query."""
    question: str
    answer: str
    entities: List[GraphNode] = field(default_factory=list)
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    related_questions: List[str] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class QueryParser:
    """Parses natural language queries to identify intent and entities."""

    QUERY_PATTERNS = {
        "what_is": r"what\s+is\s+(.+)\??",
        "who_is": r"who\s+is\s+(.+)\??",
        "how_related": r"how\s+is\s+(.+)\s+related\s+to\s+(.+)\??",
        "what_new": r"what\s+new\s+(.+)\s+(?:appeared|emerged|came out)\s+(.+)\??",
        "what_connects": r"what\s+(?:connects|links|bridges)\s+(.+)\s+and\s+(.+)\??",
        "find_papers": r"(?:find|show|list)\s+(?:research\s+)?papers?\s+(?:about|on|connecting)\s+(.+)",
        "what_is_evolution": r"(?:what|how)\s+is\s+the\s+evolution\s+of\s+(.+)\??",
        "most_important": r"what\s+are\s+the\s+most\s+important\s+(.+)\s+in\s+(.+)\??",
        "compare": r"compare\s+(.+)\s+(?:and|with|vs)\s+(.+)",
    }

    def parse(self, question: str) -> Dict[str, Any]:
        """Parse a question and extract intent and entities."""
        question_lower = question.lower().strip()

        for intent, pattern in self.QUERY_PATTERNS.items():
            match = re.search(pattern, question_lower)
            if match:
                return {
                    "intent": intent,
                    "entities": list(match.groups()),
                    "original": question
                }

        # Fallback: extract key nouns
        words = question.split()
        key_terms = [w for w in words if len(w) > 3 and w.lower() not in
                    {"what", "how", "who", "when", "where", "which", "that",
                     "this", "are", "the", "and", "for", "with"}]

        return {
            "intent": "general",
            "entities": key_terms[:3],
            "original": question
        }


class EntityResolver:
    """Resolves entity mentions to graph nodes."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def resolve(self, entity_text: str) -> List[GraphNode]:
        """Find graph nodes matching an entity mention."""
        # Direct lookup
        node = self.graph.find_node_by_name(entity_text)
        if node:
            return [node]

        # Search
        return self.graph.search(entity_text, limit=5)


class AnswerGenerator:
    """Generates natural language answers from graph query results."""

    def generate_entity_answer(self, entity: GraphNode, question: str) -> str:
        """Generate answer for entity-type questions."""
        parts = [f"'{entity.name}' is a {entity.entity_type.value}"]

        if entity.description:
            parts.append(f". {entity.description[:300]}")

        if entity.aliases and len(entity.aliases) > 1:
            other_names = [a for a in entity.aliases[:3] if a != entity.name]
            if other_names:
                parts.append(f". Also known as: {', '.join(other_names)}")

        if entity.sources:
            parts.append(f". Sources: {len(entity.sources)} reference(s)")

        return "".join(parts)

    def generate_path_answer(
        self,
        source: GraphNode,
        target: GraphNode,
        path: List[GraphNode]
    ) -> str:
        """Generate answer for relationship queries."""
        if not path:
            return f"No direct connection found between '{source.name}' and '{target.name}'."

        if len(path) == 2:
            return f"'{source.name}' is directly connected to '{target.name}'."

        intermediate = [n.name for n in path[1:-1]]
        return (
            f"'{source.name}' connects to '{target.name}' through: "
            f"{' -> '.join([source.name] + intermediate + [target.name])}. "
            f"Path length: {len(path) - 1} steps."
        )

    def generate_list_answer(
        self,
        items: List[GraphNode],
        query_topic: str
    ) -> str:
        """Generate answer for list-type queries."""
        if not items:
            return f"No results found for '{query_topic}'."

        item_names = [f"'{n.name}'" for n in items[:5]]
        answer = f"Found {len(items)} result(s) for '{query_topic}': "
        answer += ", ".join(item_names)

        if len(items) > 5:
            answer += f", and {len(items) - 5} more."

        return answer

    def generate_cross_domain_answer(
        self,
        domain1: str,
        domain2: str,
        connecting_nodes: List[GraphNode]
    ) -> str:
        """Generate answer for cross-domain reasoning queries."""
        if not connecting_nodes:
            return f"No direct connections found between '{domain1}' and '{domain2}'."

        connector_names = [n.name for n in connecting_nodes[:3]]
        return (
            f"The domains '{domain1}' and '{domain2}' are connected through: "
            f"{', '.join(connector_names)}. "
            f"Found {len(connecting_nodes)} bridging concept(s)."
        )


class CrossDomainReasoner:
    """Performs reasoning across different knowledge domains."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def find_connections(
        self,
        domain1: str,
        domain2: str
    ) -> List[GraphNode]:
        """Find concepts that bridge two knowledge domains."""
        nodes_d1 = set(n.node_id for n in self.graph.query_by_domain(domain1))
        nodes_d2 = set(n.node_id for n in self.graph.query_by_domain(domain2))

        # Find nodes that connect to both domains
        bridging = []
        all_nodes = list(self.graph._node_index.values())

        for node in all_nodes[:500]:  # Sample for performance
            neighbors = set(n.node_id for n in self.graph.get_neighbors(node.node_id, depth=2))
            if (neighbors & nodes_d1) and (neighbors & nodes_d2):
                bridging.append(node)

        return bridging[:10]

    def reason_analogy(
        self,
        source_concept: str,
        target_domain: str
    ) -> Optional[str]:
        """Find analogous concepts across domains."""
        source_node = self.graph.find_node_by_name(source_concept)
        if not source_node:
            return None

        # Find nodes in target domain with similar connectivity
        domain_nodes = self.graph.query_by_domain(target_domain)
        if not domain_nodes:
            return None

        # Find the most structurally similar node
        source_degree = self.graph._graph.degree(source_node.node_id)
        best_match = None
        min_diff = float("inf")

        for node in domain_nodes[:50]:
            node_degree = self.graph._graph.degree(node.node_id)
            diff = abs(source_degree - node_degree)
            if diff < min_diff:
                min_diff = diff
                best_match = node

        return best_match.name if best_match else None


class QueryEngine:
    """
    Natural language query interface to the knowledge graph.

    Supports complex queries:
    - "What new ML techniques appeared this year?"
    - "What research papers connect LLMs and robotics?"
    - "How is attention mechanism related to memory?"
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.parser = QueryParser()
        self.resolver = EntityResolver(graph)
        self.answer_gen = AnswerGenerator()
        self.cross_domain = CrossDomainReasoner(graph)
        self._query_count = 0

    def answer(self, question: str) -> QueryResult:
        """Answer a natural language question about the knowledge graph."""
        self._query_count += 1
        logger.info(f"Query #{self._query_count}: {question}")

        # Parse the question
        parsed = self.parser.parse(question)
        intent = parsed["intent"]
        entity_texts = parsed["entities"]

        # Dispatch to appropriate handler
        handlers = {
            "what_is": self._handle_what_is,
            "who_is": self._handle_what_is,
            "how_related": self._handle_how_related,
            "what_new": self._handle_what_new,
            "what_connects": self._handle_what_connects,
            "find_papers": self._handle_find_papers,
            "what_is_evolution": self._handle_evolution,
            "most_important": self._handle_most_important,
            "compare": self._handle_compare,
            "general": self._handle_general,
        }

        handler = handlers.get(intent, self._handle_general)
        result = handler(question, entity_texts)
        return result

    def _handle_what_is(self, question: str, entities: List[str]) -> QueryResult:
        """Handle 'what is X' questions."""
        if not entities:
            return QueryResult(question=question, answer="I couldn't identify what you're asking about.")

        entity_text = entities[0]
        nodes = self.resolver.resolve(entity_text)

        if not nodes:
            return QueryResult(
                question=question,
                answer=f"I don't have information about '{entity_text}' in my knowledge base yet.",
                confidence=0.0
            )

        primary = nodes[0]
        answer = self.answer_gen.generate_entity_answer(primary, question)

        # Find related entities
        related = self.graph.get_neighbors(primary.node_id)

        return QueryResult(
            question=question,
            answer=answer,
            entities=[primary] + related[:3],
            confidence=primary.confidence,
            sources=primary.sources[:3],
            related_questions=[
                f"How is {primary.name} related to AI?",
                f"What are the applications of {primary.name}?",
                f"Who discovered {primary.name}?"
            ]
        )

    def _handle_how_related(self, question: str, entities: List[str]) -> QueryResult:
        """Handle 'how is X related to Y' questions."""
        if len(entities) < 2:
            return QueryResult(question=question, answer="Please specify two entities to compare.")

        source_nodes = self.resolver.resolve(entities[0])
        target_nodes = self.resolver.resolve(entities[1])

        if not source_nodes or not target_nodes:
            missing = entities[0] if not source_nodes else entities[1]
            return QueryResult(
                question=question,
                answer=f"Could not find '{missing}' in the knowledge base."
            )

        source = source_nodes[0]
        target = target_nodes[0]
        path = self.graph.find_path(source.name, target.name)
        answer = self.answer_gen.generate_path_answer(source, target, path or [])

        return QueryResult(
            question=question,
            answer=answer,
            entities=[source, target],
            confidence=0.8 if path else 0.3,
            reasoning_path=[n.name for n in (path or [])]
        )

    def _handle_what_new(self, question: str, entities: List[str]) -> QueryResult:
        """Handle 'what new X appeared' questions."""
        topic = entities[0] if entities else "technology"
        time_ref = entities[1] if len(entities) > 1 else "recently"

        # Search for recent nodes matching the topic
        results = self.graph.search(topic, limit=10)

        # Filter to recent ones (based on created_at)
        recent_results = sorted(
            results,
            key=lambda n: n.created_at,
            reverse=True
        )[:5]

        answer = self.answer_gen.generate_list_answer(recent_results, f"new {topic}")

        return QueryResult(
            question=question,
            answer=answer,
            entities=recent_results,
            confidence=0.7,
            related_questions=[
                f"What are the key features of {topic}?",
                f"How has {topic} evolved?",
                f"Who are the main researchers in {topic}?"
            ]
        )

    def _handle_what_connects(self, question: str, entities: List[str]) -> QueryResult:
        """Handle 'what connects/links X and Y' questions."""
        if len(entities) < 2:
            return QueryResult(question=question, answer="Please specify two domains to connect.")

        d1, d2 = entities[0], entities[1]
        connecting_nodes = self.cross_domain.find_connections(d1, d2)
        answer = self.answer_gen.generate_cross_domain_answer(d1, d2, connecting_nodes)

        return QueryResult(
            question=question,
            answer=answer,
            entities=connecting_nodes[:5],
            confidence=0.75 if connecting_nodes else 0.2
        )

    def _handle_find_papers(self, question: str, entities: List[str]) -> QueryResult:
        """Handle 'find papers about X' questions."""
        topic = entities[0] if entities else ""
        results = self.graph.search(topic, limit=10)

        # Filter for paper-type entities
        papers = [n for n in results if n.entity_type == EntityType.PAPER]
        if not papers:
            papers = results[:5]  # Fall back to any matching entities

        answer = self.answer_gen.generate_list_answer(papers, f"papers on {topic}")

        return QueryResult(
            question=question,
            answer=answer,
            entities=papers,
            confidence=0.6 if papers else 0.1
        )

    def _handle_evolution(self, question: str, entities: List[str]) -> QueryResult:
        """Handle knowledge evolution queries."""
        topic = entities[0] if entities else "AI"
        nodes = self.graph.search(topic, limit=1)

        if not nodes:
            return QueryResult(question=question, answer=f"No evolution data for '{topic}'.")

        node = nodes[0]
        history_count = len(node.sources)
        version = node.version

        answer = (
            f"The concept '{node.name}' has evolved through {version} version(s) "
            f"with {history_count} source reference(s). "
            f"First seen: {node.created_at[:10]}, "
            f"Last updated: {node.updated_at[:10]}."
        )

        return QueryResult(
            question=question,
            answer=answer,
            entities=[node],
            confidence=0.75
        )

    def _handle_most_important(self, question: str, entities: List[str]) -> QueryResult:
        """Handle 'most important X in Y' questions."""
        most_connected = self.graph.get_most_connected(top_k=5)
        items = [node for node, _ in most_connected]

        domain = entities[1] if len(entities) > 1 else "the knowledge graph"
        answer = self.answer_gen.generate_list_answer(items, f"important concepts in {domain}")

        return QueryResult(
            question=question,
            answer=answer,
            entities=items,
            confidence=0.8
        )

    def _handle_compare(self, question: str, entities: List[str]) -> QueryResult:
        """Handle 'compare X and Y' questions."""
        if len(entities) < 2:
            return QueryResult(question=question, answer="Please specify two things to compare.")

        nodes_a = self.resolver.resolve(entities[0])
        nodes_b = self.resolver.resolve(entities[1])

        if not nodes_a or not nodes_b:
            return QueryResult(question=question, answer="Could not find one or both entities.")

        a, b = nodes_a[0], nodes_b[0]
        a_degree = self.graph._graph.degree(a.node_id) if a.node_id in self.graph._graph else 0
        b_degree = self.graph._graph.degree(b.node_id) if b.node_id in self.graph._graph else 0

        answer = (
            f"Comparison: '{a.name}' vs '{b.name}'. "
            f"'{a.name}' has {a_degree} connections, "
            f"'{b.name}' has {b_degree} connections. "
            f"Both are of types: {a.entity_type.value} and {b.entity_type.value}. "
        )

        # Check if they're connected
        path = self.graph.find_path(a.name, b.name)
        if path:
            answer += f"They are connected by {len(path) - 1} step(s)."
        else:
            answer += "No direct connection found between them."

        return QueryResult(
            question=question,
            answer=answer,
            entities=[a, b],
            confidence=0.7
        )

    def _handle_general(self, question: str, entities: List[str]) -> QueryResult:
        """Handle general queries."""
        if not entities:
            stats = self.graph.get_stats()
            return QueryResult(
                question=question,
                answer=(
                    f"The knowledge graph contains {stats['total_nodes']} entities "
                    f"and {stats['total_edges']} relationships across "
                    f"{len(stats.get('domains', []))} domains."
                ),
                confidence=1.0
            )

        # Search for any matching entities
        all_results = []
        for entity_text in entities:
            results = self.graph.search(entity_text, limit=5)
            all_results.extend(results)

        if not all_results:
            return QueryResult(
                question=question,
                answer=f"No information found for the query: '{question}'",
                confidence=0.0
            )

        # Deduplicate
        seen = set()
        unique_results = []
        for n in all_results:
            if n.node_id not in seen:
                seen.add(n.node_id)
                unique_results.append(n)

        answer = self.answer_gen.generate_list_answer(unique_results, question)

        return QueryResult(
            question=question,
            answer=answer,
            entities=unique_results[:5],
            confidence=0.6
        )

    def get_query_count(self) -> int:
        return self._query_count
