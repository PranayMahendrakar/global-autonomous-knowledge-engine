"""
Knowledge Graph Construction and Management.

Stores all extracted knowledge in a graph structure:
- Nodes: people, concepts, technologies, events, papers
- Edges: relationships (discovered, related_to, part_of, etc.)

Supports multiple backends:
- NetworkX (in-memory, for development)
- Neo4j (production graph database)
- RDFLib (semantic web / linked data)
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator
from dataclasses import dataclass, field, asdict

import networkx as nx

from gake.config import GraphConfig
from gake.extraction.knowledge_extractor import (
    Entity, Relation, Concept, ExtractedKnowledge,
    EntityType, RelationType
)

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    node_id: str
    entity_type: EntityType
    name: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    version: int = 1


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""
    edge_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    context: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    Core knowledge graph that stores and manages all extracted knowledge.

    Provides:
    - Entity (node) management with deduplication
    - Relationship (edge) management
    - Graph traversal and querying
    - Knowledge evolution tracking
    - Persistence (save/load)
    - Statistics and analytics
    """

    def __init__(self, config: GraphConfig):
        self.config = config
        self._graph = nx.MultiDiGraph()
        self._node_index: Dict[str, GraphNode] = {}      # node_id -> GraphNode
        self._alias_index: Dict[str, str] = {}            # alias -> node_id
        self._edge_index: Dict[str, GraphEdge] = {}       # edge_id -> GraphEdge
        self._domain_index: Dict[str, Set[str]] = {}      # domain -> Set[node_ids]
        self._evolution_log: List[Dict] = []              # Change history

        self._stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "total_integrations": 0,
            "last_update": None
        }

        # Load existing graph if available
        persistence_path = Path(config.persistence_path)
        if persistence_path.exists():
            self._load(persistence_path)
            logger.info(f"Loaded existing knowledge graph: "
                       f"{self._stats['total_nodes']} nodes, "
                       f"{self._stats['total_edges']} edges")
        else:
            logger.info("Initialized new knowledge graph")

    async def integrate(self, knowledge: ExtractedKnowledge) -> Dict[str, int]:
        """
        Integrate extracted knowledge into the graph.
        Handles entity resolution, deduplication, and relationship creation.

        Returns counts of added/updated nodes and edges.
        """
        stats = {
            "new_nodes": 0,
            "updated_nodes": 0,
            "new_edges": 0,
            "new_concepts": 0
        }

        # Process entities -> nodes
        entity_to_node: Dict[str, str] = {}  # entity text -> node_id

        for entity in knowledge.entities:
            node_id = self._resolve_entity(entity)

            if node_id is None:
                # New entity - create node
                node_id = self._create_node(entity, knowledge.source_url)
                stats["new_nodes"] += 1
            else:
                # Existing entity - update with new info
                self._update_node(node_id, entity, knowledge.source_url)
                stats["updated_nodes"] += 1

            entity_to_node[entity.canonical_name] = node_id

        # Process relations -> edges
        for relation in knowledge.relations:
            subj_id = entity_to_node.get(relation.subject.canonical_name)
            obj_id = entity_to_node.get(relation.obj.canonical_name)

            if subj_id and obj_id:
                edge_id = self._create_edge(
                    subj_id, obj_id, relation,
                    knowledge.source_url
                )
                if edge_id:
                    stats["new_edges"] += 1

        # Process concepts
        for concept in knowledge.concepts:
            if concept.name and len(concept.name) > 2:
                concept_entity = Entity(
                    text=concept.name,
                    entity_type=EntityType.CONCEPT,
                    description=concept.definition,
                    confidence=concept.confidence
                )
                node_id = self._resolve_entity(concept_entity)
                if node_id is None:
                    self._create_node(concept_entity, knowledge.source_url)
                    stats["new_concepts"] += 1

        # Update domain index
        if knowledge.domain:
            if knowledge.domain not in self._domain_index:
                self._domain_index[knowledge.domain] = set()
            for node_id in entity_to_node.values():
                self._domain_index[knowledge.domain].add(node_id)

        # Log evolution
        self._evolution_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "source": knowledge.source_url,
            "changes": stats
        })

        # Update stats
        self._stats["total_integrations"] += 1
        self._stats["last_update"] = datetime.utcnow().isoformat()
        self._stats["total_nodes"] = len(self._node_index)
        self._stats["total_edges"] = len(self._edge_index)

        return stats

    def _resolve_entity(self, entity: Entity) -> Optional[str]:
        """
        Find if an entity already exists in the graph.
        Uses canonical name and aliases for matching.
        """
        # Check by canonical name
        if entity.canonical_name in self._alias_index:
            return self._alias_index[entity.canonical_name]

        # Check by aliases
        for alias in entity.aliases:
            alias_lower = alias.lower()
            if alias_lower in self._alias_index:
                return self._alias_index[alias_lower]

        return None

    def _create_node(self, entity: Entity, source_url: str = "") -> str:
        """Create a new node in the graph."""
        node_id = f"{entity.entity_type.value}:{entity.canonical_name}"

        node = GraphNode(
            node_id=node_id,
            entity_type=entity.entity_type,
            name=entity.text,
            description=entity.description,
            aliases=entity.aliases + [entity.text, entity.canonical_name],
            sources=[source_url] if source_url else [],
            confidence=entity.confidence,
            metadata=entity.metadata
        )

        self._graph.add_node(node_id, **asdict(node))
        self._node_index[node_id] = node

        # Update alias index
        self._alias_index[entity.canonical_name] = node_id
        for alias in node.aliases:
            self._alias_index[alias.lower()] = node_id

        return node_id

    def _update_node(self, node_id: str, entity: Entity, source_url: str = ""):
        """Update an existing node with new information."""
        node = self._node_index.get(node_id)
        if not node:
            return

        # Add new source
        if source_url and source_url not in node.sources:
            node.sources.append(source_url)

        # Update confidence (average)
        node.confidence = (node.confidence + entity.confidence) / 2

        # Add new aliases
        for alias in entity.aliases:
            if alias not in node.aliases:
                node.aliases.append(alias)
                self._alias_index[alias.lower()] = node_id

        # Update metadata
        node.metadata.update(entity.metadata)
        node.updated_at = datetime.utcnow().isoformat()
        node.version += 1

        # Update graph node data
        self._graph.nodes[node_id].update(asdict(node))

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        relation: Relation,
        source_url: str = ""
    ) -> Optional[str]:
        """Create a new edge in the graph."""
        edge_id = f"{source_id}-{relation.predicate.value}-{target_id}"

        # Check if edge already exists
        if edge_id in self._edge_index:
            # Update existing edge
            edge = self._edge_index[edge_id]
            if source_url not in edge.sources:
                edge.sources.append(source_url)
            edge.weight += 0.1  # Reinforce with each confirmation
            return None  # No new edge

        edge = GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation.predicate,
            confidence=relation.confidence,
            sources=[source_url] if source_url else [],
            context=relation.context,
            metadata=relation.metadata
        )

        self._graph.add_edge(
            source_id, target_id,
            key=edge_id,
            **{k: v for k, v in asdict(edge).items() if v is not None}
        )
        self._edge_index[edge_id] = edge

        return edge_id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._node_index.get(node_id)

    def find_node_by_name(self, name: str) -> Optional[GraphNode]:
        """Find a node by name or alias."""
        node_id = self._alias_index.get(name.lower())
        if node_id:
            return self._node_index.get(node_id)
        return None

    def get_neighbors(
        self,
        node_id: str,
        relation_types: Optional[List[RelationType]] = None,
        depth: int = 1
    ) -> List[GraphNode]:
        """Get neighboring nodes."""
        if node_id not in self._graph:
            return []

        neighbors = []

        if depth == 1:
            for _, target_id in self._graph.edges(node_id):
                target_node = self._node_index.get(target_id)
                if target_node:
                    neighbors.append(target_node)
        else:
            # BFS traversal
            visited = {node_id}
            queue = [(node_id, 0)]

            while queue:
                curr_id, curr_depth = queue.pop(0)
                if curr_depth >= depth:
                    continue

                for _, next_id in self._graph.edges(curr_id):
                    if next_id not in visited:
                        visited.add(next_id)
                        node = self._node_index.get(next_id)
                        if node:
                            neighbors.append(node)
                        queue.append((next_id, curr_depth + 1))

        return neighbors

    def find_path(
        self,
        source_name: str,
        target_name: str
    ) -> Optional[List[GraphNode]]:
        """Find shortest path between two entities."""
        source_node = self.find_node_by_name(source_name)
        target_node = self.find_node_by_name(target_name)

        if not source_node or not target_node:
            return None

        try:
            path_ids = nx.shortest_path(
                self._graph,
                source_node.node_id,
                target_node.node_id
            )
            return [self._node_index[nid] for nid in path_ids
                   if nid in self._node_index]
        except nx.NetworkXNoPath:
            return None

    def query_by_domain(self, domain: str) -> List[GraphNode]:
        """Get all nodes in a specific knowledge domain."""
        node_ids = self._domain_index.get(domain, set())
        return [self._node_index[nid] for nid in node_ids
                if nid in self._node_index]

    def get_most_connected(self, top_k: int = 10) -> List[Tuple[GraphNode, int]]:
        """Get most connected nodes (hubs in the knowledge graph)."""
        degrees = [(nid, self._graph.degree(nid))
                  for nid in self._graph.nodes()]
        degrees.sort(key=lambda x: x[1], reverse=True)

        return [
            (self._node_index[nid], degree)
            for nid, degree in degrees[:top_k]
            if nid in self._node_index
        ]

    def search(self, query: str, limit: int = 10) -> List[GraphNode]:
        """Search nodes by text similarity."""
        query_lower = query.lower()
        results = []

        for node in self._node_index.values():
            score = 0
            if query_lower in node.name.lower():
                score += 2
            if any(query_lower in alias.lower() for alias in node.aliases):
                score += 1
            if query_lower in node.description.lower():
                score += 0.5

            if score > 0:
                results.append((node, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in results[:limit]]

    async def optimize(self):
        """Optimize graph structure - remove low confidence edges, merge duplicates."""
        removed_edges = 0
        for edge_id, edge in list(self._edge_index.items()):
            if edge.confidence < 0.3:
                self._graph.remove_edges_from(
                    [(edge.source_id, edge.target_id, edge_id)]
                )
                del self._edge_index[edge_id]
                removed_edges += 1

        if removed_edges > 0:
            logger.info(f"Graph optimization: removed {removed_edges} low-confidence edges")

    async def save(self):
        """Persist knowledge graph to disk."""
        path = Path(self.config.persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "graph": self._graph,
            "node_index": self._node_index,
            "alias_index": self._alias_index,
            "edge_index": self._edge_index,
            "domain_index": self._domain_index,
            "evolution_log": self._evolution_log[-1000:],  # Keep last 1000 entries
            "stats": self._stats
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Knowledge graph saved: {path}")

    def _load(self, path: Path):
        """Load knowledge graph from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._graph = data["graph"]
        self._node_index = data["node_index"]
        self._alias_index = data["alias_index"]
        self._edge_index = data["edge_index"]
        self._domain_index = data.get("domain_index", {})
        self._evolution_log = data.get("evolution_log", [])
        self._stats = data.get("stats", self._stats)

    def get_stats(self) -> dict:
        """Return graph statistics."""
        stats = dict(self._stats)
        stats["total_nodes"] = len(self._node_index)
        stats["total_edges"] = len(self._edge_index)
        stats["domains"] = list(self._domain_index.keys())
        stats["node_types"] = {}

        for node in self._node_index.values():
            type_name = node.entity_type.value
            stats["node_types"][type_name] = stats["node_types"].get(type_name, 0) + 1

        return stats

    def get_evolution_log(self, last_n: int = 100) -> List[Dict]:
        """Return recent knowledge evolution events."""
        return self._evolution_log[-last_n:]

    def export_to_dict(self) -> Dict:
        """Export graph as a dictionary for serialization."""
        return {
            "nodes": [asdict(n) for n in self._node_index.values()],
            "edges": [asdict(e) for e in self._edge_index.values()],
            "stats": self.get_stats()
        }
