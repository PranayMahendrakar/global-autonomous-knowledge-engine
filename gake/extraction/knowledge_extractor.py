"""
Knowledge Extraction Pipeline.
Extracts entities, relationships, concepts, and events from raw sources.

Example:
    "Einstein discovered relativity in 1905"
    -> Entity: Einstein (Person)
    -> Entity: Relativity (Theory)  
    -> Relation: discovered(Einstein, Relativity)
    -> Event: Discovery, date=1905
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from gake.config import ExtractionConfig
from gake.discovery.agent_manager import KnowledgeSource

logger = logging.getLogger(__name__)


class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    EVENT = "event"
    PAPER = "paper"
    DATASET = "dataset"
    ALGORITHM = "algorithm"
    METRIC = "metric"
    UNKNOWN = "unknown"


class RelationType(Enum):
    DISCOVERED = "discovered"
    DEVELOPED = "developed"
    USED_IN = "used_in"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    IMPROVES = "improves"
    CITES = "cites"
    AUTHORED_BY = "authored_by"
    APPLIES_TO = "applies_to"
    ACHIEVES = "achieves"
    BASED_ON = "based_on"
    COMPARED_TO = "compared_to"


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    entity_type: EntityType
    canonical_name: str = ""
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0
    source_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self.text.strip().lower()


@dataclass
class Relation:
    """Represents an extracted relationship between entities."""
    subject: Entity
    predicate: RelationType
    obj: Entity
    confidence: float = 0.0
    context: str = ""
    source_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Concept:
    """Represents an extracted concept or abstract idea."""
    name: str
    definition: str = ""
    domain: str = ""
    related_concepts: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source_url: str = ""


@dataclass
class Event:
    """Represents an extracted event."""
    name: str
    description: str = ""
    date: Optional[str] = None
    participants: List[Entity] = field(default_factory=list)
    location: str = ""
    confidence: float = 0.0
    source_url: str = ""


@dataclass
class ExtractedKnowledge:
    """Container for all knowledge extracted from a source."""
    source_url: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    concepts: List[Concept] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    domain: str = ""
    extraction_confidence: float = 0.0


class EntityExtractor:
    """Extracts named entities from text using NLP."""

    # Technology/AI keywords for domain-specific extraction
    AI_TERMS = {
        "transformer", "bert", "gpt", "llm", "neural network", "deep learning",
        "reinforcement learning", "attention mechanism", "embedding", "tokenizer",
        "fine-tuning", "pre-training", "zero-shot", "few-shot", "rag",
        "knowledge graph", "graph neural network", "diffusion model", "vae",
        "gan", "cnn", "rnn", "lstm", "gru", "resnet", "vit", "clip",
    }

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._nlp = None

    def _get_nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self.config.model_name)
            except (ImportError, OSError):
                logger.warning("spaCy model not available, using rule-based extraction")
                self._nlp = "rule-based"
        return self._nlp

    def extract(self, text: str, source_url: str = "") -> List[Entity]:
        """Extract entities from text."""
        entities = []
        nlp = self._get_nlp()

        if nlp == "rule-based":
            entities = self._rule_based_extraction(text, source_url)
        else:
            entities = self._spacy_extraction(text, source_url, nlp)

        # Add AI/tech specific entities
        entities.extend(self._extract_tech_entities(text, source_url))

        # Deduplicate
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e.canonical_name, e.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        return unique_entities[:self.config.max_entities_per_doc]

    def _spacy_extraction(self, text: str, source_url: str, nlp) -> List[Entity]:
        """Extract entities using spaCy NER."""
        entities = []
        doc = nlp(text[:10000])  # Limit text length

        type_map = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "PRODUCT": EntityType.TECHNOLOGY,
            "EVENT": EntityType.EVENT,
            "WORK_OF_ART": EntityType.CONCEPT,
        }

        for ent in doc.ents:
            entity_type = type_map.get(ent.label_, EntityType.UNKNOWN)
            entity = Entity(
                text=ent.text,
                entity_type=entity_type,
                confidence=0.85,
                source_url=source_url
            )
            entities.append(entity)

        return entities

    def _rule_based_extraction(self, text: str, source_url: str) -> List[Entity]:
        """Simple rule-based entity extraction fallback."""
        entities = []

        # Extract capitalized sequences (potential named entities)
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        matches = re.findall(pattern, text)

        for match in matches[:20]:
            if len(match.split()) >= 2:  # Multi-word proper nouns
                entity = Entity(
                    text=match,
                    entity_type=EntityType.UNKNOWN,
                    confidence=0.5,
                    source_url=source_url
                )
                entities.append(entity)

        return entities

    def _extract_tech_entities(self, text: str, source_url: str) -> List[Entity]:
        """Extract AI/ML technology terms."""
        entities = []
        text_lower = text.lower()

        for term in self.AI_TERMS:
            if term in text_lower:
                entity = Entity(
                    text=term,
                    entity_type=EntityType.TECHNOLOGY,
                    confidence=0.9,
                    source_url=source_url,
                    metadata={"domain": "AI/ML"}
                )
                entities.append(entity)

        return entities


class RelationExtractor:
    """Extracts semantic relationships between entities."""

    # Relation patterns
    PATTERNS = [
        (r'(\w+)\s+(?:discovered|found|invented)\s+(\w+)', RelationType.DISCOVERED),
        (r'(\w+)\s+(?:developed|created|built)\s+(\w+)', RelationType.DEVELOPED),
        (r'(\w+)\s+(?:improves?|outperforms?)\s+(\w+)', RelationType.IMPROVES),
        (r'(\w+)\s+(?:is based on|extends|builds on)\s+(\w+)', RelationType.BASED_ON),
        (r'(\w+)\s+(?:uses?|applies?)\s+(\w+)', RelationType.USED_IN),
        (r'(\w+)\s+(?:is part of|belongs to)\s+(\w+)', RelationType.PART_OF),
    ]

    def __init__(self, config: ExtractionConfig):
        self.config = config

    def extract(self, text: str, entities: List[Entity],
                source_url: str = "") -> List[Relation]:
        """Extract relationships between entities."""
        relations = []

        if not self.config.extract_relations:
            return relations

        # Create entity lookup
        entity_map = {e.canonical_name: e for e in entities}
        entity_texts = list(entity_map.keys())

        # Pattern-based relation extraction
        for pattern, rel_type in self.PATTERNS:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                subj_text = match.group(1)
                obj_text = match.group(2)

                # Find matching entities
                subj = entity_map.get(subj_text)
                obj_entity = entity_map.get(obj_text)

                if subj and obj_entity:
                    relation = Relation(
                        subject=subj,
                        predicate=rel_type,
                        obj=obj_entity,
                        confidence=0.7,
                        context=match.group(0),
                        source_url=source_url
                    )
                    relations.append(relation)

        return relations


class ConceptExtractor:
    """Extracts abstract concepts and their definitions."""

    def __init__(self, config: ExtractionConfig):
        self.config = config

    def extract(self, text: str, source_url: str = "") -> List[Concept]:
        """Extract concepts from text."""
        if not self.config.extract_concepts:
            return []

        concepts = []

        # Look for definition patterns
        patterns = [
            r'([A-Z][\w\s]+)\s+(?:is|are|refers to|means)\s+([^.]+)',
            r'(?:The concept of|The notion of)\s+([\w\s]+)\s+(?:is|refers to)\s+([^.]+)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    concept = Concept(
                        name=match.group(1).strip(),
                        definition=match.group(2).strip()[:200],
                        confidence=0.65,
                        source_url=source_url
                    )
                    concepts.append(concept)

        return concepts[:20]


class KeywordExtractor:
    """Extracts keywords and topics from text."""

    def __init__(self):
        self.stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "been", "be",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "of", "in",
            "to", "for", "on", "at", "by", "from", "with", "about", "as"
        }

    def extract(self, text: str, top_k: int = 20) -> List[str]:
        """Extract top keywords using TF-IDF-like scoring."""
        words = re.findall(r'\b[a-z][a-z-]+\b', text.lower())
        word_freq: Dict[str, int] = {}

        for word in words:
            if word not in self.stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_k]]


class KnowledgeExtractor:
    """
    Main knowledge extraction pipeline.
    Orchestrates all specialized extractors.
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.entity_extractor = EntityExtractor(config)
        self.relation_extractor = RelationExtractor(config)
        self.concept_extractor = ConceptExtractor(config)
        self.keyword_extractor = KeywordExtractor()

        self._stats = {
            "total_extracted": 0,
            "total_entities": 0,
            "total_relations": 0,
            "total_concepts": 0,
        }

        logger.info("KnowledgeExtractor initialized")

    async def extract(self, source: KnowledgeSource) -> ExtractedKnowledge:
        """
        Extract all knowledge from a source.
        Returns structured ExtractedKnowledge object.
        """
        text = source.content or source.title
        if not text:
            return ExtractedKnowledge(source_url=source.url)

        # Extract entities
        entities = self.entity_extractor.extract(text, source.url)

        # Extract relations between entities
        relations = self.relation_extractor.extract(text, entities, source.url)

        # Extract abstract concepts
        concepts = self.concept_extractor.extract(text, source.url)

        # Extract keywords
        keywords = self.keyword_extractor.extract(text)

        # Detect domain
        domain = self._detect_domain(text, keywords)

        # Create summary
        summary = self._create_summary(source, entities, keywords)

        knowledge = ExtractedKnowledge(
            source_url=source.url,
            entities=entities,
            relations=relations,
            concepts=concepts,
            keywords=keywords,
            domain=domain,
            summary=summary,
            extraction_confidence=self._calculate_confidence(entities, relations)
        )

        # Update stats
        self._stats["total_extracted"] += 1
        self._stats["total_entities"] += len(entities)
        self._stats["total_relations"] += len(relations)
        self._stats["total_concepts"] += len(concepts)

        logger.debug(
            f"Extracted from {source.url}: "
            f"{len(entities)} entities, {len(relations)} relations, "
            f"{len(concepts)} concepts"
        )

        return knowledge

    def _detect_domain(self, text: str, keywords: List[str]) -> str:
        """Detect the knowledge domain of the text."""
        domain_keywords = {
            "machine_learning": {"neural", "learning", "model", "training", "dataset"},
            "nlp": {"language", "text", "token", "sentence", "nlp", "bert", "gpt"},
            "computer_vision": {"image", "vision", "pixel", "object", "detection"},
            "robotics": {"robot", "motion", "sensor", "actuator", "control"},
            "physics": {"quantum", "particle", "energy", "wave", "force"},
            "biology": {"gene", "protein", "cell", "dna", "organism"},
        }

        text_lower = text.lower()
        domain_scores = {}

        for domain, terms in domain_keywords.items():
            score = sum(1 for term in terms if term in text_lower)
            domain_scores[domain] = score

        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain

        return "general"

    def _create_summary(
        self,
        source: KnowledgeSource,
        entities: List[Entity],
        keywords: List[str]
    ) -> str:
        """Create a brief summary of extracted knowledge."""
        entity_names = [e.text for e in entities[:5]]
        return (
            f"Source: {source.title or source.url}. "
            f"Key entities: {', '.join(entity_names[:5])}. "
            f"Keywords: {', '.join(keywords[:5])}."
        )

    def _calculate_confidence(
        self,
        entities: List[Entity],
        relations: List[Relation]
    ) -> float:
        """Calculate overall extraction confidence."""
        if not entities:
            return 0.0

        entity_conf = sum(e.confidence for e in entities) / len(entities)
        relation_conf = (
            sum(r.confidence for r in relations) / len(relations)
            if relations else 0.5
        )

        return (entity_conf * 0.6 + relation_conf * 0.4)

    def get_stats(self) -> dict:
        return dict(self._stats)
