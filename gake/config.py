"""
Configuration management for the Global Autonomous Knowledge Engine.
Loads and validates engine configuration from YAML files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class DiscoveryConfig:
    """Configuration for the information discovery module."""
    interval_seconds: int = 300
    max_concurrent_agents: int = 10
    sources: List[str] = field(default_factory=lambda: [
        "arxiv", "semantic_scholar", "github", "wikipedia",
        "news_api", "web_crawler"
    ])
    arxiv_categories: List[str] = field(default_factory=lambda: [
        "cs.AI", "cs.LG", "cs.CL", "cs.RO", "cs.CV",
        "physics", "math", "q-bio"
    ])
    max_papers_per_cycle: int = 100
    max_web_pages_per_cycle: int = 50
    github_topics: List[str] = field(default_factory=lambda: [
        "machine-learning", "artificial-intelligence", "deep-learning",
        "nlp", "computer-vision", "robotics"
    ])


@dataclass
class ExtractionConfig:
    """Configuration for the knowledge extraction module."""
    model_name: str = "en_core_web_trf"
    relation_extraction_model: str = "bert-base-uncased"
    min_confidence: float = 0.7
    max_entities_per_doc: int = 100
    extract_relations: bool = True
    extract_events: bool = True
    extract_concepts: bool = True
    batch_size: int = 32
    use_llm_extraction: bool = True
    llm_model: str = "gpt-4"


@dataclass
class GraphConfig:
    """Configuration for the knowledge graph module."""
    backend: str = "networkx"  # Options: networkx, neo4j, rdflib
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    persistence_path: str = "data/knowledge_graph.pkl"
    max_nodes: int = 1_000_000
    max_edges: int = 5_000_000
    enable_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384


@dataclass
class ValidationConfig:
    """Configuration for the knowledge validation module."""
    min_source_count: int = 2
    confidence_threshold: float = 0.75
    enable_cross_validation: bool = True
    enable_temporal_validation: bool = True
    enable_logical_consistency: bool = True
    contradiction_detection: bool = True
    fact_check_api_key: str = ""


@dataclass
class LearningConfig:
    """Configuration for the continuous learning module."""
    interval_seconds: int = 3600
    enable_concept_discovery: bool = True
    enable_relation_inference: bool = True
    enable_knowledge_evolution: bool = True
    graph_embedding_model: str = "TransE"
    embedding_epochs: int = 100
    link_prediction_threshold: float = 0.8
    cluster_new_concepts: bool = True


@dataclass
class EngineConfig:
    """Master configuration for the GAKE engine."""
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    maintenance_interval_seconds: int = 86400
    log_level: str = "INFO"
    log_file: str = "logs/gake.log"
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    enable_api: bool = True
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, config_path: Path) -> "EngineConfig":
        """Load configuration from a YAML file."""
        if not config_path.exists():
            return cls()  # Return defaults if no config file

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        config = cls()

        # Override with values from YAML
        if "discovery" in raw:
            d = raw["discovery"]
            config.discovery = DiscoveryConfig(**{
                k: v for k, v in d.items()
                if hasattr(DiscoveryConfig, k)
            })

        if "extraction" in raw:
            e = raw["extraction"]
            config.extraction = ExtractionConfig(**{
                k: v for k, v in e.items()
                if hasattr(ExtractionConfig, k)
            })

        if "graph" in raw:
            g = raw["graph"]
            # Override Neo4j password from environment
            if "neo4j_password" not in g:
                g["neo4j_password"] = os.environ.get("NEO4J_PASSWORD", "")
            config.graph = GraphConfig(**{
                k: v for k, v in g.items()
                if hasattr(GraphConfig, k)
            })

        if "validation" in raw:
            v = raw["validation"]
            config.validation = ValidationConfig(**{
                k: val for k, val in v.items()
                if hasattr(ValidationConfig, k)
            })

        if "learning" in raw:
            l = raw["learning"]
            config.learning = LearningConfig(**{
                k: v for k, v in l.items()
                if hasattr(LearningConfig, k)
            })

        # Top-level config
        for key in ["maintenance_interval_seconds", "log_level", "log_file",
                    "api_host", "api_port", "enable_api", "data_dir", "checkpoint_dir"]:
            if key in raw:
                setattr(config, key, raw[key])

        return config

    def to_yaml(self, path: Path):
        """Save configuration to a YAML file."""
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
