"""
Agent Manager for Information Discovery.
Coordinates multiple specialized agents that search different sources:
- ArXiv research papers
- Semantic Scholar
- GitHub repositories
- Wikipedia
- News APIs
- General web crawling
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from gake.config import DiscoveryConfig

logger = logging.getLogger(__name__)


class SourceType(Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    GITHUB = "github"
    WIKIPEDIA = "wikipedia"
    NEWS = "news"
    WEB = "web"
    DATASET = "dataset"


@dataclass
class KnowledgeSource:
    """Represents a discovered knowledge source."""
    url: str
    source_type: SourceType
    title: str = ""
    authors: List[str] = field(default_factory=list)
    published_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: str = ""
    relevance_score: float = 0.0


class BaseAgent:
    """Base class for all discovery agents."""

    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.discovered_count = 0
        self.error_count = 0

    async def discover(self) -> List[KnowledgeSource]:
        """Discover knowledge sources. Override in subclasses."""
        raise NotImplementedError

    def get_stats(self) -> dict:
        return {
            "discovered": self.discovered_count,
            "errors": self.error_count
        }


class ArxivAgent(BaseAgent):
    """Agent for discovering research papers from ArXiv."""

    async def discover(self) -> List[KnowledgeSource]:
        """Fetch recent papers from ArXiv across configured categories."""
        sources = []
        try:
            import arxiv
            for category in self.config.arxiv_categories[:3]:  # Limit for demo
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=self.config.max_papers_per_cycle // len(self.config.arxiv_categories),
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                for result in search.results():
                    source = KnowledgeSource(
                        url=result.entry_id,
                        source_type=SourceType.ARXIV,
                        title=result.title,
                        authors=[str(a) for a in result.authors],
                        published_date=result.published,
                        content=result.summary,
                        metadata={
                            "categories": result.categories,
                            "doi": result.doi,
                            "pdf_url": result.pdf_url,
                        },
                        relevance_score=0.8
                    )
                    sources.append(source)
                    self.discovered_count += 1

        except ImportError:
            logger.warning("arxiv package not installed. Using mock data.")
            sources = self._mock_sources()
        except Exception as e:
            logger.error(f"ArXiv discovery error: {e}")
            self.error_count += 1

        return sources

    def _mock_sources(self) -> List[KnowledgeSource]:
        """Return mock sources for testing."""
        return [
            KnowledgeSource(
                url="https://arxiv.org/abs/2301.00001",
                source_type=SourceType.ARXIV,
                title="Advances in Large Language Models",
                authors=["Researcher A", "Researcher B"],
                content="This paper explores advances in LLMs...",
                relevance_score=0.9
            )
        ]


class SemanticScholarAgent(BaseAgent):
    """Agent for discovering papers from Semantic Scholar."""

    async def discover(self) -> List[KnowledgeSource]:
        """Fetch papers from Semantic Scholar API."""
        sources = []
        try:
            import aiohttp
            queries = ["machine learning 2024", "artificial intelligence", "deep learning"]

            async with aiohttp.ClientSession() as session:
                for query in queries[:2]:
                    url = f"https://api.semanticscholar.org/graph/v1/paper/search"
                    params = {
                        "query": query,
                        "limit": 10,
                        "fields": "title,authors,abstract,year,url,citationCount"
                    }
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for paper in data.get("data", []):
                                source = KnowledgeSource(
                                    url=paper.get("url", ""),
                                    source_type=SourceType.SEMANTIC_SCHOLAR,
                                    title=paper.get("title", ""),
                                    authors=[a["name"] for a in paper.get("authors", [])],
                                    content=paper.get("abstract", ""),
                                    metadata={
                                        "year": paper.get("year"),
                                        "citations": paper.get("citationCount", 0)
                                    },
                                    relevance_score=0.75
                                )
                                sources.append(source)
                                self.discovered_count += 1

        except Exception as e:
            logger.error(f"Semantic Scholar discovery error: {e}")
            self.error_count += 1

        return sources


class GitHubAgent(BaseAgent):
    """Agent for discovering knowledge from GitHub repositories."""

    async def discover(self) -> List[KnowledgeSource]:
        """Search GitHub for relevant repositories."""
        sources = []
        try:
            import aiohttp
            headers = {"Accept": "application/vnd.github.v3+json"}

            async with aiohttp.ClientSession(headers=headers) as session:
                for topic in self.config.github_topics[:3]:
                    url = "https://api.github.com/search/repositories"
                    params = {
                        "q": f"topic:{topic}",
                        "sort": "stars",
                        "per_page": 5
                    }
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for repo in data.get("items", []):
                                source = KnowledgeSource(
                                    url=repo["html_url"],
                                    source_type=SourceType.GITHUB,
                                    title=repo["full_name"],
                                    content=repo.get("description", ""),
                                    metadata={
                                        "stars": repo["stargazers_count"],
                                        "language": repo.get("language"),
                                        "topics": repo.get("topics", []),
                                    },
                                    relevance_score=min(repo["stargazers_count"] / 10000, 1.0)
                                )
                                sources.append(source)
                                self.discovered_count += 1

        except Exception as e:
            logger.error(f"GitHub discovery error: {e}")
            self.error_count += 1

        return sources


class WikipediaAgent(BaseAgent):
    """Agent for discovering knowledge from Wikipedia."""

    SEED_TOPICS = [
        "Artificial intelligence", "Machine learning", "Neural network",
        "Natural language processing", "Computer vision", "Robotics",
        "Quantum computing", "Blockchain", "Climate change", "Genomics"
    ]

    async def discover(self) -> List[KnowledgeSource]:
        """Fetch Wikipedia articles on key topics."""
        sources = []
        try:
            import wikipedia
            for topic in self.SEED_TOPICS[:5]:
                try:
                    page = wikipedia.page(topic, auto_suggest=False)
                    source = KnowledgeSource(
                        url=page.url,
                        source_type=SourceType.WIKIPEDIA,
                        title=page.title,
                        content=page.summary,
                        metadata={
                            "categories": page.categories[:10],
                            "links": page.links[:20],
                        },
                        relevance_score=0.7
                    )
                    sources.append(source)
                    self.discovered_count += 1
                except Exception:
                    pass

        except ImportError:
            logger.warning("wikipedia package not installed")
        except Exception as e:
            logger.error(f"Wikipedia discovery error: {e}")
            self.error_count += 1

        return sources


class AgentManager:
    """
    Manages and orchestrates all discovery agents.
    Runs agents concurrently and aggregates results.
    """

    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self._stats = {
            "total_discovered": 0,
            "cycles_completed": 0,
            "last_run": None
        }

        # Initialize agents based on configured sources
        self.agents: Dict[str, BaseAgent] = {}
        source_map = {
            "arxiv": ArxivAgent,
            "semantic_scholar": SemanticScholarAgent,
            "github": GitHubAgent,
            "wikipedia": WikipediaAgent,
        }

        for source_name in config.sources:
            if source_name in source_map:
                self.agents[source_name] = source_map[source_name](config)

        logger.info(f"AgentManager initialized with {len(self.agents)} agents")

    async def discover_sources(self) -> List[KnowledgeSource]:
        """
        Run all discovery agents concurrently and aggregate results.
        Returns a deduplicated list of knowledge sources.
        """
        self._stats["last_run"] = datetime.utcnow()
        all_sources = []

        # Run agents concurrently (with concurrency limit)
        semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

        async def run_agent(name: str, agent: BaseAgent):
            async with semaphore:
                try:
                    logger.info(f"Running {name} agent...")
                    sources = await agent.discover()
                    logger.info(f"{name} agent found {len(sources)} sources")
                    return sources
                except Exception as e:
                    logger.error(f"Agent {name} failed: {e}")
                    return []

        tasks = [run_agent(name, agent) for name, agent in self.agents.items()]
        results = await asyncio.gather(*tasks)

        for source_list in results:
            all_sources.extend(source_list)

        # Deduplicate by URL
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)

        # Sort by relevance score
        unique_sources.sort(key=lambda s: s.relevance_score, reverse=True)

        self._stats["total_discovered"] += len(unique_sources)
        self._stats["cycles_completed"] += 1

        logger.info(f"Discovery cycle complete: {len(unique_sources)} unique sources found")
        return unique_sources

    def get_stats(self) -> dict:
        """Return aggregated statistics."""
        stats = dict(self._stats)
        stats["agent_stats"] = {
            name: agent.get_stats()
            for name, agent in self.agents.items()
        }
        return stats
