from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500,
                       example="latest developments in artificial intelligence")
    top_k: int = Field(default=5, ge=1, le=20,
                       description="Number of recommendations to return")
    category: Optional[str] = Field(default=None,
                                    example="technology",
                                    description="Filter by category")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning breakthroughs 2024",
                "top_k": 5,
                "category": "technology"
            }
        }


class ArticleResult(BaseModel):
    id: int
    title: str
    description: str
    category: str
    url: str
    published_at: str
    similarity_score: float


class RecommendResponse(BaseModel):
    query: str
    results: List[ArticleResult]
    total_results: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    articles_indexed: int
    message: str
