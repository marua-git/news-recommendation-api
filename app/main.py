from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.recommender import NewsRecommender
from app.models import RecommendRequest, RecommendResponse, HealthResponse
import time

# Global recommender instance
recommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load recommender on startup"""
    global recommender
    print("Loading news recommender...")
    recommender = NewsRecommender()
    recommender.load_data()
    recommender.fit()
    print(f"✅ Recommender ready — {recommender.n_articles:,} articles indexed")
    yield
    print("Shutting down...")

app = FastAPI(
    title="📰 News Recommendation API",
    description="""
## News Recommendation Engine

Personalized news recommendations using **TF-IDF vectorization** and **Cosine Similarity**.

### Features:
- Real-time recommendations in < 150ms
- Filter by category
- Similar article discovery
- Search by keyword

### How it works:
1. Articles are vectorized using TF-IDF (title + description + content)
2. User query is transformed to the same vector space
3. Cosine similarity ranks articles by relevance
4. Top-K results returned as JSON
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        articles_indexed=recommender.n_articles if recommender else 0,
        message="News Recommendation API is running 🚀"
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        articles_indexed=recommender.n_articles if recommender else 0,
        message="All systems operational"
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Get personalized news recommendations.

    - **query**: Text describing what you want to read
    - **top_k**: Number of recommendations (default: 5)
    - **category**: Optional category filter
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")

    start_time = time.time()

    results = recommender.recommend(
        query=request.query,
        top_k=request.top_k,
        category=request.category
    )

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return RecommendResponse(
        query=request.query,
        results=results,
        total_results=len(results),
        latency_ms=latency_ms
    )


@app.get("/recommend/{article_id}")
async def similar_articles(article_id: int, top_k: int = 5):
    """
    Get articles similar to a given article ID.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")

    if article_id >= recommender.n_articles:
        raise HTTPException(status_code=404, detail=f"Article {article_id} not found")

    results = recommender.similar(article_id=article_id, top_k=top_k)
    return {"article_id": article_id, "similar_articles": results}


@app.get("/categories")
async def get_categories():
    """List all available news categories"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    return {"categories": recommender.categories}


@app.get("/stats")
async def get_stats():
    """Dataset and model statistics"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    return recommender.stats()
