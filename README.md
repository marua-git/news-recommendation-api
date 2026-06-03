# 📰 News Recommendation API

> Content-based news recommendation engine built with **FastAPI**, **TF-IDF**, and **Docker**.  
> Returns personalized recommendations in **< 150ms** at the 95th percentile.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 🚀 Quick Start

### Option 1 — Docker (recommended)

```bash
docker build -t news-api .
docker run -p 8000:8000 news-api
```

### Option 2 — Local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive API documentation.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
TF-IDF Vectorizer (50k features, bigrams)
    │
    ▼
Cosine Similarity (query vs 100k+ articles)
    │
    ▼
Top-K Results (filtered by category if specified)
    │
    ▼
JSON Response < 150ms
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed status |
| `POST` | `/recommend` | Get recommendations |
| `GET` | `/recommend/{id}` | Similar articles |
| `GET` | `/categories` | List categories |
| `GET` | `/stats` | Model statistics |

---

## 📋 Example Request

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning breakthroughs artificial intelligence",
    "top_k": 5,
    "category": "technology"
  }'
```

### Response

```json
{
  "query": "machine learning breakthroughs artificial intelligence",
  "results": [
    {
      "id": 42,
      "title": "OpenAI releases GPT-5 with improved reasoning",
      "description": "The latest model shows significant improvements...",
      "category": "technology",
      "url": "https://example.com/gpt5",
      "similarity_score": 0.8734
    }
  ],
  "total_results": 5,
  "latency_ms": 23.4
}
```

---

## ⚙️ How It Works

1. **Data loading** — Articles indexed from CSV (title + description + content)
2. **TF-IDF vectorization** — 50,000 features, bigrams, sublinear TF scaling
3. **Query processing** — User query transformed to same vector space
4. **Cosine similarity** — Ranked against all articles in O(n) time
5. **Category filtering** — Optional post-filtering by news category
6. **Response** — Top-K articles returned as structured JSON

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| P95 latency | < 150ms |
| Articles indexed | 100,000+ |
| Vocabulary size | 50,000 terms |
| Supported categories | 6 |

---

## 🗂️ Project Structure

```
news-recommendation-api/
├── app/
│   ├── main.py          # FastAPI application & routes
│   ├── recommender.py   # TF-IDF engine & cosine similarity
│   └── models.py        # Pydantic request/response schemas
├── data/
│   └── fetch_news.py    # Script to fetch real news data
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `data/news.csv` | Path to news CSV file |
| `MAX_FEATURES` | `50000` | TF-IDF vocabulary size |
| `PORT` | `8000` | API server port |

---

## 👩‍💻 Author

**Marua Makpyr** — ML Engineer  
[LinkedIn](https://linkedin.com) · [Kaggle](https://kaggle.com/maruamakpyr)

---

## 📄 License

MIT License — feel free to use and modify.
