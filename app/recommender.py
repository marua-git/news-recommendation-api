import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict, Any
import os
import json


class NewsRecommender:
    """
    Content-based news recommendation engine.

    Uses TF-IDF vectorization of article text (title + description + content)
    and cosine similarity to find relevant articles for a given query.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,       # Apply log normalization
            strip_accents='unicode',
            analyzer='word',
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.df = None
        self.n_articles = 0
        self.categories = []

    def load_data(self, data_path: str = "data/news.csv"):
        """Load and preprocess news articles."""

        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
            print(f"Loaded {len(self.df):,} articles from {data_path}")
        else:
            # Generate sample dataset if no file exists
            print("No data file found — generating sample dataset...")
            self.df = self._generate_sample_data()
            os.makedirs("data", exist_ok=True)
            self.df.to_csv(data_path, index=False)
            print(f"Sample dataset saved to {data_path}")

        # Preprocess
        self.df = self.df.dropna(subset=['title'])
        self.df['category'] = self.df.get('category', pd.Series(['general'] * len(self.df)))
        self.df['category'] = self.df['category'].fillna('general').str.lower()
        self.df['description'] = self.df.get('description', pd.Series([''] * len(self.df))).fillna('')
        self.df['content'] = self.df.get('content', pd.Series([''] * len(self.df))).fillna('')

        # Combine fields for vectorization
        self.df['text'] = (
            self.df['title'].fillna('') + ' ' +
            self.df['title'].fillna('') + ' ' +   # title twice = higher weight
            self.df['description'].fillna('') + ' ' +
            self.df['content'].fillna('')
        )

        # Build text column for vectorization
        self.df['text'] = (
            self.df['title'].fillna('') + ' ' +
            self.df['title'].fillna('') + ' ' +
            self.df['description'].fillna('') + ' ' +
            self.df['content'].fillna('')
        )

        self.n_articles = len(self.df)
        self.categories = sorted(self.df['category'].unique().tolist())

    def fit(self):
        """Fit TF-IDF vectorizer and build article matrix."""
        print("Building TF-IDF matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text'])
        print(f"Matrix shape: {self.tfidf_matrix.shape}")

    def recommend(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend articles for a given query.

        Args:
            query: User query string
            top_k: Number of results to return
            category: Optional category filter

        Returns:
            List of article dicts with similarity scores
        """
        # Vectorize query
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Apply category filter
        if category and category.lower() in self.categories:
            mask = self.df['category'] == category.lower()
            scores[~mask.values] = 0

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k * 2]  # get extra, filter later
        results = []

        for idx in top_indices:
            if scores[idx] < 0.01:
                break
            if len(results) >= top_k:
                break

            row = self.df.iloc[idx]
            results.append({
                "id": int(idx),
                "title": str(row.get('title', '')),
                "description": str(row.get('description', ''))[:300],
                "category": str(row.get('category', 'general')),
                "url": str(row.get('url', '')),
                "published_at": str(row.get('published_at', '')),
                "similarity_score": round(float(scores[idx]), 4)
            })

        return results

    def similar(self, article_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find articles similar to a given article by ID."""
        article_vec = self.tfidf_matrix[article_id]
        scores = cosine_similarity(article_vec, self.tfidf_matrix).flatten()
        scores[article_id] = 0  # exclude self

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "id": int(idx),
                "title": str(row.get('title', '')),
                "category": str(row.get('category', 'general')),
                "similarity_score": round(float(scores[idx]), 4)
            })
        return results

    def stats(self) -> Dict[str, Any]:
        """Return dataset and model statistics."""
        return {
            "total_articles": self.n_articles,
            "categories": self.categories,
            "category_counts": self.df['category'].value_counts().to_dict(),
            "vocab_size": len(self.vectorizer.vocabulary_),
            "matrix_shape": list(self.tfidf_matrix.shape),
            "avg_title_length": round(self.df['title'].str.split().str.len().mean(), 1)
        }

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate a sample news dataset for demonstration."""
        categories = ['technology', 'business', 'health', 'sports', 'science', 'entertainment']

        samples = {
            'technology': [
                ("OpenAI releases GPT-5 with improved reasoning", "The latest model shows significant improvements in mathematical reasoning and code generation.", "https://example.com/gpt5"),
                ("Apple announces new MacBook Pro with M4 chip", "The new silicon chip delivers 40% faster performance compared to previous generation.", "https://example.com/macbook"),
                ("Google DeepMind achieves breakthrough in protein folding", "AlphaFold 3 can now predict protein interactions with unprecedented accuracy.", "https://example.com/deepmind"),
                ("Meta releases open-source AI model for developers", "Llama 3 outperforms many proprietary models on standard benchmarks.", "https://example.com/meta"),
                ("Cybersecurity threats increase by 300% in 2024", "Organizations face record number of ransomware attacks globally.", "https://example.com/cyber"),
            ],
            'business': [
                ("Stock markets reach all-time high amid economic optimism", "S&P 500 crosses 5000 points for the first time in history.", "https://example.com/stocks"),
                ("Tesla reports record quarterly earnings", "EV manufacturer beats analyst expectations with $25B in revenue.", "https://example.com/tesla"),
                ("Federal Reserve holds interest rates steady", "Central bank signals potential cuts later in the year amid cooling inflation.", "https://example.com/fed"),
                ("Amazon expands same-day delivery to 50 new cities", "The retail giant continues logistics investment despite cost-cutting elsewhere.", "https://example.com/amazon"),
                ("Startup funding rebounds after two-year slump", "Venture capital investment surges in AI and clean energy sectors.", "https://example.com/startups"),
            ],
            'health': [
                ("New cancer immunotherapy shows 90% success rate", "Clinical trials demonstrate breakthrough results for pancreatic cancer treatment.", "https://example.com/cancer"),
                ("WHO warns of new respiratory virus spreading in Asia", "Health officials urge precautionary measures as cases rise in three countries.", "https://example.com/who"),
                ("Daily exercise reduces dementia risk by 45%", "A 10-year study of 50,000 participants confirms the cognitive benefits of physical activity.", "https://example.com/exercise"),
                ("FDA approves first AI diagnostic tool for radiology", "The algorithm detects tumors with greater accuracy than experienced radiologists.", "https://example.com/fda"),
                ("Obesity rates decline for first time in two decades", "Weight-loss medications and lifestyle programs credited with the trend reversal.", "https://example.com/obesity"),
            ],
            'sports': [
                ("Manchester City wins Premier League title again", "Pep Guardiola's side clinches fourth consecutive championship with two games to spare.", "https://example.com/mancity"),
                ("LeBron James retires after record-breaking NBA career", "The four-time champion ends his 22-season career as the all-time leading scorer.", "https://example.com/lebron"),
                ("Novak Djokovic wins record 25th Grand Slam", "The Serbian tennis legend defeats Carlos Alcaraz in an epic Wimbledon final.", "https://example.com/djokovic"),
                ("Olympics 2024 Paris breaks viewership records", "Over 3.8 billion people tune in globally to the summer games.", "https://example.com/olympics"),
                ("Formula 1 introduces new car regulations for 2025", "The changes aim to increase overtaking and reduce aerodynamic complexity.", "https://example.com/f1"),
            ],
            'science': [
                ("NASA discovers water ice on Mars south pole", "New rover data confirms large deposits of frozen water beneath the surface.", "https://example.com/mars"),
                ("Scientists create first room-temperature superconductor", "The breakthrough could revolutionize energy transmission and storage.", "https://example.com/superconductor"),
                ("Quantum computing milestone achieved by IBM", "1000-qubit processor solves problem impossible for classical computers.", "https://example.com/quantum"),
                ("New species of deep-sea creature discovered in Pacific", "The bioluminescent organism challenges our understanding of deep-sea ecosystems.", "https://example.com/deepsea"),
                ("Climate scientists predict faster ice melt than expected", "New models suggest Arctic could be ice-free by 2030 under current trajectories.", "https://example.com/climate"),
            ],
            'entertainment': [
                ("Dune Part 3 confirmed with original cast returning", "Denis Villeneuve will complete the trilogy adaptation of Herbert's sci-fi saga.", "https://example.com/dune"),
                ("Taylor Swift breaks streaming record with new album", "The Eras Tour album surpasses 1 billion streams in its first week.", "https://example.com/swift"),
                ("Netflix announces price increase for 2025", "The streaming giant raises subscription prices citing content investment costs.", "https://example.com/netflix"),
                ("Cannes Film Festival announces lineup", "This year's selection features films from 45 countries across all continents.", "https://example.com/cannes"),
                ("Video game industry revenue surpasses Hollywood", "Gaming generates $200B annually compared to $100B for traditional film.", "https://example.com/gaming"),
            ],
        }

        rows = []
        for cat, articles in samples.items():
            for title, desc, url in articles:
                rows.append({
                    'title': title,
                    'description': desc,
                    'content': desc + ' ' + desc,
                    'category': cat,
                    'url': url,
                    'published_at': '2024-01-01'
                })

        # Repeat to simulate larger dataset
        df = pd.DataFrame(rows)
        df = pd.concat([df] * 200, ignore_index=True)
        df['title'] = df['title'] + ' ' + df.index.astype(str)  # make unique
        return df
