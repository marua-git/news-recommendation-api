"""
fetch_news.py — Download real news articles for the recommendation engine.

Uses the free NewsAPI (https://newsapi.org) to fetch articles.
Get your free API key at: https://newsapi.org/register

Usage:
    python data/fetch_news.py --api-key YOUR_KEY --articles 10000
"""

import argparse
import pandas as pd
import requests
import time
import os


def fetch_news(api_key: str, total_articles: int = 10000) -> pd.DataFrame:
    """Fetch news articles from NewsAPI."""
    categories = ['technology', 'business', 'health', 'sports', 'science', 'entertainment']
    all_articles = []
    per_category = total_articles // len(categories)

    for category in categories:
        print(f"Fetching {category}...")
        page = 1
        fetched = 0

        while fetched < per_category:
            url = (
                f"https://newsapi.org/v2/top-headlines"
                f"?category={category}"
                f"&language=en"
                f"&pageSize=100"
                f"&page={page}"
                f"&apiKey={api_key}"
            )
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                print(f"  Error: {response.status_code}")
                break

            data = response.json()
            articles = data.get('articles', [])

            if not articles:
                break

            for article in articles:
                all_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'category': category,
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', '')
                })

            fetched += len(articles)
            page += 1
            time.sleep(0.5)  # rate limiting

        print(f"  ✅ {fetched} articles fetched for {category}")

    df = pd.DataFrame(all_articles)
    df = df.dropna(subset=['title'])
    df = df[df['title'] != '[Removed]']
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch news articles")
    parser.add_argument("--api-key", required=True, help="NewsAPI key")
    parser.add_argument("--articles", type=int, default=10000, help="Total articles to fetch")
    parser.add_argument("--output", default="data/news.csv", help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = fetch_news(api_key=args.api_key, total_articles=args.articles)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Saved {len(df):,} articles to {args.output}")
