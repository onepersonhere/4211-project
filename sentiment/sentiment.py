import json
import requests
import ssl
import pandas as pd
import numpy as np
import datetime as dt
from transformers import pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ----- SSL Bypass (if needed) -----
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ----- Load API Key from secrets.json -----
def load_api_key():
    try:
        with open('secrets.json', 'r') as f:
            secrets = json.load(f)
            api_key = secrets.get("news_api_key")
            if not api_key:
                raise ValueError("news_api_key not found in secrets.json")
            return api_key
    except Exception as e:
        print("Error loading API key:", e)
        exit(1)


NEWS_API_KEY = load_api_key()
NEWS_API_URL = 'https://newsapi.org/v2/everything'


# ----- Function to fetch news using a given query -----
def get_news(query, page_size=20):
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': page_size,
        'apiKey': NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('articles', [])
    else:
        print(f"Error fetching news for query {query}: {response.status_code}")
        return []


# ----- Combine news from multiple queries for a given ticker -----
def get_news_combined(ticker):
    # Query 1: just the ticker (in quotes for exact match)
    query1 = f'"{ticker}"'
    articles1 = get_news(query1, page_size=20)

    # Query 2: ticker with financial terms to capture finance-focused articles
    query2 = f'"{ticker}" AND ("financial" OR "market" OR "stock" OR "crypto")'
    articles2 = get_news(query2, page_size=20)

    # Deduplicate based on URL
    combined = {article['url']: article for article in articles1 + articles2}.values()
    return list(combined)


# ----- Initialize FinBERT Sentiment Pipeline -----
# FinBERT is widely used by quant traders for financial sentiment analysis.
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone",
    tokenizer="yiyanghkust/finbert-tone"
)


def analyze_sentiment(text):
    """Analyze sentiment using FinBERT and convert output to a numeric compound score."""
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']
    # Define a compound score: POSITIVE => +score, NEGATIVE => -score, NEUTRAL => 0.
    compound = score if label.upper() == "POSITIVE" else -score if label.upper() == "NEGATIVE" else 0
    return {"label": label.upper(), "score": score, "compound": compound}


# ----- Aggregate article sentiment on a daily basis -----
def aggregate_daily_sentiment(articles):
    records = []
    for article in articles:
        pub_date = article.get('publishedAt', None)
        if pub_date:
            # Convert ISO date to Python date (ignoring time)
            date_obj = dt.datetime.fromisoformat(pub_date.replace("Z", "+00:00")).date()
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title}. {description}"
            sentiment = analyze_sentiment(text)
            records.append({"date": date_obj, "compound": sentiment["compound"]})
    if records:
        df = pd.DataFrame(records)
        # Compute average compound sentiment for each day
        daily_sentiment = df.groupby("date")["compound"].mean().reset_index()
        return daily_sentiment
    else:
        return pd.DataFrame(columns=["date", "compound"])


# ----- Main Function -----
def main():
    # Load tickers from tickers.json
    try:
        with open('tickers.json', 'r') as f:
            data = json.load(f)
            tickers = data.get("tickers", [])
    except FileNotFoundError:
        print("tickers.json file not found. Please create it with the format:")
        print('{"tickers": ["AAPL", "TSLA", "BTC", "ETH"]}')
        return

    if not tickers:
        print("No tickers found.")
        return

    # Dictionaries to store daily sentiment per ticker and detailed article results.
    ticker_sentiments = {}
    detailed_results = []

    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        articles = get_news_combined(ticker)
        print(f"Found {len(articles)} articles for {ticker}.")
        if not articles:
            continue

        # Aggregate daily sentiment for this ticker.
        daily_df = aggregate_daily_sentiment(articles)
        if daily_df.empty:
            print(f"No valid publication dates found for {ticker}.")
            continue
        ticker_sentiments[ticker] = daily_df.set_index("date")

        # Save detailed sentiment result for each article.
        for article in articles:
            pub_date = article.get('publishedAt', '')
            try:
                date_obj = dt.datetime.fromisoformat(pub_date.replace("Z", "+00:00")).date()
            except Exception:
                date_obj = None
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title}. {description}"
            sentiment = analyze_sentiment(text)
            detailed_results.append({
                "ticker": ticker,
                "date": date_obj,
                "title": title,
                "compound": sentiment["compound"],
                "label": sentiment["label"],
                "score": sentiment["score"],
                "url": article.get("url", "")
            })

    # Combine aggregated daily sentiment data across tickers.
    if ticker_sentiments:
        # Build a complete date range across all tickers.
        all_dates = pd.concat([df for df in ticker_sentiments.values()]).index.unique()
        start_date = min(all_dates)
        end_date = max(all_dates)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame(index=date_range)
        for ticker, df in ticker_sentiments.items():
            df = df.reindex(date_range)
            sentiment_df[ticker] = df["compound"]
        # Forward-fill missing values; if still missing, fill with 0.
        sentiment_df.fillna(method="ffill", inplace=True)
        sentiment_df.fillna(0, inplace=True)
        sentiment_df.to_csv("aggregated_daily_sentiment.csv")
        print("\nAggregated daily sentiment saved to aggregated_daily_sentiment.csv")

        # --- Correlation Analysis ---
        corr_matrix = sentiment_df.corr()
        print("\nCorrelation Matrix between tickers' daily sentiment scores:")
        print(corr_matrix)

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix of Daily Sentiment Scores")
        plt.savefig("correlation_heatmap.png")
        plt.show()

        # --- PCA Analysis ---
        pca_data = sentiment_df.dropna().values
        n_components = min(len(tickers), pca_data.shape[1])
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(pca_data)
        explained_var = pca.explained_variance_ratio_
        print("\nPCA Explained Variance Ratios:")
        for i, ratio in enumerate(explained_var):
            print(f"Component {i + 1}: {ratio:.4f}")

        # Scatter plot of first two principal components (if available)
        if pca_data.shape[0] > 1 and principal_components.shape[1] >= 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(principal_components[:, 0], principal_components[:, 1])
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.title("PCA of Daily Sentiment Scores")
            plt.savefig("pca_scatter.png")
            plt.show()
    else:
        print("No aggregated sentiment data available.")

    # Write detailed article-level sentiment results to CSV.
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv("detailed_sentiment_results.csv", index=False)
        print("Detailed sentiment results saved to detailed_sentiment_results.csv")
    else:
        print("No detailed sentiment results to save.")


if __name__ == "__main__":
    main()
