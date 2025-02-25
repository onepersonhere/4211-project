import json
import requests
import csv
import ssl
from transformers import pipeline

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
        print("Error loading API key from secrets.json:", e)
        exit(1)


NEWS_API_KEY = load_api_key()
NEWS_API_URL = 'https://newsapi.org/v2/everything'


# ----- Function to Fetch News Articles for a Given Ticker -----
def get_news(ticker, page_size=20):
    """
    Fetch news articles related to the given ticker.
    The ticker is enclosed in quotes to ensure an exact match.
    """
    query = f'"{ticker}"'
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
        print(f"Error fetching news for {ticker}: {response.status_code}")
        return []


# ----- Initialize FinBERT Sentiment Pipeline -----
# FinBERT is widely used by quant traders for financial sentiment analysis.
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone",
    tokenizer="yiyanghkust/finbert-tone",
    framework="pt"  # Force PyTorch usage
)



def analyze_sentiment(text):
    """
    Uses FinBERT to perform sentiment analysis on the input text.
    Returns a dictionary with:
      - label: one of 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
      - score: the model's confidence
      - compound: a numeric score (+score for POSITIVE, -score for NEGATIVE, 0 for NEUTRAL)
    """
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']
    # Assign a numeric compound score:
    compound = score if label.upper() == 'POSITIVE' else -score if label.upper() == 'NEGATIVE' else 0
    # Return all values in a unified dictionary.
    return {"label": label.upper(), "score": score, "compound": compound}


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
        print("No tickers found in the JSON file.")
        return

    # Lists to collect detailed article results and summary per ticker.
    results = []
    summary = {}

    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        articles = get_news(ticker)
        if not articles:
            print(f"No articles found for {ticker}.")
            continue

        # Initialize summary statistics for this ticker.
        summary[ticker] = {"count": 0, "total_compound": 0}

        for article in articles:
            title = article.get('title', 'No title')
            description = article.get('description', '')
            published_at = article.get('publishedAt', '')
            url = article.get('url', '')
            # Combine title and description for analysis.
            text = f"{title}. {description}"
            sentiment = analyze_sentiment(text)

            # Update summary.
            summary[ticker]["count"] += 1
            summary[ticker]["total_compound"] += sentiment["compound"]

            # Append detailed result.
            results.append({
                "ticker": ticker,
                "title": title,
                "description": description,
                "publishedAt": published_at,
                "url": url,
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "sentiment_compound": sentiment["compound"]
            })

            print("Title:", title)
            print("FinBERT Output:", sentiment)
            print("-" * 60)

    # Write detailed results to CSV.
    if results:
        fieldnames = [
            "ticker", "title", "description", "publishedAt", "url",
            "sentiment_label", "sentiment_score", "sentiment_compound"
        ]
        csv_filename = "sentiment_results.csv"
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed sentiment analysis results saved to {csv_filename}")
    else:
        print("No detailed results to write to CSV.")

    # Print sentiment summary per ticker.
    print("\nSentiment Summary per Ticker:")
    for ticker, stats in summary.items():
        count = stats["count"]
        if count > 0:
            avg_compound = stats["total_compound"] / count
            print(f"\nTicker: {ticker}")
            print(f"  Articles Processed: {count}")
            print(f"  Average Compound Score: {avg_compound:.4f}")
        else:
            print(f"\nTicker: {ticker} - No articles processed.")


# ----- Run the Script -----
if __name__ == "__main__":
    main()
