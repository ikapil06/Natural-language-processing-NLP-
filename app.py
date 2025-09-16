"""
Flask web application for lexicon‑based sentiment analysis on product reviews.

The application reads a CSV file of product reviews, preprocesses the
text, classifies each review into positive, negative or neutral
sentiment and computes aggregated statistics per product.  A simple
HTML form allows users to submit their own review and receive a
sentiment prediction.  A bar chart visualising the distribution of
sentiments in the dataset and a summary table by product are
displayed on the page.

This approach falls into the lexicon‑based category of sentiment
analysis, which uses predefined lists of positive and negative words
to determine the emotional tone of the text【127290445729235†L141-L146】.
"""

from __future__ import annotations

import os
import pandas as pd
from flask import Flask, render_template, request
from typing import List
import matplotlib
# Use a non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

# Optional SpaCy support
try:
    import spacy  # type: ignore
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

# Minimal stopwords list
STOPWORDS = set([
    "the", "and", "is", "it", "to", "a", "of", "this", "i", "very",
    "for", "in", "that", "with", "on", "was", "have", "ever", "be", "but"
])

POSITIVE_WORDS = set([
    "fantastic", "great", "love", "amazing", "excellent", "superb",
    "best", "wonderful", "awesome", "value", "recommend", "satisfied",
    "good", "friendly", "quality", "happy", "pleased"
])
NEGATIVE_WORDS = set([
    "worst", "terrible", "disappointing", "poor", "horrible", "hate",
    "bad", "awful", "useless", "regret", "defective", "slow",
    "broke", "broken", "unhappy", "dissatisfied", "annoying"
])

def preprocess(text: str) -> List[str]:
    import re
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    if _nlp is not None:
        doc = _nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc if not token.is_stop]
    return tokens
def index():
    global _df, _summary, _chart_data
    sentiment = None
    user_input = ""
    product = None
    # Normalize product names to lowercase for comparison and display
    product_list = sorted(set(p.lower() for p in _df["product"].unique())) if not _df.empty else []
    ask_product = False
    if request.method == "POST":
        user_input = request.form.get("review", "")
        product = request.form.get("product", "")
        # Normalize product name to lowercase
        product = product.strip().lower() if product else ""
        # If product is not provided, ask user to select from existing products
        if not product:
            ask_product = True
        else:
            tokens = preprocess(user_input)
            sentiment = classify(tokens)
            # Append new review to CSV (store product as lowercase)
            new_row = pd.DataFrame({
                "product": [product],
                "review": [user_input]
            })
            new_row.to_csv(DATA_PATH, mode="a", header=False, index=False)
            # Reload all data and update global variables
            _df, _summary, _chart_data = reload_data()
    # Render summary with index starting from 1
    summary_html = _summary.reset_index(drop=True)
    summary_html.index += 1
    summary_html = summary_html.to_html(classes="table table-striped", index=True)
    return render_template(
        "index.html",
        sentiment=sentiment,
        review_text=user_input,
        summary=summary_html,
        chart_data=_chart_data,
        product_list=product_list,
        ask_product=ask_product,
        selected_product=product
    )
    if _nlp is not None:
        doc = _nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc if not token.is_stop]
    return tokens


def classify(tokens: List[str]) -> str:
    pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    if pos_count > neg_count:
        return "positive"
    if neg_count > pos_count:
        return "negative"
    return "neutral"


def analyze_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tokens"] = df["review"].apply(preprocess)
    df["sentiment"] = df["tokens"].apply(classify)
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize product names to lowercase for grouping
    df = df.copy()
    df["product"] = df["product"].str.lower()
    summary = df.groupby(["product", "sentiment"]).size().unstack(fill_value=0)
    summary["total"] = summary.sum(axis=1)
    summary["positive_ratio"] = summary.get("positive", 0) / summary["total"]
    summary = summary.reset_index()
    return summary


def generate_bar_chart(df: pd.DataFrame) -> str:
    """Generate a bar chart and return it as a base64 string."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="sentiment", data=df, palette="pastel", ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return img_b64


# Initialise application

app = Flask(__name__)

# Path to the reviews CSV
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "reviews.csv")

# Functions to reload data
def reload_data():
    df = analyze_dataset(DATA_PATH)
    summary = compute_summary(df)
    chart_data = generate_bar_chart(df)
    return df, summary, chart_data

# Initial load
_df, _summary, _chart_data = reload_data()


@app.route("/", methods=["GET", "POST"])
def index():
    global _df, _summary, _chart_data
    sentiment = None
    user_input = ""
    product = None
    # Normalize product names to lowercase for comparison and display
    product_list = sorted(set(p.lower() for p in _df["product"].unique())) if not _df.empty else []
    ask_product = False
    if request.method == "POST":
        user_input = request.form.get("review", "")
        product = request.form.get("product", "")
        # Normalize product name to lowercase
        product = product.strip().lower() if product else ""
        # If product is not provided, ask user to select from existing products
        if not product:
            ask_product = True
        else:
            tokens = preprocess(user_input)
            sentiment = classify(tokens)
            # Append new review to CSV (store product as lowercase)
            new_row = pd.DataFrame({
                "product": [product],
                "review": [user_input]
            })
            new_row.to_csv(DATA_PATH, mode="a", header=False, index=False)
            # Reload all data and update global variables
            _df, _summary, _chart_data = reload_data()
    # Render template with summary table, bar chart, and optional sentiment
    # Render summary with index starting from 1
    summary_html = _summary.reset_index(drop=True)
    summary_html.index += 1
    summary_html = summary_html.to_html(classes="table table-striped", index=True)
    return render_template(
        "index.html",
        sentiment=sentiment,
        review_text=user_input,
        summary=summary_html,
        chart_data=_chart_data,
        product_list=product_list,
        ask_product=ask_product,
        selected_product=product
    )


if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=True)