"""Create the TF-IDF artifacts consumed by the CineSense API."""
import ast
import pickle
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent


def names(value: object) -> str:
    """Safely parse TMDB's stringified genre/collection dictionaries."""
    if not isinstance(value, str) or not value.strip(): return ""
    try:
        data = ast.literal_eval(value)
        if isinstance(data, dict): return data.get("name", "")
        return " ".join(item.get("name", "") for item in data if isinstance(item, dict))
    except (SyntaxError, ValueError): return ""


def clean(text: object, stops: set[str], lemmatizer: WordNetLemmatizer) -> str:
    words = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower()).split()
    return " ".join(lemmatizer.lemmatize(word) for word in words if word not in stops)


def main() -> None:
    nltk.download("stopwords", quiet=True); nltk.download("wordnet", quiet=True)
    raw = pd.read_csv(ROOT / "movies_metadata.csv", low_memory=False).drop_duplicates()
    wanted = ["id", "title", "genres", "overview", "vote_average", "vote_count", "popularity", "tagline", "belongs_to_collection"]
    df = raw[[column for column in wanted if column in raw]].copy().dropna(subset=["title"]).reset_index(drop=True)
    for column in ("overview", "tagline", "genres", "belongs_to_collection"):
        if column in df: df[column] = df[column].fillna("")
    df["genres"] = df["genres"].apply(names)
    df["collection"] = df["belongs_to_collection"].apply(names) if "belongs_to_collection" in df else ""
    df["tags"] = df["title"] + " " + df["genres"] + " " + df["overview"] + " " + df["tagline"]
    stops, lemmatizer = set(stopwords.words("english")), WordNetLemmatizer()
    df["tags"] = df["tags"].apply(lambda text: clean(text, stops, lemmatizer))
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(df["tags"])
    indices = pd.Series(df.index, index=df["title"])
    indices = indices[~indices.index.duplicated(keep="first")]
    for filename, value in {"movies.pkl": df, "tfidf_matrix.pkl": matrix, "tfidf_vectorizer.pkl": vectorizer, "indices.pkl": indices}.items():
        with open(ROOT / filename, "wb") as file: pickle.dump(value, file)
    print(f"Saved {len(df):,} movies with {matrix.shape[1]:,} TF-IDF features.")


if __name__ == "__main__": main()
