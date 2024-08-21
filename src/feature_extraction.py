
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization
def extract_tfidf_features(corpus):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(corpus), vectorizer
