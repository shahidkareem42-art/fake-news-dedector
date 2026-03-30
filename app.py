import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
df = pd.read_csv(url)

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['text'] = df['text'].apply(clean_text)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train model
model = LogisticRegression()
model.fit(X, y)

# UI
st.title("📰 Fake News Detector")

news = st.text_input("Enter News")

if st.button("Predict"):
    if news:
        cleaned = clean_text(news)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)

        if pred[0] == "REAL":
            st.success("🟢 This news is REAL")
        else:
            st.error("🔴 This news is FAKE")