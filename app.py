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
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.markdown("### Check whether a news is **Real or Fake**")

user_input = st.text_area("Enter News Here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("Analyzing news..."):
            transformed_input = vectorizer.transform([user_input])
            prediction = model.predict(transformed_input)[0]

        if prediction == "FAKE" or prediction == 0:
            st.error("🚨 This news is FAKE")
        else:
            st.success("✅ This news is REAL")

st.markdown("---")
st.caption("Built by Shahid Kareem 🚀")
