from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data" / "messages.csv"

st.set_page_config(page_title="Spam Message Detector", page_icon="📩", layout="wide")
st.title("📩 Spam Message Detector")
st.write("Machine learning text classification using TF-IDF and Logistic Regression.")

@st.cache_data
def load_data():
    data = pd.read_csv(CSV_PATH).dropna(subset=["message", "label"])
    data["message"] = data["message"].astype(str)
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    return data

@st.cache_resource
def build_model():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.25,
        random_state=42, stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]),
    }
    return model, metrics

data = load_data()
if set(data["label"].unique()) != {"ham", "spam"}:
    st.error("The CSV must contain exactly the labels 'ham' and 'spam'.")
    st.stop()

model, metrics = build_model()
m1, m2 = st.columns(2)
m1.metric("Messages in dataset", len(data))
m2.metric("Held-out test accuracy", f"{metrics['accuracy']:.1%}")

st.subheader("Try your own message")
message = st.text_area("Message", placeholder="Congratulations! You won a free prize. Claim now!")
if st.button("Detect spam", type="primary"):
    if not message.strip():
        st.warning("Enter a message first.")
    else:
        label = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        confidence = probabilities[list(model.classes_).index(label)]
        if label == "spam":
            st.error(f"🚨 Likely SPAM — estimated probability: {confidence:.1%}")
        else:
            st.success(f"✅ Likely NOT SPAM (ham) — estimated probability: {confidence:.1%}")
        st.caption("The probability is an estimate, not a guarantee that a message is safe.")

with st.expander("Model evaluation"):
    st.dataframe(pd.DataFrame(metrics["report"]).transpose().round(3), use_container_width=True)
    st.write("Confusion matrix: rows are actual labels; columns are predicted labels (ham, spam).")
    st.dataframe(pd.DataFrame(metrics["matrix"],
        index=["Actual ham", "Actual spam"],
        columns=["Predicted ham", "Predicted spam"]), use_container_width=True)

with st.expander("View sample dataset"):
    st.dataframe(data, use_container_width=True)

st.info("Educational project using a small sample dataset. Use a larger representative dataset for serious applications.")
