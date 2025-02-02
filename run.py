import streamlit as st
import string
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/bot_detection_results.csv")
    return df

# Load the model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessing():
    model = joblib.load("models/bot_detection_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    behavior_scaler = joblib.load("models/behavior_scaler.pkl")
    minmax_scaler = joblib.load("models/minmax_scaler.pkl")
    return model, vectorizer, behavior_scaler, minmax_scaler

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

# Feature Engineering
def extract_features(df, vectorizer, behavior_scaler):
    # Text features
    X_text = vectorizer.transform(df["clean_text"]).toarray()

    # Behavioral features
    features = ["Retweet Count", "Mention Count", "Follower Count", "Verified"]
    X_behavior = df[features].fillna(0)
    X_behavior = behavior_scaler.transform(X_behavior)

    # Combine features
    X_combined = np.hstack((X_text, X_behavior))
    return X_combined

# Streamlit App
def main():
    st.title("Bot Detection Model Evaluation")

    # Load data
    df = load_data()
    df["clean_text"] = df["Tweet"].astype(str).apply(preprocess_text)

    # Load model and preprocessing objects
    model, vectorizer, behavior_scaler, minmax_scaler = load_model_and_preprocessing()

    # Extract features
    X_combined = extract_features(df, vectorizer, behavior_scaler)
    y_labels = df["is_bot"].astype(int)

    # Evaluate model
    st.header("Model Evaluation Metrics")
    predictions = model.predict(X_combined)
    proba_predictions = model.predict_proba(X_combined)[:, 1]

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_labels, predictions, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    precisions, recalls, thresholds = precision_recall_curve(y_labels, proba_predictions)
    st.line_chart(pd.DataFrame({"Precision": precisions, "Recall": recalls}))

    # AUC-ROC Score
    st.subheader("AUC-ROC Score")
    roc_auc = roc_auc_score(y_labels, proba_predictions)
    st.write(f"AUC-ROC Score: {roc_auc:.4f}")

    # Best Threshold
    best_threshold = thresholds[np.argmax(precisions * recalls)]
    st.write(f"Best Threshold: {best_threshold:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    confusion_matrix = pd.crosstab(y_labels, predictions, rownames=['Actual'], colnames=['Predicted'])
    st.dataframe(confusion_matrix)

if __name__ == "__main__":
    main()
