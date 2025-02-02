## This code train the model on the given dataset and export the trained model into model/ folder
import pandas as pd
import numpy as np
import re
import string
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

# Load dataset
# Assumes a CSV file with tweets and metadata features
# Ensure "Tweet" column contains textual data for processing
# Other numerical features like "Retweet Count", "Mention Count" are used for behavioral analysis
df = pd.read_csv("dataset/bot_detection_data.csv")

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase to maintain uniformity
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

# Apply text preprocessing to all tweets
df["clean_text"] = df["Tweet"].astype(str).apply(preprocess_text)

# Feature Engineering - Convert text into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit feature size for performance
X_text = vectorizer.fit_transform(df["clean_text"]).toarray()

# Selecting behavioral features
features = ["Retweet Count", "Mention Count", "Follower Count", "Verified"]  # Example features
X_behavior = df[features].fillna(0)  # Fill missing values with 0

# Standardizing behavioral features for better model performance
scaler = StandardScaler()
X_behavior = scaler.fit_transform(X_behavior)

# Combine text-based and behavioral features into one array
X_combined = np.hstack((X_text, X_behavior))

# Train Isolation Forest for anomaly detection in user behavior
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest.fit(X_behavior)
behavioral_scores = iso_forest.decision_function(X_behavior)  # Get anomaly scores

# Normalize anomaly scores between 0 and 1
minmax_scaler = MinMaxScaler()
behavioral_scores = minmax_scaler.fit_transform(behavioral_scores.reshape(-1, 1)).flatten()

# Load Pre-trained Transformer Model for NLP-based Bot Detection
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"  # Pre-trained model for sentiment analysis

# Load tokenizer and model for processing text
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to("cuda")  # Move model to GPU for faster inference

def predict_bot(text):
    # Tokenize and convert text to tensor format for model input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")
    
    # Disable gradient calculation as we are only making predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probability scores and return bot probability
    return torch.softmax(outputs.logits, dim=1)[0][1].item()

# Apply NLP model to all tweets to get bot likelihood scores
df["text_bot_score"] = df["clean_text"].apply(predict_bot)

# Normalize text bot scores between 0 and 1
df["text_bot_score"] = minmax_scaler.fit_transform(df[["text_bot_score"]])

# Final bot score - Averaging NLP and behavioral anomaly scores
df["final_bot_score"] = (df["text_bot_score"] + behavioral_scores[:len(df)]) / 2

# Initial bot classification based on a threshold
df["is_bot"] = df["final_bot_score"] > 0.5  # Default threshold of 0.5

# Save results to CSV file
df.to_csv("bot_detection_results.csv", index=False)

# Ensure dataset consistency by filtering out any NaN values
valid_indices = ~df["is_bot"].isna()
X_combined = X_combined[valid_indices]
y_labels = df.loc[valid_indices, "is_bot"].astype(int)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_labels, test_size=0.2, random_state=42)

# Apply SMOTE to balance dataset and avoid bias towards majority class
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train a traditional machine learning model (RandomForest) for bot detection
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)  # Fit model on training data

# Make predictions on test data
predictions = rf_model.predict(X_test)

# Adjust bot classification threshold using Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
best_threshold = thresholds[np.argmax(precisions * recalls)]  # Optimal threshold selection

df["is_bot"] = df["final_bot_score"] > best_threshold  # Apply optimized threshold
# Add this to your original code after training the model
import joblib

# Save the trained model
joblib.dump(rf_model, "models/bot_detection_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Save the StandardScaler for behavioral features
joblib.dump(scaler, "models/behavior_scaler.pkl")

# Save the MinMaxScaler for normalization
joblib.dump(minmax_scaler, "models/minmax_scaler.pkl")
# Print model evaluation metrics
print(classification_report(y_test, predictions))
