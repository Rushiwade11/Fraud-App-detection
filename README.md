# Fraud-App-detection
pip install flask scikit-learn nltk pandas
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load your dataset
data = pd.read_csv('fraud_data.csv')

# Preprocessing
X = data['text']
y = data['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Save the model and vectorizer
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model trained and saved.")
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load('fraud_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)
    
    result = {
        'text': text,
        'prediction': int(prediction[0]),
        'label': 'Fraud' if prediction[0] == 1 else 'Non-fraud'
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
text,label
"Transaction was successful",0
"Fraudulent activity detected",1
"Payment processed without issues",0
"Unauthorized transaction occurred",1
python train_model.py
python app.py
curl -X POST -H "Content-Type: application/json" -d '{"text": "Fraudulent transaction attempted"}' http://127.0.0.1:5000/predict
flask
scikit-learn
nltk
pandas
joblib
