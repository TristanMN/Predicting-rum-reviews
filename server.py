from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import joblib  # Use joblib for scikit-learn models
import pandas as pd
nltk.download('averaged_perceptron_tagger_eng')
print("Initializing Flask app")
app = Flask(__name__)
app.static_folder = 'static'
print("Flask app initialized")

# Define preprocessing functions
def preprocess_text(text):
    text = ''.join(char for char in text if char not in string.punctuation)
    words = [word.lower() for word in text.split() if word not in stopwords.words('english')]
    lemm = WordNetLemmatizer()
    tokens = words
    tagged = nltk.pos_tag(tokens)
    lemmatized = " ".join(lemm.lemmatize(word, pos='n') for word, tag in tagged)
    return lemmatized

print("Loading TFIDF vectorizer")
# Load TFIDF vectorizer and model
tfidf = TfidfVectorizer()
print("Loading processed dataset")
df = pd.read_csv('processed_data.csv')
print("Preprocessing")
df['Opinion'] = df['Opinion'].dropna().apply(preprocess_text)
print("Fitting vectorizer")
tfidf.fit(df['Opinion'])

print("Loading model")
# Load the RandomForestClassifier model using joblib
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loded")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the form data
    review_text = request.form.get("review", "").strip()
    
    if not review_text:
        return jsonify({"error": "No review text provided"}), 400

    # Preprocess the review text
    preprocessed_review = preprocess_text(review_text)
    
    # Transform the review text into a vector
    review_vector = tfidf.transform([preprocessed_review]).toarray()
    
    # Predict the rating
    rating = model.predict(review_vector)[0]
    
    # Return the predicted rating as JSON
    return jsonify({"rating": int(rating)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
