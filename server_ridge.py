from flask import Flask, request, jsonify, render_template
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.static_folder = 'static'

with open('opinion_vectorizer.pkl', 'rb') as f:
    tfidf_opinion = pickle.load(f)
with open('title_vectorizer.pkl', 'rb') as f:
    tfidf_title = pickle.load(f)
with open('optimized_ridge_regression.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_text(text):
    text = ''.join(char for char in text if char not in string.punctuation)
    words = [word.lower() for word in text.split() if word not in stopwords.words('english')]
    lemm = WordNetLemmatizer()
    tokens = words
    tagged = nltk.pos_tag(tokens)
    lemmatized = " ".join(lemm.lemmatize(word, pos='n') for word, tag in tagged)
    return lemmatized

@app.route('/')
def home():
    return render_template('index_ridge.html')

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form.get("review", "").strip()
    title_text = request.form.get("title", "").strip()
    
    if not review_text or not title_text:
        return jsonify({"error": "Both title and review text are required."}), 400

    processed_review = preprocess_text(review_text)
    processed_title = preprocess_text(title_text)

    review_vector = tfidf_opinion.transform([processed_review]).toarray()
    title_vector = tfidf_title.transform([processed_title]).toarray()

    feature_vector = np.hstack([review_vector, title_vector])
    rating = model.predict(feature_vector)[0]
    
    return jsonify({"rating": round(rating, 1)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)