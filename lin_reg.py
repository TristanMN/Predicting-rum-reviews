import numpy as np
import pandas as pd
import optuna
import pickle
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge  # Using Ridge instead of LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load and preprocess data
logger.info("Loading and preprocessing data...")
df = pd.read_csv('processed_data.csv')
df.dropna(inplace=True)

# Preprocessing similar to original script
def preprocess_text(text):
    import string
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet

    # Download necessary NLTK resources
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)

    # Remove punctuation and lowercase
    text = ''.join(char for char in text if char not in string.punctuation)
    text = ' '.join([word.lower() for word in text.split() if word not in stopwords.words('english')])
    
    # Lemmatization
    lemm = WordNetLemmatizer()
    tokens = text.split()
    tagged = nltk.pos_tag(tokens)
    converted = " ".join([lemm.lemmatize(word, pos=wordnet.NOUN) for word, tag in tagged])
    
    return converted

# Preprocess text columns
logger.info("Preprocessing text columns...")
df['ProcessedOpinion'] = df['Opinion'].apply(preprocess_text)
df['ProcessedOpinionTitle'] = df['OpinionTitle'].apply(preprocess_text)

# Combine features
logger.info("Preparing feature matrix...")
try:
    with open('opinion_vectorizer.pkl', 'rb') as f:
        tfidf_opinion = pickle.load(f)
    with open('title_vectorizer.pkl', 'rb') as f:
        tfidf_title = pickle.load(f)
except FileNotFoundError:
    print("Not found vectorizers")
    tfidf_opinion = TfidfVectorizer()
    tfidf_title = TfidfVectorizer()
    X = np.hstack([
        tfidf_opinion.fit_transform(df["ProcessedOpinion"]).toarray(),
        tfidf_title.fit_transform(df["ProcessedOpinionTitle"]).toarray()
    ])
    y = df["Rating"]

    with open('opinion_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_opinion, f)
    with open('title_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_title, f)

# Split data
logger.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e5)
    
    # Create model with suggested hyperparameters
    model = Ridge(alpha=alpha)
    
    # Use cross-validation to get robust performance estimate
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    return -scores.mean()

# Create Optuna study
logger.info("Starting Optuna optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

logger.info(f"Best trial: {study.best_trial.number}")
logger.info(f"Best hyperparameters: {study.best_params}")

# Train final model with best hyperparameters
logger.info("Training final model...")
final_model = Ridge(alpha=study.best_params['alpha'])
final_model.fit(X_train, y_train)

# Predictions and performance metrics
logger.info("Evaluating model performance...")
y_pred_train = final_model.predict(X_train)
y_pred_test = final_model.predict(X_test)

print("Optuna Optimized Ridge Regression Performance:")
print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.2f}")
print(f"Test MSE: {mean_squared_error(y_test, y_pred_test):.2f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
print(f"Train R2: {r2_score(y_train, y_pred_train):.2f}")
print(f"Test R2: {r2_score(y_test, y_pred_test):.2f}")

# Export model and vectorizers
logger.info("Exporting model and vectorizers...")
with open('optimized_ridge_regression.pkl', 'wb') as f:
    pickle.dump(final_model, f)

logger.info("Process completed successfully!")