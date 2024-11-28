import numpy as np
import pandas as pd
import optuna
import pickle
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Loading and preprocessing data...")
try:
    df = pd.read_pickle('preprocessed_dataframe.pkl')
except FileNotFoundError:
    df = pd.read_csv('processed_data.csv')
    df.dropna(inplace=True)

    def preprocess_text(text):
        import string
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet

        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('wordnet', quiet=True)

        text = ''.join(char for char in text if char not in string.punctuation)
        text = ' '.join([word.lower() for word in text.split() if word not in stopwords.words('english')])
        
        lemm = WordNetLemmatizer()
        tokens = text.split()
        tagged = nltk.pos_tag(tokens)
        converted = " ".join([lemm.lemmatize(word, pos=wordnet.NOUN) for word, tag in tagged])
        
        return converted

    logger.info("Preprocessing text columns...")
    df['ProcessedOpinion'] = df['Opinion'].apply(preprocess_text)
    df['ProcessedOpinionTitle'] = df['OpinionTitle'].apply(preprocess_text)
    
    df.to_pickle('preprocessed_dataframe.pkl')

logger.info("Preparing feature matrix...")
try:
    with open('opinion_vectorizer.pkl', 'rb') as f:
        tfidf_opinion = pickle.load(f)
    with open('title_vectorizer.pkl', 'rb') as f:
        tfidf_title = pickle.load(f)
except FileNotFoundError:
    logger.info("Vectorizers not found, creating new ones...")
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

logger.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    model = LinearRegression()
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    return -scores.mean()

logger.info("Starting Optuna optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

logger.info(f"Best trial: {study.best_trial.number}")

logger.info("Training final model...")
final_model = LinearRegression()
final_model.fit(X_train, y_train)

logger.info("Evaluating model performance...")
y_pred_train = final_model.predict(X_train)
y_pred_test = final_model.predict(X_test)

print("Optuna Optimized Linear Regression Performance:")
print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.2f}")
print(f"Test MSE: {mean_squared_error(y_test, y_pred_test):.2f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
print(f"Train R2: {r2_score(y_train, y_pred_train):.2f}")
print(f"Test R2: {r2_score(y_test, y_pred_test):.2f}")

logger.info("Exporting model...")
with open('optimized_linear_regression.pkl', 'wb') as f:
    pickle.dump(final_model, f)

logger.info("Process completed successfully!")