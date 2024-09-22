from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import html
import unicodedata
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Load the saved model and TF-IDF Vectorizer
with open('model_with_vectorizer.pkl', 'rb') as file:
    nb_model, tfidf_vectorizer = pickle.load(file)

app = FastAPI()

# Define request body
class Review(BaseModel):
    text: str

# Preprocessing functions
def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))

def remove_non_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def to_lowercase(text):
    return text.lower()

def replace_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

def stem_words(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def text2words(text):
    return word_tokenize(text)

def normalize_text(text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = text2words(text)
    words = remove_stopwords(words)
    words = stem_words(words)
    return ' '.join(words)

# Prediction function
def predict_review(review):
    processed_text = normalize_text(review)
    vector = tfidf_vectorizer.transform([processed_text])  # تحويل النص إلى متجه باستخدام TF-IDF
    prediction = nb_model.predict(vector)
    return prediction[0]

@app.post("/predict/")
def predict(review: Review):
    prediction = predict_review(review.text)
    recommendation = "Recommended" if prediction == 1 else "Not Recommended"
    return {"review": review.text, "recommendation": recommendation}

# To run the app, use the command: uvicorn main:app --reload
