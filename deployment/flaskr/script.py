from flask import Flask, request, render_template

import pickle
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preprocessing function
def preprocess(text):
    punctuations = string.punctuation
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    corpus = [word.lower() for word in words if word not in stop_words and word not in punctuations]
    corpus = [lemmatizer.lemmatize(word) for word in corpus]
    corpus = ' '.join(corpus)
    return corpus

# Data transformation
def data_transformer(corpus):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(corpus)
    return vectors

# Prediction function
def ValuePredictor(to_predict_list):
    to_predict = [preprocess(text) for text in to_predict_list]
    x = data_transformer(to_predict)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(x)
    return result[0]

app = Flask(__name__, template_folder='../templates')

@app.route('/home')
def home():
    return render_template("form.html")

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        text_input = request.form.get('x')  # Assuming 'x' is the name of the input field
        to_predict_list = [text_input]
        result = ValuePredictor(to_predict_list)
        prediction = 'Spam' if int(result) == 1 else 'Ham'
        return render_template("result.html", prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True)
