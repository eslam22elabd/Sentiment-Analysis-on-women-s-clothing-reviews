# Women's E-Commerce Clothing Reviews NLP

## Introduction
This project aims to analyze the sentiment of women's clothing reviews using machine learning techniques. The model is based on the Naive Bayes algorithm to classify reviews as positive or negative.

## Requirements
- Python 3.7 or higher
- Required libraries:
  - pandas
  - numpy
  - gensim
  - scikit-learn
  - nltk
  - fastapi
  - uvicorn

You can install the requirements using:
```bash
pip install -r requirements.txt

project-directory/
│
├── model_with_vectorizer.pkl  # Naive Bayes model with TF-IDF vectorizer
├── main.py                     # FastAPI application                    # Testing script for the model
├── README.md                   # This file
└── requirements.txt            # Project dependencies

How to Use
Run the Model: You can start the application using the following command:

uvicorn main:app --reload