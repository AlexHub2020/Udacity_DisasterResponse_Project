#import necessary libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import pickle
nltk.download('stopwords')

#load data out of the database
def load_data(database_filepath):
    #create database engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    #read in file
    df = pd.read_sql_table('disaster_data', engine)
    #define features and label arrays
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    Y.replace({'related': 2}, 1, inplace = True)
    #get category names
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    #normalize all words to lowercase
    text = text.lower()
    #remove special characters - keep alpha / numeric letters
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    #tokenization of the message
    words = word_tokenize(text)
    #remove stopwords out of the meassage
    words = [w for w in words if w not in stopwords.words('english')]
    return words


def build_model():
    # text processing and model pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
    # define parameters for GridSearchCV
    parameters = {'clf__estimator__n_estimators': [50, 75],
                  'clf__estimator__learning_rate': [0.75, 1]}

    # create gridsearch object and return as final model pipeline
    final_model = GridSearchCV(pipeline, param_grid = parameters)
    return final_model

def evaluate_model(model, X_test, Y_test, category_names):
    #predict data using the trained model
    pred_data = model.predict(X_test)
    #print report on prediction scores
    print(classification_report(Y_test, pred_data, target_names = category_names))

def save_model(model, model_filepath):
    #save the trained model
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
