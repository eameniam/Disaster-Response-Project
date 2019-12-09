# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#import word cleaning libraries

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# import sklearn libraries

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    # read in file
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df=pd.read_sql_table("SELECT * FROM disaster_response", engine)
    X = df['message']
    y = df.iloc[:,4:]
    return X, y



def tokenize(text):
    #normalize text
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    tokens = [w for w in words if w not in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
     # text processing and model pipeline
     pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {'clf__estimator__max_depth': [10,50,None],
              'clf__estimator__min_samples_leaf': [2,5,10]}
    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, parameters)

    return model_pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]), target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))



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