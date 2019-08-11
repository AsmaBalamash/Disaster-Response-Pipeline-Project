import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import re
import pickle

def load_data(database_filepath):
   '''
   Function: Load dataset from database
   Input: 
   database_filepath -> sql file path
   Output: 
   X -> feature
   Y -> target variable
   category_names -> target columns' names
   '''
   engine = create_engine('sqlite:///{}'.format(database_filepath))
   df = pd.read_sql_table('mytable',engine)
   X = df['message']
   Y = df.drop(['message','genre', 'id', 'original'],axis=1)
   category_names = Y.columns
   return X, Y, category_names


def tokenize(text):
    '''
    Function: tokenization function to process text data
    Input: Text 
    Ouptut: Cleaned tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    This function will build a model using MultiOutputClassifier for predicting multiple target variables and then improving it using grid search.
    Output: return the model
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters =  {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [10, 25], 
              'clf__estimator__min_samples_split': [2, 4]}
    
    #Use grid search to find better parameters.
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function used for testing the model using classification_report and accurancy.
    Input: model, X_test, Y_test, category_names
    Output: classification_report
    '''
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print('('+str(i+1)+') '+ category_names[i])
        print(classification_report(Y_test[category_names[i]], y_pred[:, i]))

        
def save_model(model, model_filepath):
    '''
    This function for exporting the model as a pickle file
    Input: 
    model -> contains the trained model    
    model_filepath -> the path for saving the pickle file
    '''
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