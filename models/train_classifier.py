import sys
import pandas as pd
import numpy as np
import sqlalchemy
import re
from nltk import word_tokenize
import nltk
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
import pickle

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
pd.set_option('display.max_columns',None)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    engine=sqlalchemy.create_engine('sqlite:///'+database_filepath)
    df=pd.read_sql('SELECT * FROM messages',engine)
    X=df['message'].values
    print(df.columns)
    y= df.iloc[:, 4:]
    cat_names=y.columns
    return X,y,cat_names



def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_url = re.findall(url_regex, text)
    for url in detected_url:
        text.replace(url, 'urlplaceholder')
    tokens = word_tokenize(text)
    lemmitizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmitizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens



def build_model():
    """
    Build Model function
    
    This function output is a Scikit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
    """
    
    # define parameters for GridSearchCV
    parameters = {
        # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'features__text_pipeline__vect__max_df': (0.75, 1.0),
        # 'features__text_pipeline__vect__max_features': (None, 5000),
        # 'features__text_pipeline__tfidf__use_idf': (True, False),
    }

    # create gridsearch object and return as final model pipeline
    pipeline = Pipeline([('features',FeatureUnion(
        [
            ('text_pipeline',Pipeline(
            [   ('vect',CountVectorizer(tokenizer=tokenize)),
                ('tfidf',TfidfTransformer())
            ])),
            ('verb_extractor',StartingVerbExtractor())
        ]
        )),
        ('clf', MultiOutputClassifier(estimator=AdaBoostClassifier(),n_jobs=-1))
                        ])
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return model_pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies ML pipeline to a test set and prints out
    model performance (accuracy and recall and f1score)
    
    Arguments:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names (multi-output)
    """
    
    # output model test results
    y_pred=model.predict(X_test)
    y_pred_pd=pd.DataFrame(y_pred,columns=Y_test.columns)
    print(category_names)
    for column in Y_test.columns:
        print('=======================================================================')
        print('[INFO] feature {}\n'.format(column))
        print(classification_report(Y_test[column],y_pred_pd[column]))
    
    overall_accuracy = (y_pred == Y_test).mean().mean()
    print('over all accuracy : {}'.format(overall_accuracy))


def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


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