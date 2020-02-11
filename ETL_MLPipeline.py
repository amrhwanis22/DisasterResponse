# import packages
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

class VerbExtractor(BaseEstimator, TransformerMixin):

    def verb_extractor(self, text):
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
        X_tagged = pd.Series(X).apply(self.verb_extractor)
        return pd.DataFrame(X_tagged)
def load_data(messages,categories):
    '''
     Load data function
     Arguments:
         messages->path the messages csv file
        categories->path to the categories csv file
    Output:
     df->loaded as pandas DataFrame
    '''
    # load datasets
    messages_df = pd.read_csv(messages)
    categories_df = pd.read_csv(categories)
    df=messages_df.merge(categories_df,left_on='id',right_on='id')
    return df
def clean_data(df):
    '''
    Input
        df -> dataframe combined from messages and categories datasets
    Output
        df ->  clean pandas ready for traning

    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop('categories', axis=1,inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df

def save_data(df,database_path):
    '''

    :param df->to save
    :param database_path->path of db to save the .db file

    output-> if the database already exists then it's deleted then recreated again by calling
     the same function at the except else it will be created within the try statement
    '''
    try:
        engine = sqlalchemy.create_engine('sqlite:///'+ database_path)
        df.to_sql('messages',engine,index=False)
    except:
        os.remove(database_path)
        save_data(df,database_path)
        print('[INFO] database already exists')



def load_from_db(database_path):
    engine=sqlalchemy.create_engine('sqlite:///'+database_path)
    df=pd.read_sql('SELECT * FROM messages',engine)
    X=df['message'].values
    print(df.columns)
    y= df.iloc[:, 4:]
    category_names=y.columns
    return X , y , category_names


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

    # define parameters for GridSearchCV
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
    }

    # create gridsearch object and return as final model pipeline
    pipeline = Pipeline([('features',FeatureUnion(
        [
            ('text_pipeline',Pipeline(
            [   ('vect',CountVectorizer(tokenizer=tokenize)),
                ('tfidf',TfidfTransformer())
            ])),
            ('verb_extractor',VerbExtractor())
        ]
        )),
        ('clf', MultiOutputClassifier(estimator=AdaBoostClassifier(),n_jobs=-1))
                        ])
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return model_pipeline


def train(X, y, model):
    # train test split
    print(y)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    # fit model
    model.fit(X_train,y_train)

    # output model test results
    y_pred=model.predict(X_test)
    y_pred_pd=pd.DataFrame(y_pred,columns=y_test.columns)
    for column in y_test.columns:
        print('=======================================================================')
        print('[INFO] feature {}\n'.format(column))
        print(classification_report(y_test[column],y_pred_pd[column]))

    return model


def export_model(model):
    #Export model as a pickle file
    filename = 'classifier.sav'
    pickle.dump(model, open(filename, 'wb'))


def run_pipeline(data_file):

    if len(data_file)==4:
        messages_file_path,categories_file_path,database_path=data_file[1:]

        print('[INFO] loading data'.format())
        # run ETL pipeline
        df = load_data(messages_file_path, categories_file_path)
        print('[INFO] cleaning data')

        df=clean_data(df)
        save_data(df,database_path)
        X,y=load_from_db(database_path)
        print(len(y))
        print(len(X))
        model = build_model()  # build model pipeline
        model = train(X, y, model)  # train model pipeline
        export_model(model)  # save model
    else:
        print('Please pass the message dataset path followed'
              ' by the categories dataset path then by the database location')


if __name__ == '__main__':
    data_file = sys.argv[:]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline