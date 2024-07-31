import pandas as pd
import numpy as np
from preprocess import *
from shallow_learning import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class PreprocessDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessing_funcs, column='text', **kwargs):
        self.preprocessing_funcs = preprocessing_funcs
        self.column = column
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_data = preprocess_data(X, self.preprocessing_funcs, self.column, **self.kwargs)
        return processed_data

# Create a custom transformer class for vectorization
class VectorizationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer='tf-idf', vector=None, target=None, **kwargs):
        self.vectorizer = vectorizer
        self.kwargs = kwargs
        self.vector = vector
        self.target = target

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        elif isinstance(X, pd.Series):
            data = pd.DataFrame({'prep': X})
        else:
            data = pd.DataFrame({'prep': [X]})
        self.vector, _ = vectorization(data, vectorizer=self.vectorizer, **self.kwargs)
        return self

    def transform(self, X):
        if self.target in X.columns:
            vectorized_data = vectorization_transform(X, self.vectorizer, self.vector, target=self.target, **self.kwargs)
        else:
            vectorized_data = vectorization_transform(X, self.vectorizer, self.vector, target=None, **self.kwargs)
        return vectorized_data

class TrainingDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, algorithm='rf', target=None, **kwargs):
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.target = target
        self.model = None

    def fit(self, X, target):
        X_train, X_test, y_train, y_test = split(X, target=self.target, **self.kwargs)
        self.model = training_model(X_train, X_test, y_train, y_test, algorithm=self.algorithm, **self.kwargs)
        return self

def NLP_helper(preprocessing_funcs, vectorizer='tf-idf', algorithm='nb', target='add_target', **kwargs):
    pipeline = Pipeline([
        ('preprocess', PreprocessDataTransformer(preprocessing_funcs=preprocessing_funcs, **kwargs)),
        ('vectorize', VectorizationTransformer(vectorizer, **kwargs)),
        #('model', TrainingDataTransformer(algorithm, target, **kwargs)),
    ])
    return pipeline

class NLP(object):
    def __init__(self, preprocessing_funcs, vectorizer='tf-idf', algorithm='nb', target='add_target', **kwargs):
        self.preprocessing_funcs = preprocessing_funcs
        self.vectorizer = vectorizer
        self.algorithm = algorithm
        self.target = target
        self.kwargs = kwargs

    def preprocess_data(self, X):
        # Aplica os métodos de preprocessamento necessários em X
        # Retorna o resultado do preprocessamento
        return PreprocessDataTransformer(preprocessing_funcs=self.preprocessing_funcs, **self.kwargs).transform(X)
    
    def vectorize_data(self, X):
        # Aplica a vetorização em X
        # Retorna o resultado da vetorização
        return VectorizationTransformer(vectorizer=self.vectorizer, **self.kwargs).transform(X)
    
    def fit(self, X):
        pipeline = Pipeline([
            ('preprocess', preprocess_data(self.preprocess_data)),
            ('vectorize', vectorize_data(self.vectorize_data)),
        ])
        return pipeline
    '''def fit(self, X):
        pipeline = Pipeline([
        ('preprocess', PreprocessDataTransformer(preprocessing_funcs=self.preprocessing_funcs, **self.kwargs).transform(X)),
        ('vectorize', VectorizationTransformer(vectorizer=self.vectorizer, **self.kwargs).fit(X)),
        #('model', TrainingDataTransformer(algorithm, target, **kwargs).fit()),
    ])
        return pipeline'''