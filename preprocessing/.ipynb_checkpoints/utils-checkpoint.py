import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import gensim
from gensim.models import Word2Vec
import spacy
from tqdm import tqdm
from nltk.stem import SnowballStemmer

nltk.download('punkt')

acepted_languages = {
    'english': 'en_core_web_sm',
    'portuguese': 'pt_core_news_sm',
    'spanish': 'es_core_news_sm',
    'french': 'fr_core_news_sm',
    'italian': 'it_core_news_sm'
}

#Preprocess functions

def transform_to_lowercase(data, verbose=0, **kwargs):
    if verbose > 0:
        print('Converting text to lowercase...')
    data['prep'] = data['prep'].str.lower()
    if verbose > 0:
        print('Done!\n')
    return data

def remove_special_characters(data, verbose=0, **kwargs):
    if verbose > 0:
        print('Removing special characters from text...')
    data['prep'] = data['prep'].apply(lambda x: re.sub(r'[^a-zA-Z0-9À-ÿ\s]', ' ', unidecode(x)))
    if verbose > 0:
        print('Done!\n')
    return data

def remove_stopwords(data, stopwords=None, verbose=0, **kwargs):
    if verbose > 0:
        print('Getting stopword list and removing from text...')
    if stopwords is None:
        stopwords = getting_stopwords(custom_stopword_list=stopwords, verbose=verbose, **kwargs)
    data['prep'] = data['prep'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    if verbose > 0:
        print('Done!\n')
    return data

def remove_specific_phrases(data, phrases=[], verbose=0, **kwargs):
    if verbose > 0:
        print('Removing phrases from text...')
    data['prep'] = data['prep'].apply(lambda x: remove_phrases_from_text(x, phrases))
    if verbose > 0:
        print('Done!\n')
    return data

def remove_phrases_from_text(data, phrases=list, **kwargs):
    data = str(data)
    for phrase in phrases:
        data = data.replace(phrase, ' ')
    return data

def remove_missing_values(data, column, **kwargs):
    data[column].replace('', np.nan, inplace=True)
    data['prep'].replace('', np.nan, inplace=True)
    data.dropna(subset=[column, 'prep'], inplace=True)
    return data


def getting_stopwords(custom_stopword_list=None, language='english', list_del_stopwords=None, 
                      list_add_stopwords=None, verbose=0, **kwargs):
    if custom_stopword_list:
        stop_words = custom_stopword_list
    else:
        nltk.download('stopwords')
        valid_languages = set(stopwords.fileids())
        if language not in valid_languages:
            raise ValueError(f"The language '{language}' is not valid for stopwords in NLTK.\n")
        stop_words = stopwords.words(language)

    if list_del_stopwords:
        stop_words = [word for word in stop_words if word not in list_del_stopwords]

    if list_add_stopwords:
        stop_words.extend(list_add_stopwords)
  
    for i in range(len(stop_words)):
        stop_words[i] = unidecode(stop_words[i]).lower()
    
    return stop_words


def white_space_tokenizer(data):
    tokenizer = WhitespaceTokenizer()
    data['prep'] = data['prep'].apply(lambda x: ' '.join(tokenizer.tokenize(x)))
    return data


# Stemming and Lemmatization

def perform_stemming(data, language='english', verbose=0, **kwargs):
    """
    Perform stemming on a specific column of a DataFrame using NLTK SnowballStemmer.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the data.
        language (str): The language to use for stemming. Default is 'english'.

    Returns:
        pandas.DataFrame: A new DataFrame with the stemmed values in the specified column.
    """
    
    if verbose > 0:
        print('Performing the stemmization of the text...')

    # Check if the specified language is supported
    if language not in SnowballStemmer.languages:
        raise ValueError(f"Stemmer for '{language}' language is not supported.\n")

    # Initialize the stemmer
    stemmer = SnowballStemmer(language)
    tqdm.pandas(desc="Processing column {}".format('prep'))

    # Perform stemming on the specified column
    data['prep'] = data['prep'].progress_apply(
        lambda x: " ".join([stemmer.stem(word) for word in nltk.word_tokenize(x)])
    )
    
    if verbose > 0:
        print('Done!\n')

    return data


def perform_lemmatization(data, language='english', verbose=0, **kwargs):
    """
    Perform lemmatization on a specific column of a DataFrame using spaCy.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the data.
        language (str): The language to use for lemmatization. Default is 'english'.

    Returns:
        pandas.DataFrame: A new DataFrame with the lemmatized values in the specified column.
    """
    
    if verbose > 0:
        print('Performing the lemmatization of the text...')

    # Load the spaCy language model for the specified language
    if language not in acepted_languages:
        raise ValueError(f"Lemmatizer for '{language}' language is not supported.\n")
        
    nlp = spacy.load(acepted_languages[language])
    tqdm.pandas(desc="Processing column {}".format('prep'))

    # Perform lemmatization on the specified column
    data['prep'] = data['prep'].progress_apply(
        lambda x: " ".join([token.lemma_ for token in nlp(x)])
    )
    
    if verbose > 0:
        print('Done!\n')
        
    return data

# Vectorization

def tokenize(data, **kwargs):
    data['tokens'] = data['prep'].apply(lambda x: word_tokenize(str(x).lower()))
    return data


def get_word2vec_features(tokens, model):
    features = []
    for token in tokens:
        if token in model.wv:
            features.append(model.wv[token])
    return sum(features) / len(features) if features else []


def vectorization(data, vectorizer='tf-idf', target=None, verbose=0, **kwargs):
    if verbose > 0:
        print('=============================== RUNNING THE VECTORIZATION ===============================\n')
        print(f'Applying {vectorizer} vectorizer...')
        
    if vectorizer == 'tf-idf':
        vector = TfidfVectorizer().fit(data['prep'])
        vectorized_data = vector.transform(data['prep'])
        vectorized_data = pd.DataFrame.sparse.from_spmatrix(vectorized_data)
        vectorized_data.columns = vector.get_feature_names_out()
        
    elif vectorizer == 'bow':
        vector = CountVectorizer().fit(data['prep'])
        vectorized_data = vector.transform(data['prep'])
        vectorized_data = pd.DataFrame.sparse.from_spmatrix(vectorized_data)
        vectorized_data.columns = vector.get_feature_names_out()
        
    elif vectorizer == 'word2vec':
        data = tokenize(data)
        vector = Word2Vec(data['tokens'], vector_size=100, window=5, min_count=1, workers=4)
        data['word2vec_features'] = data['tokens'].apply(get_word2vec_features, args=(vector,))
        vectorized_data = pd.DataFrame(data['word2vec_features'].tolist())
        vectorized_data.columns = vectorized_data.columns.astype(str)
        vectorized_data.fillna(0, inplace=True)
        
    else:
        if verbose > 0:
            print("Invalid option, choose 'tf-idf', 'bow', or 'word2vec'.")
        return None, None

    if target is not None:
        vectorized_data = pd.concat([vectorized_data, data[target].reset_index(drop=True)], axis=1)
    
    if verbose > 0:
        print('Done!\n')
        print('=============================== END OF THE VECTORIZATION ================================\n')
        
    return vector, vectorized_data

def vectorization_transform(data, vectorizer='tf-idf', vector=None, target=None, verbose=0, **kwargs):
    if verbose > 0:
        print('=============================== RUNNING THE VECTORIZATION ===============================\n')
        print(f'Applying {vectorizer} vectorizer...')
        
    if vectorizer == 'tf-idf':
        vectorized_data = vector.transform(data['prep'])
        vectorized_data = pd.DataFrame.sparse.from_spmatrix(vectorized_data)
        vectorized_data.columns = vector.get_feature_names_out()
        
    elif vectorizer == 'bow':
        vectorized_data = vector.transform(data['prep'])
        vectorized_data = pd.DataFrame.sparse.from_spmatrix(vectorized_data)
        vectorized_data.columns = vector.get_feature_names_out()
        
    elif vectorizer == 'word2vec':
        data = tokenize(data)
        data['word2vec_features'] = data['tokens'].apply(get_word2vec_features, args=(vector,))
        vectorized_data = pd.DataFrame(data['word2vec_features'].tolist())
        vectorized_data.columns = vectorized_data.columns.astype(str)
        vectorized_data.fillna(0, inplace=True)
        
    else:
        if verbose > 0:
            print("Invalid option, choose 'tf-idf', 'bow', or 'word2vec'.")
        return None, None

    if target is not None:
        vectorized_data = pd.concat([vectorized_data, data[target].reset_index(drop=True)], axis=1)
    
    if verbose > 0:
        print('Done!\n')
        print('=============================== END OF THE VECTORIZATION ================================\n')
        
    return vectorized_data