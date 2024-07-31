import sys
sys.path.append('../preprocessing/')

import pandas as pd
import numpy as np
from preprocess import *
from tqdm import tqdm
import itertools
from collections import OrderedDict
import yake
from nltk.util import ngrams
import nltk
import math
from collections import Counter

acepted_languages = {
    'english': 'en',
    'portuguese': 'pt',
    'spanish': 'es',
    'french': 'fr',
    'italian': 'it'
}

nltk.download('punkt')

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results


def tfidf_extractor(data, number_to_extract=1000, number_of_tokens=1, verbose=0, **kwargs):
    if verbose > 0:
        print('Applying tfidf measure...')
        
    cv = CountVectorizer(max_df=0.80, ngram_range=(number_of_tokens, number_of_tokens))
    word_count_vector = cv.fit_transform(data['prep'])
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    
    # you only needs to do this once, this is a mapping of index to 
    feature_names = cv.get_feature_names_out()

    #generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([data['prep'].to_string()]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 1000
    keywords = extract_topn_from_vector(feature_names, sorted_items, -1)
        
    keywords = {k: keywords[k] for k in list(keywords)[:number_to_extract]}

    extract = []
    for kw in keywords:
        extract.append(kw)
    return extract

######################

def yake_extractor(data, number_to_extract=1000, number_of_tokens=1, verbose=0, **kwargs):
    if verbose > 0:
        print('Applying yake measure...')
        
    data = ' '.join(data['prep'])
    kw_extractor = yake.KeywordExtractor(top=number_to_extract, n=number_of_tokens)
    keywords = kw_extractor.extract_keywords(data)
    extract = []
    for kw in keywords:
        extract.append(kw[0])
    return extract

######################

def compute_ngram_counts(data, column_name, n):
    text_column = data[column_name]
    flattened_text = [token for sublist in text_column for token in sublist]
    ngram_counts = Counter(zip(*(flattened_text[i:] for i in range(n))))
    return ngram_counts

def compute_joint_frequency(words_occurrences, total_count):
    log_probability = 0.0
    for i in range(len(words_occurrences)):
        result = words_occurrences[i] / total_count
        log_probability = log_probability + math.log(result, 2)
    return log_probability

def pmi_extractor(ngram_counts, unigram_counts, total_count, number_to_extract=1000, verbose=0, **kwargs):
    '''
    Pointwise mutual information
    '''
    if verbose > 0:
        print('Applying pmi measure...')
        
    pmi_scores = {}
    
    for ngram, count in ngram_counts.items():
        ngram_str = ' '.join(ngram)  # Convert the tuple back to a string
            
        words = ngram_str.split()
        words_occurrences = []
        
        for word in words:
            words_occurrences.append(unigram_counts[word,])
        
        joint_frequency = compute_joint_frequency(words_occurrences, total_count)
        probability_ngram = math.log((count/total_count), 2)
        
        pmi_scores[ngram_str] = probability_ngram - joint_frequency #log2(P(X, Y) / (P(X) * P(Y))) -> logP(X, Y, Z) - (logP(X) + logP(Y) + logP(Z))
        
    pmi_df = pd.DataFrame.from_dict(pmi_scores, orient='index', columns=['pmi'])
    pmi_df.sort_values(by=['pmi'], ascending=False, inplace=True)
        
    pmi_df = pmi_df.head(number_to_extract)
    extract = list(pmi_df.index)
        
    return extract

######################

def dice_extractor(ngram_counts, unigram_counts, number_to_extract=1000, verbose=0, **kwargs): 
    '''
    Sørensen–Dice coefficient
    '''
    if verbose > 0:
        print('Applying dice measure...')
        
    dice_scores = {}
    
    for ngram, count in ngram_counts.items():
        ngram_str = ' '.join(ngram)  # Convert the tuple back to a string
            
        words = ngram_str.split()
        words_occurrences = []
        
        for word in words:
            words_occurrences.append(unigram_counts[word,])
        
        dice_scores[ngram_str] = (len(words_occurrences) * count) / sum(words_occurrences)
        
    dice_df = pd.DataFrame.from_dict(dice_scores, orient='index', columns=['dice'])
    dice_df.sort_values(by=['dice'], ascending=False, inplace=True)
        
    dice_df = dice_df.head(number_to_extract)
    extract = list(dice_df.index)
        
    return extract

######################

def mle_extractor(data, ngram_counts, total_count, number_to_extract=1000, number_of_tokens=1, verbose=0, **kwargs):
    '''
    Maximum Likelihood Estimate
    '''
    if verbose > 0:
        print('Applying mle measure...')
        
    mle_scores = {}
    if number_of_tokens > 1:
        ngram_minus1_counts = compute_ngram_counts(data, 'tokens', number_of_tokens-1)
    
    for ngram, count in ngram_counts.items():
        ngram_str = ' '.join(ngram)
        if len(ngram) > 1:
            mle_scores[ngram_str] = count / ngram_minus1_counts[ngram[:-1]]
        else:
            mle_scores[ngram_str] = count / total_count
        
    mle_df = pd.DataFrame.from_dict(mle_scores, orient='index', columns=['mle'])
    mle_df.sort_values(by=['mle'], ascending=False, inplace=True)
        
    mle_df = mle_df.head(number_to_extract)
    extract = list(mle_df.index)
        
    return extract

######################

def count_extractor(ngram_counts, unigram_counts, number_to_extract=1000, verbose=0, **kwargs):
    if verbose > 0:
        print('Applying count measure...')
        
    count_scores = {}
    
    for ngram, count in ngram_counts.items():
        ngram_str = ' '.join(ngram)  # Convert the tuple back to a strin
        count_scores[ngram_str] = count
        
    count_df = pd.DataFrame.from_dict(count_scores, orient='index', columns=['count'])
    count_df.sort_values(by=['count'], ascending=False, inplace=True)
        
    count_df = count_df.head(number_to_extract)
    extract = list(count_df.index)
        
    return extract

######################

def extract_terms(data, measures, number_to_extract=1000, number_of_tokens=1, verbose=0, threshold_percentage=1):
    if verbose > 0:
        print('============================ RUNNING THE TERMS IDENTIFICATION ===========================\n')
        print(f'Performing the terms identification with {measures} measures...\n')
        
    data['tokens'] = data['prep'].apply(lambda x: nltk.word_tokenize(x.lower()))
    ngram_counts = compute_ngram_counts(data, 'tokens', number_of_tokens)
    unigram_counts = compute_ngram_counts(data, 'tokens', 1)
    total_count = sum(unigram_counts.values())
    extract = {}
    
    if threshold_percentage != 1:
        threshold_percentage = 1-threshold_percentage
        number_to_extract = int(len(ngram_counts) * threshold_percentage)
        
    if verbose > 0:
        print(f"You're gonna identify {number_to_extract} terms per measure!")
        if (number_to_extract >= 1000) and ('yake' in measures):
            print('\033[1;91m' + f'Warning: This number is high, yake measure can take several minutes to complete!' + '\033[0m')
        print('')
    for measure in measures:
        try:
            extractor_function_name = f"{measure}_extractor"
            extractor_function = globals()[extractor_function_name]
            extract[measure] = extractor_function(data=data,
                                                  ngram_counts=ngram_counts, 
                                                  unigram_counts=unigram_counts, 
                                                  total_count=total_count,
                                                  number_of_tokens=number_of_tokens,
                                                  number_to_extract=number_to_extract,
                                                  verbose=verbose)
        except Exception as e:
            print("ERROR : "+str(e))
            raise ValueError(f"The measure '{measure}' is not valid for word extractor, use: 'yake', 'tfidf', 'pmi', 'mle', 'dice', or 'count'.\n")

    if verbose > 0:
        print('\nDone!\n')
        print('============================ END OF THE TERMS IDENTIFICATION ============================\n')
    return extract

###########################
###########################
###########################

def getting_word_list(extract):
    word_list = [word for item in extract for word in item.split()]
    return word_list

def generate_combinations(lst, number_of_tokens):
    combinations = []
    for word in lst:
        for i in range(number_of_tokens, 0, -1):
            for subset in itertools.combinations(lst, i):
                if subset[0] == word:
                    combinations.append(' '.join(subset))
    return list(OrderedDict.fromkeys(combinations))

def extract_words(data, extract_set, word_list, number_of_tokens):
        
    words = data.split()
    
    intersection = set(words) & set(word_list)
    filtered_list = [word for word in words if word in intersection]
    
    combinations = generate_combinations(filtered_list, number_of_tokens)
    result = filter_candidates(combinations, extract_set)
    
    return result


def filter_candidates(candidates, extract_set):
    filtered_list = [candidate for candidate in candidates if candidate in extract_set]
    return filtered_list


def terms_extractor(data, target_column='column_text', result_column='your list', 
                    measures=['yake', 'tfidf', 'pmi', 'mle', 'dice', 'count'], extract=None, 
                    number_of_tokens=1, number_to_extract=1000, stopwords=False, language='english', verbose=0, 
                    threshold_percentage=1, perform_term_extraction=True, **kwargs):
    
    if stopwords:
        preprocessing_funcs = ['transform_to_lowercase', 'remove_special_characters', 'remove_stopwords']
    else:
        preprocessing_funcs = ['transform_to_lowercase', 'remove_special_characters']
    
    data = preprocess_data(data=data, 
                           preprocessing_funcs=preprocessing_funcs,
                           column=target_column,
                           language=language,
                           verbose=verbose)
    
    if extract is None:
        extract = extract_terms(data=data, 
                                number_to_extract=number_to_extract, 
                                threshold_percentage=threshold_percentage,
                                number_of_tokens=number_of_tokens,
                                measures=measures, 
                                verbose=verbose)
        
    if type(extract) is not dict:
        extract = {result_column: extract}
        
    if 'tokens' in data.columns:
        data.drop('tokens', axis=1, inplace=True)
    
    if perform_term_extraction:
        
        if verbose > 0:
            print('============================== RUNNING THE TERMS EXTRACTION =============================\n')
            print(f'This stage can take several minutes...\n')

        for measure in extract:
            word_list = getting_word_list(extract[measure])
            extract_set = set(extract[measure])

            if verbose > 0:
                print(f'Terms extraction with {measure}...')

            tqdm.pandas(desc="Processing column {}".format('prep'))
            data[measure] = data['prep'].progress_apply(
                lambda x: extract_words(x, extract_set, word_list, number_of_tokens)
            )

        if verbose > 0:
            print('Done!\n')
            print('============================== END OF THE TERMS EXTRACTION ==============================\n')

    return extract, data

###########################
###########################
###########################

def common_elements(extract, n=2):
    element_counter = Counter()
    for key, value in extract.items():
        element_counter.update(value)

    # Select elements that appear at least n times
    common_elements_at_least_n = [element for element, count in element_counter.items() if count >= n]
    return common_elements_at_least_n


def unique_elements(extract):
    unique_elements_by_list = {}
    for current_list_name, current_list_values in extract.items():
        unique_elements = set(current_list_values)  # Initialize with the current list's values

        for other_list_name, other_list_values in extract.items():
            if current_list_name != other_list_name:
                unique_elements -= set(other_list_values)  # Remove elements from other lists

        unique_elements_by_list[current_list_name] = list(unique_elements)

    return unique_elements_by_list


def merge_lists(extract):
    merged_list = []
    for value in extract.values():
        merged_list.extend(value)

    unique_merged_list = list(set(merged_list))

    return unique_merged_list