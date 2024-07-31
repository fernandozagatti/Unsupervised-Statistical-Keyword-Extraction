from utils import *
import warnings
warnings.filterwarnings("ignore")

functions = {
    'transform_to_lowercase': transform_to_lowercase, 
    'remove_special_characters': remove_special_characters, 
    'remove_stopwords': remove_stopwords,
    'remove_specific_phrases': remove_specific_phrases,
    'perform_lemmatization': perform_lemmatization,
    'perform_stemming': perform_stemming,
}

def preprocess_data(data, preprocessing_funcs, column, verbose=0, **kwargs):
    if(verbose > 0):
        print('=============================== RUNNING THE PREPROCESSING ===============================\n')
        print('Defined pipeline:', preprocessing_funcs, '\n')
        
    if isinstance(data, str):
        data = pd.DataFrame({column: [data]})
    
    if isinstance(data, list):
        data = pd.DataFrame({column: data})
    
    processed_data = data.copy()
    processed_data['prep'] = data[column]
    
    processed_data['prep'].fillna("", inplace=True)

    for preprocess_func in preprocessing_funcs:
        if(verbose > 0):
            print(f"Preprocess --> {preprocess_func}")
        if preprocess_func in functions:
            try:
                processed_data = functions[preprocess_func](data=processed_data, verbose=verbose, **kwargs)
            except Exception as e:
                if(verbose > 0):
                    print(e)
        else:
            if(verbose > 0):
                print(f"Skipping invalid preprocessing function: {preprocess_func}\n")

    processed_data = white_space_tokenizer(processed_data)   
    processed_data = remove_missing_values(processed_data, column)

    #vector, processed_data = vectorization(processed_data, **kwargs)
    if(verbose > 0):
        print('=============================== END OF THE PREPROCESSING ================================\n')
    return processed_data