# Meta-AutoNLP Prototype

**Using the _Text data preprecessing_**

---

## How use _Data preprecessing_ :

[Go to Example Notebook](../Example%20notebooks/Preprocessing%20example.ipynb)

```python
import sys
sys.path.append('../preprocessing/')
from NLPipeline import *

# Define your arguments
preprocess_functions = ['transform_to_lowercase', 'remove_special_characters', 'remove_stopwords']
vectorizer = 'word2vec'
column = 'review_text'

# Create the pipeline
pipeline = NLP_helper(preprocessing_funcs=preprocess_functions, 
                      vectorizer=vectorizer, 
                      column=column)

pipeline = pipeline.fit(df)
```

or

```python
import sys
sys.path.append('../preprocessing/')
from preprocess import *

# Define your arguments
preprocessing_functions = ['transform_to_lowercase', 'remove_special_characters', 'remove_stopwords']

# Make the preprocessing
processed_df = preprocess_data(data=df, 
                               preprocessing_funcs=preprocess_functions, 
                               column='column_text', 
                               verbose=1)

# Make the vectorization
vector, vectorized_df = vectorization(data=processed_df, 
                                      verbose=1, 
                                      vectorizer='tf-idf')
```

where, for the **preprocess_data**: 

* **data**: <pandas.DataFrame> Your DataFrame.
* **preprocess_funcs**: <list> list with all the functions you want to use.
* **column**: <str> Name of the column with the texts that must be preprocessed.
* **verbose**: <int, default=0> If the number is greater than 0, it will print extra info about the processing.
* **\*\*kwargs**: Additional hyperparameters.

and for the **vectorization**:

* **data**: <pandas.DataFrame> Your preprocessed DataFrame;
* **vectorizer**: <str, default='tf-idf'> Name of the vectorization technique you want to use. The valid options are: 'tf-idf', 'bow', or 'word2vec'.

The functions that can be added to the "preprocess\_funcs" hyperparameter, as well as the additional hyperparameters that can be added, will be explained below.

## Available functions:

### transform_to_lowercase

Used to transform all text into lowercase.

### remove_special_characters

Used to remove all special characters (accents and symbols) from the text.

### remove_stopwords

Used to remove all stopwords from the text.
Additional hyperparameters can be used, namely:

* **stopwords**: <list, default=None> If you already have a list of stopwords, simply add the list here.
    * Example: stopwords = ['e', 'ou', 'que', 'qual']
* **language**: <str, default='english'> If there is no list in the "stopwords" hyperparameter, the list will be obtained by the NLTK library through the assigned language.
    * Example: language = 'portuguese' 
* **list_del_stopwords** <list, default=None> Add words you want to remove from the original stopword list.
    * Example: list_del_stopwords = ['e', 'ou']
* **list_add_stopwords**: <list, default=None> Add words you want to include in the original stopword list. 
    * Example: list_add_stopwords = ['ao', 'sem'] 

### remove_specific_phrases

Removes specific phrases (with 2 or more n-grams) from the text.
Additional hyperparameters can be used, namely:

* **phrases**: <list, default=None> List with all the phrases you want to remove.
    * Example: phrases = ['bom dia', 'boa tarde', 'boa noite']

### perform_lemmatization

Perform lemmatization using spaCy.
Additional hyperparameters can be used, namely:

* **language**: <str, default='english'> Add the language you want to use for lemmatization.
    * Example: language = 'portuguese'

### perform_stemming

Perform stemming using NLTK SnowballStemmer.
Additional hyperparameters can be used, namely:

* **language**: <str, default='english'> Add the language you want to use for stemming.
    * Example: language = 'portuguese'
