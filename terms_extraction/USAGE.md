# Meta-AutoNLP Prototype

**Using the _Terms extraction_**

---

## How use _Terms extraction_ :

[Go to Example Notebook](../Example%20notebooks/Terms%20extractor%20example.ipynb)

```python
import sys
sys.path.append('../terms_extraction/')

from terms_extractor import terms_extractor

extract, processed_df = terms_extractor(data=df, measures=['yake', 'tfidf', 'pmi', 'mle', 'dice', 'count'], 
                                        target_column='column_text', verbose=1)
display(processed_df)
```

where, for the **terms_extractor**: 

* **data**: <pandas.DataFrame> Your DataFrame.
* **measures**: <list> list with all the techniques you want to use.
* **target_column**: <str> Name of the column with the texts that must be extracted the terms.
* **\*\*kwargs**: Additional hyperparameters.

The techniques that can be used in the "measures" hyperparameter, as well as the additional hyperparameters that can be added, will be explained below.

## Additional hyperparameters

* **extract**: <list, default=None> If you already have a list of terms that you want to extract, simply add the list here. 
    * Example: extract = ['notebook', 'mouse', 'teclado', 'processador']
* **result_column**: <str, default='your list'> If you use your list through the "extract" hyperparameter, the resulting column in the output DataFrame will have the name assigned here.
    * Example: result_column = 'my_column' 
* **number_of_tokens**: <int, default=1> The extracted terms will have up to the assigned number of tokens.
    * Example: number_of_tokens = 2 
* **number_to_extract**: <int, default=1000> Number of terms that will be extracted.
    * Example: number_to_extract = 800
* **threshold_percentage**: <float, default=1> Number of terms that will be extracted based in a percentage threshold. For instance, if your text have 1000 words and your threshold is 0.7, will be extracted 300 terms.
    * Example: threshold_percentage = 0.9
* **stopwords**: <bool, default=False> If _True_, the list will be obtained by the NLTK library through the assigned language; else, stopwords will not be removed.
    * Example: stopwords = True 
* **language**: <str, default='english'> Add the language you want to use for stopwords removal, the list will be obtained by the NLTK library.
    * Example: language ='portuguese' 
* **perform_term_extraction**: <bool, default=True> If _True_, the terms will be obtained (list with terms) and extracted from the text (result in DataFrame). Else, only the terms (list) will be obtained.
    * Example: perform_term_extraction = False
* **verbose**: <int, default=0> If the number is greater than 0, it will print extra info about the processing.
    * Example: verbose = 1

## Available techniques (measures):

### Yet Another Keyword Extractor (Yake)

This is a method that extracts keywords or key phrases from text documents by preprocessing and identifying candidates through techniques like term and document frequency, co-occurrences, position and length; these features are used to score and rank the candidates. Github project can be accessed [here](https://github.com/LIAAD/yake).

### Term Frequency–Inverse Document Frequency (TF-IDF)

The TF-IDF identifies keywords that are both frequently occurring within a specific document and relatively rare across the entire corpus; in this way, this module assigns a numerical weight to each word and perform a ranking based on these scores.

### Pointwise Mutual Information (PMI)

This is a statistical measure used to determine the association or co-occurrence between two events (or words) within a text corpus. A high PMI score indicates a strong association, implying that the occurrence of one word makes the presence of the other more likely, while a low or negative PMI suggests independence or repulsion.

### Maximum Likelihood Estimator (MLE)

This is a technique to estimate the probability distribution of words (or phrases) in a corpus, assuming that the probability of a word occurring in a specific context is proportional to the number of times that word appears within the training corpus.

### Sørensen–Dice coefficient (Dice)

This is a measure used to quantify the degree of overlap or similarity between two sets. The score ranges from 0 to 1, with higher values indicating greater similarity.

### _N_-grams count

These are contiguous sequences of _n_ items, which can be characters, words, or even more extended units like phrases; in this context, it is checked how many times each _n_-gram appears in the text.