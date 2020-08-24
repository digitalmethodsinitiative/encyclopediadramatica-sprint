
import collections
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re


# get stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))


# initial cleaning

def clean_tweets(txt):
    '''
    remove URLs and mentions (@person) from tweet txt string
    '''
    no_links = re.sub(r'http\S+', '', txt)
    no_links_mentions = re.sub("@[A-Za-z0-9]+", "", no_links)

    return no_links_mentions


# top ngrams
def sum_counts(old, old_counts, new, new_counts):
    '''
    adds two lists of counts
    '''
    old_counts = collections.Counter(dict(zip(old, old_counts)))
    new_counts = collections.Counter(dict(zip(new, new_counts)))
    result = old_counts + new_counts
    result = result.most_common(20)
    a, b = zip(*result)  # .items())
    return list(a), list(b)


def get_top_n_words(corpus, n_gram_range,  n=None):
    '''
    return top n_grams, n_gram_range = (min_ngram, max_ngram), e.g. (2,2) for only bigrams
    '''
    vec1 = CountVectorizer(stop_words=stop_words, ngram_range=n_gram_range,
                           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]


def calculate_ngrams(corpus, n_gram_range):
    top_n = get_top_n_words(corpus, n_gram_range, n=20)
    top_n_df = pd.DataFrame(top_n)
    top_n_df.columns = ["Ngram", "Freq"]
    top_n_df.set_index('Ngram', inplace=True)
    return top_n_df




# language innovation / neologisms


def load_reference_corpus():
    '''
    google one million books 2008 words with + 80 occurrences
    '''
    reference = set(line.strip() for line in open('data_wordlist50.txt'))
    return reference


def get_neologisms(string, reference):
    """
        matches the words in the data to the words in the reference dictionary
        and returns a dictionary of words that were not in the reference dictonary
        along with the frequencies
        input: reference corpus (set)
               string data
        output: dictionary with word frequency
    """

    candidates = dict()
    string = re.sub(r'\d+', '', string)  # remove numbers
    string = re.sub('[^A-Za-z0-9]+', ' ', string)  # remove punctuation

    word_tokens = word_tokenize(string)
    word_tokens_nostops = [w for w in word_tokens if not w in stop_words]


    for w in word_tokens_nostops:
        if w.lower() not in reference:
            if w in candidates:
                candidates[w] = candidates[w] + 1
            else:
                candidates[w] = 1

    return  candidates



def get_doc_size(string):
    """
    get tokens of document
"""

   
    string = re.sub(r'\d+', '', string)  # remove numbers
    string = re.sub('[^A-Za-z0-9]+', ' ', string)  # remove punctuation

    word_tokens = word_tokenize(string)
    word_tokens_nostops = [w for w in word_tokens if not w in stop_words]
    doc_size = len(word_tokens_nostops)

    return doc_size