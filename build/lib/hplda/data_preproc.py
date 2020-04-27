#!/usr/bin/env python
# coding: utf-8




import string 
import nltk
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer as stemmer
from nltk.corpus import stopwords
import nltk.stem

from collections import Counter

nltk.download('wordnet')
nltk.download('stopwords')
stemmer = stemmer("english")



def preprocess(doc):
    '''Function that lemmatizes words in abstract by verbs'''
    
    return [stemmer.stem(WordNetLemmatizer().lemmatize(w, pos='v')) 
            for w in doc.translate(str.maketrans('','', string.punctuation)).lower().split(' ')]


def rm_stopwords_and_short_words(words, st_words):
    '''Function removes stop words and those with length < 3'''
    results = []
    for i in words:
        if not i in st_words and len(i)  > 3:
            results.append(i)
    return results

def full_preprocess(doc, st_words):
    '''Performs word lemmatization and stopword removal'''
    return rm_stopwords_and_short_words(preprocess(doc), st_words)


def tf(docs, st_words):
    '''Term frequency matrix function, calculates the term frequencies of word from an text-document paired dictionary input. 
       The output is a term frequency table '''
    
    # generate counts per document
    counts = {k: Counter(full_preprocess(txt, st_words)) for k, txt in docs.items()}
    tf_df = pd.DataFrame.from_dict(counts).fillna(0).astype(int) # build pandas df, fill empty vals with 0s
    
    return(tf_df)


def token_filtering(tf_df):
    '''Filters out tokens that appear in fewer than 3 abstracts and tokens that appear in more than half the abstracts '''
    filtered_df = tf_df[(tf_df.sum(axis=1) > 3)]
    filtered_df = filtered_df[(filtered_df.astype(bool).sum(axis=1) / tf_df.shape[1] < 0.5)]
    
    return filtered_df
    
def get_docs(df):
    '''quickly build a dictionary based on filtered dataframe, get words w/ unique ids'''
    df.reset_index(inplace=True)
    filt_words = pd.DataFrame.to_dict(df.drop(columns='index'))
    
    return [[word for word, cnt in words.items() if cnt!=0] for dkeys, words in filt_words.items()]
    
    
def data_preproc(file_path):
    '''Data pre-processing function
       Input -> url to data in CSV format where each row is a document text'''
    
    df = pd.read_csv(file_path)
    
    in_docs = {k: str(txt[0]) for k,txt in enumerate(df.values)}
    
    st_words = stopwords.words('english')
    
    tf_df = tf(in_docs, st_words)
    
    filtered_df = token_filtering(tf_df)
    
    vocab = filtered_df.index.values
        
    docs = get_docs(filtered_df)
    
    return [vocab, docs]



