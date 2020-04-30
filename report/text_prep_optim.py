#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import cppimport

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer as stemmer
from nltk.corpus import stopwords
import nltk.stem

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
stemmer = stemmer("english")

## Import C++ functions
preproc_cpp = cppimport.imp("hdp_preproc")


#### MAIN FUNCTIONS ####

    
def run_preprocess_optim(file_path):
    '''Data pre-processing function
       Input -> url to data in CSV format where each row is a document text'''
    
    df = pd.read_csv(file_path)
        
    st_words = set(stopwords.words('english'))
    
    in_docs = [full_preprocess(d, st_words) for d in df.values]
    
    vocab, filtered_df = tf(in_docs)
            
    docs = get_docs(filtered_df)
    
    return vocab, docs


def preprocess(doc):
    '''Function that lemmatizes words in abstract by verbs'''
    
    return [stemmer.stem(WordNetLemmatizer().lemmatize(w, pos='v')) 
            for w in preproc_cpp.text_cleanup(doc[0])]

def full_preprocess(doc, st_words):
    '''Performs word lemmatization and stopword removal'''
    return preproc_cpp.rm_stops_n_shorts(preprocess(doc), st_words, 3)


def tf(in_docs):
    '''Term frequency matrix function, calculates the term frequencies of word from an 
       a list of documents text. Then filtered according to frequency criteria to keep shared 
       yet low occurence words.
       The output is the filtered term frequency table and associated vocabulary'''
    
    v = preproc_cpp.generate_vocab(in_docs) # generates vocab
    tf = preproc_cpp.tf_cpp(in_docs, v) # generates tf matrix
    
    filt_df = preproc_cpp.filter_tf_cpp(tf) # filters tf matrix (last column is vocab indexes)
    
    # Filter vocab indexes
    v_idx = filt_df[:, filt_df.shape[1]-1].astype(int) # get vocab indexes
    vocab = np.array(v)[v_idx]
    
    filt_df = np.delete(filt_df, filt_df.shape[1]-1, axis=1)
    
    return vocab, filt_df



def get_docs(df):
    '''Get list of sublists (len = documents), with each sublist containing unique word ids per document'''
    
    return preproc_cpp.get_docs(df)




