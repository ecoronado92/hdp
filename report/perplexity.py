import numpy as np
import pandas as pd
from scipy.special import gammaln
from text_prep import run_preprocess
import string
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import gensim
from gensim.models import LdaModel
from gensim.test.utils import common_corpus
import matplotlib.pyplot as plt
from gensim.models import HdpModel
import random


##### FUNCTIONS ######

# Function to reconvert docs back into words (the format the gensim likes)
def back_to_words(docs, vocab):
    '''Function to reconvert docs back into words (the format the gensim likes)'''
    return [[vocab[i] for i in j] for j in docs]

# Takes doc_topic distibution (list of tuples of topic_idx and proportion) and returns topic-length array with props in topic_idx and zeros elsewhere
def special_array(doc, num_topics):
    '''Takes doc_topic distibution (list of tuples of topic_idx and proportion) and 
    returns topic-length array with props in topic_idx and zeros elsewhere'''
    topic_idx = [i[0] for i in doc]
    props = [i[1] for i in doc] 
    arr = np.zeros(num_topics)
    
    arr[topic_idx] = props
    
    return(arr)

def model_to_dist(model, common_corpus, common_dictionary, topic_num):
    '''Takes Gensim LDA Model and common corpus and dictionary objects and returns doc-topic distribution and word-topic distribution'''
    
    doc_topic_dist = [model.get_document_topics(item) for item in common_corpus]
    doc_topic_dist = [special_array(i, topic_num) for i in doc_topic_dist]
    doc_topic_dist = np.vstack(doc_topic_dist)
    
    word_topic_dist = [model.get_term_topics(i, minimum_probability = 1e-4) for i in range(len(common_dictionary))]
    word_topic_dist = [special_array(word, topic_num) for word in word_topic_dist]
    word_topic_dist = np.vstack(word_topic_dist)
    
    return doc_topic_dist, word_topic_dist


def perplex_func(doc_topic_dist, word_topic_dist, corpus_key):
    '''Computes the perplexity score'''
    LL = 0
    N = 0
    
    word_prob_lst = []
    
    for doc_dist, word_idx in zip(doc_topic_dist, corpus_key):
    
        N += len(word_idx)
        
        word_prob_lst.append(word_topic_dist[word_idx] @ doc_dist)
        
        word_probs = np.hstack(word_prob_lst)
        
    return np.exp(-np.sum(np.log(word_probs[word_probs!=0]))/N)


def doc_arrays_to_doc_topic_distribution(doc_arrays, topic_idx):
    '''Takes doc_arrays and topic_idx from hdp algorithm and returns a document-topic distribution'''
    
    def spec_array(test_arr, topic_id):
        '''Helper function for doc_arrays_to_doc_topic_distribution'''
        thing = np.zeros(max(topic_id))
        arr_idx = np.array([i-1 for i in test_arr[1:,0] if i < (topic_id[-1]+1)])
        contents = test_arr[:,1][np.where(test_arr[:,1][test_arr[:,0] < (topic_id[-1]+1)] > 0)[0]]
        if len(arr_idx) > len(contents):
            arr_idx = arr_idx[:len(contents)]
        thing[arr_idx] = contents
        return thing/np.sum(thing)
     
    # Pulling relevant data from model output 
    k_jt_fin = [i['k_jt'] for i in doc_arrays]
    n_jt_fin = [i['n_jt'] for i in doc_arrays]
    
    # Converting model output to word-topic and document-topic distributions
    doc_topic_key = [np.column_stack([pd.factorize(k_jt)[1], 
                                      np.bincount(pd.factorize(k_jt)[0], n_jt).astype(n_jt.dtype)]) for k_jt, n_jt in zip(k_jt_fin, n_jt_fin)]
    
    doc_dist = [spec_array(item, topic_idx) for item in doc_topic_key]
    doc_dist = np.vstack(doc_dist)
    doc_dist = doc_dist[:,[i-1 for i in topic_idx[1:]]]
    
    return doc_dist


def n_kv_to_word_dist(n_kv, topic_idx):
    '''Takes n_kv from hdp algorithm and returns a word-topic distribution'''
    word_dist = [idx for idx in (n_kv[:,topic_idx[1:]] - .5)/np.sum(n_kv[:,topic_idx[1:]]-.5)]
    return np.vstack(word_dist)