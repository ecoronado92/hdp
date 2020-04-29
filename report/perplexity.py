import numpy as np
from scipy.special import gammaln
import data_preproc
from data_preproc import data_preproc
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
    return [[vocab[i] for i in j] for j in docs]

# Takes doc_topic distibution (list of tuples of topic_idx and proportion) and returns topic-length array with props in topic_idx and zeros elsewhere
def special_array(doc, num_topics):
    
    topic_idx = [i[0] for i in doc]
    props = [i[1] for i in doc] 
    arr = np.zeros(num_topics)
    
    arr[topic_idx] = props
    
    return(arr)

def model_to_dist(model, common_corpus, common_dictionary):
    '''Takes Gensim LDA Model and common corpus and dictionary objects and returns doc-topic distribution and word-topic distribution'''
    
    doc_topic_dist = [lda.get_document_topics(item) for item in common_corpus]
    doc_topic_dist = [special_array(i, topic_num) for i in doc_topic_dist]
    doc_topic_dist = np.vstack(doc_topic_dist)
    
    word_topic_dist = [lda.get_term_topics(i, minimum_probability = 1e-4) for i in range(len(common_dictionary))]
    word_topic_dist = [special_array(word, topic_num) for word in word_topic_dist]
    word_topic_dist = np.vstack(word_topic_dist)
    
    return doc_topic_dist, word_topic_dist


def perplex_func(doc_topic_dist, word_topic_dist, corpus_key):

    LL = 0
    N = 0
    
    word_prob_lst = []
    
    for doc_dist, word_idx in zip(doc_topic_dist, corpus_key):
    
        N += len(word_idx)
        
        word_prob_lst.append(word_topic_dist[word_idx] @ doc_dist)
        
        word_probs = np.hstack(word_prob_lst)
        
    return np.exp(-np.sum(np.log(word_probs[word_probs!=0]))/N)