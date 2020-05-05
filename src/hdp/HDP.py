#!/usr/bin/env python
# coding: utf-8

#### Code implementation inspired by https://github.com/shuyo
#### Optimized by Eduardo Coronado and Andrew Carr, Duke University

import numpy as np

from scipy.special import gammaln
from hdp.perplexity import get_perplexity

import hdp_funcs as hdp_cpp


###########################################
########### HDP ALGORITHM ##########
##########################################

def run_hdp(docs, voca, gamma, alpha, beta, epochs=1):
    '''Wrapper function to run the HDP inference algorithm'''
    
    #### INITIALIZE PARAMETERS #####
    
    V = voca.shape[0] # length of vocabulary

    D = len(docs) # numb docs
    
    topic_idx = [0] # topic indexes
    
    # table specific indexes and counters
    doc_arrays = [{'t_j': [0],
              'k_jt': np.array([0],dtype=int),
              'n_jt': np.array([0],dtype=int),
              't_ji': np.zeros(len(docs[j]), dtype=int) - 1 } for j in range(D)]

    n_jtw = [[None] for j in range(D)] # word-table specific assignment/counts

    # Topic specific counts
    m_k = np.ones(1, dtype=int) # tables
    n_kv = np.ones((V, 1)) # words-topic count matrix (shape will be V x k)
    
    x_ji = docs # list of sublists
    
    perplex =[]
    
    ##### INFERENCE LOOPS ######
    for z in range(epochs):
        
        ### Infer t
        for j, x_i in enumerate(x_ji):
            doc_j = doc_arrays[j]
            
            for i, w in enumerate(x_i):
                doc_j, topic_idx, n_jtw, n_kv, m_k = sample_t(j, i, w, doc_j, topic_idx, n_jtw, n_kv, m_k, gamma, alpha, beta, V)
                
            doc_arrays[j] = doc_j

            
        ### Infer k
        for j in range(D):
            doc_j = doc_arrays[j]
            
            for tbl in doc_j['t_j']:
                doc_j, topic_idx, n_kv, m_k = sample_k(j, tbl, doc_j, topic_idx, n_jtw, n_kv, m_k, beta, V, gamma)

            doc_arrays[j] = doc_j
        
        
        perplex.append(get_perplexity(x_ji, doc_arrays, topic_idx, n_kv, m_k, beta, alpha, gamma, V))


    return doc_arrays, topic_idx, n_kv, m_k, perplex




###########################################
########### SAMPLING T FUNCTIONS ##########
##########################################
    
def sample_t(j, i, w, doc_j, topic_idx, n_jtw, n_kv, m_k, gamma, alpha, beta, V):
    '''For each word in document j (doc_j), sample for posterior distribution of t and update
       table and topic assignments, as well as other count structures within jt_info, n_jtw, m_k, and n_kv
       Output: updated doc_j, m_k, n_kv, and topic idx (k_idx)'''

    tbl = doc_j['t_ji'][i]

    ### Remove word if assigned to table (i.e. -x_ji) 
    if tbl>0:

        doc_j, n_jtw, n_kv, m_k, topic_idx = remove_xji(j, doc_j, topic_idx, w, tbl, n_jtw, n_kv, m_k)

    #### Sampling t ####
    fk = hdp_cpp.fk_cpp(w, n_kv)
    
    # Posterior pvals
    t_j = doc_j['t_j']
    n_jt = doc_j['n_jt']
    k_jt = doc_j['k_jt']
    post_pvals_t = hdp_cpp.posterior_t_cpp(t_j, n_jt, k_jt, m_k, fk, gamma, V, alpha )
    

    post_t_idx = hdp_cpp.argmax_multinomial_cpp(1, post_pvals_t)

    # Get most likely table selection
    new_t = doc_j['t_j'][post_t_idx]

    # If new table
    if new_t == 0:

        ### Sampling k when t is NEW ###
        post_pvals_kt = posterior_k_new_t(topic_idx, m_k, fk, gamma, V)
        post_pvals_kt /= post_pvals_kt.sum()

        post_kt_idx = hdp_cpp.argmax_multinomial_cpp(1, post_pvals_kt)

        # Select most likely topic for new table
        new_k = topic_idx[post_kt_idx]

        ## New topic selected
        if new_k == 0:
            
            # Create new topic
            new_k, topic_idx, m_k, n_kv = new_topic(topic_idx, m_k, n_kv, beta, V)

        # Add new table
        new_t, doc_j, n_jtw = new_table(j, new_k, doc_j, n_jtw)

        m_k[new_k] += 1 #add to table cnt for topic k

    # Assign word to table
    doc_j, n_kv, n_jtw = assign_to_table(j, i, w, new_t, doc_j, n_kv, n_jtw)
            
    return doc_j, topic_idx, n_jtw, n_kv, m_k   


def remove_xji(j, doc_j, topic_idx, w, tbl, n_jtw, n_kv, m_k):
    '''Remove word if assigned to table (i.e. -x_ji), calls on remove_table helper function
       Inputs: table idx, topic for table t, word
       Outputs: updated n_kv, plus additional 
                 updates on doc_j, m_k (tables in topic k), and k_idx (topics) from remove_table fcn '''
    
    topic = doc_j['k_jt'][tbl]
    
    # decrease counts
    doc_j['n_jt'][tbl] -=1
    n_jtw[j][tbl][w] -=1
    n_kv[w, topic] -= 1
    
    # Empty table, remove it
    if doc_j['n_jt'][tbl] == 0:
        
        doc_j, m_k, topic_idx = remove_table(doc_j, tbl, topic_idx, m_k)
        
    return doc_j, n_jtw, n_kv, m_k, topic_idx


def remove_table(doc_j, tbl, topic_idx, m_k):
    '''Empty tables (i.e. n_jt == 0) are removed
       Inputs: table idx, doc_j and m_k (tables in topic k)
       Outputs: Updated doc_j, m_k, k_idx '''
    
    # Delete table 
    doc_j['t_j'].remove(tbl)
    topic = doc_j['k_jt'][tbl]
    m_k[topic] -=1 # decreate table topic count
    
    #if no more tables with topic k, remove topic
    if m_k[topic] == 0:
        topic_idx.remove(topic)
        
    return doc_j, m_k, topic_idx



def posterior_k_new_t(topic_idx, m_k, fk, gamma, V):
    '''If new table selected, generate posterior pvals for selecting a new or existing topic'''
    post_k = (m_k*fk)[topic_idx] #existing topic
    post_k[0] = gamma / V # new topic
    
    return post_k


def new_topic(topic_idx, m_k, n_kv, beta, V):
    '''If new topic selected, get new topic k and extend structures k_idx (topic idx), n_kv (word-topic matrix), 
       m_k (tables per topic) for later updates. 
       Output: new topic and extended structures'''
    
    # Search through topic indexes, find correct extension to array given topic might have been removed
    for k_idx, k in enumerate(topic_idx):
        if k_idx != k:
            break
    else:
        k_idx = len(topic_idx)
        if k_idx >= n_kv.shape[1]:
            m_k = np.resize(m_k, k_idx + 1)
            n_kv = np.c_[n_kv, np.ones((V,1), dtype=int) * beta]

    # Add new topic index, set new topic table count to zero and add beta values to new n_kv column
    topic_idx.insert(k_idx, k_idx)
    m_k[k_idx] = 0
    n_kv[:, k_idx] = np.ones(V, dtype=int) * beta
    
    return k_idx, topic_idx, m_k, n_kv


def new_table(j, new_k, doc_j, n_jtw):
    '''If new table selected, get new table idx and extend structures doc_j jt_info and n_jtw for 
       later updates
       Output: new table and extended structures'''
    
    # Search through table indexes, find correct extension to array given tables might have been removed
    for t_idx, t in enumerate(doc_j['t_j']):
        if t_idx != t:
            break
    else:
        t_idx = len(doc_j['t_j'])
        doc_j['n_jt'] = np.resize(doc_j['n_jt'], t_idx + 1)
        doc_j['k_jt'] = np.resize(doc_j['k_jt'], t_idx + 1)
        n_jtw[j].append(None)
            
    # Add new table index, set count to zero and initialize word-specific count dictionary
    doc_j['t_j'].insert(t_idx, t_idx)
    doc_j['n_jt'][t_idx] = 0
    n_jtw[j][t_idx] = {}
    
    doc_j['k_jt'][t_idx] = new_k # set new table topic to (new or old) topic
        
    return t_idx, doc_j, n_jtw


def assign_to_table(j, i, word, new_t, doc_j, n_kv, n_jtw):
    '''Assign word to table new_t with topic new_k in doc_j, add counts to overall table count,  
       word-topic matrix and discretized table word counts
       Outputs: updated doc_j and n_kv'''
    
    # Get word to table (either new or old) and add to table count
    doc_j['t_ji'][i] = new_t
    doc_j['n_jt'][new_t] +=1
    
    # Get topic table
    new_k = doc_j['k_jt'][new_t]
    
    # Add to table word-specific counts and topic-word specific counts
    n_kv[word, new_k] += 1
    
    n_jtw[j][new_t][word] = n_jtw[j][new_t].get(word, 0) + 1
    
    return doc_j, n_kv, n_jtw




###########################################
########### SAMPLING K FUNCTIONS ##########
##########################################

def sample_k(j, tbl,  doc_j, topic_idx, n_jtw, n_kv, m_k, beta, V, gamma):
    '''For each TABLE in document j (doc_j), sample for posterior distribution of k and update
       table and topic assignments, as well as other count structures within jt_info, n_jtw, m_k, and n_kv
       Output: updated doc_j, m_k, n_kv, and topic idx (k_idx)'''
    
    #### START of Sampling k loop through tables, (skip first index always, 0 = dummy idx) ####
    if tbl != 0:

        # Get topic k, remove all components from table t associated with topic k
        doc_j, topic_idx, m_k = remove_Xvec_ji(tbl, doc_j, topic_idx, m_k)

        # Samples posterior p-vals K
        post_pvals_k = hdp_cpp.posterior_k_cpp(tbl, doc_j['k_jt'], doc_j['n_jt'], topic_idx, n_kv, 
                                                m_k,  n_jtw[j][tbl], gamma, V, beta)

        # Select most likely topic for table
        post_k_idx = hdp_cpp.argmax_multinomial_cpp(1, post_pvals_k)
        
        new_k = topic_idx[post_k_idx]

        ## New topic selected
        if new_k == 0:

            # Create new topic
            new_k, topic_idx, m_k, n_kv = new_topic(topic_idx, m_k, n_kv, beta, V)

        # Add table to topic k count
        m_k[new_k] += 1

        # Rearrange individual word-topic counts based on potential new_k reassignment
        doc_j, n_kv = rearranging_k_counts(j,tbl, new_k, doc_j, n_jtw, n_kv)

    return doc_j, topic_idx, n_kv, m_k


def posterior_k(j, doc_j, tbl, topic_idx, n_jtw, n_kv, m_k, V, beta):
    '''Compute explicit posterior multinomial-dirichlet posterior distribution'''
    
    # Topic k of table t
    k = doc_j['k_jt'][tbl]
    n_jt = doc_j['n_jt'][tbl]
    
    # Remove all counts associated with topic k in table t, from overall topic counts (n_k)
    n_kv = n_kv.copy()
    n_k = n_kv.sum(axis = 0)
    n_k[k] -= n_jt
    n_k = n_k[topic_idx]
    
    # Initialized k posterior in log-form for simplicity, this computes f_k^{-X_ji} 
    # has Dirichlet-Multinomial form
    log_post_k = np.log(m_k[topic_idx]) + gammaln(n_k) - gammaln(n_k + n_jt)
    log_post_k_new = np.log(gamma) + gammaln(V*beta) - gammaln((V*beta) + n_jt)

    
    # Remove individual word counts associated with topic k
    # add their contributions to k posterior
    for w_key, w_cnt in n_jtw[j][tbl].items():

        if w_cnt == 0: # if word count is 0 skip
            continue

        # For word w, get counts across topics
        w_cnt_k = n_kv[w_key, :]
        
        # For specific topic k, remove count from associated table t
        w_cnt_k[k] -= w_cnt
        w_cnt_k = w_cnt_k[topic_idx]
        w_cnt_k[0] = 1

        # Add contributions of individual observations (words)
        log_post_k += gammaln(w_cnt_k  + w_cnt) - gammaln(w_cnt_k)
        log_post_k_new += gammaln(beta + w_cnt) - gammaln(beta)
    
    # p-val for new k
    log_post_k[0] = log_post_k_new

    # Bring back to non-log realm, normalize k-posterior 
    post_k = np.exp(log_post_k - log_post_k.max())

    return post_k


def remove_Xvec_ji(tbl, doc_j, topic_idx, m_k):
    '''Remove table from topic k (i.e. related removing all components associated to table t later)
       If table becomes empty, remove topic'''
    
    # Get topic k, remove all components from table t associated with topic k
    k_idx = doc_j['k_jt'][tbl]
 
    m_k[k_idx] -=1 # remove from table-topic vector
    
    # if no more tables with topic k, remove topic k and set table's topic to 0
    if m_k[k_idx] == 0:
        topic_idx.remove(k_idx)
        doc_j['k_jt'][tbl] = 0
        
    return doc_j, topic_idx, m_k


def rearranging_k_counts(j, tbl, new_k, doc_j, n_jtw, n_kv):
    '''For sampled k, rearrange counts for topics accordingly (i.e. if a new k was selected, subtract
       from previous k and add to new k in word-topic matrix)'''
    
    k = doc_j['k_jt'][tbl] 
    
    # If new topic for table t is selected, set topic to new topic
    if k != new_k:
        doc_j['k_jt'][tbl] = new_k
        
        # On word-topic matrix, move counts from old topic to new topic
        for w_key, cnt in n_jtw[j][tbl].items():
            if k != 0: 
                n_kv[w_key, k] -= cnt
    
            n_kv[w_key, new_k] += cnt
            
    return doc_j, n_kv










