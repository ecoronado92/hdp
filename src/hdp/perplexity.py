#!/usr/bin/env python
# coding: utf-8

#### Code implementation inspired by https://github.com/shuyo
#### Optimized by Eduardo Coronado and Andrew Carr, Duke University

import numpy as np



def get_perplexity(x_ji, doc_arrays, topic_idx, n_kv, m_k, beta, alpha, gamma, V):
    '''Computes the models perplexity given inferences at epoch i, in other words provides a metric
        related to the likelihood of each work pertaining to a topic'''
    phi = word_topic_dist(n_kv, topic_idx, beta, V)
    theta = doc_topic_dist(doc_arrays, topic_idx, m_k, alpha, gamma)
    
    log_lik = 0
    N = 0
    
    for x_i, p_jk in zip(x_ji, theta):
        for word in x_i:
            p_w = sum(p_k *p_kv for p_k, p_kv in zip(p_jk, phi[word,:]))
            log_lik -= np.log(p_w)
        N += len(x_i)
    
    perplex = np.exp(log_lik/N)
    
    return perplex
            
        

def doc_topic_dist(doc_arrays, topic_idx, m_k, alpha, gamma):
    '''Computes effects of each doc and topic into a probability distribution'''
    topic_tbl_effect = np.array(m_k, dtype=float)
    topic_tbl_effect[0] = gamma
    topic_tbl_effect *= alpha / topic_tbl_effect[topic_idx].sum()

    theta = []

    for j, doc_j in enumerate(doc_arrays):
        n_jt = doc_j['n_jt']
        p_jk = topic_tbl_effect.copy()
        
        for tbl in doc_j['t_j']:
            
            if tbl == 0: continue
            k = doc_j['k_jt'][tbl]
            p_jk[k] += n_jt[tbl]
            
        p_jk = p_jk[topic_idx]
        p_jk /= p_jk.sum()
        theta.append(p_jk)
    
    return np.array(theta)

def word_topic_dist(n_kv, topic_idx, beta, V):
    '''Compute word probability distribution per topic'''
    phi = n_kv/n_kv.sum(axis=0)[None, :]
    phi = phi[:, topic_idx]
    phi[:,0] = beta / V
    
    return phi