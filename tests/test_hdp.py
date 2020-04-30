
import hdp.HDP as HDP
import hdp.text_prep as txtp
import numpy as np
import unittest
import copy

from scipy.special import gammaln


class TestHDP(unittest.TestCase):
    
    def setUp(self):
        self.j = 0
        self.gamma = 0.866
        self.alpha = 1.715
        self.beta = 0.5
        self.V = 5
        self.doc_j ={'t_j':         [0, 1, 2, 3],
                    'k_jt':np.array([0, 1, 1, 2], dtype=int),
                    'n_jt':np.array([0, 1, 2, 2], dtype=int),
                    't_ji':np.array([1, 2, 3, 3, 2], dtype=int)}
        self.njtw = [[None, {0:1}, {1:1, 4:1}, {2:1, 3:1}]]
        self.topics = [0, 1, 2]
        self.m_k = np.array([0, 2, 1])
        self.x_i = [0, 1, 2 ,3, 4]
        self.n_kv = np.array([
            [0.5, 1.5, 0.5],
            [0.5, 1.5, 0.5],
            [0.5, 0.5, 1.5],
            [0.5, 0.5, 1.5],
            [0.5, 1.5, 0.5]
        ])

        
    ######### SAMPLING T TESTS #############  
    def test_rm_xji(self):
        '''Test x_ji gets removed on all structures'''
        j, word, tbl, topic  = 0, 0, 1, 1
        doc_j = copy.deepcopy(self.doc_j)
        njtw = copy.deepcopy(self.njtw)
        n_kv = copy.deepcopy(self.n_kv)
        m_k = copy.deepcopy(self.m_k)
        d, n_dict, n_kv2, m_k2, topic_idx2 = HDP.remove_xji(j, doc_j, self.topics, word,
                                                            tbl, njtw, n_kv, m_k)
        
        self.assertEqual(d['n_jt'][tbl], self.doc_j['n_jt'][tbl] -1)
        self.assertEqual(n_dict[j][tbl][word], self.njtw[j][tbl][word] -1)
        self.assertEqual(n_kv2[word, topic ], self.n_kv[word, topic] -1)
        
    
    
    def test_remove_tbl(self):
        '''Remove empty tables and topics'''
        doc_j = copy.deepcopy(self.doc_j)
        topic_idx = copy.deepcopy(self.topics)
        m_k = self.m_k.copy()
        tbl = 3
        k = doc_j['k_jt'][tbl]
        
        d, m_k2, topic_idx2 = HDP.remove_table(doc_j, tbl, topic_idx, m_k)
        
        self.assertNotIn(tbl, d['t_j'])
        self.assertEqual(m_k2[k], self.m_k[k] -1)
        self.assertNotIn(topic_idx2, self.topics) # being removed when no more topics k

        
    def test_fk(self):
        '''check fk distribution doesn't have negative vals'''
        w = 2
        n_kv = self.n_kv.copy()
        
        fk = HDP.hdp_cpp.fk_cpp(w, n_kv)
        
        self.assertEqual(len(fk), self.n_kv.shape[1])
        self.assertEqual(fk[0], 0)
        for i in fk[1:]:
            self.assertGreater(i, 0)
            
            
    def test_posterior_t(self):
        '''pvals of sampling t, positive or zero'''
        t_j = self.doc_j['t_j']
        n_jt = self.doc_j['n_jt']
        k_jt = self.doc_j['k_jt']
        w =2 
        fk = HDP.hdp_cpp.fk_cpp(w, self.n_kv)
        pvals = HDP.hdp_cpp.posterior_t_cpp(t_j, n_jt, k_jt, self.m_k, fk, self.gamma, self.V, self.alpha)
        
        self.assertAlmostEqual(np.sum(pvals), 1)
        self.assertFalse(any(pvals < 0))
        self.assertTrue(len(pvals), len(self.doc_j['k_jt']))
        
    
    def test_multinomial_sample(self):
        '''cpp multinomial gives argmax idx'''
        n = 100
        pvals = np.array([0.3, 0.1, 0.1, 0.05, 0.4, 0.05])
        idx = HDP.hdp_cpp.argmax_multinomial_cpp(n, pvals)
        
        self.assertIn(idx, [0,4])
        
        
    def test_newt_topic(self):
        '''new t + topic k sampling'''
        w = 2
        fk = HDP.hdp_cpp.fk_cpp(w, self.n_kv)
        
        pval = HDP.posterior_k_new_t(self.topics, self.m_k, fk, self.gamma, self.V)
        
        self.assertAlmostEqual(np.sum(pval/pval.sum()), 1)
        self.assertFalse(any(pval < 0))
        
        
    
    def test_new_topic(self):
        '''new topic is added'''
        topic_idx = self.topics.copy()
        m_k = self.m_k.copy()
        n_kv = self.n_kv.copy()
        
        k_idx, topic_idx, m_k, n_kv = HDP.new_topic(topic_idx, m_k, n_kv, self.beta, self.V)
        
        self.assertEqual(len(topic_idx), len(self.topics)+1)
        self.assertIn(k_idx, topic_idx)
        self.assertEqual(m_k[k_idx], 0)
        self.assertEqual(n_kv[:,k_idx].sum(), 2.5)
        self.assertGreater(n_kv.shape[1], self.n_kv.shape[1])
    
    
    def test_new_table(self):
        '''new table is added'''
        doc_j = copy.deepcopy(self.doc_j)
        k = 2
        njtw = copy.deepcopy(self.njtw)
        
        t_idx, doc_j, njtw = HDP.new_table(self.j, k, doc_j, njtw)
        
        self.assertEqual(len(doc_j['n_jt']), len(self.doc_j['n_jt'])+ 1)
        self.assertEqual(len(doc_j['k_jt']), len(self.doc_j['k_jt']) + 1)
        self.assertEqual(len(njtw[self.j]), len(self.njtw[self.j])+ 1)
        self.assertIn(t_idx, doc_j['t_j'])
        self.assertEqual(doc_j['k_jt'][t_idx], k)
        
        
    def test_assign_table(self):
        '''word is assigned to table'''
        doc_j = copy.deepcopy(self.doc_j)
        njtw = copy.deepcopy(self.njtw)
        n_kv = self.n_kv.copy()
        i, w = 4, 4
        new_t = 1
        
        doc_j, n_kv, n_jtw = HDP.assign_to_table(self.j, i, w, new_t, 
                                                 doc_j, n_kv, njtw)
        
        self.assertEqual(doc_j['t_ji'][i], new_t)
        self.assertEqual(doc_j['n_jt'][new_t], self.doc_j['n_jt'][new_t] + 1)
        self.assertEqual(n_kv[w, 1], 2.5)
        self.assertEqual(njtw[self.j][new_t][w], 1)
        
        
    ######### SAMPLING K TESTS #############  
        
    def test_rm_Xvec(self):
        '''rm Xvec conditional distribution'''
        m_k = self.m_k.copy()
        doc_j = copy.deepcopy(self.doc_j)
        topic_idx = self.topics.copy()
        tbl = 3
        k = doc_j['k_jt'][tbl]
        
        doc_j, topic_idx, m_k = HDP.remove_Xvec_ji(tbl, doc_j, topic_idx, m_k)
        
        self.assertNotIn(k, topic_idx)
        self.assertEqual(m_k[k], self.m_k[k]-1)
        self.assertEqual(doc_j['k_jt'][tbl], 0)
        
    
    def test_posterior_k(self):
        "check pvals of post k are valid"
        tbl = 3
        
        pvals = HDP.hdp_cpp.posterior_k_cpp(tbl, self.doc_j['k_jt'], self.doc_j['n_jt'],
                                           self.topics, self.n_kv, self.m_k, self.njtw[self.j][tbl],
                                           self.gamma, self.V, self.beta)
        
        self.assertEqual(np.sum(pvals), 1)
        self.assertFalse(any(pvals < 0))
        
        
    def test_rearrange_cnts(self):
        '''check that counts are added to new k'''
        doc_j = copy.deepcopy(self.doc_j)
        n_kv = copy.deepcopy(self.n_kv)
        new_k = 1
        tbl =3
        k = doc_j['k_jt'][tbl]
        
        doc_j, n_kv = HDP.rearranging_k_counts(self.j, tbl, new_k, doc_j, self.njtw, n_kv)
        
        self.assertEqual(np.sum(n_kv[2:4,new_k]), np.sum(self.n_kv[2:4,k]))
        self.assertEqual(np.sum(n_kv[2:4,new_k]), np.sum(n_kv[2:4,k])+2)
        self.assertEqual(doc_j['k_jt'][tbl], new_k)
        self.assertFalse(doc_j['k_jt'][tbl] == k)
        
        
        
        
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHDP)
    unittest.TextTestRunner(verbosity=2).run(suite)
