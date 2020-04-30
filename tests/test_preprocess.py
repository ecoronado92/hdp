
import hdp.text_prep as txtp
import pandas as pd
import nltk
import unittest
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer as stemmer
from nltk.corpus import stopwords

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


class TestPreprocess(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.read_csv('./tests/data.csv')
        self.st_words = stopwords.words('english')
        self.stemmer = stemmer('english')
        self.tmp_docs = [txtp.full_preprocess(d, set(self.st_words)) for d in self.df.values]
        self.v = txtp.preproc_cpp.generate_vocab(self.tmp_docs)
        self.tf = txtp.preproc_cpp.tf_cpp(self.tmp_docs, self.v)

    def test_csv_format(self):
        '''Check csv has only 1 column'''
        self.assertEqual(self.df.shape[1], 1)
    
    def test_full_preproc(self):
        '''Check stop words removed and word len > 3'''
        tmp = txtp.full_preprocess(self.df.values[0], set(self.st_words))
        for w in tmp:
            self.assertGreater(len(w), 3)
            self.assertNotIn(w, self.st_words )
    
    def test_all_docs_preproc(self): 
        '''Check all documents are preprocessed effectively'''
        self.assertEqual(len(self.tmp_docs), self.df.shape[0])
    
    
    def test_vocab_fcn(self):
        '''Check vocab is an array and has words'''
        self.assertTrue(isinstance(self.v, list))
        self.assertGreater(len(self.v), 0)
        for v in self.v:
            self.assertTrue(isinstance(v, str))
  
    def test_tf_cpp_fcn(self):
        '''Check tf matrix is generated correctly '''
        self.assertEqual(self.tf.shape[0], len(self.v))
        self.assertTrue(isinstance(self.tf, np.ndarray))
        
    
    def test_tf_filtering(self):
        '''Check filtering reduces words in tf matrix, keeps structure'''
        filt_tf = txtp.preproc_cpp.filter_tf_cpp(self.tf)
        self.assertLess(filt_tf.shape[0], self.tf.shape[0])
        self.assertEqual(filt_tf.shape[1] -1, self.tf.shape[1])
        self.assertLessEqual(len(filt_tf[:,-1]), len(self.v))
    
    
    def test_get_docs_fcn(self):
        '''Check output structure is inference ready'''
        filt_tf = txtp.preproc_cpp.filter_tf_cpp(self.tf)
        filt_tf = filt_tf[:,1:filt_tf.shape[1]] # returned by tf function
        tmp = txtp.preproc_cpp.get_docs(filt_tf)
        
        self.assertTrue(isinstance(tmp, list))
        self.assertEqual(len(tmp), self.df.shape[0])
        for i in tmp[0]:
            self.assertTrue(isinstance(i, int))

    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocess)
    unittest.TextTestRunner(verbosity=2).run(suite)
