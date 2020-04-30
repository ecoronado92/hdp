# hdp - Hierarchical Dirichlet Process (or Hierarichical LDA)

The `hdp` package provides tools to set-up and train a Hierarchical Dirichlet Process (HDP) for topic modeling. This is similar to a Latent Dirichlet Allocation (LDA) model, with one major difference -  HDPs are non-parametric in that the topics are learned from the data rather than user-specified.

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [API](#api)

## Dependencies

This packages has the following dependencies 

- Python >=3.0 (stability not tested on Python 2)
- Numpy >= 1.15
- Pandas >= 1.0
- Scipy >= 1.2
- Pybind11 = 2.5

- Eigen (C++ linear algebra library)

To install the Python dependencies, open up the terminal and run the following commands

```
pip3 install -r requirements.txt
```

To install the external C++ libraries, run the following command on terminal (note: it is important that you do so on the hdp root folder).

```
python3 ./src/clone_extensions.py
```

The scripts won't install anything if everything is up to date.

## Installation

Once the dependencies are installed, run the following script on terminal

```
chmod 755 ./INSTALL.sh
./INSTALL.sh
```

or individuallyou run the following commands

```
python3 setup.py build_ext -i
python3 setup.py build
python3 setup.py install
```


## Getting Started

Let's get started by importing the package and it's main functions `run_preprocess` and `run_hdp`. The former provides an API to preprocess text data in CSV format while the later computes the inference. 
```
import hdp

from hdp import run_preprocess
from hdp import run_hdp
```

You can test these functions with some test data in the `./data` folder. For example, to preprocess files and obtain the corpus vocabulary and an inference-ready document structure use the following commands
```
url = './data/tm_test_data.csv'

vocab, docs = run_preprocess(url)
```

Subsequently, you can generate inferences using the `hdp` function with the following commands

```
import numpy as np

it = 5

# Hyperparameters (user-defined)
beta = 0.5 # topic concentration (LDA)
alpha = np.random.gamma(1,1) # DP mixture hyperparam
gamma = np.random.gamma(1,1) # Base DP hyperparam

doc_arrays, topic_idx, n_kv, m_k = run_hdp(docs, vocab, gamma, alpha, beta, epochs=it)
```

For more information on how to run these functions see the [api](#api) section below or visit the `./report` folder which provides context and theory behind the implementation.


## API

### run_preprocess(file_url)

**Parameters**

- **file_url**: string
    - Path to CSV formatted data. **Note**: This should be 1 column only, where each row is the text data to be pre-processed (e.g. abstracts)


**Returns**
   
- **vocab**:array
    - corpus vocabulary
    
- **docs**:list(sub_list), len = num of docs
    - list containing document sublists, each sublists is composed of word unique indexes from vocabulary



### run_hdp(docs, vocab, gamma, alpha, beta, epochs)

Computes inference on document corpus and vocabulary over a user defined number of epochs (iterations). Additionally, user must provide prior distribution hyperparameters similar to those needed in LDA


**Parameters**

- **docs**:list(sub_list), len = num of docs
    - list containing document sublists, each sublists is composed of word unique indexes from vocabulary
    
- **vocab**:array
    - corpus vocabulary

- **gamma**:float
    - Base Dirichlet Process distribution $G_0$ hyperparameter
    
- **alpha**:float
    - Dirichlet Process Mixture $G_j$ hyperparameter
    
- **beta**:float
    - Word concentration parameter (similar to LDA)
    
- **epochs**:int (default = 1)
    - Defines number of iterations for which to train the model

**Returns**

- **doc_arrays**: list(Dict), len = num of docs
    - For each document, there's a dictionary with the following keys to the following arrays (all the same length)
        - `t_j`: array of table indexes in document 
        - `k_jt`: array of topics assigned to tables in `t_j`
        - `n_jt`: count array of words assigned to each table/topic

- **topic_idx**: list
    - Inferred topic indexes

- **n_kv**: ndarray
    - Word-topic matrix, rows = words and columns = topics. **Note:** num of cols will not equal topic_idx since this is not dynamically updated as `topic_idx` during inference. Use the `topic_idx` to select the appropriate cols.
    - Using n_kv.sum(axis=0) provides a summary of the total number of words assigned per topic 

- **m_k**:array
    - Number of tables per topic
