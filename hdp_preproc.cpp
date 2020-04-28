<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['eigen']
setup_pybind11(cfg)
%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iterator>
#include <iostream>
#include <fstream>
#include <set>


namespace py = pybind11;

// Text clean-up function, removes non-ASCIIs, punctuations
// turns to lower case and split into vector
std::vector<std::string> text_cleanup(std::string& text){
    
    // Remove non-ASCII values
    auto non_ascii = [] (char c) {  return !(c >= 0 && c < 128);};
    text.erase(std::remove_if(text.begin(), text.end(), non_ascii), text.end());

   
    // Remove punctuations
    auto check_punct = [](char c) {return std::ispunct(static_cast<unsigned char>(c)); };
    text.erase(std::remove_if(text.begin(), text.end(), check_punct), text.end());
    
    // To lower
    auto to_lower = [](char c){ return std::tolower(static_cast<unsigned char>(c));};
    std::transform(text.begin(), text.end(), text.begin(), to_lower);
    
    // Split
    std::istringstream iss(text);
    std::vector<std::string> str_array(std::istream_iterator<std::string>{iss},
                                 std::istream_iterator<std::string>());
    
    
    return str_array;
}


// Remove stop words and words larger than specific cutoff given a vector of strings
std::vector<std::string> rm_stops_n_shorts(std::vector<std::string> txt, 
                                          std::set<std::string>& st_words,
                                          unsigned long cutoff){
    
    std::vector<std::string> result;
    
    // Include non-stop words and those larger than cutoff
    for (auto word : txt){
        if ((st_words.find(word) == st_words.end()) && word.size() > cutoff )
            result.push_back(word);
        
    }
    
    return result;
}



// Get vocabulary 
// code inspired by Zhaojie Tao, https://github.com/taozhaojie
std::vector<std::string> generate_vocab(std::vector<std::vector<std::string>>& stemmed_docs){
    
    
    std::vector<std::string> vocab;
    std::set<std::string> vocab_set;
    
    // Loop through words in documents, insert into set to get unique words
    for (std::vector<std::string> doc : stemmed_docs){
        for (std::string word : doc)
            vocab_set.insert(word);
    }
    
    // Copy elements of set into type string vector
    std::copy(vocab_set.begin(), vocab_set.end(), std::back_inserter(vocab));
    
    return vocab;
}



// TF Matrix generation
// code inspired by Zhaojie Tao, https://github.com/taozhaojie
Eigen::MatrixXf tf_cpp(std::vector<std::vector<std::string>>& stemmed_docs, 
                   std::vector<std::string>& vocab){

    std::vector<Eigen::VectorXf> cnt_vec; //allocate memory, list of lists
    
    // Loop through doucments
    for (std::vector<std::string> doc : stemmed_docs){ 
        
        std::vector<float> tmp_vec(vocab.size(), 0); // pre-allocate temp vector

        // Creates binary vector, 1 for words found in text and 0s if not
        for (std::string word : doc){

            size_t idx = std::find(vocab.begin(), vocab.end(), word) - vocab.begin();

            if (idx == vocab.size())
                std::cout << "word: " << word << "not found" << std::endl;

            else
                tmp_vec.at(idx) += 1;
        };

        // Convert vector to Eigen vector
        Eigen::Map<Eigen::VectorXf> vec(tmp_vec.data(),tmp_vec.size());
    
    
        cnt_vec.push_back(vec); //a add to main list
    };
    
    // Generate term count matrix for output using list of sublists
    int nrow = cnt_vec[0].size();
    int ncol = cnt_vec.size();
    
    Eigen::MatrixXf tf = Eigen::MatrixXf::Zero(nrow,ncol);

    for (int i = 0; i < ncol ; ++i){
        tf.col(i) = cnt_vec[i];
    };
        
    return tf;     
}




// Filter term-frequency matrix for words that 
// 1) appear in more than 3 documents, and
// 2) are in less than half the corpus
 Eigen::MatrixXf  filter_tf_cpp(Eigen::MatrixXf tf){
    
    // Allocate memory
    int nrow = tf.rows();
    int ncol = tf.cols();
    double cols = tf.cols();
    Eigen::MatrixXf filt_tf = Eigen::MatrixXf::Zero(nrow, ncol);
    std::vector<float> filt_vocab;
     
    // For each word
    for(int i=0; i < nrow; ++i){
        
        // Sum total word count accross documents
        double cnt = 0;
        cnt += tf.row(i).sum();
        
        // Filtering criteria
        if(tf.row(i).sum() > 3.0 && cnt/cols < 0.5){
            filt_tf.row(i) = tf.row(i);
            filt_vocab.push_back(i);
        }
                
    }
    
    // Find non-zero rows
    Eigen::Matrix<bool, Eigen::Dynamic, 1> non_zeros = filt_tf.cast<bool>().rowwise().any();

    // Allocate output matrix
    Eigen::MatrixXf res(non_zeros.count(), filt_tf.cols());

    // Fill output matrix
    Eigen::Index j=0;
    for(Eigen::Index i=0; i<nrow; ++i)
    {
        if(non_zeros(i))
            res.row(j++) = filt_tf.row(i);
    }

    // Add vocabulary word indexes as last column
    int cols_new = res.cols();
    res.conservativeResize(Eigen::NoChange, cols_new+1); // preserves data while resizing
    Eigen::VectorXf vec = Eigen::Map<Eigen::VectorXf>(filt_vocab.data(),filt_vocab.size());
    res.col(cols_new) = vec;
     
     
    return res;
}


// Use filtered term-frequency matrix and generate topic model analysis ready structure
// Output: List of sublists with integers, each integer is a vocabulary word index
std::vector<std::vector<int>>  get_docs(Eigen::MatrixXf& filt_tf){
    
    // Allocate memory
    int ncol = filt_tf.cols();
    int nrow = filt_tf.rows();
    std::vector<std::vector<int>> res;
    
    // Get words(row) in each document (column) (i.e. count > 0.0), 
    // and store the word unique idx in array
    for (int i=0; i< ncol; i++){
        std::vector<int> tmp_vec;
        
        for(int j=0; j<nrow; j++){
            
            if(filt_tf(j,i) >0.0){
                tmp_vec.push_back(j);
            }      
        } 
        
        res.push_back(tmp_vec);
    }
    
    return res;  
}


// Pybind11 module referencing
PYBIND11_MODULE(hdp_preproc, m) {
    m.doc() = "pybind11 cpp functions used text preprocessing for topic modeling",
    m.def("text_cleanup", &text_cleanup, "Vector of lowercase alpha-num strings"),
    m.def("rm_stops_n_shorts", &rm_stops_n_shorts, "Filter out stop short words"),
    m.def("generate_vocab", &generate_vocab, "Extract complete vocab of document set"),
    m.def("tf_cpp", &tf_cpp, "Create term frequency matrix"),
    m.def("filter_tf_cpp", &filter_tf_cpp, "Filter tf matrix w/ associated vocab"),
    m.def("get_docs", &get_docs, "List of sublist data format for hdp inference");
}
