#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <math.h>


#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <random>

namespace py = pybind11;

// POSTERIOR PVAL FOR T 
// Generate posterior pvals selecting a new or existing table
Eigen::VectorXd posterior_t_cpp(Eigen::VectorXd t_j, Eigen::VectorXd n_jt, 
                           Eigen::VectorXd k_jt, Eigen::VectorXd m_k, Eigen::VectorXd fk,
                           double gamma, double V, double alpha){
    
    // Get table specific topic and counts
    Eigen::VectorXd k_idx = k_jt(t_j);
    Eigen::VectorXd n_idx = n_jt(t_j);
    
    // If t is not new
    Eigen::VectorXd post_t = n_idx.cwiseProduct(fk(k_idx)); // element-wise product (scalar * vector)
    
    // If t is new
    double post_t_new = m_k.dot(fk) + gamma/V;
    post_t(0) = post_t_new * alpha / (gamma + m_k.sum());
    
    post_t /= post_t.sum(); // Normalize to generate pvals
    
    return post_t;
    
}
    

// Conditional density of x_ji given k and all data items except x_j
Eigen::VectorXd fk_cpp(int word, Eigen::MatrixXd  n_kv){
    
    Eigen::VectorXd lik = n_kv.row(word).cwiseQuotient(n_kv.colwise().sum()); // element-wise division(vec / scalar)
    lik[0] = 0; // First should always be zero
    
    return lik;
}
    
    
// POSTERIOR PVAL FOR K  
// Compute explicit posterior multinomial-dirichlet posterior distribution
Eigen::VectorXd posterior_k_cpp(int tbl, Eigen::VectorXd k_jt, 
                                Eigen::VectorXd n_jt, Eigen::VectorXd topic_idx, Eigen::MatrixXd n_kv,
                                Eigen::VectorXd m_k, const std::unordered_map<int, int> n_jtw,
                                double gamma, double V, double beta){
    
    // Topic k of table t
    int k = k_jt(tbl);
    double cnt_t = n_jt(tbl);
    double vbeta = V*beta;
    
    // Remove all counts associated with topic k in table t, from overall topic counts (n_k)
    Eigen::VectorXd n_k = n_kv.colwise().sum();
    n_k(k) -= cnt_t;
    n_k = n_k(topic_idx);
    
    // Initialized k posterior in log-form for simplicity, this computes f_k^{-X_ji} 
    // has Dirichlet-Multinomial form
    double log_post_k_new = std::log(gamma) + std::lgamma(vbeta) - std::lgamma(vbeta + cnt_t); // for new k
    
    // for old ks, needs to loop since it's a vector
    m_k = m_k(topic_idx);
    Eigen::VectorXd log_post_k = m_k.array().log();
    for (int i = 0; i < log_post_k.size() ; i++){
        log_post_k(i) += std::lgammaf(n_k(i)) - std::lgammaf(n_k(i) + cnt_t);
    };
    
    // Remove individual word counts associated with topic k
    // add their contributions to k posterior
    for (auto item : n_jtw){
        
        if (item.second == 0) continue; //if word count is 0 skip
        
        //For word w, get counts across topics
        Eigen::VectorXd w_cnt_k = n_kv.row(item.first);
        
        // For specific topic k, remove count from associated table t
        w_cnt_k(k) -= item.second;
        w_cnt_k = w_cnt_k(topic_idx);
        w_cnt_k(0) = 1;
        
        // Add contributions of individual observations (words) for k new and k old
        log_post_k_new += std::lgammaf(item.second + beta) - std::lgammaf(beta);
        
        for (int j = 0; j < w_cnt_k.size(); j++){
            log_post_k(j) += std::lgammaf(w_cnt_k(j) + item.second) - std::lgammaf(w_cnt_k(j));
        
        };      
    };
  
    // p-val for new k
    log_post_k(0) = log_post_k_new;
    
    // Bring back to non-log realm
    double pk_max = log_post_k.maxCoeff();
        
    for (int s = 0; s < log_post_k.size(); s++){
        log_post_k(s) -= pk_max;
    };
    
    log_post_k = log_post_k.array().exp();
    
    
    log_post_k /= log_post_k.sum(); // Normalize
    
    
    return log_post_k;

}


// Random Multinomial Sampling in Eigen w/ argmax selection
// Given a set of pvals and N, gives back index most likely category
// Ref: 
//   C.S. David, The computer generation of multinomial random variates,
//   Comp. Stat. Data Anal. 16 (1993) 205-217
// 

int argmax_multinomial_cpp(int N, Eigen::VectorXd pvals){

    // Init random number generated
    std::random_device rd;
    std::mt19937 re(rd());
    
    // Allocate memory
    double sum_p = 0.0;
    double norm = 0.0;
    int sum_n = 0;
    int K = pvals.size(); // size of vector (i.e. # of components)
    Eigen::MatrixXd n_vec(1,K);
    Eigen::MatrixXd::Index maxRow, maxCol;

    norm += pvals.sum(); // = 1 if pvals are input (sum of components)
    
    // For each component (except last one) sample from binomial
    for (int i=0; i < K-1; i++){
        
        // Binomial sampler varies across loop, pvals and remaining N get adjusted 
        std::binomial_distribution<int> binomial(N-sum_n, pvals(i)/(norm - sum_p) );
       

        if (pvals(i) > 0.0){
            n_vec(0,i) = binomial(re);
        }
        else {
            n_vec(i) = 0;
        };
        
        sum_p += pvals(i); // adjustment for pvals and remaining n < N
        sum_n += n_vec(0,i);
 
    };
    
    // For last component allocate remaining N
    n_vec(0,K-1) = N - sum_n;
    
    int max = n_vec.maxCoeff(&maxRow, &maxCol); // Get argmax of vector
 
    return maxCol;
     
}


// Pybind11 module referencing
PYBIND11_MODULE(hdp_funcs, m) {
    m.doc() = "pybind11 cpp functions used in hierarchical dirichlet processes topic modeling",
    m.def("posterior_t_cpp", &posterior_t_cpp, "Generate posterior pvals to allocate word to table"),
    m.def("fk_cpp", &fk_cpp, "Conditional distribution necessary for posterior sampling"),
    m.def("posterior_k_cpp", &posterior_k_cpp, "Generate posterior pvals to allocate word to topic"),
    m.def("argmax_multinomial_cpp", &argmax_multinomial_cpp, "Vectorized multinomial random sampling");
}
