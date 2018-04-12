#include <iostream>
#include <functional>
#include <Eigen/Dense>
#include <cmath>
#include <boost/math/distributions.hpp>
#include <stack>


#include "partitions.h"
#include "utils.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::Ref;       


double log_p_x_exact_word(const Ref<const VectorXd> x, const Ref<const VectorXd> phi, const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
    int num_factors = x.size();
    VectorXd log_ps(num_factors);
    double log_p = 0;
    for (int f = 0 ; f < num_factors; ++f) {
        double nb_p = phi(f) * p(f) / (1 + p(f)* (phi(f) - 1)); 
        //boost::math::negative_binomial nb(r, nb_p);
        //log_ps(f) = log_neg_bin_pdf(x(f), r(f), nb_p);
        log_p += log_neg_bin_pdf(x(f), r(f), nb_p);
    }
    return log_p;
    //return log_sum_exp_vec(log_ps);
}

double log_p_y_exact_word(double y, const Ref<const VectorXd> phi,  const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
    int num_factors = phi.size();
    auto partitions = partitions_size_up_to(y, num_factors);
    int num_partitions = partitions.size();
    VectorXd log_ps(num_partitions);
    int i = 0;
    for (auto x : partitions) {
        log_ps(i) = log_p_x_exact_word(x, phi, r, p);
        ++i;
    }
    return log_sum_exp_vec(log_ps);
}


double log_p_x_exact_doc(const Ref<const MatrixXd> x, const Ref<const MatrixXd> Phi, 
  const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
  int num_factors = Phi.rows();
  double log_p = 0.0;

  //VectorXd p_neg_multinomial(Phi.cols());
  double alpha, beta, sum , factor;
  for (int k = 0 ; k < num_factors; k++) {
    auto phi_k = Phi.row(k);
    alpha = r(k); 
    beta = (1 - p(k)) / p(k);
    //beta = p(k) / (1 - p(k)) ;
    sum = phi_k.sum();
    factor = 1 / (beta + sum);
    log_p += log_neg_multinomial_pdf(x.col(k), alpha, phi_k * factor );
  }
  return log_p;
}

double sparse_log_p_x_exact_doc(const Ref<const MatrixXd> x, std::vector<int> &non_zeros, const Ref<const VectorXd> alpha,
                                const Ref<const MatrixXd> cache_log_p, const Ref<const VectorXd> cache_log_gamma) {
  int num_factors = x.cols();
  //std::cout << "Num factors:" << num_factors << std::endl; 
  //double log_p = 0.0;
  //VectorXd p_neg_multinomial(Phi.cols());
  //double alpha, beta, sum , factor;
  VectorXd log_p(num_factors);
  //#pragma omp parallel for
  // NOTICE: When uncommented, this pragma makes performance far worse!
  for (int k = 0 ; k < num_factors; k++) {
    log_p(k) = sparse_log_neg_multinomial_pdf(x.col(k), non_zeros, alpha(k), cache_log_p.row(k), cache_log_gamma);
    //std::cout << "log_p_k(" << k << ")=" << log_p(k)<< std::endl;
  }
  return log_p.sum();
}



double log_p_y_exact_doc(const Ref<const VectorXd> y, const Ref<const MatrixXd> Phi, 
  const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
  //std::cout << "y = " << y << std::endl;
  int num_factors = Phi.rows();
  int num_words = Phi.cols();
  //double prob = 0;
  int count = 0;
  long long num_partitions = compute_num_partitions(y, num_factors);
  
  int max_gamma = std::round(y.sum() + 2);
  VectorXd cache_log_gamma(max_gamma);
  
  #pragma omp parallel for
  for (int i = 0 ; i < max_gamma; i++) {
      cache_log_gamma(i) = lgamma(i);
  }
  //std::cout << "Assessed gammas" << std::endl; 
  std::vector<int> non_zeros;
  compute_non_zeros(y, non_zeros);
  //std::cout << std::endl;
  //std::cout << "Assessed non_zeros" << std::endl; 
  VectorXd alpha(num_factors), beta(num_factors), sum(num_factors), Phi_factor(num_factors);
  MatrixXd cache_p(num_factors, num_words);
  MatrixXd cache_log_p(num_factors, num_words+1); 
  #pragma omp parallel for
  for (int k = 0 ; k < num_factors; k++) {
    //auto phi_k = Phi.row(k);
    alpha(k) = r(k); 
    beta(k)  = (1 - p(k)) / p(k);
    //beta = p(k) / (1 - p(k)) ;
    sum(k) = Phi.row(k).sum();
    Phi_factor(k) = 1 / (beta(k) + sum(k));
    //std::cout << "cache_log_p(1)" << std::endl; 
    cache_p.row(k) = (Phi.row(k) * Phi_factor(k));
    cache_log_p.row(k).head(num_words) = cache_p.row(k).array().log();
    //std::cout << "cache_log_p(2)" << std::endl; 
    cache_log_p(k, num_words) = log(1.0 - cache_p.row(k).sum());
    //std::cout << "cache_log_p(3)" << std::endl; 
  }
  //std::cout << "Assessed cache_log_p" << std::endl; 
  //std::cout << "Cache_log_p " << cache_log_p << std::endl;
  VectorXd log_p(num_partitions);
  //int num_partitions = log_binom()

  MatrixXdVisitor x_iterator = [&non_zeros, &count, &log_p, &alpha, &cache_log_p, &cache_log_gamma](const Ref<const MatrixXd> x) {
    //std::cout << x << std::endl;
    //std::cout << std::endl;
    //for (int i :non_zeros){
    //    std::cout << x.row(i) << std::endl;
    //}
    //std::cout << std::endl;
    log_p(count) = sparse_log_p_x_exact_doc(x, non_zeros, alpha, cache_log_p, cache_log_gamma);
    //std::cout << "log_p("<< count << ")=" << log_p(count) << std::endl;
    //prob += exp();
    ++count;
    //if (count == 1000) {
    //    return true;
    //}
    return false;
  };

  //std::cout << "y=" << y << std::endl;
  for_each_combination_MatrixXd(y, num_factors, x_iterator);
  //std::cout << "Num partitions:" << count << std::endl;
  return log_sum_exp_vec(log_p);
  //return log(prob);
}


double log_p_y_exact_docs(const Ref<const MatrixXd> y, const Ref<const MatrixXd> Phi, 
  const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
    int num_words = Phi.cols();
    int num_factors = Phi.rows();
    int num_documents = y.rows();

    VectorXd llikes(num_documents);

    for(int n = 0; n < num_documents; n++) {
      //std::cout << n << std::endl;
      llikes(n) = log_p_y_exact_doc(y.row(n).transpose(), Phi, r, p);
    }
    //std::cout << log_sum_exp_vec(llikes) << std::endl; 
    return llikes.array().sum();
}