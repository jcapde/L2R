#pragma once
#include <vector>
#include <Eigen/Dense>

double log_sum_exp_vec(const Eigen::Ref<const Eigen::VectorXd>);

// Returns a vector with all the partitions of integer y into at most num_factors parts
std::vector<Eigen::VectorXd> partitions_size_up_to(int y, int num_factors);

// Returns the probability of a particular combination of a word having a vector of counts per factor $x_{ip.}$
double log_p_x_exact_word(const Eigen::Ref<const Eigen::VectorXd> x, const Eigen::Ref<const Eigen::VectorXd> phi, 
                          const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p);

// Returns the probability of a word havin a particular count $y_{ip}$
double log_p_y_exact_word(double y, const Eigen::Ref<const Eigen::VectorXd> phi,  
                          const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p);

// Returns the probability of a particular document $y_{i.}$
double log_p_y_exact_doc(const Eigen::Ref<const Eigen::VectorXd> y, const Eigen::Ref<const Eigen::MatrixXd> Phi, 
  const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p);

// Returns the probability of a particular set of documents $y_{..}$
double log_p_y_exact_docs(const Eigen::Ref<const Eigen::MatrixXd> y, const Eigen::Ref<const Eigen::MatrixXd> Phi, 
        const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p);
        
double exact_calculation(const Eigen::Ref<const Eigen::MatrixXd> y, const Eigen::Ref<const Eigen::MatrixXd> Phi, 
  const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p);

double sparse_log_p_x_exact_doc(const Eigen::Ref<const Eigen::MatrixXd> x,
                                const Eigen::Ref<const Eigen::VectorXd> alpha, const Eigen::Ref<const Eigen::MatrixXd> cache_log_p,
                                const Eigen::Ref<const Eigen::VectorXd> cache_log_gamma);
