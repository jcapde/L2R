#include <iostream>
#include <cmath>
#include <unordered_map>
#include <random>
#include <omp.h>
#include <set>
#include <math.h>

#include "MyEigen.h"
#include "utils.h"
#include "random.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::Ref;

void gamma_sampler(InEVector gam_shape, InEVector gam_scale, OutEVector theta, MTGen& gen) {
  std::gamma_distribution<> g;
  typedef std::gamma_distribution<>::param_type param_type;
  
  for (int i = 0 ; i < gam_shape.size() ; ++i) {
    theta(i) = g(gen.get(), param_type(gam_shape(i), gam_scale(i)));
  }
}

double log_prob_multinomial(int n, InEVector p, InEVector x){
  double log_prob = lgamma(n+1);
  int K = p.size();
  for(int k = 0; k < K; k++){
    log_prob += x(k)*log(p(k)) - lgamma(x(k)+1);
  }
  return log_prob;
}
  
void multinomial_sampler(int n, InEVector mult_para, OutEVector values, MTGen& gen) {
  int num_options = mult_para.size();
  double remainder = n;
  double weight_remainder = 1.0;
  std::binomial_distribution<> b;
  values.setZero();
  typedef std::binomial_distribution<>::param_type param_type;
  for (int i = 0 ; (i < num_options-1) && (remainder > 0) ; i++) {
    auto p = param_type(remainder, mult_para(i) / weight_remainder);
    values(i) = b(gen.get(), p);
    remainder -= values(i);
    weight_remainder -= mult_para(i);
  }
  values(num_options-1) = remainder;
}

void extract_indexes_vec(InEVector y, const std::vector<int> &indx_selected, OutEVector sel) {
    for (int i = 0 ; i < indx_selected.size() ; ++i) {
        sel(i) = y(indx_selected[i]);
    }
}
void extract_by_cols(InEMatrix M, const std::vector<int> &indx_selected, OutEMatrix sel) {
    for (int i = 0 ; i < indx_selected.size() ; ++i) {
        sel.col(i) = M.col(indx_selected[i]);
    }
}

int get_non_zeros(InEVector y){
  int num_non_zeros = 0;
  for(int w = 0; w < y.size(); w++){
    if(y[w]>0){
      num_non_zeros++;
    }
  }
  return num_non_zeros;
}

void compute_non_zeros(InEVector y, std::vector<int> &non_zeros) {
  non_zeros.clear();
  for (int i = 0 ; i < y.size() ; ++i) {
      if (y(i) > 0) {
          //std::cout << y(i) << " ";
          non_zeros.push_back(i);
      }
  }
}

void compute_non_zeros_counts(InEVector y, std::vector<int> &non_zeros, OutEVector non_zeros_counts) {
  non_zeros.clear();
  int j = 0;
  for (int i = 0 ; i < y.size() ; ++i) {
    if (y(i) > 0) {
      //std::cout << y(i) << " ";
      non_zeros.push_back(i);
      non_zeros_counts(j) = int(y(i));
      j++;
    }
  }
}

Eigen::PermutationMatrix<Eigen::Dynamic> compute_non_zeros_permutation(InEVector y, std::vector<int> &non_zeros) {
  non_zeros.clear();
  std::set<int> non_zeros_set;
  for (int i = 0 ; i < y.size() ; ++i) {
      if (y(i) > 0) {
          //std::cout << i << "has number " << y(i) << std::endl;
          non_zeros.push_back(i);
          non_zeros_set.emplace(i);
      }
  }
  Eigen::PermutationMatrix<Eigen::Dynamic> perm(y.size());
  perm.setIdentity();
  int free_pos_candidate = 0 ;
  for (int i = 0 ; i < non_zeros.size() ; ++i) {
      if (non_zeros[i] >= non_zeros.size()) {
          while (non_zeros_set.find(free_pos_candidate) != non_zeros_set.end()) {
              free_pos_candidate++;
          }
          //std::cout << "Moving "<< non_zeros[i] << " to position " <<
          //free_pos_candidate << std::endl;
          perm.applyTranspositionOnTheRight(non_zeros[i], free_pos_candidate);
          free_pos_candidate++;
        }
  }

  return perm;
}

bool relatively_equal(double x, double y) {
    return std::fabs((x-y) / std::min(x,y)) < 1e-7;
}

VectorXd partial_sum(const Ref<const VectorXd> v) {
    VectorXd ps(v.size());
    ps(0) = v(0);
    for (int i = 1; i < v.size() ; ++i) {
        ps(i) = v(i) + ps(i-1);
    }
    return ps;
}

void cumsum_rows(InEMatrix M, OutEMatrix CM) {
  if (CM.cols() > 0) {
    CM.col(0) = M.col(0);
    for (int i = 1 ; i < M.cols() ; ++i ) {
      CM.col(i) = CM.col(i-1) + M.col(i);
    }
  }
}

void cumsum_cols(InEMatrix M, OutEMatrix CM) {
  if (CM.rows() > 0) {
    CM.row(0) = M.row(0);
    for (int i = 1 ; i < M.rows() ; ++i ) {
      CM.row(i) = CM.row(i-1) + M.row(i);
    }
  }
}

double log_sum_exp_vec(const Ref<const VectorXd> v) {
    double max_log = v.maxCoeff();
    return log((v.array() - max_log).exp().sum()) + max_log;
}

VectorXd log_sum_exp_rows(const Ref<const MatrixXd> mat){
  int rows = mat.rows();
  VectorXd res(rows);
  for(int i = 0; i < rows; i++) {
    res(i) = log_sum_exp_vec(mat.row(i));
  }
  return res;
}

//VectorXd log_sum_exp_rows(const Ref<const MatrixXd> mat){
//    int num_samples = mat.cols();
//    auto max = mat.rowwise().maxCoeff();
//    return (mat - max.replicate(1, num_samples)).array().exp().matrix().rowwise().sum().array().log().matrix() + max;
// }

long long binom(int n, int k) {
    return (long long) std::round(exp(log_binom((double)n, (double)k)));
}

double log_binom(double n, double k) {
    return lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1);
}

double log_gamma_pdf(double x, double k, double theta) {
  return -lgamma(k) - k * std::log(theta) + (k-1) * std::log(x) - x/theta;
}

double log_poisson_pdf(double lambda, int k) {
    return -lambda + k * std::log(lambda) - lgamma(k+1);
}

double log_poisson_pdf_wo_lambda(double lambda, int k) {
  return  k * std::log(lambda) - lgamma(k+1);
}

double log_neg_bin_pdf(double k, double r, double p) {
    return log_binom(k+r-1, k) + r * std::log1p(- p) + k * std::log(p);
}

double log_neg_multinomial_pdf(const Ref<const VectorXd> x, double k0, const Ref<const VectorXd> p) {
    double sum = x.sum() + k0;
    double p0 = 1 - p.sum();
    double log_p = lgamma(sum) + k0 * std::log(p0) - lgamma(k0);
    for (int i = 0 ; i < p.size() ; ++i) {
        if (x(i) > 0) {
            log_p += x(i) * log(p(i)) - lgamma(x(i)+1);
        }
    }
    return log_p;
}

double log_multinomial_pdf(const Ref<const VectorXd> x, int n, const Ref<const VectorXd> p) {
  double log_p = lgamma(n+1);
  for(int i = 0; i < p.size(); i++){
    if(x(i) > 0){
      log_p += x(i) * log(p(i)) - lgamma(x(i)+1);
    }
  }
  return log_p;
}

static std::unordered_map<double, double> lgamma_dict;
double memo_lgamma(double x) {
    auto got = lgamma_dict.find(x);
    if (got == lgamma_dict.end()) {
        double v = lgamma(x);
        lgamma_dict[x] = v;
        return v;
    }// else {
    //    std::cout << "A" << std::endl;
    //}
    return got->second;
}

static std::unordered_map<double, double> log_dict;
double memo_log(double x) {
    auto got = log_dict.find(x);
    if (got == log_dict.end()) {
        double v = log(x);
        log_dict[x] = v;
        return v;
    }// else {
    //    std::cout << "A" << std::endl;
    //}
    return got->second;
}


double sparse_log_neg_multinomial_pdf_Xnonzeros(const Ref<const VectorXd> x, double k0,
                                      const Ref<const VectorXd> cache_log_p, const Ref<const VectorXd> cache_log_gamma) {
    double sum = x.sum() + k0;
    int num_non_zeros = x.size();
//    std::cout << "num_non_zeros: " << num_non_zeros << std::endl;
//    std::cout << "cache_log_p: "<< cache_log_p.transpose() <<std::endl;
//    std::cout << "x.transpose(): " << x.transpose() <<std::endl;
//    std::cout << "cache_log_gamma: " << cache_log_gamma.transpose() << std::endl;

    //double p0 = 1 - p.sum();
    //std::cout << "Sum is " << sum << std::endl;
    double log_p = lgamma(sum) + k0 * cache_log_p(cache_log_p.size()-1) - cache_log_gamma(std::round(k0));
    for (int iw = 0; iw < num_non_zeros; iw++) {
        //std::cout << " x(iw): " <<  x(iw) << std::endl;
        log_p += x(iw) * cache_log_p(iw) - cache_log_gamma(std::round(x(iw)+1));
    }
    return log_p;
}


double sparse_log_neg_multinomial_pdf(const Ref<const VectorXd> x, std::vector<int> &non_zeros, double k0,
                                      const Ref<const VectorXd> cache_log_p, const Ref<const VectorXd> cache_log_gamma) {
  //std::cout << x << std::endl;
  double sum = x.sum() + k0;
  //double p0 = 1 - p.sum();
  //std::cout << "Sum is " << sum << std::endl;
  double log_p = lgamma(sum) + k0 * cache_log_p(cache_log_p.size()-1) - cache_log_gamma(std::round(k0));
  //std::cout << "Initial log_p " << log_p << std::endl;
  for (int i : non_zeros) {
    log_p += x(i) * cache_log_p(i) - cache_log_gamma(std::round(x(i)+1));
    //std::cout << "log_p_non_zeros(" << i << ")=" << log_p << std::endl;
  }
  return log_p;
}
// double sparse_log_neg_multinomial_pdf(const Ref<const VectorXd> x, std::vector<int> &non_zeros, double k0, const Ref<const VectorXd> p, 
//                                       const Ref<const VectorXd> cache_log_p, const Ref<const VectorXd> cache_log_gamma) {
//     double sum = x.sum() + k0;
//     double p0 = 1 - p.sum();
//     double log_p = memo_lgamma(sum) + k0 * memo_log(p0) - memo_lgamma(k0);
//     for (int i : non_zeros) {
//         log_p += x(i) * memo_log(p(i)) - memo_lgamma(x(i)+1);
//     }
//     return log_p;
// }

void sort_indexes(EVector probs, std::vector<int> &idx, bool decreasing){
  if(decreasing){
    std::sort(idx.begin(), idx.end(),  [&probs](size_t i1, size_t i2) {return probs[i1] > probs[i2];});
  }else{
    std::sort(idx.begin(), idx.end(),  [&probs](size_t i1, size_t i2) {return probs[i1] < probs[i2];});
  }
}

double log_factorial(int x) {
   return lgamma(x + 1);
}