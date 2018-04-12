#include <iostream>
#include <functional>
#include <cmath>
#include <stack>
#include <tuple>
#include <numeric>

#include <Eigen/Dense>

#include "utils.h"
#include "partitions.h"
#include "combinatorial.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::Ref;

/*
PartitionsIt partitions_size_up_to_it(int y, int num_factors) {
    VectorXd zeros(num_factors);
    PartitionsIt pi;
    pi.stack.push(std::make_tuple(y, num_factors, zeros, 0, num_factors));
    return pi;
}*/


int determine_new_k(VectorXd probs, std::vector<int> &idx,  double prob_thr){
  double accum = 0.0;
  int k = 0;
  sort_indexes(probs, idx, true);
  for(auto i : idx){
    //std::cout << i << std::endl;
    accum += probs[i];
    k++;
    if(accum > prob_thr) {
        return k;
    }
  }
}

void for_each_combination_VectorXd_mask(int n, int k, Eigen::VectorXi mask, VectorXdVisitor f) {

  int new_k = mask.sum();
  //std::cout << "mask " << mask << std::endl;
  //std::cout << "new_k " << new_k << std::endl;
  int N = n + new_k - 1;
  //std::cout << "Num. partitions from " << binom(n + k -1 , k - 1) <<" to " << binom(N, new_k-1) << std::endl;
  std::vector<int> v(N);
  std::iota(v.begin(), v.end(), 1);
  std::vector<int>::iterator r = v.begin() + new_k - 1;
  VectorXd v_it(k);
  v_it.setZero();

  std::vector<int> idx(new_k);
  int j = 0;
  for(int i = 0; i < k; i++){
    if (mask(i)==1){
      idx[j] = i;
      //std::cout << idx[j] << std::endl;
      j++;
    }
  }

  auto f_out = [&N, &f, &v_it, &idx](std::vector<int>::iterator start, std::vector<int>::iterator end) {
    int j = 0;
    int prev = 1;
    for (std::vector<int>::iterator i = start ; i < end; ++j, ++i) {
      //std::cout << *i << " ";
      v_it(idx[j]) = *i - prev;
      prev = *i + 1;
    }
    //std::cout << std::endl;
    v_it(idx[j]) = N - prev + 1;

    //std::cout << v_it << std::endl;
    return f(v_it);
  };

  for_each_combination(v.begin(), r, v.end(), f_out);
}
double compute_partition_probability(VectorXd x, VectorXd probs){

  int num_factors = x.size();
  double prob = log_factorial(x.sum());
  for(int k=0; k<num_factors; k++){
    prob += x(k)*log(probs(k));
    prob -= log_factorial(x(k));
  }
  return prob;
}

void for_each_combination_VectorXd_prob(int n, int k, VectorXd probs,  VectorXdVisitor f) {

  std::vector<int> idx(probs.size());
  std::iota(idx.begin(), idx.end(),0);
  sort_indexes(probs, idx, false);

  VectorXd new_probs(k);
  for(int i = 0; i<k; i++){
    new_probs[i] = probs[idx[i]];
  }
  //std::cout << new_probs << std::endl;

  //int new_k = determine_new_k(probs, idx, prob_thr);
  //int N = n + new_k - 1;
  int N = n + k - 1;
  //std::cout << "Num. partitions from " << binom(n + k -1 , k - 1) <<" to " << binom(N, new_k-1) << std::endl;
  std::vector<int> v(N);
  std::iota(v.begin(), v.end(), 1);
  //std::vector<int>::iterator r = v.begin() + new_k - 1;
  std::vector<int>::iterator r = v.begin() + k - 1;
  VectorXd v_it(k);
  v_it.setZero();

  double likelihoods = 0.0;

  auto f_out = [&N, &f, &v_it, &probs, &idx, &likelihoods](std::vector<int>::iterator start, std::vector<int>::iterator end) {
    int j = 0;
    int prev = 1;
    for (std::vector<int>::iterator i = start ; i < end; ++j, ++i) {
      //std::cout << *i << " ";
      v_it(idx[j]) = *i - prev;
      prev = *i + 1;
    }
    //std::cout << std::endl;             
    v_it(idx[j]) = N - prev + 1;

    //std::cout << probs.transpose() << std::endl;

    double ll = compute_partition_probability(v_it, probs);
    if(likelihoods < 0.75){
      likelihoods += exp(ll);
      //std::cout << likelihoods << std::endl;
      return f(v_it);
    }else{
      return false;
    }

  };

  for_each_combination(v.begin(), r, v.end(), f_out);
}


void for_each_combination_VectorXd(int n, int k, VectorXdVisitor f) {
  int N = n + k - 1;
  std::vector<int> v(N);
  std::iota(v.begin(), v.end(), 1);
  std::vector<int>::iterator r = v.begin() + k-1;
  VectorXd v_it(k);
  
  auto f_out = [&N, &f, &v_it](std::vector<int>::iterator start, std::vector<int>::iterator end) {
    int j = 0;
    int prev = 1;
    for (std::vector<int>::iterator i = start ; i < end; ++j, ++i) {
      //std::cout << *i << " ";
      v_it(j) = *i - prev;
      prev = *i + 1;
    }
    //std::cout << std::endl;             
    v_it(j) = N - prev + 1;

    //std::cout << v_it << std::endl;
    return f(v_it);
  };
  for_each_combination(v.begin(), r, v.end(), f_out);
}
  

//typedef std::function<bool(const Ref<const MatrixXd>)> MatrixXdVisitor;
//typedef std::function<bool(const Ref<const VectorXd>, int)> RecVectorVisitor;

class RecCaller {
public:
  RecCaller(Ref<MatrixXd> x, int num_words, const Ref<const VectorXd> y, int num_factors, 
            MatrixXdVisitor f): _x(x), _num_words(num_words), _y(y), _num_factors(num_factors), _f(f), _non_zeros() {
              for (int i = 0 ; i < _y.size() ; ++i) {
                if (y(i) > 0) {
                  _non_zeros.push_back(i);
                }
            }
  }
  bool rec_visit(const Ref<const VectorXd> v, int w) {
    //std::cout << "w = " << w << std::endl;
    _x.row(_non_zeros[w]) = v;
    int z = w+1;
    if (z < _non_zeros.size()) {
      auto f_next = std::bind(&RecCaller::rec_visit, this, std::placeholders::_1, z);
      for_each_combination_VectorXd(_y(_non_zeros[z]), _num_factors, f_next); 
    } else {
      _f(_x);
    }
    return false;
  }
  int non_zero_index(int i) {
    return _non_zeros[i];
  }
private:
  Ref<MatrixXd> _x;
  int _num_words;
  const Ref<const VectorXd> _y; 
  int _num_factors; 
  MatrixXdVisitor _f;
  std::vector<int> _non_zeros;
};


class RecCallerProbs {
public:
  RecCallerProbs(Ref<MatrixXd> x, int num_words, const Ref<const VectorXd> y, int num_factors,
            MatrixXdVisitor f, MatrixXd probs, double per): _x(x), _num_words(num_words), _y(y), _num_factors(num_factors), _f(f), _probs(probs), _per(per), _non_zeros() {
              for (int i = 0 ; i < _y.size() ; ++i) {
                if (y(i) > 0) {
                  _non_zeros.push_back(i);
                  _prob_thr.push_back(log(per)-log(binom(y(i)+_num_factors-1,_num_factors-1)));
                }
            }
  }
  bool rec_visit(const Ref<const VectorXd> v,  int w) {
    //std::cout << "w = " << w << std::endl;
    //std::cout << exp(logprob) << " " << exp(_prob_thr[w]) << std::endl;

    //std::cout << v.transpose() << std::endl;
    _x.row(_non_zeros[w]) = v;

    int z = w + 1;
    if (z < _non_zeros.size()) {
      auto f_next = std::bind(&RecCallerProbs::rec_visit, this, std::placeholders::_1, z);
      for_each_combination_VectorXd_prob(_y(_non_zeros[z]), _num_factors, _probs.row(_non_zeros[z]), f_next);
    } else {
      _f(_x);
    }
  }

  int non_zero_index(int i) {
    return _non_zeros[i];
  }
private:
  Ref<MatrixXd> _x;
  int _num_words;
  const Ref<const VectorXd> _y;
  std::vector<double> _prob_thr;
  int _num_factors; 
  MatrixXdVisitor _f;
  std::vector<int> _non_zeros;
  MatrixXd _probs;
  double _per;
};

class RecCallerMask {
  public:
  RecCallerMask(Ref<MatrixXd> x, int num_words, const Ref<const VectorXd> y, int num_factors,
                 MatrixXdVisitor f, Eigen::MatrixXi mask): _x(x), _num_words(num_words), _y(y), _num_factors(num_factors), _f(f), _mask(mask), _non_zeros() {
    for (int i = 0 ; i < _y.size() ; ++i) {
      if (y(i) > 0) {
        _non_zeros.push_back(i);
      }
    }
  }
  bool rec_visit(const Ref<const VectorXd> v, int w) {
    //std::cout << "w = " << w << " " << _non_zeros[w] << " in " << _non_zeros.size() << std::endl;
    _x.row(_non_zeros[w]) = v;
    int z = w + 1;
    if (z < _non_zeros.size()) {
      auto f_next = std::bind(&RecCallerMask::rec_visit, this, std::placeholders::_1, z);
      for_each_combination_VectorXd_mask(_y(_non_zeros[z]), _num_factors, _mask.row(_non_zeros[z]), f_next);
    } else {
      _f(_x);
    }
    return false;
  }
  int non_zero_index(int i) {
    return _non_zeros[i];
  }
  private:
  Ref<MatrixXd> _x;
  int _num_words;
  const Ref<const VectorXd> _y;
  double _prob_thr;
  int _num_factors;
  MatrixXdVisitor _f;
  std::vector<int> _non_zeros;
  Eigen::MatrixXi _mask;
};

long long compute_num_partitions_prob(const Ref<const VectorXd> y, int k, MatrixXd eta, double p_th) {

  std::vector<int> idx(k);
  long long n_combinations = 1;
  for (int i = 0 ; i < y.size() ; ++i) {
    std::cout << "Theoretical number of combinations ("<< i << ") " << n_combinations << std::endl;
    std::iota(idx.begin(), idx.end(),0);
    int k_new = determine_new_k(eta.row(i), idx, p_th);
    n_combinations *= binom(y(i) + k_new - 1 , k_new - 1);
  }
  //std::cout << "Theoretical number of combinations: " << n_combinations << std::endl;
  return n_combinations;
}

long long compute_num_partitions(const Ref<const VectorXd> y, int k) {
  long long n_combinations = 1;
  for (int i = 0 ; i < y.size() ; ++i) {
    //std::cout << "Theoretical number of combinations ("<< i << ") " << n_combinations << std::endl;
    n_combinations *= binom(y(i) + k - 1 , k - 1);
  }
  //std::cout << "Theoretical number of combinations: " << n_combinations << std::endl;
  return n_combinations;
}

int for_each_combination_MatrixXd_prob(const Ref<const VectorXd> y, int k, MatrixXd probs, double prob_thr, MatrixXdVisitor f) {
  int num_words = y.size();
  MatrixXd x;
  x.setZero(num_words, k);

  //int w = 0;
  RecCallerProbs rc(x, num_words, y, k, f, probs, prob_thr);
  auto f_next = std::bind(&RecCallerProbs::rec_visit, rc, std::placeholders::_1, 0);
  for_each_combination_VectorXd_prob(y(rc.non_zero_index(0)), k, probs.row(rc.non_zero_index(0)), f_next);
}

int for_each_combination_MatrixXd_mask(const Ref<const VectorXd> y, int k, Eigen::MatrixXi mask, MatrixXdVisitor f) {
  int num_words = y.size();
  MatrixXd x;
  x.setZero(num_words, k);

  //int w = 0;
  RecCallerMask rc(x, num_words, y, k, f, mask);
  auto f_next = std::bind(&RecCallerMask::rec_visit, rc, std::placeholders::_1, 0);
  for_each_combination_VectorXd_mask(y(rc.non_zero_index(0)), k, mask.row(rc.non_zero_index(0)), f_next);
}

int for_each_combination_MatrixXd(const Ref<const VectorXd> y, int k, MatrixXdVisitor f) {
  int num_words = y.size();
  MatrixXd x;
  x.setZero(num_words, k);
  
  //int w = 0;
  RecCaller rc(x, num_words, y, k, f);
  auto f_next = std::bind(&RecCaller::rec_visit, rc, std::placeholders::_1, 0);
  for_each_combination_VectorXd(y(rc.non_zero_index(0)), k, f_next);
}

//void ipartitions_size_up_to(int y, int num_factors, std::vector<VectorXd> &l, const Ref<const VectorXd> seed, int start, int size);
void ipartitions_size_up_to(int y, int num_factors, std::vector<VectorXd> &l, const Ref<const VectorXd> seed, int start, int size) {
    if (start == (size - 1)) {
        VectorXd new_p(seed);
        new_p(start) =  y;
        l.push_back(new_p);
    } else {
        for (int i = 0 ; i <= y; ++i) {
            VectorXd new_p(seed);
            new_p(start) =  i;
            ipartitions_size_up_to(y - i, num_factors, l, new_p, start + 1, size);
        }
    }
}

std::vector<VectorXd> partitions_size_up_to(int y, int num_factors) {
    VectorXd zeros(num_factors);
    std::vector<VectorXd> l;
    ipartitions_size_up_to(y, num_factors, l, zeros, 0, num_factors);
    return l;
}
