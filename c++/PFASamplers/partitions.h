#pragma once
#include <vector>
#include <tuple>
#include <stack>
#include <functional>
#include <Eigen/Dense>

/*
typedef std::tuple<int, int, const Eigen::Ref<const Eigen::VectorXd>, int, int> CallInfo;

typedef class {
  CallInfo current;
  int i;
  std::stack<CallInfo> stack;
  
  PartitionsIt(CallInfo first_call): current(first_call), stack(), i(0) {};
  bool end() {return stack.empty() && (i == std::get<0>(current));}
  bool next() {}
} PartitionsIt;

// Returns a pseudo-iterator (this should be adequately implemented using boost::iterator) over the partitions of y in num_factors where each partition is 
PartitionsIt partitions_size_up_to_it(int y, int num_factors);*/


typedef std::function<bool(const Eigen::Ref<const Eigen::VectorXd>)> VectorXdVisitor;
typedef std::function<bool(const Eigen::Ref<const Eigen::VectorXd>, double)> VectorXdVisitor2;
typedef std::function<bool(const Eigen::Ref<const Eigen::MatrixXd>)> MatrixXdVisitor;
typedef std::function<bool(const Eigen::Ref<const Eigen::MatrixXd>, const Eigen::Ref<const Eigen::VectorXd>)> MatrixXdVisitor2;


void for_each_combination_VectorXd(int n, int k, VectorXdVisitor f);
void for_each_combination_VectorXd_prob(int n, int k, Eigen::VectorXd probs, double prob_thr, VectorXdVisitor f);
long long compute_num_partitions(const Eigen::Ref<const Eigen::VectorXd> y, int k);
long long  compute_num_partitions_prob(const Eigen::Ref<const Eigen::VectorXd> y, int k, Eigen::MatrixXd eta, double p_th);

int for_each_combination_MatrixXd_mask(const Eigen::Ref<const Eigen::VectorXd> y, int k, Eigen::MatrixXi mask,  MatrixXdVisitor f);
int for_each_combination_MatrixXd(const Eigen::Ref<const Eigen::VectorXd> y, int k, MatrixXdVisitor f);
int for_each_combination_MatrixXd_prob(const Eigen::Ref<const Eigen::VectorXd> y, int k, Eigen::MatrixXd probs, double prob_thr, MatrixXdVisitor f);
std::vector<Eigen::VectorXd> partitions_size_up_to(int y, int num_factors);