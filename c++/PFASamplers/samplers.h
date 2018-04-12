#pragma once
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <functional>
#include <unordered_map>
#include "random.h"
#include "MyEigen.h"

// A theta_sampler leaves the per document weights of the samples in the first vector and the 
// per document thetas in the matrix.

typedef std::function<void(OutEMatrix, OutEVector)> posterior_sampler;
typedef std::function<void(Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::MatrixXd>)> theta_sampler_ds;

posterior_sampler post_sample(int burn_in_iter, InEVector non_zeros_counts, InEMatrix new_Phi, InEVector r, InEVector p, MTGen& gen);

theta_sampler_ds prior_sampler(const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p, MTGen& gen);

void gibbs_sampler(int num_Gibbs_iterations, InEVector y, int w, InEMatrix Phi, InEVector r, InEVector p,  OutEMatrix xips,  MTGen& gen);

std::unordered_map<int, Eigen::VectorXd> importance_sampling(const Eigen::Ref<const Eigen::MatrixXd> y, std::vector<std::vector<int>> non_zeros, 
									const Eigen::Ref<const Eigen::MatrixXd> Phi, int num_samples, int num_partials, theta_sampler_ds sampler, int num_threads);

std::unordered_map<int,double> hm_sampling(InEVector non_zeros_counts,  InEMatrix Phi, InEMatrix new_Phi, InEVector r,
								 	InEVector p, int num_samples, int num_partials, posterior_sampler sampler);


