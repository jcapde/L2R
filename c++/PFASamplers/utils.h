#pragma once 
#include <vector>

#include "MyEigen.h"
#include "random.h"

void gamma_sampler(InEVector gam_shape, InEVector gam_scale, OutEVector theta, MTGen& gen);
 
void multinomial_sampler(int n, InEVector mult_para, OutEVector values, MTGen& gen);

void extract_indexes_vec(InEVector y, const std::vector<int> &indx_selected, OutEVector sel);

void extract_by_cols(InEMatrix M, const std::vector<int> &indx_selected, OutEMatrix sel);

int get_non_zeros(InEVector y);
void compute_non_zeros(InEVector y, std::vector<int> &non_zeros);
void compute_non_zeros_counts(InEVector y, std::vector<int> &non_zeros, OutEVector non_zeros_counts);

Eigen::PermutationMatrix<Eigen::Dynamic> compute_non_zeros_permutation(InEVector y, std::vector<int> &non_zeros);

bool relatively_equal(double x, double y);

EVector partial_sum(InEVector v);

void cumsum_rows(InEMatrix M, OutEMatrix CM);

void cumsum_cols(InEMatrix M, OutEMatrix CM);

double log_sum_exp_vec(InEVector v);

EVector log_sum_exp_rows(InEMatrix);

long long binom(int n, int k);

double log_binom(double n, double k);

double log_neg_bin_pdf(double k, double r, double p);

double log_gamma_pdf(double x, double k, double theta);

double log_poisson_pdf(double lambda, int k);

double log_poisson_pdf_wo_lambda(double lambda, int k);

double log_neg_multinomial_pdf(InEVector x, double k0, InEVector p);

double log_multinomial_pdf(InEVector x, int n, InEVector p);

double sparse_log_neg_multinomial_pdf(InEVector x, std::vector<int> &non_zeros, double k0,
                                      InEVector cache_log_p, InEVector cache_log_gamma);


double sparse_log_neg_multinomial_pdf_Xnonzeros(InEVector x, double k0,
                                      InEVector cache_log_p, InEVector cache_log_gamma);

void sort_indexes(EVector probs, std::vector<int> &idx, bool decreasing);
double log_factorial(int x);
double log_prob_multinomial(int n, InEVector p, InEVector x);