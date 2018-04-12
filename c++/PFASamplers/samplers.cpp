#include <iostream>
#include <unordered_map>
#include <map>
#include <boost/functional/hash.hpp>
#include <omp.h>

#include "samplers.h"
#include "utils.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Ref;

template<typename T>
struct matrix_hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const& matrix) const {
    size_t seed = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      boost::hash_combine(seed, elem);
    }
    return seed;
  }
};

void theta_given_x_sampler(OutEVector thetas, InEVector r, InEVector p, InEMatrix xips, MTGen& gen){
  int num_factors = r.size();
  InEVector gam_shape = xips.rowwise().sum().transpose() + r.transpose();
  gamma_sampler(gam_shape, p, thetas, gen);
}

void x_given_theta_sampler(OutEMatrix xips, InEVector non_zeros_counts, InEMatrix new_Phi, InEVector theta, MTGen& gen ){

  int num_factors = new_Phi.rows();
  int num_non_zeros = new_Phi.cols();

  EVector tmp_row(num_factors);
  EVector mult_para(num_factors);
  EVector log_psi(num_factors);

  for(int iw = 0; iw < num_non_zeros; ++iw){
    log_psi = new_Phi.col(iw).array().log() + theta.array().log();
    mult_para = (log_psi.matrix() - VectorXd::Constant(num_factors, log_sum_exp_vec(log_psi.matrix()))).array().exp().matrix();
    multinomial_sampler(non_zeros_counts(iw),  mult_para, tmp_row, gen);
    xips.col(iw) = tmp_row;
  }
}

void gibbs_sample(InEVector non_zeros_counts, InEMatrix new_Phi, InEVector r, InEVector p, InEVector thetao, MTGen& gen, OutEMatrix xip, OutEVector theta){

  if(theta.isZero(0)){
    x_given_theta_sampler(xip, non_zeros_counts, new_Phi, thetao, gen);
  }else{
    x_given_theta_sampler(xip, non_zeros_counts, new_Phi, theta, gen);
  }

  theta_given_x_sampler(theta, r, p, xip, gen);

}

posterior_sampler post_sample(int burn_in_iter, InEVector non_zeros_counts, InEMatrix new_Phi, InEVector r, InEVector p, MTGen& gen){

  int num_factors = new_Phi.rows();
  int num_non_zeros = new_Phi.cols();

  MatrixXd xip(num_factors, num_non_zeros);
  VectorXd thetao(num_factors);
  VectorXd ones(num_factors);
  ones.fill(1.);

  gamma_sampler(r,  p.cwiseQuotient(ones - p), thetao, gen);
  for(int i = 0; i < burn_in_iter; i++){
    x_given_theta_sampler(xip, non_zeros_counts, new_Phi, thetao, gen);
    theta_given_x_sampler(thetao, r, p, xip, gen);
  }

  return std::bind(gibbs_sample, non_zeros_counts, new_Phi, r, p, thetao, gen, std::placeholders::_1, std::placeholders::_2);
}

void gibbs_sampler(int num_Gibbs_iterations, InEVector y, int w, InEMatrix Phi, InEVector r, InEVector p,  OutEMatrix xips,  MTGen& gen) {
      int num_factors = r.size();
      
      EVector gam_shape(num_factors);
      //xips.setZero();
      //std::cout << "xips: "<< xips << std::endl;
      EVector gam_scale(num_factors);
      //std::cout <<"Phi_block: " << Phi.block(0, 0, num_factors, w)<< std::endl;
      auto sub1 = Phi.block(0, 0, num_factors, w).rowwise().sum();
      //std::cout << "sub1:" << sub1 << std::endl;
      auto sub2 = (p.array() * sub1.array()).matrix();
      //std::cout << "sub2:" << sub2 << std::endl;
      auto sub3 = (1 - p.array()).matrix();
      //std::cout << "sub3:" << sub3 << std::endl;
      auto denom = sub3 + sub2;
      //std::cout << "denom:" << denom << std::endl;
      gam_scale = p.cwiseQuotient(denom.matrix());
      //std::cout << "gam_scale:" << gam_scale.transpose() << std::endl;
      
      EVector theta(num_factors);
      EVector mult_para(num_factors);
      double sum_mult_para;
      EVector tmp_row(num_factors); 
      for (int i = 0 ; i < num_Gibbs_iterations; ++i) {
            auto sub4 = xips.block(0, 0, w, num_factors).colwise().sum().transpose();
            //std::cout << "Sub4: "<< sub4 << std::endl;
            gam_shape = r + sub4;
            //std::cout << "gam_shape:" << gam_shape.transpose() << std::endl;
            gamma_sampler(gam_shape, gam_scale, theta, gen);
            //std::cout << "theta:" << theta.transpose() << std::endl; 
            for (int wp = 0 ; wp < w ; ++wp) {
                mult_para = (Phi.col(wp).array() * theta.array()).matrix();
                sum_mult_para = mult_para.sum();
                mult_para /= sum_mult_para;
                //std::cout << "mult_para("<< wp <<"):" << mult_para.transpose() << std::endl; 
                multinomial_sampler(y(wp), mult_para, tmp_row, gen);
                //std::cout << "multinomial sample ("<< y(wp) << "):" << tmp_row << std::endl;
                xips.row(wp) = tmp_row;
            }
      }
}

void real_prior_sampler(std::vector<std::vector<std::gamma_distribution<double>>> &distributions, 
                        MTGen& gen, Eigen::Ref<Eigen::VectorXd> logw, Eigen::Ref<Eigen::MatrixXd> theta) {
    int rank = omp_get_thread_num();
    std::vector<std::gamma_distribution<double>> &this_thread_distributions = distributions[rank];
    //std::cout << "Correctly obtained distributions " << std::endl;
    for (int d = 0 ; d < theta.rows(); d++) {
        logw(d) = 0.0;
        for (int f = 0 ; f < theta.cols(); f++) {
            theta(d,f) = (this_thread_distributions[f])(gen.get(rank));
        }
    }
}

theta_sampler_ds prior_sampler(const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p, MTGen& gen) {
    // Prepare the distributions from which we will sample
    int num_threads = gen.num_threads();
    //std::cout << "Creating distributions for "<< num_threads <<" threads"<<std::endl;
    std::vector<std::vector<std::gamma_distribution<double>>> distributions(num_threads);
    for (int t = 0; t < num_threads ; t++) {
        std::vector<std::gamma_distribution<double>> this_thread_distributions(r.rows());
        for (int f = 0 ; f < r.rows(); f++) {
            double shape = r(f);
            double scale = p(f)/(1.0-p(f));
            assert(shape>0);
            assert(scale>0);
            //std::cout << "shape:" << shape << " scale:" << scale << std::endl;
            std::gamma_distribution<double> distribution(shape, scale);
            this_thread_distributions[f] = distribution;
        }
        distributions[t] = this_thread_distributions;
    }
    //auto f = std::bind(real_prior_sampler<URNG>, _1, std::cref(distributions));
    return std::bind(real_prior_sampler, distributions, gen, std::placeholders::_1, std::placeholders::_2);
}
         

void log_p_y_given_theta_phi(const Ref<const MatrixXd> y, std::vector<std::vector<int>> non_zeros, const Ref<const MatrixXd> Phi,
                             const Ref<const MatrixXd> theta, Ref<VectorXd> log_p) {
  int num_documents = y.rows();
  int num_factors = Phi.rows();
  int num_words = Phi.cols();

  MatrixXd poisson_means = theta * Phi;
  for (int d = 0 ; d < num_documents; d++) {
    log_p(d) = 0.0;
    for (int w : non_zeros[d]) {
      log_p(d) += log_poisson_pdf_wo_lambda(poisson_means(d,w), y(d,w));
    }
    log_p(d) -= poisson_means.row(d).sum();
  }
}

double compute_sum_partial(const Ref<const MatrixXd> log_p, const Ref<const MatrixXd> weights, int doc_id, int partial_num_samples) {
  auto mat = log_p.block(doc_id, 0, 1, partial_num_samples-1) + weights.block(doc_id, 0, 1, partial_num_samples-1);
  //std::cout << "mat" << mat << std::endl;
  auto log_p_docs = log_sum_exp_rows(mat);
  //std::cout << "log_p_docs" << log_p_docs << std::endl;
  auto norm = log_sum_exp_rows(weights.block(doc_id, 0, 1, partial_num_samples-1));
  //std::cout << "norm" << norm << std::endl;
  return log_p_docs(0) - norm(0);
}

double compute_sum_partial_vDiscrete(const Ref<const VectorXd> log_p, const Ref<const VectorXd> weights, int partial_num_samples) {
  //std::cout << "partial_num_samples" << partial_num_samples << std::endl;
  //std::cout << "weights" << weights << std::endl;

  auto mat = log_p.head(partial_num_samples) + weights.head(partial_num_samples);
  //std::cout << "mat" << mat << std::endl;
  double log_p_docs = log_sum_exp_vec(mat);

  //std::cout << "log_p_docs" << log_p_docs << std::endl;
  //auto norm = log(partial_num_samples);
  return log_p_docs - log(partial_num_samples);
}

double compute_sum_partial_HM(const Ref<const VectorXd> log_p, int partial_num_samples) {

  auto mat = - log_p.head(partial_num_samples);

  double log_p_docs = log_sum_exp_vec(mat);

  return log(partial_num_samples) - log_p_docs;
}

double compute_sum_partial_vDiscrete_ds(const Ref<const MatrixXd> log_p, const Ref<const MatrixXd> weights, int doc_id, int partial_num_samples) {
    //std::cout << "partial_num_samples" << partial_num_samples << std::endl;
    //std::cout << "weights" << weights << std::endl;

    auto mat = log_p.block(doc_id, 0, 1, partial_num_samples-1) + weights.block(doc_id, 0, 1, partial_num_samples-1);
    //std::cout << "mat" << mat << std::endl;
    VectorXd log_p_docs = log_sum_exp_rows(mat);
    //std::cout << "log_p_docs" << log_p_docs << std::endl;
    //auto norm = log(partial_num_samples);
    return log_p_docs(0) - log(partial_num_samples);
}


std::unordered_map<int, Eigen::VectorXd> importance_sampling(const Ref<const MatrixXd> y, std::vector<std::vector<int>> non_zeros, const Ref<const MatrixXd> Phi,
                         int num_samples, int num_partials, theta_sampler_ds sampler, int num_threads) {

    int num_words = Phi.cols();
    int num_factors = Phi.rows();
    int num_documents = y.rows();
    
    MatrixXd log_p(num_documents, num_samples);
    MatrixXd weights(num_documents, num_samples);

    std::vector<MatrixXd> thetas;
    for (int i = 0 ; i < num_threads ; i++) {
        MatrixXd theta(num_documents, num_factors);
        thetas.push_back(theta);
    }

    int count = 0;
    #pragma omp parallel for num_threads(num_threads)
    for (int s = 0; s < num_samples; s++) {
      #pragma omp critical
      {
        count++;
        if (!(count%(num_samples/10))) std::cout << "Sample " << count << std::endl;
      };
      int rank = omp_get_thread_num();
      sampler(weights.col(s), thetas[rank]);
      log_p_y_given_theta_phi(y, non_zeros, Phi, thetas[rank], log_p.col(s));
    }

    std::cout << "Assessing " << num_partials << " partials" << std::endl;
    std::unordered_map<int, Eigen::VectorXd> log_probs;
    
    int inc = std::floor(num_samples/num_partials);
    VectorXd doc_log_probs(num_documents);

    for (int s = inc; s <= num_samples; s+=inc) {
      #pragma omp parallel for num_threads(num_threads)
      for(int n = 0; n < num_documents; n++){
          doc_log_probs(n) = compute_sum_partial(log_p, weights, n, s);
      }
      log_probs[s] = doc_log_probs;
    }
    if (inc * num_partials  != num_samples) {
      #pragma omp parallel for num_threads(num_threads)
      for (int n = 0; n <= num_documents; n++) {
        doc_log_probs(n) = compute_sum_partial(log_p, weights, n, num_samples);

      }
      log_probs[num_samples] = doc_log_probs;
    }

    return log_probs;
}



void log_p_xnp_given_p_r_phi(const Ref<const VectorXd> r, const Ref<const MatrixXd> cache_log_p,
                             const Ref<const VectorXd> cache_log_gamma, const Ref<const MatrixXd> xnp, double& log_p) {

  int num_factors = cache_log_p.rows();
  log_p = 0.0;
  for (int k = 0; k < num_factors; k++) {
    log_p += sparse_log_neg_multinomial_pdf_Xnonzeros(xnp.row(k), r(k), cache_log_p.row(k),  cache_log_gamma);
  }

}


void hm(const Ref<const VectorXd> non_zeros_counts, const Ref<const MatrixXd> Phi,
        const Ref<const MatrixXd> new_Phi,
        const Ref<const VectorXd> theta, double& log_p) {

  int num_non_zeros = non_zeros_counts.size();
  log_p = 0;
  for(int iw = 0; iw < num_non_zeros; iw++){
    log_p += log_poisson_pdf_wo_lambda(theta.transpose()*new_Phi.col(iw),non_zeros_counts[iw]);
  }
  log_p -= (theta.transpose()*Phi).sum();
  //std::cout << "log_p:" << log_p << std::endl;

}

std::unordered_map<int,double> hm_sampling(const Ref<const VectorXd> non_zeros_counts,
                                           const Ref<const MatrixXd> Phi, const Ref<const MatrixXd> new_Phi,  const Ref<const VectorXd> r,
                                           const Ref<const VectorXd> p, int num_samples, int num_partials,
                                           posterior_sampler sampler) {

  int num_factors = new_Phi.rows();
  int num_non_zeros = new_Phi.cols();

  VectorXd log_p(num_samples);
  VectorXd theta(num_factors);
  MatrixXd xip(num_factors, num_non_zeros);
  theta.setZero();
  xip.setZero();
  for (int s = 0; s < num_samples; s++) {
    sampler(xip, theta);
    hm(non_zeros_counts, Phi, new_Phi, theta, log_p(s));
  }

  std::unordered_map<int, double> log_probs;
  int inc = std::floor(num_samples/num_partials);
  for (int s = inc; s < num_samples; s += inc) {
    log_probs[s] = compute_sum_partial_HM(log_p,  s);
  }
  log_probs[num_samples] = compute_sum_partial_HM(log_p, num_samples);

  return log_probs;
}


