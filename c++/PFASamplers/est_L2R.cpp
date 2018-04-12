#include <set>
#include <algorithm>
#include <boost/math/special_functions/digamma.hpp>

#include "pfa_inference.h"
#include "exact.h"
#include "random.h"
#include "samplers.h" 
#include "utils.h"
#include "partitions.h"
#include "MyEigen.h"

using namespace boost::math;


class L2RInferenceV2 : public SamplingInference {
public:
  L2RInferenceV2() {
    name = "L2R";
    register_method();
  }
  
  virtual std::unique_ptr<PFAInferenceMethod> make_new() {
    std::unique_ptr<PFAInferenceMethod> p(new L2RInferenceV2());
    return p;
  }


  virtual void init(nlohmann::json params) {
    SamplingInference::init(params);
  }

  
  void log_p_y_word_given_previous_words_sampling(int w, int num_non_zeros, InEVector y, InEMatrix Phi, 
                                                          InEVector r, InEVector p, InEMatrix nbin_p, 
                                                          OutEMatrix xips, OutEVector log_p_y_w) {
    //std::cout << "Starting word " << w << std::endl;
    int num_words = y.size();
    int num_factors = xips.cols();
    //std::cout << "Num factors " << num_factors << std::endl;
    EVector pseudocounts(num_factors);
    EVector nbin_r(num_factors);
    //std::cout << "y=" << y(w) << std::endl;

    //xips.setZero();
    //std::cout << "xips: "<< xips << std::endl;

    EVector tmp_row(num_factors);
    EVector mult_para(num_factors);
    double sum_mult_para;
    double logW;
    double log_p_y;
    for (int s = 0 ; s < _num_samples ; s++) {
        
        //if (!(s%(_num_samples/10))) std::cout << "Sample " << s << std::endl;
        if (w > 0) gibbs_sampler(_num_gibbs_iter, y, w, Phi, r, p, xips, *_mt_gen_ptr);
        
        //std::cout << "Obtained sample" << xips.block(0, 0, w, num_factors) << std::endl;
        pseudocounts = xips.block(0, 0, w, num_factors).colwise().sum();


        //std::cout << "pseudocounts:" << pseudocounts << std::endl;
        nbin_r = r + pseudocounts;
        //std::cout << "Assess nbin_r" <<  nbin_r  << std::endl;
        if (w < num_non_zeros) {
          //std::cout << "Assessing word "<< w <<  std::endl;
          mult_para = (nbin_r.array() * nbin_p.col(w).array() * Phi.col(w).array()).matrix();
          sum_mult_para = mult_para.sum();
          mult_para /= sum_mult_para;
          
          EVector log_p_y_inner(_num_samples_inner);
          for(int sp = 0; sp < _num_samples_inner ; sp++){
              multinomial_sampler(y(w), mult_para, tmp_row, *_mt_gen_ptr);
              logW = - log_multinomial_pdf(tmp_row, y(w), mult_para);
              log_p_y = 0;
              for (int k = 0; k < num_factors ; ++k) {
                log_p_y += log_neg_bin_pdf(tmp_row(k), nbin_r(k), nbin_p(k, w));
              }
              log_p_y_inner(sp) = log_p_y + logW;
          }

          log_p_y_w(s) = log_sum_exp_vec(log_p_y_inner) - log(_num_samples_inner);

          xips.row(w) = tmp_row;
          //std::cout << "log_p_y "<< log_p_y <<  std::endl;
          //std::cout << "logW "<< logW <<  std::endl;
          
          //std::cout << "Computed probability of word " << w << " sample " << s<< ":" << log_p_y_w(s) << std::endl;
        } else {
          //std::cout << "Assessing last words"<< std::endl;
          //std::cout << "nbin_p(" << nbin_p.rows()  <<" , " << nbin_p.cols() <<  std::endl;
          auto sub5 = (1. - nbin_p.array()).block(0, num_non_zeros, num_factors, num_words - num_non_zeros).array().log().matrix();
          //std::cout << "Sub5:"<< sub5 << std::endl;
          log_p_y_w(s) = (sub5.transpose() * nbin_r  ).sum();
        }
    }
  }
  
  // VERIF
  void compute_nbin_p(InEMatrix Phi, InEVector p, OutEMatrix nbin_p) {
    int num_factors = Phi.rows();
    int num_words = Phi.cols();
    EMatrix Phi_row_cum(num_factors, num_words);
    cumsum_rows(Phi, Phi_row_cum); 
    //std::cout << "Phi_row_cum:"  << Phi_row_cum << std::endl;
    auto p_matrix = p.replicate(1, num_words);
    //std::cout << "3" <<std::endl;
    auto num = Phi.array() * p_matrix.array();
    //std::cout << "num(" << num.rows() << "," << num.cols() << ")" <<std::endl;
    auto denom = ((-p_matrix.array()  + Phi_row_cum.array() * p_matrix.array()).array() + 1).matrix();
    //std::cout << "denom(" << denom.rows() <<" , " << denom.cols() << ")" << std::endl;
    nbin_p =  (((num.matrix().cwiseQuotient(denom))).array() ).matrix();
    //std::cout << "nbin_p "<< nbin_p <<std::endl;
  }

//     
  //void log_p_y_zero_words_given_previous_words_sampling(y, num_non_zeros, Phi, r, p);

  virtual double log_p_ds(InEMatrix y, InEMatrix Phi, InEVector r, InEVector p) {
    int num_documents = y.rows();
    int num_factors = Phi.rows();
    int num_words = Phi.cols();

    std::vector<std::vector<int>> non_zeros(num_documents);
    std::vector<EVector> log_samples(num_documents);
    std::vector<EMatrix> log_p_y_s(num_documents);

    int count = 0;
    #pragma omp parallel for num_threads(_num_threads)
    for (int n = 0; n < num_documents; n++){
      #pragma omp critical
      {
        count++;
        if (!(count%(num_documents/10))) std::cout << "Document " << count << std::endl;
      };
      auto perm = compute_non_zeros_permutation(y.row(n), non_zeros[n]);
      long int num_non_zeros =  non_zeros[n].size();

      EMatrix xips(num_non_zeros, num_factors);
      xips.setZero();
      log_p_y_s[n] = EMatrix(_num_samples, num_non_zeros + 1);

      EVector y_perm = perm * y.row(n).transpose();
      EMatrix Phi_perm = (perm * Phi.transpose()).transpose();
      EMatrix nbin_p_perm(num_factors, num_words);
      compute_nbin_p(Phi_perm,  p, nbin_p_perm);


      for (int w = 0; w <= num_non_zeros; ++w) {
        log_p_y_word_given_previous_words_sampling(w,
                                                   num_non_zeros,
                                                   y_perm,
                                                   Phi_perm,
                                                   r,
                                                   p,
                                                   nbin_p_perm,
                                                   xips,
                                                   log_p_y_s[n].col(w));
      }
      log_samples[n] = EVector(num_non_zeros+1);
    }

    _log_ps.clear();
    _log_ps_ds.clear();


    int inc = std::max((int)std::floor( _num_samples / _num_partials), 1);
    std::cout << "Assessing partials at increments of "<< inc << std::endl;


    EVector doc_log_probs(num_documents);
    for (int i = inc ; i < _num_samples; i += inc) {
      #pragma omp parallel for num_threads(_num_threads)
      for (int n = 0; n < num_documents; n++) {
        long int num_non_zeros =  non_zeros[n].size();
        log_samples[n].fill(log((double) i));
        auto mat = log_p_y_s[n].block(0, 0, i, num_non_zeros + 1).transpose();
        doc_log_probs(n) = (log_sum_exp_rows(mat) - log_samples[n]).array().sum();
      }
      _log_ps_ds[i] = doc_log_probs;
    }

    #pragma omp parallel for num_threads(_num_threads)
    for (int n = 0; n < num_documents; n++) {
        log_samples[n].fill(log((double) _num_samples));
        auto mat = log_p_y_s[n].transpose();
        doc_log_probs(n) = (log_sum_exp_rows(mat) - log_samples[n]).array().sum();
    }
    _log_ps_ds[_num_samples] = doc_log_probs;

    for (auto it = _log_ps_ds.cbegin() ; it != _log_ps_ds.end() ; ++it) {
      _log_ps[it->first] = it->second.sum();
    }

    return _log_ps_ds[_num_samples].sum();
  }

  virtual double log_p(InEVector y, InEMatrix Phi, InEVector r, InEVector p) {
    int num_documents = 1;
    int num_factors = Phi.rows();
    int num_words = Phi.cols();

    std::vector<int> non_zeros;
    auto perm = compute_non_zeros_permutation(y, non_zeros);
    int num_non_zeros = non_zeros.size();
    //std::cout << "perm("<< perm.rows() <<") :" << perm.indices().transpose() << std::endl;
    //std::cout << "Non_zeros:" << num_non_zeros << std::endl;
    
    
    EMatrix xips(num_non_zeros, num_factors);
    xips.setZero();
    EVector sample_log_p(_num_samples);
    EVector log_p_y_zeros(_num_samples);
    EMatrix log_p_y_s(_num_samples, num_non_zeros + 1); // Space for the log_p_y of each y different from zero + 1 for the rest of the y's
    
    //std::cout << "y:" << y.transpose() << std::endl;
    EVector y_perm = perm * y;
    //std::cout << "y_perm:" << y_perm.transpose() << std::endl;
    //std::cout << "Phi:" << Phi << std::endl;
    EMatrix Phi_perm = (perm * Phi.transpose()).transpose();
    //std::cout << "Phi_perm:" << Phi_perm << std::endl;
    //EVector r_perm = perm * r;
    //std::cout << "r_perm:" << r_perm << std::endl;
    //EVector p_perm = perm * p;
    //std::cout << "Phi_perm:" << Phi_perm << std::endl;
    EMatrix nbin_p_perm(num_factors, num_words);
    compute_nbin_p(Phi_perm, p, nbin_p_perm);
    for (int w = 0 ; w <= num_non_zeros  ; ++w) {
      //std::cout << "Word "<< w << std::endl;
      log_p_y_word_given_previous_words_sampling(w, num_non_zeros,  y_perm,
                                                         Phi_perm,
                                                         r,  
                                                         p, 
                                                         nbin_p_perm,
                                                         xips, 
                                                         log_p_y_s.col(w));
    }
    //std::cout << "log_p_ys:" << log_p_y_s << std::endl;

    EVector log_samples(num_non_zeros+1);
    log_samples.fill(log((double)_num_samples));

    //std::cout <<  "log_sum_exp - log_samples" << log_sum_exp_rows(log_p_y_s.transpose()) - log_samples  << std::endl; 
    //std::cout <<  "res " << (log_sum_exp_rows(log_p_y_s.transpose()) - log_samples).array().sum() << std::endl;
    
    //sample_log_p = log_p_y_s.rowwise().sum();
    //std::cout << sample_log_p << std::endl;
    _log_ps.clear();
    
    int inc = std::max((int)std::floor( _num_samples / _num_partials), 1);   
    //std::cout << "Assessing partials at increments of "<< inc << std::endl;
    for (int i = inc ; i < _num_samples; i += inc) {
      log_samples.fill(log((double)i));
      _log_ps[i] = (log_sum_exp_rows(log_p_y_s.block(0,0,i,num_non_zeros+1).transpose()) - log_samples).array().sum();
    } 
    //if (inc * _num_partials  != _num_samples) {
      log_samples.fill(log((double)_num_samples));
      _log_ps[_num_samples] = (log_sum_exp_rows(log_p_y_s.transpose()) - log_samples).array().sum();
    //}
    //std::cout <<  "res " << _log_ps[_num_samples] << std::endl;
    return _log_ps[_num_samples];
  }
  
  virtual ~L2RInferenceV2() {};

private:
  std::unique_ptr<EMatrix> _xips_ptr;
};

static L2RInferenceV2 l2r_inference_v2_exemplar;
