//
// Created by Joan Capdevila Pujol on 15/3/18.
//


#include <set>
#include <Eigen/Dense>
#include "pfa_inference.h"
#include "samplers.h"
#include "utils.h"

using Eigen::Ref;
using Eigen::VectorXd;
using Eigen::MatrixXd;

class HMSampling : public SamplingInference {
  public:
  HMSampling() {
    name = "HM";
    register_method();
    _burn_in = 100;
  }

  virtual std::unique_ptr<PFAInferenceMethod> make_new() {
    std::unique_ptr<PFAInferenceMethod> p(new HMSampling());
    return p;
  }

  virtual void init(nlohmann::json params) {
    SamplingInference::init(params);
    if (params.find("burn_in") != params.end()) {
      _burn_in = params["burn_in"].get<int>();
    }
  }


  virtual nlohmann::json get_additional_info() {
    nlohmann::json j;
    nlohmann::json log_ps = nlohmann::json({});

    std::set<int> samples;
    for (auto it = _log_ps.cbegin() ; it != _log_ps.end() ; ++it) {
      samples.emplace(it->first);
    }

    j["sample_sizes"] = samples;
    std::vector<double> values;
    std::vector<std::vector<double>> values_doc;

    for (int sample_size : samples) {
      values.push_back(_log_ps[sample_size]);
      std::vector<double> aux(_log_ps_ds[sample_size].data(), _log_ps_ds[sample_size].data() + _log_ps_ds[sample_size].rows()*_log_ps_ds[sample_size].cols()) ;
      values_doc.push_back(aux);
    }

    j["values"] = values;
    j["value"] = values[values.size()-1];
    j["values_doc"] = values_doc;
    j["value_doc"] = values_doc[values_doc.size()-1];
    return j;
  }


  virtual double log_p_ds(const Ref<const MatrixXd> y, const Ref<const MatrixXd> Phi, const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
    int num_documents = y.rows();
    int num_factors = Phi.rows();
    int num_words = Phi.cols();

    std::vector<std::unordered_map<int, double>> output(num_documents);

    int count = 0;
    #pragma omp parallel for num_threads(_num_threads)
    for(int n = 0; n < num_documents; n++){

      int num_non_zeros = get_non_zeros(y.row(n));
      //std::cout << "num_non_zeros: " << num_non_zeros << std::endl;
      //std::cout << "y: " << y.row(n) << std::endl;
      std::vector<int> non_zeros;
      VectorXd non_zeros_counts(num_non_zeros);


      compute_non_zeros_counts(y.row(n), non_zeros, non_zeros_counts);

      MatrixXd new_Phi(num_factors, num_non_zeros);
      for(int iw = 0; iw < num_non_zeros; ++iw){
        int w = non_zeros[iw];
        new_Phi.col(iw) = Phi.col(w);
      }

      if(_num_threads < num_documents) {
      #pragma omp critical
        {
          if (!(count%(num_documents/10))) std::cout << "Document " << count << std::endl;
          count++;
        };
      }

      posterior_sampler sampler = post_sample(_burn_in, non_zeros_counts, new_Phi, r, p, *_mt_gen_ptr);
      output[n] =  hm_sampling(non_zeros_counts, Phi, new_Phi, r, p, _num_samples, _num_partials,  sampler);
    }

    std::cout << "Regenerating partials" << std::endl;
    int inc = std::floor(_num_samples/_num_partials);
    for (int s = inc; s < _num_samples; s += inc) {
      _log_ps_ds[s] = VectorXd(num_documents);
      _log_ps[s] = 0.0;
    }
    _log_ps_ds[_num_samples] = VectorXd(num_documents);
    _log_ps[_num_samples] = 0.0;

    for(int n = 0; n < num_documents; n++){
      for (int s = inc; s < _num_samples; s += inc) {
        _log_ps_ds[s](n) = output[n][s];
        _log_ps[s] += output[n][s];
      }
      _log_ps_ds[_num_samples](n) = output[n][_num_samples];
      _log_ps[_num_samples] += output[n][_num_samples];
    }


    return _log_ps[_num_samples];

  }

  virtual double log_p(const Ref<const VectorXd> y, const Ref<const MatrixXd> Phi, const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
    int num_documents = 1;
    int num_factors = Phi.rows();
    int num_words = Phi.cols();


    std::vector<int> non_zeros;
    int num_non_zeros = get_non_zeros(y);

    VectorXd non_zeros_counts(num_non_zeros);
    compute_non_zeros_counts(y, non_zeros, non_zeros_counts);

    MatrixXd new_Phi(num_factors, num_non_zeros);
    for(int iw = 0; iw < num_non_zeros; ++iw){
      int w = non_zeros[iw];
      new_Phi.col(iw) = Phi.col(w);
    }

//    std::vector<EVector> thetas(_num_samples);
//    std::vector<EMatrix> xips(_num_samples);
//    gibbs_sampling(_num_samples, _burn_in, non_zeros_counts, Phi, new_Phi, r, p, thetas, xips, *_mt_gen_ptr);
//    _log_ps =  hm_sampling(non_zeros_counts, Phi, new_Phi, r, p, _num_samples, _num_partials,  thetas);

    posterior_sampler sampler = post_sample(_burn_in, non_zeros_counts, new_Phi, r, p, *_mt_gen_ptr);
    _log_ps =  hm_sampling(non_zeros_counts, Phi, new_Phi, r, p, _num_samples, _num_partials,  sampler);

    return _log_ps[_num_samples];
  }

  virtual ~HMSampling() {};

  private:
  int _burn_in;
};

static HMSampling hm_exemplar;
