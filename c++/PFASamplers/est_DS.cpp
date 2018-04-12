#include <set>
#include <Eigen/Dense>
#include "pfa_inference.h"
#include "samplers.h"
#include "utils.h"

using Eigen::Ref;
using Eigen::VectorXd;
using Eigen::MatrixXd;

class PriorImportanceSamplingInference : public SamplingInference {
public:
  PriorImportanceSamplingInference() {
    name = "DS";
    register_method();
  }
  
  virtual std::unique_ptr<PFAInferenceMethod> make_new() {
    std::unique_ptr<PFAInferenceMethod> p(new PriorImportanceSamplingInference());
    return p;
  }
  
  virtual void init(nlohmann::json params) {
    SamplingInference::init(params);
  }


  virtual double log_p_ds(const Ref<const MatrixXd> y, const Ref<const MatrixXd> Phi, const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
    int num_documents = y.rows();
    int num_words = Phi.cols();

    auto sampler = prior_sampler(r, p, *_mt_gen_ptr);

    std::vector<std::vector<int>> non_zeros(num_documents);

    for(int n = 0; n< num_documents; n++){
      compute_non_zeros(y.row(n), non_zeros[n]);
    }

    _log_ps_ds = importance_sampling(y, non_zeros, Phi, _num_samples, _num_partials, sampler, _num_threads);

    for (auto it = _log_ps_ds.cbegin() ; it != _log_ps_ds.end() ; ++it) {
      _log_ps[it->first] = it->second.sum();
    }
    return _log_ps[_num_samples];
  }
  
  virtual double log_p(const Ref<const VectorXd> y, const Ref<const MatrixXd> Phi, const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
    auto sampler = prior_sampler(r, p, *_mt_gen_ptr);
    int num_words = y.size();
    MatrixXd yM(1, num_words);
    yM.row(0) = y;
    std::vector<std::vector<int>> non_zeros(1);
    compute_non_zeros(y, non_zeros[0]);

    _log_ps_ds = importance_sampling(yM, non_zeros, Phi, _num_samples, _num_partials, sampler, _num_threads);
    return _log_ps_ds[_num_samples](0);
  }
  
  virtual ~PriorImportanceSamplingInference() {};
};

static PriorImportanceSamplingInference prior_importance_sampling_inference_exemplar;