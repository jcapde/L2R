#include <set>
#include <Eigen/Dense>
#include "pfa_inference.h"
#include "utils.h"
#include "exact.h"

using Eigen::Ref;
using Eigen::VectorXd;
using Eigen::MatrixXd;

class ExactInference : public PFAInferenceMethodExemplar {
public:
  ExactInference() {
    name = "Exact";
    register_method();
  }
  virtual std::unique_ptr<PFAInferenceMethod> make_new() {
    std::unique_ptr<PFAInferenceMethod> p(new ExactInference());
    return p;
  }
  virtual void init(nlohmann::json params) {
    if (params.find("num_threads") != params.end()) {
      _num_threads = params["num_threads"].get<int>();
    }
  };

  virtual double log_p_ds(const Ref<const MatrixXd> y, const Ref<const MatrixXd> Phi, const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
    int num_documents = y.rows();
    log_p_value_ds = VectorXd(num_documents);
    log_p_value = 0;
    int count = 0;

    #pragma omp parallel for num_threads(_num_threads) if(_num_threads<num_documents)
    for(int n = 0; n < num_documents; n++){
      if(_num_threads < num_documents) {
        #pragma omp critical
        {
            if (!(count % (num_documents / 10))) std::cout << "Document " << count << std::endl;
            count++;
        };
      }
      log_p_value_ds(n) = log_p_y_exact_doc(y.row(n), Phi, r, p);
      log_p_value += log_p_value_ds(n);
    }
    return log_p_value;
  };

  virtual double log_p(const Ref<const VectorXd> y, const Ref<const MatrixXd> Phi, const Ref<const VectorXd> r, const Ref<const VectorXd> p) {
    log_p_value = log_p_y_exact_doc(y, Phi, r, p);
    return log_p_value;
  };

  virtual nlohmann::json get_additional_info() {
    nlohmann::json j;
    j["value"] = log_p_value;
    j["value_doc"]  = std::vector<double>(log_p_value_ds.data(), log_p_value_ds.data() + log_p_value_ds.rows()*log_p_value_ds.cols());
    return j;
  }

  virtual ~ExactInference() {};
private:
  double log_p_value;
  VectorXd log_p_value_ds;
};

static ExactInference exact_inference_exemplar;

