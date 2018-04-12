#include <set>
#include <Eigen/Dense>
#include <memory>

#include "pfa_inference.h"

#include "random.h"
#include "samplers.h"
#include "utils.h"

using Eigen::Ref;
using Eigen::VectorXd;
using Eigen::MatrixXd;

void PFAInferenceMethodExemplar::register_method() {
    PFAInferenceFactory::register_method(name, this);
}

std::unordered_map<std::string,PFAInferenceMethodExemplar*>  PFAInferenceFactory::_exemplars;
std::vector<std::string> PFAInferenceFactory::_registered_methods;

std::string run_pfa_inference_ds(std::string name, std::string params, const Eigen::Ref<const Eigen::MatrixXd> y, const Eigen::Ref<const Eigen::MatrixXd> Phi,
                              const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p) {
  auto m = PFAInferenceFactory::get(name, nlohmann::json::parse(params));
  m->log_p_ds(y, Phi, r, p);
  return m->get_additional_info().dump();
}

std::string run_pfa_inference(std::string name, std::string params, const Eigen::Ref<const Eigen::VectorXd> y, const Eigen::Ref<const Eigen::MatrixXd> Phi, 
                       const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p) {
  auto m = PFAInferenceFactory::get(name, nlohmann::json::parse(params));
  m->log_p(y, Phi, r, p);
  return m->get_additional_info().dump();
}
