#include <iostream>
#include <string>
#include <unordered_map>

#include <Eigen/Dense>

#include "json.hpp"

class PFAInferenceMethod {
public:
  PFAInferenceMethod(): _num_threads(8) {}
  virtual void init(nlohmann::json params) = 0;
  virtual double log_p(const Eigen::Ref<const Eigen::VectorXd> y, const Eigen::Ref<const Eigen::MatrixXd> Phi, 
                       const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p) = 0;
  virtual double log_p_ds(const Eigen::Ref<const Eigen::MatrixXd> y, const Eigen::Ref<const Eigen::MatrixXd> Phi,
                          const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p) = 0;
  virtual nlohmann::json get_additional_info() {return nlohmann::json({});};
  virtual ~PFAInferenceMethod() {};
  int _num_threads;
};

class PFAInferenceMethodExemplar : public PFAInferenceMethod {
public:
  virtual void register_method(); 
  virtual std::unique_ptr<PFAInferenceMethod> make_new() = 0;
  virtual ~PFAInferenceMethodExemplar() {};
  std::string name;
};

class PFAInferenceFactory {
public:
  static std::unique_ptr<PFAInferenceMethod> get(std::string name, nlohmann::json params) {
    auto it = PFAInferenceFactory::_exemplars.find(name);
    if (it != PFAInferenceFactory::_exemplars.end()) {
      std::unique_ptr<PFAInferenceMethod> m = PFAInferenceFactory::_exemplars[name]->make_new();
      m->init(params);
      return m;
    } else {
      std::cout << "Inference method " << name << " not registered" << std::endl;
      return std::unique_ptr<PFAInferenceMethod>(nullptr);
    }
  };
  
  static void register_method(std::string name, PFAInferenceMethodExemplar *method) {
    auto it = PFAInferenceFactory::_exemplars.find(name);
    //std::cout << name << std::endl;
    if (it == PFAInferenceFactory::_exemplars.end()) {
      //std::cout << "Inserting exemplar for " << name << std::endl;
      PFAInferenceFactory::_exemplars[name] = method;
      _registered_methods.push_back(name);
    }
  }
  const std::vector<std::string> registered_methods() {return _registered_methods;}
private:
  static std::unordered_map<std::string, PFAInferenceMethodExemplar*> _exemplars;
  static std::vector<std::string> _registered_methods;
};


std::string run_pfa_inference(std::string name, std::string params, const Eigen::Ref<const Eigen::VectorXd> y, const Eigen::Ref<const Eigen::MatrixXd> Phi, 
                       const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p);

std::string run_pfa_inference_ds(std::string name, std::string params, const Eigen::Ref<const Eigen::MatrixXd> y, const Eigen::Ref<const Eigen::MatrixXd> Phi,
                              const Eigen::Ref<const Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> p);


#include <set>
#include "random.h"

class SamplingInference :  public PFAInferenceMethodExemplar {
public:
  SamplingInference() : _num_samples(1000), _num_partials(100), _num_gibbs_iter(1), _num_samples_inner(1), _log_ps(), _mt_gen_ptr() {}
  
//   virtual std::unique_ptr<PFAInferenceMethod> make_new() {
//     std::unique_ptr<PFAInferenceMethod> p(new PriorImportanceSamplingInference());
//     return p;
//   }
  
  virtual void init(nlohmann::json params) {
    if (params.find("seed") != params.end()) {
      _seed = params["seed"].get<int>();
    }
    if (params.find("num_threads") != params.end()) {
      _num_threads = params["num_threads"].get<int>();
    }
    _mt_gen_ptr = std::make_unique<MTGen>(_num_threads, _seed);
    if (params.find("num_samples") != params.end()) {
      _num_samples = params["num_samples"].get<int>();
    }
    if (params.find("num_samples_inner") != params.end()) {
      _num_samples_inner = params["num_samples_inner"].get<int>();
    }
    if (params.find("num_gibbs_iter") != params.end()) {
      _num_gibbs_iter = params["num_gibbs_iter"].get<int>();
    }
    if (params.find("num_partials") != params.end()) {
      _num_partials = params["num_partials"].get<int>();
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
  
  virtual ~SamplingInference() {};

protected:
  int _num_samples;
  int _num_samples_inner;
  int _num_gibbs_iter;
  int _num_partials;
  int _seed;
  double _prob_threshold;
  int _max_partitions;
  std::unordered_map<int,double> _log_ps;
  std::unordered_map<int, Eigen::VectorXd> _log_ps_ds;
  std::unique_ptr<MTGen> _mt_gen_ptr;
};