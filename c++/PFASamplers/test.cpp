#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>
#include <iomanip>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"
#include <boost/math/tr1.hpp>

#include "loader.h"
#include "exact.h"
#include "partitions.h"
#include "random.h"
#include "utils.h"
#include "pfa_inference.h"
#include "json.hpp"


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::Ref;

void log_init() {
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
    );
}


void test(std::string method) {
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> sec;
  typedef std::chrono::duration<double, std::nano> ns;
  
  std::string dataset = "NIPS";
  int num_docs = 5802;
  int num_words = 100;
  int num_factors = 5;
  int doc_id = 182;

  int num_samples = 10000;
  int num_partials = 100;
  int num_threads = 4;
  int burn_in = 100;
  int seed = 123;

  auto gen = std::mt19937_64(0);
  std::uniform_real_distribution<double> d;
  VectorXd p(num_factors);
  VectorXd r(num_factors);
  MatrixXd Phi(num_factors, num_words);
  MatrixXd y(num_docs, num_words);

  std::unique_ptr<PFAInferenceMethod> inference_method = PFAInferenceFactory::get(method, nlohmann::json({{"num_samples" , num_samples},{"num_threads" , num_threads}, {"seed", seed}}));
  
  std::string filepath = "data/"+dataset+"_GaP_"+ std::to_string(num_docs) +"N_"+ std::to_string(num_words) +"W_"+ std::to_string(num_factors) +"T.mat";
  load_from_mat(filepath.c_str(), p, r, Phi, y);
  
  Clock::time_point t0 = Clock::now();
  double ll = inference_method->log_p(y.row(doc_id), Phi, r, p);
  Clock::time_point t1 = Clock::now();
  sec s0 = t1 - t0;
  std::cout << "ll = " << ll << " "
          << "\tTime = " << s0.count() << " seconds" << std::endl;

}

int main(int argc, char *argv[]) {
   if( argc == 2 ) {
      printf("Computing the marginal document likelihood with %s method \n", argv[1]);
   }
   else if( argc > 2 ) {
      printf("Too many arguments supplied.\n");
   }
   else {
      printf("One argument expected.\n");
   }
  log_init();
  std::string method(argv[1]);
  test(method);
}

 