#include <Python.h>
#include <boost/python.hpp>
#include <Eigen/Dense>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
//#include <boost/python/numpy.hpp>

#include "boopy.hpp"
#include "pfa_inference.h"
#include "loader.h"

using Eigen::Map;
using Eigen::VectorXd;
using Eigen::MatrixXd;

boost::python::str out(const char *msg1, const char *msg2) {
    //std::cout << msg1 << std::endl;
   //std::cout << msg2 << std::endl;
   //std::cout << "y=" << y << std::endl;
   //std::cout << PyArray_Size(y) << std::endl;
   //Map<VectorXd> _y((double *) PyArray_DATA(y), PyArray_Size(y));
   //std::cout << "y:" << _y  << std::endl;
   return "Outy";
}

std::string run_pfa_inference_python_ds(const char *name, const char *params, MatrixXd y,
                                     MatrixXd Phi, VectorXd r, VectorXd p) {
  return run_pfa_inference_ds(name, params, y, Phi, r, p);
}


std::string run_pfa_inference_python(const char *name, const char *params, VectorXd y,
                              MatrixXd Phi, VectorXd r, VectorXd p) {
    return run_pfa_inference(name, params, y, Phi, r, p);
}

VectorXd run_loader_data_python(const char *filename, int doc_id, int num_words){
  return load_doc_from_arff(filename, doc_id, num_words);
}

boost::python::tuple run_loader_model_python(const char *file, int num_docs, int num_words, int num_factors){
  MatrixXd Phi(num_factors, num_words);
  VectorXd r(num_factors), p(num_factors);
  MatrixXd y(num_docs, num_words);
  load_from_mat(file, p, r, Phi, y);
  return boost::python::make_tuple(y, p, r, Phi);
}

int imp_solver() {
    import_array();
    return 0;
}

BOOST_PYTHON_MODULE(pfa)
{
    imp_solver();
    //import_array();
    using namespace boost::python;
    to_python_converter<Eigen::MatrixXd,
                          boopy::EigenMatrix_to_python_matrix<Eigen::MatrixXd> >();
    boopy::EigenMatrix_from_python_array<Eigen::MatrixXd>();

    to_python_converter<Eigen::VectorXd,
                          boopy::EigenMatrix_to_python_matrix<Eigen::VectorXd> >();
    boopy::EigenMatrix_from_python_array<Eigen::VectorXd>();

    def("out", out);
    def("inference", run_pfa_inference_python);
    def("inference_ds", run_pfa_inference_python_ds);
    def("load_data", run_loader_data_python);
    def("load_model", run_loader_model_python);
}
