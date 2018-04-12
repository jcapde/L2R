#pragma once
#include <Eigen/Dense>


void load_from_mat(const char *file, Eigen::Ref<Eigen::VectorXd> p, Eigen::Ref<Eigen::VectorXd> r, Eigen::Ref<Eigen::MatrixXd> Phi, Eigen::Ref<Eigen::MatrixXd> y);
void load_from_mat2(const char *file, Eigen::Ref<Eigen::VectorXd> p, Eigen::Ref<Eigen::VectorXd> r, Eigen::Ref<Eigen::MatrixXd> Phi, Eigen::Ref<Eigen::MatrixXd> y);
void load_from_arff(const char *filename, Eigen::Ref<Eigen::MatrixXd> y, int ndocs, int nwords);
Eigen::VectorXd load_doc_from_arff(const char *filename, int doc_id, int num_words);
