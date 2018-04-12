#pragma once 
#include <Eigen/Dense>

typedef Eigen::VectorXd EVector;
typedef Eigen::MatrixXd EMatrix;
typedef const Eigen::Ref<const Eigen::VectorXd> InEVector;
typedef Eigen::Ref<Eigen::VectorXd> OutEVector;
typedef const Eigen::Ref<const Eigen::MatrixXd> InEMatrix;
typedef Eigen::Ref<Eigen::MatrixXd> OutEMatrix;

