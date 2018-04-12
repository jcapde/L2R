#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <string>
#include <Eigen/Dense>
#include "matio.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::Ref;

void load_from_mat2(const char *file, Eigen::Ref<Eigen::VectorXd> p, Eigen::Ref<Eigen::VectorXd> r, Eigen::Ref<Eigen::MatrixXd> Phi, Eigen::Ref<Eigen::MatrixXd> y) {
  // We read the mat file using the MATLAB library functions
  // I have to rewrite this to automatise the load for different configurations.

  int num_factors = Phi.rows();


  mat_t *matfp;
  matfp = Mat_Open(file, MAT_ACC_RDONLY);
  if ( NULL == matfp ) {
    std::cerr << "Error opening MAT file " << file;
    return;
  }
  matvar_t* mvt_q = Mat_VarRead(matfp, "pr");
  double *qr_avg = (double *)mvt_q->data;

  for(int k=0; k<num_factors; k++){
    p(k) = qr_avg[k];
  }

  //std::cout << "p:  "<< p << std::endl;

  matvar_t* mvt_s = Mat_VarRead(matfp, "r");
  double *s = (double *) mvt_s->data;
  for(int k=0; k<num_factors; k++){
    r(k) = s[0];
  }

  //std::cout << "r:  "<< r << std::endl;

  matvar_t* mvt_u = Mat_VarRead(matfp, "phi");
  double *U_avg = (double *)mvt_u->data;

  int nrows = mvt_u->dims[0];
  int ncols = mvt_u->dims[1];
  //printf("U_avg rows: %d\n",nrows);
  //printf("U_avg cols: %d\n",ncols);

  for(int i=0; i<nrows; i++) {
    for (int j=0; j<ncols; j++) {
      Phi(i, j) = U_avg[j * nrows + i];
    }
  }

  //std::cout << "phi:  "<< Phi.rowwise().sum() << std::endl;

  matvar_t* mvt_y = Mat_VarRead(matfp, "x");
  double *y_input = (double *)mvt_y->data;
  nrows = mvt_y->dims[0];
  ncols = mvt_y->dims[1];
  //printf("y rows: %d\n",nrows);
  //printf("y cols: %d\n",ncols);

  for(int i=0; i < nrows; i++) {
    for (int j=0; j < ncols; j++) {
      y(i, j) = y_input[j * nrows + i];
    }
  }

  //std::cout << "y:  "<< y.rowwise().sum() << std::endl;

  Mat_Close(matfp);
  return;
}

void load_from_mat(const char *file, Eigen::Ref<Eigen::VectorXd> p, Eigen::Ref<Eigen::VectorXd> r, Eigen::Ref<Eigen::MatrixXd> Phi, Eigen::Ref<Eigen::MatrixXd> y) {
    // We read the mat file using the MATLAB library functions
    // I have to rewrite this to automatise the load for different configurations.

    int num_factors = Phi.rows();

    //std::cout << "num_factors:  "<< num_factors << std::endl;

    mat_t *matfp;
    matfp = Mat_Open(file, MAT_ACC_RDONLY);
    if ( NULL == matfp ) {
      //std::cout << "Error opening MAT file"<< std::endl;
      std::cerr << "Error opening MAT file " << file;
        return;
    }
    matvar_t* mvt_q = Mat_VarRead(matfp,"qr_avg");
    double *qr_avg = (double *)mvt_q->data;
    
    for(int k=0; k<num_factors; k++){
      p(k) = qr_avg[k];
    }

    //std::cout << "p:  "<< p << std::endl;

    matvar_t* mvt_s = Mat_VarRead(matfp,"s");
    double *s = (double *) mvt_s->data;
    for(int k=0; k<num_factors; k++){
      r(k) = s[0];
    }

    //std::cout << "r:  "<< r << std::endl;

    matvar_t* mvt_u = Mat_VarRead(matfp,"U_avg");
    double *U_avg = (double *)mvt_u->data;

    int nrows = mvt_u->dims[0];
    int ncols = mvt_u->dims[1];
    //printf("U_avg rows: %d\n",nrows);
    //printf("U_avg cols: %d\n",ncols);

    for(int i=0; i<nrows; i++) {
      for (int j=0; j<ncols; j++) {
        Phi(j, i) = U_avg[j * nrows + i];
      }
    }

    matvar_t* mvt_y = Mat_VarRead(matfp,"y");
    double *y_input = (double *)mvt_y->data;
    nrows = mvt_y->dims[0];
    ncols = mvt_y->dims[1];
    //printf("y rows: %d\n",nrows);
    //printf("y cols: %d\n",ncols);

    for(int i=0; i<nrows; i++) {
      for (int j=0; j<ncols; j++) {
        y(j, i) = y_input[j * nrows + i];
      }
    }

    Mat_Close(matfp);
    return;
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

VectorXd load_doc_from_arff(const char *filename, int doc_id, int num_words){

    VectorXd y(num_words);
    std::ifstream file(filename);
    std::string str; 

    int line = 1004 + doc_id - 1;
    int n = 0;

    while (std::getline(file, str))
    {
      if(n==line){
        std::vector<std::string> x = split(str, ',');
        int j = 0;
        for(std::string word : x) {
          if(j < num_words){
            y(j) = std::stoi(word);
          }
          j++;
        }
        break;
      }
      n++;
    }

    return y;

}


void load_from_arff(const char *filename, Ref<MatrixXd> y, int ndocs, int nwords){

     // We read the arff file as a text file in which documents can be found after the START line, 
  //with word counts separated by commas.

    std::ifstream file(filename);
    std::string str; 
    int n = 0;


    const int doc_id = 1;
    const int START = 1004 + doc_id - 1;

    while (std::getline(file, str))
    {
      if((n>=START) && (n<(START+ndocs))){
        std::vector<std::string> x = split(str, ',');
        int j = 0;
        //printf("%lu\n",x.size());
        for(std::string word : x) {
          if(j < nwords){
            //printf("%d - %d", n, j);
            //printf(": %s\n", word.c_str());
            y(n-START,j) = std::stoi(word);
          }
          j++;
        }
      }
      n++;
    }

}
