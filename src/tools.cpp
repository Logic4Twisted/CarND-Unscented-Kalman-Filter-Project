#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  //std::cout << "CalculateRMSE" << std::endl;
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  if (estimations.size() == 0) return rmse;
  for (int i = 0; i < estimations.size(); i++) {
    VectorXd d = estimations[i].array()-ground_truth[i].array();
    d = d.array() * d.array();
    rmse = rmse + d;
  }
  rmse /= estimations.size();
  rmse = sqrt(rmse.array());
  return rmse;
}
