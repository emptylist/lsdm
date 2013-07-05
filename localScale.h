#ifndef LOCAL_SCALE_H
#define LOCAL_SCALE_H

#include <vector>
//#include "Eigen/Dense"

struct DistArr {
  double dist;
  std::vector<double> vec;
};

bool compare(const DistArr&, const DistArr&);
double distance(const std::vector<double>&, const std::vector<double>&);
DistArr dist_y_from_x(const std::vector<double>&, const std::vector<double>&);

//double distance(const Eigen::ArrayXd&, const Eigen::ArrayXd&);
//DistArr dist_y_from_x(const Eigen::ArrayXd&, const Eigen::ArrayXd&);

#endif
