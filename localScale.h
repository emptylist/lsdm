#ifndef LOCAL_SCALE_H
#define LOCAL_SCALE_H

#include "Eigen/Dense"

struct DistArr {
  double dist;
  Eigen::ArrayXd arr;
};

bool compare(const DistArr&, const DistArr&);
double distance(const Eigen::ArrayXd&, const Eigen::ArrayXd&);

#endif
