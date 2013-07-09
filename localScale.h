#ifndef LOCAL_SCALE_H
#define LOCAL_SCALE_H

#include <vector>
//#include "Eigen/Dense"

struct DistArr {
  double dist;
  const std::vector<double> * pvec;
};

typedef std::vector<std::vector<double> > dataArray;
typedef std::vector<DistArr> distData;

bool compare(const DistArr&, const DistArr&);

double distance(const std::vector<double>&, const std::vector<double>&);
DistArr dist_y_from_x(const std::vector<double>&, const std::vector<double>&);
distData dist_from_x(const std::vector<double>&, const dataArray&);

//double calc_scaling_parameter(const std::vector<double>&, const dataArray&); //TODO
//std::vector<double> calc_scaling_parameters(const dataArray&);

//TODO: Implement MDS that works on vector<DistArr>

//double distance(const Eigen::ArrayXd&, const Eigen::ArrayXd&);
//DistArr dist_y_from_x(const Eigen::ArrayXd&, const Eigen::ArrayXd&);

#endif
