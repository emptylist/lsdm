#include <iostream>
#include "localScale.h"

//functions for computing the local scaling parameters

bool compare(const DistArr& x, const DistArr& y)
{
  return x.dist < y.dist;
}

double distance(const Eigen::ArrayXd& x, const Eigen::ArrayXd& y)
{
  //fill in function
}
