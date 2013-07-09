#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>
//#include "Eigen/Dense"
#include "localScale.h"

using namespace std;

//functions for computing the local scaling parameters

bool compare(const DistArr& x, const DistArr& y)
{
  return x.dist < y.dist;
}

double euclidean(const vector<double>& x, const vector<double>& y)
{
  if (x.size()!=y.size())
    throw domain_error("Vectors are not the same size.");

  vector<double>::size_type sz = x.size();
  double sqdist = 0;
  
  for (vector<double>::size_type i = 0; i < sz; ++i) {
    sqdist += pow((x[i] - y[i]), 2);
  }

  return sqrt(sqdist);
}

DistArr dist_y_from_x(const vector<double>& x, const vector<double>& y)
{
  DistArr da = DistArr({dist: euclidean(x,y), pvec: &y});
  return da;
}

distData dist_from_x(const vector<double>& x, const dataArray& D)
{
  dataArray::size_type sz = D.size();
  distData ret;

  for (dataArray::size_type i = 0; i < sz; ++i) {
    ret.push_back(dist_y_from_x(x, D[i]));
  }

  return ret;
}
/*
double calc_scaling_parameter

vector<double> calc_scaling_parameters(const dataArray& X)
{
  vector<double> scaling_parameters;
  dataArray::size_type sz = X.size();
  
  for (dataArray::size_type i = 0; i < sz; ++i) {
    scaling_parameters.push_back(calc_scaling_parameter(X[i], X));
  }

  return scaling_parameters;

}
*/
int main() 
{
  vector<double> x;
  vector<double> y;

  x.push_back(1.0);
  x.push_back(1.0);
  x.push_back(1.0);
  x.push_back(1.0);

  y.push_back(3.0);
  y.push_back(3.0);
  y.push_back(3.0);
  y.push_back(3.0);

  double d = euclidean(x,y);

  cout << d << endl;

  DistArr X = dist_y_from_x(x,y);
  cout << X.dist << endl;

  DistArr Y = dist_y_from_x(y,x);
  cout << compare(Y,X) << endl;

  cout << Y.dist << endl << Y.pvec << endl;

  return 0;
}
