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
  //return DistArr { dist: distance(x,y), arr: x& } ;
}

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

  return 0;
}
