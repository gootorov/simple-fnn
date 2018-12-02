#ifndef INTERNAL_HPP
#define INTERNAL_HPP

#include <Eigen/Core>

namespace NeuralNet {

namespace internal {

long int argmax(const Eigen::VectorXd& v);

double random(double);

double sigmoid(double component);

double d_sigmoid(double component);

} // namespace internal

} // namespace NeuralNet

#endif