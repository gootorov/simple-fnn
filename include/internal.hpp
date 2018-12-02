#ifndef INTERNAL_HPP
#define INTERNAL_HPP

#include <Eigen/Core>

namespace NeuralNet {

/// @brief Contains helper functions used throughout the NeuralNet library.
namespace internal {

/// @brief Given a vector, returns the index of the largest component.
/// @details E.g. given [0, -1, 3, 2] returns 2.
/// @param v Input vector.
/// @return Index of the largest component.
long int argmax(const Eigen::VectorXd& v);

/// @brief Returns a random number.
/// @details Raturns a random number within Gaussian distribution with
/// mean 0 and variance 1.
/// @return Random number.
double random(double);

/// @brief Applies the sigmoid function to the input.
/// @details Applies f(x) = 1 / (1 + e^-x).
/// @param component Vector component.
double sigmoid(double component);

/// @brief Applies the first derivative of the sigmoid function to the input.
/// @details Applies f'(x) = f(x) * (1 - f(x)).
/// @param component Vector component.
double d_sigmoid(double component);

} // namespace internal

} // namespace NeuralNet

#endif
