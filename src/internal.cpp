#include <random>

#include <Eigen/Core>

#include "internal.hpp"

namespace internal {

long int argmax(const Eigen::VectorXd& v) {
    long int argmax = 0;
    double max = v(0);
    for (long int i = 0; i < v.size(); i++) {
        if (v(i) > max) {
            max = v(i);
            argmax = i;
        }
    }
    return argmax;
}

double random(double) {
    std::random_device seed{};
    std::mt19937 engine{seed()};
    std::normal_distribution<double> random_value{0.0, 1.0};

    return random_value(engine);
}

double sigmoid(double component) {
    return 1.0 / (1.0 + std::exp(-component));
}

double d_sigmoid(double component) {
    return component * (1.0 - component);
}

} // namespace internal
