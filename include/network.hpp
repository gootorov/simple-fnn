#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Core>

#include "layer.hpp"

using TrainingData = std::vector<Eigen::VectorXd>;
using Labels = std::vector<Eigen::VectorXd>;

class Network {
private:
    /// @brief Layers of the Network.
    std::vector<Layer> layers{};

public:
    Network() = default;

    Network(std::size_t layers, std::size_t width);

    Eigen::VectorXd forward_propagate(Eigen::VectorXd input) const;

    double cost(const TrainingData& training_data, const Labels& labels) const;
};

#endif
