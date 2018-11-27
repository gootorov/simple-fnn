#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Core>

#include "layer.hpp"


class Network {
private:
    using TrainingData = std::vector<Eigen::VectorXd>;
    using Labels = std::vector<Eigen::VectorXd>;
    using Array = Eigen::Array<Eigen::VectorXd, Eigen::Dynamic, 1>;

    /// @brief Layers of the Network.
    std::vector<Layer> layers{};

    double learning_rate{};

public:
    Network() = default;

    Network(std::size_t layers, std::size_t width, double learning_rate);

    Eigen::VectorXd forward_propagate(Eigen::VectorXd input);

    Network::Array backpropagate(Eigen::VectorXd net_output, Eigen::VectorXd label) const;

    double cost(const TrainingData& training_data, const Labels& labels);
};

#endif
