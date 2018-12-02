#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Core>

#include "layer.hpp"

using Gradient = Eigen::Array<Eigen::VectorXd, Eigen::Dynamic, 1>;

namespace NeuralNet {

class Network {
private:
    using Data = std::vector<Eigen::VectorXd>;
    using Labels = std::vector<Eigen::VectorXd>;

    /// @brief Layers of the Network.
    std::vector<Layer> layers{};

    double learning_rate{};

    Eigen::VectorXd forward_propagate(Eigen::VectorXd input);

    Gradient backpropagate(Eigen::VectorXd net_output, Eigen::VectorXd label) const;

    void gradient_descent(Gradient gradient);

public:
    Network() = default;

    Network(std::size_t layers, std::size_t width, double learning_rate);

    void learn(const Data& training_data, const Labels& labels, bool debug=false);

    int accuracy(const Data& training_data, const Labels& labels);

    double cost(const Data& training_data, const Labels& labels);
};

} // namespace NeuralNet

#endif
