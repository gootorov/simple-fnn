#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Core>

#include "layer.hpp"

/// @brief Fully Connected Neural Network library.
namespace NeuralNet {

using Gradient = Eigen::Array<Eigen::VectorXd, Eigen::Dynamic, 1>;

/// @brief Type that represents a fully connected neural network.
class Network {
private:
    /// @brief Type alias for training or testing data.
    using Data = std::vector<Eigen::VectorXd>;
    /// @brief Type alias for training or testing labels.
    using Labels = std::vector<Eigen::VectorXd>;

    /// @brief Layers of the Network.
    std::vector<Layer> layers{};

    /// @brief The learning rate of the Network, i.e.
    /// the gradient scalar.
    double learning_rate{};

    /// @brief Propagate the input vector throught the Network.
    /// @param input Input vector, e.g. a MNIST image.
    /// @return Propagated vector.
    Eigen::VectorXd forward_propagate(Eigen::VectorXd input);

    /// @brief Given the output of the Network and the label,
    /// computes the output error backpropagates it.
    /// @param net_output Output of the Network, i.e. forward propagated input.
    /// @param label The desired output of the Network.
    /// @return The Gradient vector.
    Gradient backpropagate(Eigen::VectorXd net_output, Eigen::VectorXd label) const;

    /// @brief Given the gradient vector, applies it to the Network's weights and biases.
    /// @param gradient The gradient vector.
    void gradient_descent(Gradient gradient);

public:
    /// @brief Default constructor.
    Network() = default;

    /// @param layers The number of hidden Layers in this Network.
    /// @param width The width of each Layer.
    /// @param learning_rate The learning rate.
    Network(std::size_t layers, std::size_t width, double learning_rate);

    /// @brief Given the training data and labels, the Network learns.
    /// @param training_data Training data, e.g. MNIST images.
    /// @param labels Labels for to the Data, e.g. the actual numbers depicted in MNIST images.
    /// @param debug Print the cost of each training example as the Network learns.
    void learn(const Data& training_data, const Labels& labels, bool debug=false);

    /// @brief Given data, computes how accurate the Netowrk is.
    /// @param data Input data, e.g. MNIST images.
    /// @param labels Labels for to the Data, e.g. the actual numbers depicted in MNIST images.
    int accuracy(const Data& data, const Labels& labels);

    /// @brief Given data, computes the average cost over all input examples.
    /// @param data Input data, e.g. MNIST images.
    /// @param labels Labels for to the Data, e.g. the actual numbers depicted in MNIST images.
    double cost(const Data& data, const Labels& labels);
};

} // namespace NeuralNet

#endif
