#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Core>

#include "layer.hpp"

/// @brief Fully Connected Neural Network library.
namespace NeuralNet {

using Gradient = Eigen::Array<Eigen::VectorXd, Eigen::Dynamic, 1>;
using Vec = Eigen::VectorXd;

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

    /// @brief Given the output of the Network and the label,
    /// computes the output error and backpropagates it.
    /// @param net_output Output of the Network, i.e. forward propagated input.
    /// @param label The desired output of the Network.
    /// @return The Gradient vector.
    Gradient backpropagate(const Vec& net_output, const Vec& label) const;

    /// @brief Given the gradient vector, applies it to the Network's weights and biases.
    /// @param gradient The gradient vector.
    void gradient_descent(const Gradient& gradient);

public:
    Network() = default;

    /// @param layers The number of hidden Layers in this Network.
    /// @param width The width of each Layer.
    /// @param learning_rate The learning rate.
    Network(std::size_t layers, std::size_t width, double learning_rate);

    /// @param i_layer The number of neurons in the input layer.
    /// @param h_layers The number of hidden layers.
    /// @param h_width The number of neurons in the hidden layers.
    /// @param o_layer The number of neurons in the output layer.
    /// @param learning_rate The learning rate.
    Network(std::size_t i_layer, int h_layers, std::size_t h_width, std::size_t o_layer, double learning_rate);

    /// @param layers The dimensions of the layers in this Network.
    /// @param learning_rate The learning rate.
    Network(std::vector<std::size_t> layers, double learning_rate);

    /// @brief Propagates the input vector throught the Network.
    /// @param input Input vector, e.g. a MNIST image.
    /// @return Propagated vector.
    Vec forward_propagate(Vec input);

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
