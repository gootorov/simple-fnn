#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Core>

namespace NeuralNet {

using Vec = Eigen::VectorXd;

/// @brief The type that represents a Layer in the Network.
class Layer {
private:
    /// @brief The matrix that holds the weights for this Layer.
    /// @details Each nth row represents a vector of weights connected
    /// to the nth neuron.
    /// The dimension of the weight matrix is layer_width x previous_layer_width.
    Eigen::MatrixXd weights{};

    /// @brief The vector of biases.
    Eigen::VectorXd biases{};

    /// @brief Stores activation of the preceeding Layer.
    /// @details The activation of the previous Layer is used
    /// to compute the gradient descent step and the error during
    /// backpropagation.
    Eigen::VectorXd prev_activation{};

public:
    Layer() = default;

    /// @param width The width of the Layer, i.e. the number
    /// of Neurons in this Layer.
    /// @param width_prev_layer The width of the preceeding Layer.
    Layer(std::size_t width, std::size_t width_prev_layer);

    /// @param width The width of the Layer, i.e. the number
    /// of Neurons in this Layer.
    /// @param width_prev_layer The width of the preceeding Layer.
    /// @param bias The bias used to initialize Neurons in this layer.
    Layer(std::size_t width, std::size_t width_prev_layer, double bias);

    /// @brief Given an input vector, propagates it through this Layer.
    /// @param input The input vector, i.e. a MNIST image or the output of the
    /// previous Layer.
    void forward_propagate(Vec& input);

    /// @brief Backpropagates the error from the previous Layer.
    /// @param prev_err The error of the previous Layer.
    /// @param prev_layer Pointer to the previous Layer.
    /// @return The component of the Gradient vector for this Layer.
    Vec backpropagate(const Vec& prev_err, const Layer& prev_layer) const;

    /// @brief Given a Gradient vector component, applies a gradient descent step
    /// to weights and biases in this Layer.
    /// @param gradient The gradient.
    void gradient_descent(const Vec& gradient);
};

} // namespace NeuralNet

#endif
