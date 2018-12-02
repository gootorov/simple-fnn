#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Core>

namespace NeuralNet {

/// @brief The type that represents a Layer in the Network.
class Layer {
private:
    /// @brief The matrix that holds the weights for this Layer.
    /// @detail Each nth row represents a vector of weights connected
    /// to the nth neuron.
    /// The dimension of the weight matrix is layer_width x previous_layer_width.
    Eigen::MatrixXd weights{};

    /// @brief The vector of biases.
    Eigen::VectorXd neurons{};

    /// @brief Stores the activation of this Layer.
    /// @detail The activation of this Layer is used to compute
    /// the error during backpropagation.
    Eigen::VectorXd activation{};

    /// @brief Stores activation of the preceeding Layer.
    /// @detail The activation of the previous Layer is used
    /// to compute the gradient descent step.
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
    void forward_propagate(Eigen::VectorXd& input);

    /// @brief Backpropagates the error from the previous Layer.
    /// @param prev_err The error of the previous Layer.
    /// @param prev_layer Pointer to the previous Layer.
    /// @return The component of the Gradient vector.
    Eigen::VectorXd backpropagate(Eigen::VectorXd prev_err, const Layer& prev_layer) const;

    /// @brief Given a Gradient vector component, applies a gradient descent step
    /// to weights and biases in this Layer.
    /// @param gradient The gradient.
    void gradient_descent(Eigen::VectorXd gradient);
};

} // namespace NeuralNet

#endif
