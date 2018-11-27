#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Core>

/// @brief The type that represents a Layer in the Network.
class Layer {
private:
    /// @brief The matrix that holds the weights for this Layer.
    /// @detail Each nth row represents a vector of weights connected
    /// to the nth neuron.
    Eigen::MatrixXd weights{};
    /// @brief The vector of neurons in this Layer that hold the biases.
    Eigen::VectorXd neurons{};

    Eigen::VectorXd activation{};

    /// @brief Generates a random number within [-1, 1] range.
    double random() const;

public:
    Layer() = default;

    /// @param width The width of the Layer, i.e. the number
    /// of Neurons in this Layer.
    Layer(std::size_t width, std::size_t width_prev_layer);

    /// @param bias The bias used to initialize Neurons in this layer.
    Layer(std::size_t width, std::size_t width_prev_layer, double bias);

    void forward_propagate(Eigen::VectorXd& input);
};

#endif
