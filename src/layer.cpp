#include "internal.hpp"
#include "layer.hpp"

namespace NeuralNet {

Layer::Layer(std::size_t width, std::size_t width_prev_layer) :
    weights{width, width_prev_layer},
    neurons{width},
    activation{width},
    prev_activation{width_prev_layer}
{
    // Initialize weights and biases to random values.
    this->neurons = neurons.unaryExpr(&internal::random);
    this->weights = weights.unaryExpr(&internal::random);
}

Layer::Layer(std::size_t width, std::size_t width_prev_layer, double bias) :
    weights{width, width_prev_layer},
    neurons{width},
    activation{width},
    prev_activation{width_prev_layer}
{
    this->neurons.fill(bias);

    // Initialize weights and to random values.
    this->weights = weights.unaryExpr(&internal::random);
}

void Layer::forward_propagate(Eigen::VectorXd& input) {
    this->prev_activation = input;
    // propagate the vector
    this->activation = (weights * input + neurons).unaryExpr(&internal::sigmoid);
    input = activation;
}

Eigen::VectorXd Layer::backpropagate(Eigen::VectorXd prev_err, const Layer& prev_layer) const {
    return (prev_layer.weights.transpose() * prev_err)
        .cwiseProduct(activation.unaryExpr(&internal::d_sigmoid));
}

void Layer::gradient_descent(Eigen::VectorXd gradient) {
    this->weights -= gradient * prev_activation.transpose();
    this->neurons -= gradient;
}

} // namespace NeuralNet
