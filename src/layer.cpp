#include "internal.hpp"
#include "layer.hpp"
#include "network.hpp"

namespace NeuralNet {

Layer::Layer(std::size_t width, std::size_t width_prev_layer) :
    weights{width, width_prev_layer},
    biases{width},
    prev_activation{width_prev_layer}
{
    // Initialize weights and biases to random values.
    this->biases = biases.unaryExpr(&internal::random);
    this->weights = weights.unaryExpr(&internal::random);
}

Layer::Layer(std::size_t width, std::size_t width_prev_layer, double bias) :
    weights{width, width_prev_layer},
    biases{width},
    prev_activation{width_prev_layer}
{
    this->biases.fill(bias);

    // Initialize weights and to random values.
    this->weights = weights.unaryExpr(&internal::random);
}

void Layer::forward_propagate(Vec& input) {
    this->prev_activation = input;
    // propagate the vector
    input = (weights * input + biases).unaryExpr(&internal::sigmoid);
}

Vec Layer::backpropagate(const Vec& prev_err, const Layer& prev_layer) const {
    // activation of this layer.
    const auto& activation = prev_layer.prev_activation;

    return (prev_layer.weights.transpose() * prev_err)
        .cwiseProduct(activation.unaryExpr(&internal::d_sigmoid));
}

void Layer::gradient_descent(const Vec& gradient) {
    this->weights -= gradient * prev_activation.transpose();
    this->biases -= gradient;
}

} // namespace NeuralNet
