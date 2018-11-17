#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Core>

#include "layer.hpp"

class Network {
private:
    /// @brief Layers of the Network.
    std::vector<Layer> layers{};

public:
    Network() = default;

    Network(std::size_t layers, std::size_t width);

    Eigen::VectorXd forward_propagate(const Eigen::VectorXd input) const;
};

#endif
