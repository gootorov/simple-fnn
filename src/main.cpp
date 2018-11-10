#include <iostream>
#include <vector>

#include "mnist/mnist_reader.hpp"

#include <Eigen/Dense>
using Eigen::MatrixXd;

int main() {
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    return 0;
}
