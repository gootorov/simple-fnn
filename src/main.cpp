#include <iostream>

#include <Eigen/Core>

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

#include "network.hpp"

double normalize(double pixel) {
    return pixel / 255.0;
}

void load_data(std::vector<std::vector<double>>& source, std::vector<Eigen::VectorXd>& target) {
    for (auto& image : source) {
        target.push_back(Eigen::VectorXd::Map(image.data(), image.size()));
    }

    // normalize values
    for (auto& image : target) {
        image = image.unaryExpr(&normalize);
    }
}

// converts "1" into [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
void vectorize_labels(std::vector<Eigen::VectorXd>& vectorized_labels, std::vector<uint8_t>& labels) {
    for (const auto& _label : labels) {
        auto label = static_cast<std::size_t>(_label);

        Eigen::VectorXd vectorized = Eigen::VectorXd::Zero(10);
        vectorized(label) = 1.0;

        vectorized_labels.push_back(vectorized);
    }
}

int main() {
    // load mnist data
    auto mnist = mnist::read_dataset<std::vector, std::vector, double, uint8_t>(MNIST_DATA_LOCATION);

    // convert std::vector's to Eigen::vector's.
    std::vector<Eigen::VectorXd> training_images{};
    std::vector<Eigen::VectorXd> test_images{};
    load_data(mnist.training_images, training_images);
    load_data(mnist.test_images, test_images);

    // vectorize labels
    std::vector<Eigen::VectorXd> training_labels{};
    std::vector<Eigen::VectorXd> test_labels{};
    vectorize_labels(training_labels, mnist.training_labels);
    vectorize_labels(test_labels, mnist.test_labels);

    auto network = Network(3, 100, 0.79);
    network.learn(training_images, training_labels);

    std::cout << "\n\n\nAccuracy: " << network.accuracy(training_images, training_labels) << "\n\n\n";
    std::cout << "\n\n\nAccuracy: " << network.accuracy(test_images, test_labels) << "\n\n\n";

    return 0;
}
