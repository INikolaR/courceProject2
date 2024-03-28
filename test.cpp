#include "test.h"

#include <chrono>
#include <fstream>
#include <string>

#include "AdamOptimizer.h"
#include "ConstantOptimizer.h"
#include "Net.h"

namespace neural_network {
void run_all_tests() {
    test_mnist();
}

void test_echo() {
    std::cout << "ECHO TEST: just trying to return the input\n";
    Net net{{1, 10, 1}, {Net::LeakyReLU, Net::LeakyReLU}, Net::Euclid};
    std::vector<Element> dataset = {{Matrix{{1}}, Matrix{{1}}}, {Matrix{{2}}, Matrix{{2}}}, {Matrix{{3}}, Matrix{{3}}},
                                    {Matrix{{4}}, Matrix{{4}}}, {Matrix{{5}}, Matrix{{5}}}, {Matrix{{6}}, Matrix{{6}}},
                                    {Matrix{{7}}, Matrix{{7}}}, {Matrix{{8}}, Matrix{{8}}}};
    net.fit(dataset, 1, 10, AdamOptimizer(1, 0.9, 0.999, 1e-8));
    for (double i = 0; i < 12; ++i) {
        std::cout << i << " -> " << net.predict(Matrix{{i}}) << "\n";
    }
}

void test_echo_vector() {
    std::cout << "ECHO TEST: just trying to return the input\n";
    neural_network::Net net{{2, 100, 100, 2}, {Net::LeakyReLU, Net::LeakyReLU, Net::LeakyReLU}, Net::Euclid};
    std::vector<neural_network::Element> dataset = {
        {neural_network::Matrix{{1}, {0}}, neural_network::Matrix{{1}, {0}}},
        {neural_network::Matrix{{0}, {1}}, neural_network::Matrix{{0}, {1}}},
        {neural_network::Matrix{{1}, {1}}, neural_network::Matrix{{1}, {1}}},
        {neural_network::Matrix{{0}, {0}}, neural_network::Matrix{{0}, {0}}}};
    net.fit(dataset, 1, 10000, ConstantOptimizer(0.001));
    std::cout << "MSE = " << net.MSE(dataset) << "\nFor example:\n";
    for (size_t i = 0; i < dataset.size(); ++i) {
        std::cout << dataset[i].x << "\n->\n" << net.predict(dataset[i].x) << "\n";
    }
}

void test_square() {
    std::cout << "SQUARE TEST: trying to return x^2 if x is in input (x is scalar)\n";
    neural_network::Net net{{1, 20, 20, 1}, {Net::LeakyReLU, Net::LeakyReLU, Net::LeakyReLU}, Net::Euclid};
    std::vector<neural_network::Element> dataset = {{neural_network::Matrix{{1}}, neural_network::Matrix{{1}}},
                                                    {neural_network::Matrix{{2}}, neural_network::Matrix{{4}}},
                                                    {neural_network::Matrix{{3}}, neural_network::Matrix{{9}}},
                                                    {neural_network::Matrix{{4}}, neural_network::Matrix{{16}}},
                                                    {neural_network::Matrix{{5}}, neural_network::Matrix{{25}}},
                                                    {neural_network::Matrix{{6}}, neural_network::Matrix{{36}}},
                                                    {neural_network::Matrix{{7}}, neural_network::Matrix{{49}}},
                                                    {neural_network::Matrix{{8}}, neural_network::Matrix{{64}}}};
    net.fit(dataset, 1, 1000, ConstantOptimizer(0.00002));
    std::cout << net.MSE(dataset) << "\n";
    for (double i = 0; i < 12; ++i) {
        std::cout << i << " -> " << net.predict(neural_network::Matrix{{i}}) << "\n";
    }
}

int reverse_int(int i) {
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) + (static_cast<int>(c3) << 8) + c4;
}

void test_mnist() {
    const int size_of_mnist_image = 784;
    std::cout << "MNIST TRAIN:\n";
    std::ifstream file_images("F:\\Kursach\\CourseProject\\train-images-idx3-ubyte\\train-images.idx3-ubyte",
                              std::ios::binary | std::ifstream::in);

    if (!file_images.is_open()) {
        std::cout << "Cannot open training images\n";
        file_images.close();
        return;
    }

    int images_magic_number = 0;
    const int expected_images_magic_number = 2051;
    file_images.read(reinterpret_cast<char*>(&images_magic_number), sizeof(images_magic_number));
    images_magic_number = reverse_int(images_magic_number);
    if (images_magic_number != expected_images_magic_number) {
        std::cout << "Bad MNIST image file!\n";
        file_images.close();
        return;
    }
    int n_rows = 0;
    int n_cols = 0;
    int number_of_images = 0;

    file_images.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    number_of_images = reverse_int(number_of_images);
    file_images.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    n_rows = reverse_int(n_rows);
    file_images.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
    n_cols = reverse_int(n_cols);

    std::ifstream file_labels("../train-labels-idx1-ubyte/train-labels.idx1-ubyte", std::ios::binary);
    if (!file_labels.is_open()) {
        std::cout << "Cannot open training labels\n";
        file_images.close();
        file_labels.close();
        return;
    }

    int labels_magic_number = 0;
    const int expected_labels_magic_number = 2049;
    file_labels.read(reinterpret_cast<char*>(&labels_magic_number), sizeof(labels_magic_number));
    labels_magic_number = reverse_int(labels_magic_number);
    if (labels_magic_number != expected_labels_magic_number) {
        std::cout << "Bad MNIST label file!\n";
        file_images.close();
        file_labels.close();
        return;
    }

    int number_of_labels = 0;
    file_labels.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
    number_of_labels = reverse_int(number_of_labels);

    if (number_of_labels != number_of_images) {
        std::cout << "Different number of rows in images and labels!\n";
        file_images.close();
        file_labels.close();
        return;
    }

    std::vector<Element> dataset(0);
    unsigned char image[size_of_mnist_image];
    unsigned char label = 0;
    for (int i = 0; i < number_of_labels; i++) {
        file_images.read(reinterpret_cast<char*>(image), size_of_mnist_image);
        double array_image[size_of_mnist_image];
        for (int j = 0; j < size_of_mnist_image; ++j) {
            array_image[j] = static_cast<double>(image[j]) / 255.0;
        }
        Vector x = Eigen::Map<Vector>(array_image, size_of_mnist_image);
        file_labels.read(reinterpret_cast<char*>(&label), 1);
        double array_label[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        array_label[static_cast<unsigned int>(label)] = 1;
        Vector y = Eigen::Map<Vector>(array_label, 10);
        dataset.emplace_back(Element{x, y});
    }

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    std::cout
        << "TEST 1 | Architecture: 784 -> Sigmoid -> 256 -> Sigmoid -> 10 | Using 1000 batches during 10 epochs\n";
    std::cout << "Using constant step length = 0.3\n";
    Net net1{{784, 256, 10}, {Net::Sigmoid, Net::Sigmoid}, Net::Euclid};

    begin = std::chrono::steady_clock::now();
    net1.fit(dataset, 1000, 10, ConstantOptimizer(0.3));
    end = std::chrono::steady_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s\n";
    std::cout << "MSE: " << net1.MSE(dataset) << "\n";
    std::cout << "Accuracy: " << net1.accuracy(dataset) * 100 << "%\n";

    std::cout << "TEST 2 | Architecture: 784 -> Sigmoid -> 256 -> Sigmoid -> 10 | Using 10k batches during 10 epochs\n";
    std::cout << "Using constant step length = 0.3\n";
    Net net2{{784, 256, 10}, {Net::Sigmoid, Net::Sigmoid}, Net::Euclid};

    begin = std::chrono::steady_clock::now();
    net2.fit(dataset, 10000, 10, ConstantOptimizer(0.3));
    end = std::chrono::steady_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s\n";
    std::cout << "MSE: " << net2.MSE(dataset) << "\n";
    std::cout << "Accuracy: " << net2.accuracy(dataset) * 100 << "%\n";

    std::cout << "TEST 3 | Architecture: 784 -> Sigmoid -> 256 -> Sigmoid -> 10 | Using 10k batches during 8 epochs\n";
    std::cout << "Using Adam optimizer with params: start_step = 0.003, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8\n";
    Net net3{{784, 256, 10}, {Net::Sigmoid, Net::Sigmoid}, Net::Euclid};

    begin = std::chrono::steady_clock::now();
    net3.fit(dataset, 10000, 8, AdamOptimizer(0.003, 0.9, 0.999, 1e-8));
    end = std::chrono::steady_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s\n";
    std::cout << "MSE: " << net3.MSE(dataset) << "\n";
    std::cout << "Accuracy: " << net3.accuracy(dataset) * 100 << "%\n";

    file_images.close();
    file_labels.close();
}

}  // namespace neural_network
