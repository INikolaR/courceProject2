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
    Net net{{2, 100, 100, 2}, {Net::LeakyReLU, Net::LeakyReLU, Net::LeakyReLU}, Net::Euclid};
    std::vector<Element> dataset = {{Matrix{{1}, {0}}, Matrix{{1}, {0}}},
                                    {Matrix{{0}, {1}}, Matrix{{0}, {1}}},
                                    {Matrix{{1}, {1}}, Matrix{{1}, {1}}},
                                    {Matrix{{0}, {0}}, Matrix{{0}, {0}}}};
    net.fit(dataset, 1, 10000, ConstantOptimizer(0.001));
    std::cout << "MSE = " << net.MSE(dataset) << "\nFor example:\n";
    for (size_t i = 0; i < dataset.size(); ++i) {
        std::cout << dataset[i].x << "\n->\n" << net.predict(dataset[i].x) << "\n";
    }
}

void test_square() {
    std::cout << "SQUARE TEST: trying to return x^2 if x is in input (x is scalar)\n";
    Net net{{1, 20, 20, 1}, {Net::LeakyReLU, Net::LeakyReLU, Net::LeakyReLU}, Net::Euclid};
    std::vector<Element> dataset = {{Matrix{{1}}, Matrix{{1}}},  {Matrix{{2}}, Matrix{{4}}},
                                    {Matrix{{3}}, Matrix{{9}}},  {Matrix{{4}}, Matrix{{16}}},
                                    {Matrix{{5}}, Matrix{{25}}}, {Matrix{{6}}, Matrix{{36}}},
                                    {Matrix{{7}}, Matrix{{49}}}, {Matrix{{8}}, Matrix{{64}}}};
    net.fit(dataset, 1, 1000, ConstantOptimizer(0.00002));
    std::cout << net.MSE(dataset) << "\n";
    for (double i = 0; i < 12; ++i) {
        std::cout << i << " -> " << net.predict(Matrix{{i}}) << "\n";
    }
}

int reverse_int(int i) {
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) + (static_cast<int>(c3) << 8) + c4;
}

std::vector<Element> parseMNISTDataset(const std::string& path_to_images_file, const std::string& path_to_labels_file) {
    const int size_of_mnist_image = 784;
    std::ifstream file_images(path_to_images_file, std::ios::binary | std::ifstream::in);

    if (!file_images.is_open()) {
        file_images.close();
        throw std::runtime_error("Cannot open training images");
    }

    int images_magic_number = 0;
    const int expected_images_magic_number = 2051;
    file_images.read(reinterpret_cast<char*>(&images_magic_number), sizeof(images_magic_number));
    images_magic_number = reverse_int(images_magic_number);
    if (images_magic_number != expected_images_magic_number) {
        throw std::runtime_error("Bad MNIST image file!");
        file_images.close();
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

    std::ifstream file_labels(path_to_labels_file, std::ios::binary);
    if (!file_labels.is_open()) {
        file_images.close();
        file_labels.close();
        throw std::runtime_error("Cannot open training labels");
    }

    int labels_magic_number = 0;
    const int expected_labels_magic_number = 2049;
    file_labels.read(reinterpret_cast<char*>(&labels_magic_number), sizeof(labels_magic_number));
    labels_magic_number = reverse_int(labels_magic_number);
    if (labels_magic_number != expected_labels_magic_number) {
        file_images.close();
        file_labels.close();
        throw std::runtime_error("Bad MNIST label file!");
    }

    int number_of_labels = 0;
    file_labels.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
    number_of_labels = reverse_int(number_of_labels);

    if (number_of_labels != number_of_images) {
        file_images.close();
        file_labels.close();
        throw std::runtime_error("Different number of rows in images and labels!");
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
    file_images.close();
    file_labels.close();
    return dataset;
}

void test_mnist() {
    std::cout << "MNIST TRAIN:\n";

    std::vector<Element> train = parseMNISTDataset("../train-images-idx3-ubyte/train-images.idx3-ubyte",
                                                   "../train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    std::vector<Element> test = parseMNISTDataset("../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
                                                  "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    std::cout
        << "TEST 1 | Architecture: 784 -> Sigmoid -> 256 -> Sigmoid -> 10 | Using 1000 batches during 10 epochs\n";
    std::cout << "Using constant step length = 0.3\n";
    Net net1{{784, 256, 10}, {Net::Sigmoid, Net::Sigmoid}, Net::Euclid};

    begin = std::chrono::steady_clock::now();
    net1.fit(train, 1000, 10, ConstantOptimizer(0.3));
    end = std::chrono::steady_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s\n";
    std::cout << "MSE: " << net1.MSE(test) << "\n";
    std::cout << "Accuracy: " << net1.accuracy(test) * 100 << "%\n";

    std::cout << "TEST 2 | Architecture: 784 -> Sigmoid -> 256 -> Sigmoid -> 10 | Using 10k batches during 10 epochs\n";
    std::cout << "Using constant step length = 0.3\n";
    Net net2{{784, 256, 10}, {Net::Sigmoid, Net::Sigmoid}, Net::Euclid};

    begin = std::chrono::steady_clock::now();
    net2.fit(train, 10000, 10, ConstantOptimizer(0.3));
    end = std::chrono::steady_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s\n";
    std::cout << "MSE: " << net2.MSE(test) << "\n";
    std::cout << "Accuracy: " << net2.accuracy(test) * 100 << "%\n";

    std::cout << "TEST 3 | Architecture: 784 -> Sigmoid -> 256 -> Sigmoid -> 10 | Using 10k batches during 10 epochs\n";
    std::cout << "Using Adam optimizer with params: start_step = 0.003, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8\n";
    Net net3{{784, 256, 10}, {Net::Sigmoid, Net::Sigmoid}, Net::Euclid};

    begin = std::chrono::steady_clock::now();
    net3.fit(train, 10000, 20, AdamOptimizer(0.005, 0.9, 0.999, 1e-8));
    end = std::chrono::steady_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s\n";
    std::cout << "MSE: " << net3.MSE(test) << "\n";
    std::cout << "Accuracy: " << net3.accuracy(test) * 100 << "%\n";
}
}  // namespace neural_network
