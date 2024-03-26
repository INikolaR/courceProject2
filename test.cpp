#include "test.h"

#include <fstream>
#include <string>

#include "Net.h"

namespace neural_network {
    void run_all_tests() {
        test_echo();
//        test_echo_vector();
        test_square();
        test_mnist();
    }

    void test_echo() {
        std::cout << "ECHO TEST: just trying to return the input\n";
        Net net{{1, 10, 1}, {Net::LeakyReLU, Net::LeakyReLU}, Net::Euclid};
        std::vector<Element> dataset = {{Matrix{{1}}, Matrix{{1}}},
                                                        {Matrix{{2}}, Matrix{{2}}},
                                                        {Matrix{{3}}, Matrix{{3}}},
                                                        {Matrix{{4}}, Matrix{{4}}},
                                                        {Matrix{{5}}, Matrix{{5}}},
                                                        {Matrix{{6}}, Matrix{{6}}},
                                                        {Matrix{{7}}, Matrix{{7}}},
                                                        {Matrix{{8}}, Matrix{{8}}}};
        net.fit(dataset, 1, 1000, 0.01);
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
        net.fit(dataset, 1, 10000, 0.001);
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
        net.fit(dataset, 1, 1000, 0.00002);
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
        Net net{{784, 256, 10}, {Net::LeakyReLU, Net::LeakyReLU}, Net::Euclid};

        std::ifstream file_images("../train-images-idx3-ubyte/train-images.idx3-ubyte", std::ios::binary);
        if (!file_images.is_open()) {
            std::cout << "Cannot open training images\n";
            return;
        }
        int images_magic_number = 0;
        const int expected_images_magic_number = 2051;
        file_images.read(reinterpret_cast<char*>(&images_magic_number), sizeof(images_magic_number));
        images_magic_number = reverse_int(images_magic_number);
        if (images_magic_number != expected_images_magic_number) {
            std::cout << "Bad MNIST image file!\n";
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
            return;
        }

        int labels_magic_number = 0;
        const int expected_labels_magic_number = 2049;
        file_labels.read(reinterpret_cast<char*>(&labels_magic_number), sizeof(labels_magic_number));
        labels_magic_number = reverse_int(labels_magic_number);
        if (labels_magic_number != expected_labels_magic_number) {
            std::cout << "Bad MNIST label file!\n";
            return;
        }

        int number_of_labels = 0;
        file_labels.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
        number_of_labels = reverse_int(number_of_labels);

        if (number_of_labels != number_of_images) {
            std::cout << "Different number of rows in images and labels!\n";
            return;
        }
        std::vector<Element> dataset(0);
        unsigned char image[size_of_mnist_image];
        unsigned char label = 0;
        for (int i = 0; i < number_of_labels; i++) {
            file_images.read(reinterpret_cast<char*>(image), size_of_mnist_image);
            double array_image[size_of_mnist_image];
            for (int j = 0; j < size_of_mnist_image; ++j) {
                array_image[j] = static_cast<double>(image[j]);
            }
            Vector x = Eigen::Map<Vector>(array_image, size_of_mnist_image);
            file_labels.read(reinterpret_cast<char*>(&label), 1);
            double array_label[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            array_label[static_cast<unsigned int>(label)] = 1;
            Vector y = Eigen::Map<Vector>(array_label, 10);
            dataset.emplace_back(Element{x, y});
        }
        net.fit(dataset, 1000, 1, 0.001);
    }

}
