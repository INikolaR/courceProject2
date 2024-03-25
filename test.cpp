#include "test.h"
#include "Net.h"

void neural_network::run_all_tests() {
    test_echo();
    test_echo_vector();
    test_square();
}

void neural_network::test_echo() {
    std::cout << "ECHO TEST: just trying to return the input\n";
    neural_network::Net net{{1, 10, 1}, {Net::LeakyReLU, Net::LeakyReLU}, Net::Euclid};
    std::vector<neural_network::Element> dataset = {
                    {neural_network::Matrix{{1}}, neural_network::Matrix{{1}}},
                    {neural_network::Matrix{{2}}, neural_network::Matrix{{2}}},
                    {neural_network::Matrix{{3}}, neural_network::Matrix{{3}}},
                    {neural_network::Matrix{{4}}, neural_network::Matrix{{4}}},
                    {neural_network::Matrix{{5}}, neural_network::Matrix{{5}}},
                    {neural_network::Matrix{{6}}, neural_network::Matrix{{6}}},
                    {neural_network::Matrix{{7}}, neural_network::Matrix{{7}}},
                    {neural_network::Matrix{{8}}, neural_network::Matrix{{8}}}};
    net.fit(dataset, 1, 1000, 0.01);
    for (double i = 0; i < 12; ++i) {
        std::cout << i << " -> " << net.predict(neural_network::Matrix{{i}}) << "\n";
    }
}

void neural_network::test_echo_vector() {
    std::cout << "ECHO TEST: just trying to return the input\n";
    neural_network::Net net{{2, 100, 100, 2}, {Net::LeakyReLU, Net::LeakyReLU, Net::LeakyReLU}, Net::Euclid};
    std::vector<neural_network::Element> dataset = {
                    {neural_network::Matrix{{1}, {3}}, neural_network::Matrix{{1}, {3}}},
                    {neural_network::Matrix{{2}, {1}}, neural_network::Matrix{{2}, {1}}},
                    {neural_network::Matrix{{3}, {4}}, neural_network::Matrix{{3}, {4}}},
                    {neural_network::Matrix{{10}, {-5}}, neural_network::Matrix{{10}, {-5}}},
                    {neural_network::Matrix{{3}, {6}}, neural_network::Matrix{{3}, {6}}},
                    {neural_network::Matrix{{7}, {6}}, neural_network::Matrix{{7}, {6}}},
                    {neural_network::Matrix{{1}, {6}}, neural_network::Matrix{{1}, {6}}},
                    {neural_network::Matrix{{8}, {0}}, neural_network::Matrix{{8}, {0}}}};
    net.fit(dataset, 1, 1000, 0.00008);
    std::cout << "MSE = " << net.MSE(dataset) << "\nFor example:\n";
    for (double i = 0; i < 8; ++i) {
        std::cout << dataset[i].x << "\n->\n" << net.predict(dataset[i].x) << "\n";
    }
}

void neural_network::test_square() {
    std::cout << "SQUARE TEST: trying to return x^2 if x is in input (x is scalar)\n";
    neural_network::Net net{{1, 20, 20, 1}, {Net::LeakyReLU, Net::LeakyReLU, Net::LeakyReLU}, Net::Euclid};
    std::vector<neural_network::Element> dataset = {
        {neural_network::Matrix{{1}}, neural_network::Matrix{{1}}},
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
