#include <iostream>
#include "Net.h"
#include <Eigen/Dense>

int main() {
    neural_network::Net n{1, 10, 1};
    std::vector<neural_network::Batch> dataset = {
            {
                {neural_network::Matrix{{1}}, neural_network::Matrix{{1}}},
                {neural_network::Matrix{{2}}, neural_network::Matrix{{4}}},
                {neural_network::Matrix{{3}}, neural_network::Matrix{{9}}},
                {neural_network::Matrix{{4}}, neural_network::Matrix{{16}}}},
            {
                    {neural_network::Matrix{{5}}, neural_network::Matrix{{25}}},
                    {neural_network::Matrix{{6}}, neural_network::Matrix{{36}}},
                    {neural_network::Matrix{{7}}, neural_network::Matrix{{49}}},
                    {neural_network::Matrix{{8}}, neural_network::Matrix{{64}}}},
    };
    n.learn(dataset, 10);
    std::cout << n.evaluate(neural_network::Matrix{{2}}) << "\n";
    return 0;
}
