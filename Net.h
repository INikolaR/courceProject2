#include <cstdio>
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include <Eigen/Dense>
#include "Layer.h"

namespace neural_network {
    using Matrix = Eigen::MatrixXd;
    using Batch = std::vector<std::pair<Matrix, Matrix>>;

    class Net {
    public:
        Net(const std::initializer_list<int>& k) : layers_(std::list<Layer>()), x_(std::list<Matrix>()) {
            auto curr = k.begin();
            auto prev = curr++;
            for (;curr != k.end();++prev, ++curr) {
                layers_.push_back(Layer(*prev, *curr, ActivateFunction()));
            }
        }
        Matrix evaluate(const Matrix &x) {
            Matrix z = x;
            for (const auto& l : layers_) {
                z = l.evaluate(z);
            }
            return z;
        }

        void learn(std::vector<Batch> dataset, int times) {
            for (int i = 0; i < times; ++i) {
                printWeights();
                double step = StartStep / (i + 1);
                for (const auto &b: dataset) {
                    x_.clear();
                    for (const auto &xy: b) {
                        Matrix z = xy.first;
                        for (const auto &l: layers_) {
                            x_.push_back(z);
                            z = l.evaluate(z);
                        }
                        Matrix u = dLdz(z, xy.second);
                        for (auto it_layers = layers_.end(); it_layers != layers_.begin();) {
                            --it_layers;
                            it_layers->addToUpdA(step, it_layers->getGradA(u, x_.back()));
                            it_layers->addToUpdB(step, it_layers->getGradB(u, x_.back()));
                            u = it_layers->getNextU(u, x_.back());
                            x_.pop_back(); // TODO make upd after batch instd of in it.
                        }
                    }
                    for (auto &l: layers_) {
                        l.updateAB();
                    }
                }
            }
        }

        void printWeights() {
            for (const auto& l : layers_) {
                std::cout << "LAYER:\n";
                l.printAB();
            }
            std::cout << "================================\n\n\n";
        }

    private:
        std::list<Layer> layers_;
        std::list<Matrix> x_;
        constexpr static double StartStep = 0.1;

        Matrix dLdz(const Matrix &z, const Matrix &y) {
            return 2 * (z.transpose() * (z - y).asDiagonal());
        }
    };
}
