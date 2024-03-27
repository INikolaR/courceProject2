#include <cstdio>
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include <Eigen/Dense>
#include <cmath>
#include "Layer.h"
#include "ActivationFunction.h"
#include "LossFunction.h"

namespace neural_network {
    using Index = Eigen::Index;
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    struct Element {
        Vector x;
        Vector y;
    };
    using ConstElemIterator = std::vector<Element>::const_iterator;

    class Net {
    public:
        const static ActivationFunction ReLU;
        const static ActivationFunction LeakyReLU;
        const static ActivationFunction Sigmoid;

        const static LossFunction Euclid;
        Net(std::initializer_list<int> k, std::initializer_list<ActivationFunction> f, LossFunction l);
        Matrix predict(const Matrix &x) const;
        double MSE(const std::vector<Element>& dataset) const;
        double accuracy(const std::vector<Element>& dataset) const;
        Index getInputSize() const;
        Index getOutputSize() const;

        template<class T>
        void fit(const std::vector<Element>& dataset, int n_of_batches, int n_of_epochs, T optimizer) {
            int64_t size_of_batch = dataset.size() / n_of_batches;
            std::vector<ConstElemIterator> borders(0);
            for (int64_t i = 0; i < n_of_batches; ++i) {
                borders.emplace_back(dataset.begin() + i * size_of_batch);
            }
            borders.push_back(dataset.end());
            for (int epoch = 0; epoch < n_of_epochs; epoch++) {
                std::cout << "epoch = " << epoch << std::endl;
                train_one_epoch(dataset, borders, optimizer);
                std::cout << "accuracy = " << accuracy(dataset) << std::endl;
            }
        }
    private:
        template <class T>
        void train_one_epoch(const std::vector<Element> &dataset, const std::vector<ConstElemIterator>& borders, T optimizer) {
            for (size_t i = 0; i < borders.size() - 1; ++i) {
                train_one_batch(dataset, borders[i], borders[i + 1], optimizer);
            }
            optimizer.updateToNextIteration();
        }

        template <class T>
        void train_one_batch(const std::vector<Element> &dataset, ConstElemIterator start, ConstElemIterator end, T optimizer) {
            std::list<Matrix> mid_values;
            Matrix u;
            Matrix x(getInputSize(), end - start);
            Matrix y(getOutputSize(), end - start);
            for (auto i = start; i != end; ++i) {
                x.col(i - start) = i->x;
                y.col(i - start) = i->y;
            }

            for (const auto& layer : layers_) {
                mid_values.emplace_back(x);
                x = layer.evaluate(x);
            }
            u = l_.derivativeDist(x, y);
            std::list<Matrix>::reverse_iterator it_x = mid_values.rbegin();
            for (std::list<Layer>::reverse_iterator layer = layers_.rbegin(); layer != layers_.rend(); ++layer, ++it_x) {
                Matrix grad_a = layer->getGradA(u, *it_x);
                Vector grad_b = layer->getGradB(u, *it_x);
                u = layer->getNextU(u, *it_x);
                layer->updA(optimizer.getNextGradientCorrection(grad_a));
                layer->updB(optimizer.getNextGradientCorrection(grad_b));
            }
        }
        int get_index_max(Vector &v) const;

        constexpr static double StartStep = 0.001;
        constexpr static double EpsilonMSE = 0.0001;
        constexpr static double LeakyReluA = 0.01;

        std::list<Layer> layers_;
        LossFunction l_;
    };
}
