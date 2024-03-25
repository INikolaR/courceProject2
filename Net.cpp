#include "Net.h"
#include "ActivationFunction.h"

#include <limits>

namespace neural_network {
    const ActivationFunction Net::ReLU = ActivationFunction([](double x) {return (x > 0) * x;}, [](double x) {return (x > 0);});
    const ActivationFunction Net::LeakyReLU = ActivationFunction([](double x) {return (x > 0) * (1 - LeakyReluA) * x + LeakyReluA * x;}, [](double x) {return (x > 0) * (1 - LeakyReluA) + LeakyReluA;});

    const LossFunction Net::Euclid = LossFunction(
        [](const Matrix& x, const Matrix& y) {Matrix d = x - y; return (d.transpose() * d)(0, 0);},
        [](const Matrix& x, const Matrix& y) {return 2 * (x - y).transpose();}
        );

    Net::Net(std::initializer_list<int> k, std::initializer_list<ActivationFunction> f, LossFunction l) : l_(l) {
        auto curr_size = k.begin();
        auto prev_size = curr_size++;
        auto activation_function = f.begin();
        for (; curr_size != k.end(); ++prev_size, ++curr_size, ++activation_function) {
            layers_.emplace_back(Layer(*prev_size, *curr_size, *activation_function));
        }
    }

    Matrix Net::predict(const neural_network::Matrix &x) const {
        Matrix z = x;
        for (const auto &l: layers_) {
            z = l.evaluate(z);
        }
        return z;
    }

    void Net::fit(const std::vector<Element> &dataset, int n_of_batches, int n_of_epochs, double start_step) {
        int64_t size_of_batch = dataset.size() / n_of_batches;
        std::vector<ConstElemIterator> borders(0);
        for (int64_t i = 0; i < n_of_batches; ++i) {
            borders.emplace_back(dataset.begin() + i * size_of_batch);
        }
        borders.push_back(dataset.end());
        for (int epoch = 0; epoch < n_of_epochs; epoch++) {
            train_one_epoch(dataset, borders, start_step);
        }
    }

    void Net::train_one_epoch(const std::vector<Element> &dataset, const std::vector<ConstElemIterator> &borders, double start_step) {
        for (size_t i = 0; i < borders.size() - 1; ++i) {
            train_one_batch(dataset, borders[i], borders[i + 1], start_step);
        }
    }
    void Net::train_one_batch(const std::vector<Element> &dataset, ConstElemIterator start, ConstElemIterator end, double start_step) {
        double step = start_step;
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
            layer->updA(step, grad_a);
            layer->updB(step, grad_b);
        }
    }
    Index Net::getInputSize() const {
        return layers_.front().getInputSize();
    }
    Index Net::getOutputSize() const {
        return layers_.back().getOutputSize();
    }
    double Net::MSE(const std::vector<Element> &dataset) const {
        double mse = 0;
        for (int i = 0; i < dataset.size(); ++i) {
            mse += l_.dist(predict(dataset[i].x), dataset[i].y);
        }
        return mse;
    }
    }
