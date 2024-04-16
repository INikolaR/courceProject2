#include "AdamOptimizer.h"
namespace neural_network {
AdamOptimizer::AdamOptimizer(double start_alpha, double beta1, double beta2,
                             double epsilon)
    : start_alpha_(start_alpha),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      beta1_t_(beta1_),
      beta2_t_(beta2_) {
    assert(start_alpha_ > 0);
    assert(beta1_ > 0);
    assert(beta2_ > 0);
    assert(epsilon_ > 0);
}
void AdamOptimizer::operator()(std::list<Layer> *layers, const LossFunction &l,
                               const std::vector<TrainUnit> &dataset,
                               int size_of_batch, int n_of_epochs) {
    assert(layers != nullptr);
    assert(dataset.size() > 0);
    assert(size_of_batch > 0);
    assert(n_of_epochs > 0);
    m_.clear();
    v_.clear();
    for (const auto &layer : *layers) {
        m_.push_back({Matrix::Zero(layer.getInputSize(), layer.getOutputSize()),
                      Vector::Zero(layer.getOutputSize())});
        v_.push_back({Matrix::Zero(layer.getInputSize(), layer.getOutputSize()),
                      Vector::Zero(layer.getOutputSize())});
    }
    std::vector<ConstTrainUnitIterator> borders(0);
    for (auto it = dataset.begin(); it < dataset.end(); it += size_of_batch) {
        borders.emplace_back(it);
    }
    borders.push_back(dataset.end());
    for (int epoch = 0; epoch < n_of_epochs; epoch++) {
        train_one_epoch(layers, l, dataset, borders);
    }
}

void AdamOptimizer::train_one_epoch(
    std::list<Layer> *layers, const LossFunction &l,
    const std::vector<TrainUnit> &dataset,
    std::vector<ConstTrainUnitIterator> &borders) {
    for (size_t i = 0; i < borders.size() - 1; ++i) {
        train_one_batch(layers, l, borders[i], borders[i + 1]);
    }
}

void AdamOptimizer::train_one_batch(std::list<Layer> *layers,
                                    const LossFunction &l,
                                    ConstTrainUnitIterator start,
                                    ConstTrainUnitIterator end) {
    std::list<Matrix> mid_values;
    Matrix u;
    Matrix x(layers->front().getInputSize(), end - start);
    Matrix y(layers->back().getOutputSize(), end - start);
    for (auto i = start; i != end; ++i) {
        x.col(i - start) = i->x;
        y.col(i - start) = i->y;
    }

    for (const auto &layer : *layers) {
        mid_values.emplace_back(x);
        x = layer.evaluate(x);
    }
    u = l.derivativeDist(x, y);
    std::list<Matrix>::reverse_iterator it_x = mid_values.rbegin();
    std::list<Momentum>::reverse_iterator it_m = m_.rbegin();
    std::list<Momentum>::reverse_iterator it_v = v_.rbegin();
    for (std::list<Layer>::reverse_iterator layer = layers->rbegin();
         layer != layers->rend(); ++layer, ++it_x, ++it_m, ++it_v) {
        Matrix grad_a = layer->getGradA(u, *it_x);
        Vector grad_b = layer->getGradB(u, *it_x);
        *it_m = {beta1_ * it_m->a + (1 - beta1_) * grad_a,
                 beta1_ * it_m->b + (1 - beta1_) * grad_b};
        *it_v = {beta2_ * it_v->a + (1 - beta2_) * grad_a.cwiseProduct(grad_a),
                 beta2_ * it_v->b + (1 - beta2_) * grad_b.cwiseProduct(grad_b)};
        double current_step =
            start_alpha_ * sqrt(1 - beta2_t_) / (1 - beta1_t_);
        beta1_t_ *= beta1_;
        beta2_t_ *= beta2_;
        u = layer->getNextU(u, *it_x);
        layer->updA(
            current_step *
            it_m->a.cwiseProduct(
                (it_v->a.cwiseSqrt() +
                 Matrix::Ones(it_v->a.rows(), it_v->a.cols()) * epsilon_)
                    .unaryExpr([](double x) { return 1 / x; })));
        layer->updB(
            current_step *
            it_m->b.cwiseProduct(
                (it_v->b.cwiseSqrt() +
                 Matrix::Ones(it_v->b.rows(), it_v->b.cols()) * epsilon_)
                    .unaryExpr([](double x) { return 1 / x; })));
    }
}
}  // namespace neural_network
