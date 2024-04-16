#include "MomentumOptimizer.h"
namespace neural_network {
MomentumOptimizer::MomentumOptimizer(double alpha, double beta)
    : alpha_(alpha), beta_(beta) {
}

void MomentumOptimizer::operator()(std::list<Layer> *layers,
                                   const LossFunction &l,
                                   const std::vector<TrainUnit> &dataset,
                                   int size_of_batch, int n_of_epochs) {
    h_.clear();
    for (const auto &layer : *layers) {
        h_.push_back({Matrix::Zero(layer.getInputSize(), layer.getOutputSize()),
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

void MomentumOptimizer::train_one_epoch(
    std::list<Layer> *layers, const LossFunction &l,
    const std::vector<TrainUnit> &dataset,
    std::vector<ConstTrainUnitIterator> &borders) {
    for (size_t i = 0; i < borders.size() - 1; ++i) {
        train_one_batch(layers, l, borders[i], borders[i + 1]);
    }
}

void MomentumOptimizer::train_one_batch(std::list<Layer> *layers,
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
    std::list<Momentum>::reverse_iterator it_h = h_.rbegin();
    for (std::list<Layer>::reverse_iterator layer = layers->rbegin();
         layer != layers->rend(); ++layer, ++it_x, ++it_h) {
        Matrix grad_a = layer->getGradA(u, *it_x);
        Vector grad_b = layer->getGradB(u, *it_x);
        u = layer->getNextU(u, *it_x);
        it_h->a = alpha_ * grad_a + beta_ * it_h->a;
        it_h->b = alpha_ * grad_b + beta_ * it_h->b;
        layer->updA(it_h->a);
        layer->updB(it_h->b);
    }
}
}  // namespace neural_network
