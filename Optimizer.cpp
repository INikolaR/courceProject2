#include "Optimizer.h"
namespace neural_network {
Optimizer::Optimizer(
    std::function<void(std::list<Layer> *, const LossFunction &,
                       const std::vector<TrainUnit> &, int, int)> &&f)
    : f_(std::move(f)) {
}
void Optimizer::fit(std::list<Layer> *layers, const LossFunction &l,
                    const std::vector<TrainUnit> &dataset, int size_of_batch,
                    int n_of_epochs) {
    f_(layers, l, dataset, size_of_batch, n_of_epochs);
}
}  // namespace neural_network
