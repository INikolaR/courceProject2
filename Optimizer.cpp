#include "Optimizer.h"

#include "AdamOptimizer.h"
#include "ConstantOptimizer.h"
#include "MomentumOptimizer.h"
namespace neural_network {
Optimizer Optimizer::Constant(double alpha) {
    return Optimizer(ConstantOptimizer(alpha));
}
Optimizer Optimizer::Momentum(double alpha, double beta) {
    return Optimizer(MomentumOptimizer(alpha, beta));
}
Optimizer Optimizer::Adam(double start_alpha, double beta1, double beta2,
                          double epsilon) {
    return Optimizer(AdamOptimizer(start_alpha, beta1, beta2, epsilon));
}
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
