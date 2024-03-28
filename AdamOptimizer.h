#pragma once

#include <Eigen/Dense>

#include "Net.h"

namespace neural_network {

struct Momentum {
    Matrix a;
    Vector b;
};

class AdamOptimizer {
public:
    AdamOptimizer(double start_step, double beta1, double beta2, double epsilon);
    void reset(std::list<Layer> &layers);
    void trainOneBatch(std::list<Layer>& layers, const LossFunction& l, const std::vector<Element> &dataset, ConstElemIterator start,
                       ConstElemIterator end);
private:
    double start_step_;

    double beta1_;
    double beta2_;
    double epsilon_;

    double step_;
    double beta1_t_;
    double beta2_t_;

    std::list<Momentum> m_;
    std::list<Momentum> v_;
};
}
