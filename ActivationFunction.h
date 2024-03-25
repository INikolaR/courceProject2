#pragma once
#include <functional>

namespace neural_network {
    class ActivationFunction {
    public:
        ActivationFunction(std::function<double(double)>&& f0, std::function<double(double)>&& f1);
        double evaluate0(double x) const;
        double evaluate1(double x) const;
    private:
        const std::function<double(double)> f0_;
        const std::function<double(double)> f1_;
    };
}
