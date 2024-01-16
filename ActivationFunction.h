#pragma once

namespace neural_network {
    class ActivationFunction {
    public:
        static double evaluate0(double x);
        static double evaluate1(double x);
    private:
        constexpr static double A = 0.01;
    };
}
