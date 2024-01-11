#pragma once
namespace neural_network {
    class ActivateFunction {
    public:
        static double evaluate0(double value);

        static double evaluate1(double value);
    private:
        constexpr static double A = 0.01;
    };
}
