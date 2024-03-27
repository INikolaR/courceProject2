#include <cstdio>
#include <random>
#include "ActivationFunction.h"
#include <Eigen/Dense>

namespace neural_network {
    using Matrix = Eigen::MatrixXd;
    using Index = Eigen::Index;
    using Vector = Eigen::VectorXd;

    class Layer {
    public:
        static std::mt19937 engine;

        Layer(int input_dimension, int output_dimension, const ActivationFunction f);
        Matrix evaluate(const Matrix& input) const;
        Matrix getGradA(const Matrix& u, const Matrix& x);
        Matrix getGradB(const Matrix& u, const Matrix& x);
        Matrix getNextU(const Matrix& u, const Matrix& x);
        void updA(const Matrix& gradient_correction);
        void updB(const Vector& gradient_correction);
        Index getInputSize() const;
        Index getOutputSize() const;
    private:
        Matrix a_;
        Vector b_;
        Matrix to_upd_a_;
        Matrix to_upd_b_;
        const ActivationFunction f_;
    };
}
