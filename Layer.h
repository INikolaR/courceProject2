#include <cstdio>
#include <random>
#include "ActivateFunction.h"
#include <Eigen/Dense>

namespace neural_network {
    using Matrix = Eigen::MatrixXd;
    using Index = Eigen::Index;
    using Vector = Eigen::VectorXd;

    class Layer {
    public:
        Layer(int input_dimension, int output_dimension, ActivateFunction f);
        Matrix evaluate(const Matrix& input) const;
        Matrix getGradA(const Matrix& u, const Matrix& x);
        Matrix getGradB(const Matrix& u, const Matrix& x);
        Matrix getNextU(const Matrix& u, const Matrix& x);
        void addToUpdA(double step, const Matrix& grad);
        void addToUpdB(double step, const Matrix& grad);
        void updateAndResetWeights();
        Index getInputSize() const;
        Index getOutputSize() const;
    private:
        Matrix genRandomMatrix(int n, int m);
        Matrix genRandomVector(int n);

        constexpr static int Seed = 23423;

        Matrix a_;
        Vector b_;
        Matrix to_upd_a_;
        Matrix to_upd_b_;
        ActivateFunction f_;
    };
}
