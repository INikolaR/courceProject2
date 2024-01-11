#include <cstdio>
#include <random>
#include "ActivateFunction.h"
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd;

namespace neural_network {
    class Layer {
    public:
        Layer(int n, int m, ActivateFunction f) : n_(n), m_(m), f_(f), a_(genRandomMatrix(m, n)), b_(genRandomMatrix(m, 1)), to_upd_a_(Matrix::Zero(m, n)), to_upd_b_(Matrix::Zero(m, 1)) {}
        Matrix evaluate(const Matrix& input) const {
            return (a_ * input + b_).unaryExpr(&f_.evaluate0);
        }
        Matrix getGradA(const Matrix& u, const Matrix& x) {
            return ((a_ * x + b_).unaryExpr(&f_.evaluate1).asDiagonal() * u.transpose()) * x.transpose();
        }
        Matrix getGradB(const Matrix& u, const Matrix& x){
            return (a_ * x + b_).unaryExpr(&f_.evaluate1).asDiagonal() * u.transpose();
        }
        Matrix getNextU(const Matrix& u, const Matrix& x) {
            return (u * (a_ * x + b_).unaryExpr(&f_.evaluate1).asDiagonal()) * a_;
        }
        void addToUpdA(double step, const Matrix& grad) {
            to_upd_a_ += step * grad;
        }

        void addToUpdB(double step, const Matrix& grad) {
            to_upd_b_ += step * grad;
        }

        void updateAB() {
            a_ += to_upd_a_;
            b_ += to_upd_b_;
            to_upd_a_.setZero();
            to_upd_b_.setZero();
        }

        int getInputSize() const {
            return n_;
        }

        int getOutputSize() const {
            return m_;
        }

    private:
        Matrix genRandomMatrix(int n, int m) {
            std::mt19937 engine(this->SEED);
            std::normal_distribution<double> dis(0.0, 1.0);
            return Matrix::NullaryExpr(n, m ,[&](){return dis(engine);});
        }

        int n_;
        int m_;
        Matrix a_;
        Matrix b_;
        Matrix to_upd_a_;
        Matrix to_upd_b_;
        ActivateFunction f_;

        const int SEED = 23423;
    };
}
