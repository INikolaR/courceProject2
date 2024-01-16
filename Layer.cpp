#include "Layer.h"
#include <EigenRand/EigenRand>

namespace neural_network {
    std::mt19937 Layer::engine = std::mt19937(42345);

    Layer::Layer(int input_dimension, int output_dimension, ActivationFunction f) : f_(f), a_(Eigen::Rand::normal<Matrix>(output_dimension, input_dimension, engine)), b_(Eigen::Rand::normal<Matrix>(output_dimension, 1, engine)), to_upd_a_(Matrix::Zero(output_dimension, input_dimension)), to_upd_b_(Vector::Zero(output_dimension)) {}

    Matrix neural_network::Layer::evaluate(const Matrix &input) const {
        return (a_ * input + b_).unaryExpr(&f_.evaluate0);
    }

    Matrix neural_network::Layer::getGradA(const Matrix &u, const Matrix &x) {
        return ((a_ * x + b_).unaryExpr(&f_.evaluate1).asDiagonal() * u.transpose()) * x.transpose();
    }

    Matrix neural_network::Layer::getGradB(const Matrix &u, const Matrix &x) {
        return (a_ * x + b_).unaryExpr(&f_.evaluate1).asDiagonal() * u.transpose();
    }

    Matrix neural_network::Layer::getNextU(const Matrix &u, const Matrix &x) {
        return (u * (a_ * x + b_).unaryExpr(&f_.evaluate1).asDiagonal()) * a_;
    }

    void neural_network::Layer::addToUpdA(double step, const Matrix &grad) {
        to_upd_a_ += step * grad;
    }

    void neural_network::Layer::addToUpdB(double step, const Matrix &grad) {
        to_upd_b_ += step * grad;
    }

    void neural_network::Layer::updateAndResetWeights() {
        a_ += to_upd_a_;
        b_ += to_upd_b_;
        to_upd_a_.setZero();
        to_upd_b_.setZero();
    }

    Index Layer::getInputSize() const {
        return a_.cols();
    }

    Index Layer::getOutputSize() const {
        return a_.rows();
    }

//    Matrix Layer::genRandomMatrix(int n, int m) {
//        std::mt19937 engine(this->Seed);
//        std::normal_distribution<double> dis(0.0, 1.0);
//        return Matrix::NullaryExpr(n, m, [&]() { return dis(engine); });
//    }
//
//    Matrix Layer::genRandomVector(int n) {
//        std::mt19937 engine(this->Seed);
//        std::normal_distribution<double> dis(0.0, 1.0);
//        return Vector::NullaryExpr(n, [&]() { return dis(engine); });
//    }
}
