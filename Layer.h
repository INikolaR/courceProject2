#include <cstdio>
#include <random>
#include "ActivateFunction.h"
#include "Matrix.h"

template <std::size_t N, std::size_t M>
class Layer {
public:
    explicit Layer<N, M>(ActivateFunction f) : f_(f), a_(generateRandomMatrix<M, N>()), b_(generateRandomMatrix<1, M>()) {}
    Matrix<1, M> evaluate(const Matrix<1, N>& input) {
        return (a_ * input + b_).forAll(f_.evaluate0);
    }
    Matrix<M, N> getGradA(const Matrix<1, M>& u, const Matrix<N, 1>& x) {
        Matrix sigma1 = (a_ * x + b_).forAll(f_.evaluate1).transposed();
        Matrix u_sigma1 = Matrix<M, 1>();
        for (std::size_t i = 0; i < M; ++i) {
            u_sigma1[i][0] = sigma1[0][i] * u[0][i];
        }
        return u_sigma1 * x.transposed();
    }
    Matrix<M, 1> getGradB(const Matrix<M, 1>& u, const Matrix<1, M>& x){
        Matrix sigma1 = (a_ * x + b_).forAll(f_.evaluate1).transposed();
        Matrix u_sigma1 = Matrix<M, 1>();
        for (std::size_t i = 0; i < M; ++i) {
            u_sigma1[i][0] = sigma1[0][i] * u[0][i];
        }
        return u_sigma1;
    }
    Matrix<1, N> getNextU(const Matrix<M, 1>& u, const Matrix<1, M>& x) {
        Matrix sigma1 = (a_ * x + b_).forAll(f_.evaluate1).transposed();
        Matrix u_sigma1 = Matrix<1, M>();
        for (std::size_t i = 0; i < M; ++i) {
            u_sigma1[0][i] = sigma1[0][i] * u[0][i];
        }
        return u_sigma1 * a_;
    }
private:
    ActivateFunction f_;
    Matrix<M, N> a_;
    Matrix<M, 1> b_;

    template <std::size_t P, std::size_t Q>
    Matrix<P, Q> generateRandomMatrix() {
        std::mt19937 engine(time(0));
        Matrix<P, Q> result = Matrix<P, Q>();
        for (std::size_t i = 0; i < P; ++i) {
            for (std::size_t j = 0; j < Q; ++j) {
                result[i][j] = engine() / static_cast<double>(engine.max());
            }
        }
    }
};
