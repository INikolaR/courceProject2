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
        return ((a_ * x + b_).forAll(f_.evaluate1) ^ u.transposed()) * x.transposed();
    }
    Matrix<M, 1> getGradB(const Matrix<1, M>& u, const Matrix<N, 1>& x){
        return (a_ * x + b_).forAll(f_.evaluate1) ^ u.transposed();
    }
    Matrix<1, N> getNextU(const Matrix<1, M>& u, const Matrix<N, 1>& x) {
        return (u ^ (a_ * x + b_).forAll(f_.evaluate1).transposed()) * a_;
    }
    void updA(double step, const Matrix<M, N>& grad) {
        a_ = a_ - (step * grad);
    }

    void updB(double step, const Matrix<M, 1>& grad) {
        b_ = b_ - (step * grad);
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
