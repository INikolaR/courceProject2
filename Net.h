#include <cstdio>
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include "Layer.h"

template <std::size_t I, std::size_t M1, std::size_t O>
class Net {
public:
    Net<I, M1, O>() = default;
    Matrix<O, 1> evaluate(const Matrix<I, 1>& x) {
        Matrix<M1, 1> w = l1_.evaluate(x);
        Matrix<O, M1> y = l2_.evaluate(w);
        return y;
    }
    void learn(std::vector<std::pair<Matrix<I, 1>, Matrix<O, 1>>> dataset) {
        for (std::size_t i = 0; i < dataset.size(); ++i) {
            Matrix<I, 1> x = dataset[i].first;
            Matrix<M1, 1> w = l1_.evaluate(x);
            Matrix<O, 1> z = l2_.evaluate(w);
            Matrix<1, O> u1 = dLdz(z, dataset[i].second);
            Matrix<1, M1> u2 = l2_.getNextU(u1, w);
            l2_.updA();
        }
    }
private:
    Layer<I, M1> l1_ = Layer<I, M1>(ActivateFunction());
    Layer<M1, O> l2_ = Layer<M1, O>(ActivateFunction());
    Matrix<1, O> dLdz(const Matrix<O, 1>& z, const Matrix<O, 1>& y) {
        return 2 * (z ^ (z - y)).transposed();
    }
};
