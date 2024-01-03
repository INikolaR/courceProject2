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
    Matrix<O, 1> evaluate(Matrix<I, 1>) {
        
    }
private:
    Layer<I, M1> l1_ = Layer<I, M1>();
    Layer<M1, O> l2_ = Layer<M1, O>();
};
