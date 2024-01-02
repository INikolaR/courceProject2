#include <iostream>
#include "ActivateFunction.h"
#include "Matrix.h"

int main() {
    ActivateFunction f = ActivateFunction();
    std::cout << f.evaluate0(5) << std::endl;
    Matrix m({{1, 2}, {3, 4}});
    Matrix m2 = m * m;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << m2[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}