#include <iostream>
#include "ActivateFunction.h"
#include "Matrix.h"

int main() {
    ActivateFunction f = ActivateFunction();
    std::cout << f.evaluate0(5) << " " << f.evaluate1(5) << std::endl;
    Matrix<2, 2> m({{{1, 2}, {3, 4}}});
    Matrix<3, 1> m1({{{1}, {2}, {3}}});
    Matrix<1, 2> m2({1, 2});
    auto r = m1 * m2;
    std::cout << r.size().first << " " << r.size().second << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << r[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}