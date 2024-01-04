#include <iostream>
#include "ActivateFunction.h"
#include "Matrix.h"
#include "Layer.h"
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd test {{1, 2},
                            {3, 4},
                            {5, 6}};
    Eigen::MatrixXd test2 {{2, 1}, {0, 4}, {3, 4}};
    Eigen::MatrixXd test3 = test * test2;
    std::cout << test3 << std::endl;
    return 0;
}