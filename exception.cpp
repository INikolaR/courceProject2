#include <iostream>
#include "exception.h"

void neural_network::react() {
    try {
        throw;
    } catch(std::exception& e) {
        std::cout << "Exception: " << e.what() << "\n";
    } catch(...) {
        std::cout << "Unknown exception\n";
    }
}