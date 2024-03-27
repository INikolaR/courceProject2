#include "test.h"
#include "exception.h"
#include <iostream>
#include <signal.h>

#include <fstream>

int main() {
    try {
        neural_network::run_all_tests();
    } catch(...) {
        neural_network::react();
    }
    return 0;
}
