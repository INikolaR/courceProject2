#pragma once

#include "Net.h"

namespace neural_network {
void run_all_tests();
void test_echo();
void test_square();
void test_echo_vector();
std::vector<TrainUnit> parseMNISTDataset(const std::string& path_to_images_file, const std::string& path_to_labels_file);
void test_mnist();
}  // namespace neural_network
