#pragma once

#include "Net.h"

namespace neural_network {

void run_all_tests();

void test_echo();

void test_square();

void test_echo_vector();

void test_mnist();

std::vector<Element> parseMNISTDataset(const std::string& path_to_images_file, const std::string& path_to_labels_file);

int reverse_int(int i);
}  // namespace neural_network
