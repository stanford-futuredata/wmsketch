#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include "dataset.h"

namespace wmsketch {
namespace data {

SparseDataset::SparseDataset()
 : prng_(std::chrono::system_clock::now().time_since_epoch().count()),
   num_classes{0},
   feature_dim{0} { }

SparseDataset::SparseDataset(int32_t seed)
 : prng_(seed),
   num_classes{0},
   feature_dim{0} { }

SparseDataset::~SparseDataset() { };

void SparseDataset::seed(int32_t seed) {
  prng_.seed(seed);
}

uint32_t SparseDataset::num_examples() {
  return examples.size();
}

const SparseExample& SparseDataset::sample() {
  uint32_t idx = prng_() % examples.size();
  return examples[idx];
}

SparseDataset::const_iterator SparseDataset::begin() const {
  return examples.begin();
}

SparseDataset::const_iterator SparseDataset::end() const {
  return examples.end();
}

SparseDataset read_libsvm(std::string& file_path) {
  SparseDataset dataset;
  std::ifstream ifs(file_path);
  std::string line;
  std::string delim = ":";
  std::set<int> classes;

  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to read " + file_path);
  }

  while (std::getline(ifs, line)) {
    std::istringstream ls(line);
    std::string label_str, pair_str;
    ls >> label_str;

    SparseExample example;
    example.label = std::stoi(label_str);
    if (example.label == -1) {
      example.label = 0;  // normalize -1/+1 to 0/1 for LIBSVM datasets
    }
    classes.insert(example.label);

    while (ls >> pair_str) {
      size_t pos = pair_str.find(delim);
      uint32_t k = (uint32_t) std::stol(pair_str.substr(0, pos));
      float v = std::stof(pair_str.substr(pos+1, std::string::npos));
      example.features.emplace_back(std::pair<uint32_t, float>(k, v));
      if (k >= dataset.feature_dim) dataset.feature_dim = k + 1;
    }
    dataset.examples.push_back(example);
  }

  dataset.num_classes = classes.size();
  return dataset;
}

} // namespace data
} // namespace wmsketch
