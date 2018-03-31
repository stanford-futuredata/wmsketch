#ifndef DATASET_H_
#define DATASET_H_

#include <vector>
#include <random>
#include <chrono>

namespace wmsketch {
namespace data {

typedef struct SparseExample {
  int32_t label;
  std::vector<std::pair<uint32_t, float> > features;
} SparseExample;


class SparseDataset {
 private:
  std::mt19937 prng_;

 public:
  uint32_t num_classes;
  uint32_t feature_dim;
  std::vector<SparseExample> examples;
  typedef typename std::vector<SparseExample>::const_iterator const_iterator;

  SparseDataset();
  SparseDataset(int32_t seed);
  ~SparseDataset();
  void seed(int32_t seed);
  uint32_t num_examples();
  const SparseExample& sample();
  const_iterator begin() const;
  const_iterator end() const;
};

/**
 * Read a dataset in LibSVM format.
 *
 * @param file_path Path to file.
 * @return The dataset.
 */
SparseDataset read_libsvm(std::string& file_path);

} // namespace data
} // namespace wmsketch

#endif /* DATASET_H_ */
