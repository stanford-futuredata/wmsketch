#ifndef SRC_PAIRED_COUNTMIN_H_
#define SRC_PAIRED_COUNTMIN_H_

#include <vector>
#include "hash.h"
#include "binary_estimator.h"

namespace wmsketch {

class PairedCountMin : public BinaryEstimator {

 public:
  static const uint32_t MAX_LOG2_WIDTH = 30;

 private:
  const uint32_t depth_;
  const float smooth_;
  const bool consv_update_;
  uint32_t width_mask_;
  uint32_t** counts_num_;
  uint32_t** counts_den_;
  uint32_t pos_count_, neg_count_;
  hash::PolynomialHash hash_fn_;
  std::vector<uint32_t> hash_buf_;

 public:
  /**
   * Estimator for the ratios p(x_i = 1 | y = +1) / p(x_i = 1 | y = -1) using a pair of
   * Count-Min sketches.
   *
   * @param log2_width Base-2 logarithm of the sketch width.
   * @param depth Sketch depth.
   * @param seed Random seed.
   * @param smooth Laplace smoothing of count estimates.
   * @param consv_update Flag to enable conservative update heuristic.
   */
  PairedCountMin(uint32_t log2_width, uint32_t depth, int32_t seed, float smooth = 1., bool consv_update = false);
  ~PairedCountMin();
  float get(uint32_t key);
  bool update(uint32_t key, bool label);
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label);
  bool update(std::vector<float>& new_weights, const std::vector<std::pair<uint32_t, float> >& x, bool label);
  float bias();

 private:
  float update_feature(uint32_t key, bool label);
};

} // namespace wmsketch

#endif /* SRC_PAIRED_COUNTMIN_H_ */
