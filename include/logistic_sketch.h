/*
 * Logistic regression with the Weight-Median Sketch.
 */

#ifndef LOGISTIC_SKETCH_H_
#define LOGISTIC_SKETCH_H_

#include <vector>
#include "binary_estimator.h"
#include "hash.h"

namespace wmsketch {

class LogisticSketch : public BinaryEstimator {

 public:
  static const uint32_t MAX_LOG2_WIDTH = 31;

 private:
  float** weights_;
  float bias_;
  const float lr_init_;
  const float l2_reg_;
  float scale_;
  uint64_t t_;
  const uint32_t depth_;
  uint32_t width_mask_;
  const bool median_update_;
  hash::TabulationHash hash_fn_;
  std::vector<uint32_t> hash_buf_;
  std::vector<float> weight_buf_, weight_medians_, weight_means_;

 public:
  LogisticSketch(
      uint32_t log2_width,
      uint32_t depth,
      int32_t seed,
      float lr_init = 0.1,
      float l2_reg = 1e-3,
      bool median_update = false);
  ~LogisticSketch() override;
  float get(uint32_t key) override;
  float dot(const std::vector<std::pair<uint32_t, float> >& x);
  bool predict(uint32_t key);
  bool predict(const std::vector<std::pair<uint32_t, float> >& x);
  bool update(uint32_t key, bool label) override;
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label) override;
  bool update(std::vector<float>& new_weights, const std::vector<std::pair<uint32_t, float> >& x, bool label) override;
  float bias() override;
  float scale();

 private:
  float get_weight(uint32_t key, bool use_median);
  void get_weights(const std::vector<std::pair<uint32_t, float> >& x);
};

} // namespace wmsketch

#endif /* LOGISTIC_SKETCH_H_ */
