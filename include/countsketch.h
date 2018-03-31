#ifndef SRC_COUNTSKETCH_H_
#define SRC_COUNTSKETCH_H_

#include <vector>
#include "hash.h"

namespace wmsketch {

class CountSketch {

 public:
  static const uint32_t MAX_LOG2_WIDTH = 31;

 private:
  const uint32_t depth_;
  uint32_t width_mask_;
  float** weights_;
  hash::TabulationHash hash_fn_;
  std::vector<uint32_t> hash_buf_;
  std::vector<float> weight_buf_;

 public:
  CountSketch(uint32_t log2_width, uint32_t depth, int32_t seed);
  ~CountSketch();
  float get(uint32_t key);
  void update(uint32_t key, float delta);

};

} // namespace wmsketch

#endif /* SRC_COUNTSKETCH_H_ */
