#include "countsketch.h"
#include "util.h"

namespace wmsketch {

CountSketch::CountSketch(
    uint32_t log2_width,
    uint32_t depth,
    int32_t seed)
 : depth_{depth},
   hash_fn_(depth, seed),
   hash_buf_(depth, 0),
   weight_buf_(depth, 0) {

  if (log2_width > CountSketch::MAX_LOG2_WIDTH) {
    throw std::invalid_argument("Invalid sketch width");
  }

  uint32_t width = 1 << log2_width;
  width_mask_ = width - 1;

  weights_ = (float**) calloc(depth, sizeof(float*));
  weights_[0] = (float*) calloc(width * depth, sizeof(float));
  for (int i = 0; i < depth; i++) {
    weights_[i] = weights_[0] + i * width;
  }
}

CountSketch::~CountSketch() {
  free(weights_[0]);
  free(weights_);
}

float CountSketch::get(uint32_t key) {
  hash_fn_.hash(hash_buf_.data(), key);

  for (int i = 0; i < depth_; i++) {
    uint32_t h = hash_buf_[i];
    int sgn = (h >> 31) ? +1 : -1;
    weight_buf_[i] = sgn * weights_[i][h & width_mask_];
  }

  return median(weight_buf_);
}

void CountSketch::update(uint32_t key, float delta) {
  hash_fn_.hash(hash_buf_.data(), key);

  for (int i = 0; i < depth_; i++) {
    uint32_t h = hash_buf_[i];
    int sgn = (h >> 31) ? +1 : -1;
    weights_[i][h & width_mask_] += sgn * delta;
  }
}

} // namespace wmsketch
