#include "util.h"
#include <random>
#include "countmin.h"


namespace wmsketch {

CountMinSketch::CountMinSketch(uint32_t log2_width, uint32_t depth, int32_t seed, bool consv_update)
 : depth_{depth},
   consv_update_{consv_update},
   hash_fn_(depth, seed),
   hash_buf_(depth, 0) {

  if (log2_width > CountMinSketch::MAX_LOG2_WIDTH) {
    throw std::invalid_argument("Invalid sketch width");
  }

  if (depth == 0) {
    throw std::invalid_argument("Invalid sketch depth");
  }

  uint32_t width = 1 << log2_width;
  width_mask_ = width - 1;

  counts_ = (uint32_t**) calloc(depth, sizeof(uint32_t*));
  counts_[0] = (uint32_t*) calloc(depth * width, sizeof(uint32_t));

  for (int i = 0; i < depth; i++) {
    counts_[i] = counts_[0] + (i * width);
  }
}

CountMinSketch::~CountMinSketch() {
  free(counts_[0]);
  free(counts_);
}

uint32_t CountMinSketch::get(uint32_t key) {
  hash_fn_.hash(hash_buf_.data(), key);
  uint32_t min = counts_[0][hash_buf_[0] & width_mask_];
  for (int i = 1; i < depth_; i++) {
    min = MIN(min, counts_[i][hash_buf_[i] & width_mask_]);
  }
  return min;
}

uint32_t CountMinSketch::update(uint32_t key) {
  hash_fn_.hash(hash_buf_.data(), key);
  for (int i = 0; i < depth_; i++) {
    hash_buf_[i] &= width_mask_;
  }

  uint32_t c;
  if (consv_update_) {
    c = counts_[0][hash_buf_[0]];
    for (int i = 1; i < depth_; i++) {
      c = MIN(c, counts_[i][hash_buf_[i]]);
    }

    for (int i = 0; i < depth_; i++) {
      uint32_t j = hash_buf_[i];
      counts_[i][j] = MAX(c + 1, counts_[i][j]);
    }
  } else {
    c = UINT_MAX;
    for (int i = 0; i < depth_; i++) {
      uint32_t j = hash_buf_[i];
      c = MIN(c, counts_[i][j]);
      counts_[i][j]++;
    }
  }

  return c + 1;
}

} // namespace wmsketch
