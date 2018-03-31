#include "paired_countmin.h"
#include "util.h"

namespace wmsketch {

PairedCountMin::PairedCountMin(uint32_t log2_width, uint32_t depth, int32_t seed, float smooth, bool consv_update)
 : depth_{depth},
   smooth_{smooth},
   consv_update_{consv_update},
   pos_count_{0},
   neg_count_{0},
   hash_fn_(depth, seed),
   hash_buf_(depth, 0) {

  if (log2_width < 1 || log2_width > PairedCountMin::MAX_LOG2_WIDTH) {
    throw std::invalid_argument("Invalid sketch width");
  }

  if (depth == 0) {
    throw std::invalid_argument("Invalid sketch depth");
  }

  uint32_t width = 1 << (log2_width - 1);  // two count-min tables of half width
  width_mask_ = width - 1;

  counts_num_ = (uint32_t**) calloc(depth, sizeof(uint32_t*));
  counts_num_[0] = (uint32_t*) calloc(depth * width, sizeof(uint32_t));
  counts_den_ = (uint32_t**) calloc(depth, sizeof(uint32_t*));
  counts_den_[0] = (uint32_t*) calloc(depth * width, sizeof(uint32_t));

  for (int i = 0; i < depth; i++) {
    counts_num_[i] = counts_num_[0] + (i * width);
    counts_den_[i] = counts_den_[0] + (i * width);
  }
}

PairedCountMin::~PairedCountMin() {
  free(counts_num_[0]);
  free(counts_num_);
  free(counts_den_[0]);
  free(counts_den_);
}

float PairedCountMin::get(uint32_t key) {
  hash_fn_.hash(hash_buf_.data(), key);
  for (int i = 0; i < depth_; i++) {
    hash_buf_[i] &= width_mask_;
  }

  uint32_t num = counts_num_[0][hash_buf_[0]];
  uint32_t den = counts_den_[0][hash_buf_[0]];
  for (int i = 1; i < depth_; i++) {
    num = MIN(num, counts_num_[i][hash_buf_[i]]);
    den = MIN(den, counts_den_[i][hash_buf_[i]]);
  }

  float ratio = (num + smooth_) / (den + smooth_);
  return ratio / bias();
}

float PairedCountMin::update_feature(uint32_t key, bool label) {
  hash_fn_.hash(hash_buf_.data(), key);
  for (int i = 0; i < depth_; i++) {
    hash_buf_[i] &= width_mask_;
  }

  uint32_t num = UINT32_MAX;
  uint32_t den = UINT32_MAX;
  if (consv_update_) {
    for (int i = 0; i < depth_; i++) {
      uint32_t j = hash_buf_[i];
      num = MIN(num, counts_num_[i][j]);
      den = MIN(den, counts_den_[i][j]);
    }

    if (label) num++;
    else den++;

    for (int i = 0; i < depth_; i++) {
      uint32_t j = hash_buf_[i];
      if (label) counts_num_[i][j] = MAX(num, counts_num_[i][j]);
      else counts_den_[i][j] = MAX(den, counts_den_[i][j]);
    }
  } else {
    for (int i = 0; i < depth_; i++) {
      uint32_t j = hash_buf_[i];
      if (label) counts_num_[i][j]++;
      else counts_den_[i][j]++;
      num = MIN(num, counts_num_[i][j]);
      den = MIN(den, counts_den_[i][j]);
    }
  }

  float ratio = (num + smooth_) / (den + smooth_);
  return ratio / bias();
}

bool PairedCountMin::update(uint32_t key, bool label) {
  if (label) pos_count_++;
  else neg_count_++;
  update_feature(key, label);
  // TODO
  return true;
}

bool PairedCountMin::update(const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  if (label) pos_count_++;
  else neg_count_++;
  uint32_t n = x.size();
  if (n == 0) return true; // TODO
  for (auto& pair : x) {
    update_feature(pair.first, label);
  }
  return true; // TODO
}

bool PairedCountMin::update(std::vector<float>& new_weights, const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  if (label) pos_count_++;
  else neg_count_++;
  uint32_t n = x.size();
  new_weights.resize(n);
  if (n == 0) return true; // TODO
  for (int i = 0; i < n; i++) {
    new_weights[i] = update_feature(x[i].first, label);
  }
  return true; // TODO
}

float PairedCountMin::bias() {
  return (pos_count_ + smooth_) / (neg_count_ + smooth_);
}

} // namespace wmsketch
