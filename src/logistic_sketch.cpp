#include "logistic_sketch.h"
#include <iostream>
#include <numeric>
#include "util.h"

namespace wmsketch {

LogisticSketch::LogisticSketch(
    uint32_t log2_width,
    uint32_t depth,
    int32_t seed,
    float lr_init,
    float l2_reg,
    bool median_update)
 : bias_{0.f},
   lr_init_{lr_init},
   l2_reg_{l2_reg},
   scale_{1.f},
   t_{0},
   depth_{depth},
   median_update_{median_update},
   hash_fn_(depth, seed),
   hash_buf_(depth, 0),
   weight_buf_(depth, 0) {

  if (log2_width > LogisticSketch::MAX_LOG2_WIDTH) {
    throw std::invalid_argument("Invalid sketch width");
  }

  if (lr_init <= 0.) {
    throw std::invalid_argument("Initial learning rate must be positive");
  }

  uint32_t width = 1 << log2_width;
  width_mask_ = width - 1;

  weights_ = (float**) calloc(depth, sizeof(float*));
  weights_[0] = (float*) calloc(width * depth, sizeof(float));
  for (int i = 0; i < depth; i++) {
    weights_[i] = weights_[0] + i * width;
  }
}

LogisticSketch::~LogisticSketch() {
  free(weights_[0]);
  free(weights_);
}

float LogisticSketch::get(uint32_t key) {
  return scale_ * get_weight(key, true);
}

float LogisticSketch::dot(const std::vector<std::pair<uint32_t, float> >& x) {
  if (x.size() == 0) return 0.f;
  float z = 0.f;
  get_weights(x);
  for (int idx = 0; idx < x.size(); idx++) {
    float val = x[idx].second;
    if (median_update_) {
      z += val * weight_medians_[idx];
    } else {
      z += val * weight_means_[idx];
    }
  }
  z *= scale_;
  return z;
}

bool LogisticSketch::predict(const std::vector<std::pair<uint32_t, float> >& x) {
  float z = dot(x) + bias_;
  return z >= 0.;
}

bool LogisticSketch::update(uint32_t key, bool label) {
  float med = get_weight(key, true);

  int y = label ? +1 : -1;
  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);
  float z = median_update_ ? med : mean(weight_buf_);
  z *= scale_;
  z += bias_;

  float g = logistic_grad(y * z);
  scale_ *= (1 - lr * l2_reg_);
  float u = lr * y * g / scale_;
  for (int i = 0; i < depth_; i++) {
    uint32_t h = hash_buf_[i];
    int sgn = (h >> 31) ? +1 : -1;
    weights_[i][h & width_mask_] -= sgn * u;
  }

  bias_ -= lr * y * g;
  t_++;
  return z >= 0.;
}

bool LogisticSketch::update(const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  if (x.size() == 0) {
    return bias_ >= 0;
  }
  int y = label ? +1 : -1;
  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);
  float z = dot(x) + bias_;
  float g = logistic_grad(y * z);
  scale_ *= (1 - lr * l2_reg_);
  float u = lr * y * g / scale_;

  for (int idx = 0; idx < x.size(); idx++) {
    float val = x[idx].second;
    for (int i = 0; i < depth_; i++) {
      uint32_t h = hash_buf_[idx*depth_ + i];
      int sgn = (h >> 31) ? +1 : -1;
      weights_[i][h & width_mask_] -= sgn * u * val;
    }
  }

  bias_ -= lr * y * g;
  t_++;
  return z >= 0;
}

bool LogisticSketch::update(
    std::vector<float>& new_weights,
    const std::vector<std::pair<uint32_t, float> >& x,
    bool label) {
  uint64_t n = x.size();
  new_weights.resize(n);
  if (n == 0) {
    return bias_ >= 0;
  }

  int y = label ? +1 : -1;
  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);
  float z = dot(x) + bias_;
  float g = logistic_grad(y * z);
  scale_ *= (1 - lr * l2_reg_);
  float u = lr * y * g / scale_;

  for (int idx = 0; idx < n; idx++) {
    float val = x[idx].second;
    for (int i = 0; i < depth_; i++) {
      uint32_t h = hash_buf_[idx*depth_ + i];
      int sgn = (h >> 31) ? +1 : -1;
      weights_[i][h & width_mask_] -= sgn * u * val;
    }

    new_weights[idx] = weight_medians_[idx] - u * val;
  }

  bias_ -= lr * y * g;
  t_++;
  return z >= 0;
}

float LogisticSketch::bias() {
  return bias_;
}

float LogisticSketch::scale() {
  return scale_;
}

float LogisticSketch::get_weight(uint32_t key, bool use_median) {
  hash_fn_.hash(hash_buf_.data(), key);
  for (int i = 0; i < depth_; i++) {
    uint32_t h = hash_buf_[i];
    int sgn = (h >> 31) ? +1 : -1;
    weight_buf_[i] = sgn * weights_[i][h & width_mask_];
  }

  if (use_median) return median(weight_buf_);
  return mean(weight_buf_);
}

void LogisticSketch::get_weights(const std::vector<std::pair<uint32_t, float> >& x) {
  uint64_t n = x.size();
  if (hash_buf_.size() < depth_ * n) {
    hash_buf_.resize(depth_ * n);
  }

  weight_medians_.resize(n);
  if (!median_update_) weight_means_.resize(n);
  uint32_t* ph = hash_buf_.data();
  for (int idx = 0; idx < n; idx++) {
    hash_fn_.hash(ph + idx*depth_, x[idx].first);
    for (int i = 0; i < depth_; i++) {
      uint32_t h = hash_buf_[idx*depth_ + i];
      int sgn = (h >> 31) ? +1 : -1;
      weight_buf_[i] = sgn * weights_[i][h & width_mask_];
    }

    weight_medians_[idx] = median(weight_buf_);
    if (!median_update_) weight_means_[idx] = mean(weight_buf_);
  }
}

} // namespace wmsketch
