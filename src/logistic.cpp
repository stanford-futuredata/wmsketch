#include "logistic.h"
#include "util.h"

namespace wmsketch {

LogisticRegression::LogisticRegression(uint32_t dim, float lr_init, float l2_reg, bool no_bias)
 : weights_(dim, 0.0),
   bias_{0.f},
   lr_init_{lr_init},
   l2_reg_{l2_reg},
   scale_{1.0},
   dim_{dim},
   t_{0},
   no_bias_{no_bias} { }

float LogisticRegression::get(uint32_t x) {
  if (x >= dim_) {
    throw std::out_of_range("Feature index out of bounds.");
  }
  return scale_ * weights_[x];
}

float LogisticRegression::dot(const std::vector<std::pair<uint32_t, float> >& x) {
  if (x.size() == 0) return 0.f;
  float z = 0.f;
  for (auto& p : x) {
    uint32_t idx = p.first;
    float val = p.second;
    z += weights_[idx] * val;
  }
  z *= scale_;
  return z;
}

bool LogisticRegression::predict(uint32_t x) {
  float z = get(x) + bias_;
  return z >= 0.;
}

bool LogisticRegression::predict(const std::vector<std::pair<uint32_t, float> >& x) {
  float z = dot(x) + bias_;
  return z >= 0.;
}

bool LogisticRegression::update(uint32_t x, bool label) {
  if (x >= dim_) {
    throw std::out_of_range("Feature index out of bounds.");
  }

  int y = label ? +1 : -1;
  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);
  float z = scale_ * weights_[x] + bias_;
  float g = logistic_grad(y * z);
  scale_ *= (1 - lr * l2_reg_);
  weights_[x] -= lr * y * g / scale_;
  if (!no_bias_) bias_ -= lr * y * g;
  t_++;
  return z >= 0;
}

bool LogisticRegression::update(const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  int y = label ? +1 : -1;
  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);

  float z = 0.f;
  for (auto& pair : x) {
    uint32_t key = pair.first;
    float val = pair.second;
    z += weights_[key] * val;
  }
  z = scale_ * z + bias_;

  scale_ *= (1 - lr * l2_reg_);
  float g = logistic_grad(y * z);
  for (auto& pair : x) {
    uint32_t key = pair.first;
    float val = pair.second;
    weights_[key] -= lr * y * g * val / scale_;
  }

  if (!no_bias_) bias_ -= lr * y * g;
  t_++;
  return z >= 0;
}

bool LogisticRegression::update(
    std::vector<float>& new_weights,
    const std::vector<std::pair<uint32_t, float> >& x,
    bool pos) {
  bool yhat = update(x, pos);
  uint64_t n = x.size();
  new_weights.resize(n);
  for (int i = 0; i < n; i++) {
    new_weights[i] = scale_ * weights_[x[i].first];
  }
  return yhat;
}

float LogisticRegression::bias() {
  return bias_;
}

} // namespace wmsketch
