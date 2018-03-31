#include "topk.h"
#include "util.h"
#include <iostream>

namespace wmsketch {

LogisticTopK::LogisticTopK(uint32_t k, uint32_t dim, float lr_init, float l2_reg, bool no_bias)
 : TopKFeatures(k),
   lr_(dim, lr_init, l2_reg, no_bias) { }

LogisticTopK::~LogisticTopK() = default;

bool LogisticTopK::predict(const std::vector<std::pair<uint32_t, float> >& x) {
  return lr_.predict(x);
}

bool LogisticTopK::update(const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  bool yhat = lr_.update(new_weights_, x, label);
  for (int i = 0; i < x.size(); i++) {
    uint32_t key = x[i].first;
    heap_.insert_or_change(key, new_weights_[i]);
  }
  return yhat;
}

float LogisticTopK::bias() {
  return lr_.bias();
}

///////////////////////////////////////////////////////////////////////////////

TruncatedLogisticTopK::TruncatedLogisticTopK(uint32_t k, float lr_init, float l2_reg)
 : TopKFeatures(k),
   bias_{0.f},
   lr_init_{lr_init},
   l2_reg_{l2_reg},
   scale_{1.f},
   t_{0} { }

TruncatedLogisticTopK::~TruncatedLogisticTopK() = default;

void TruncatedLogisticTopK::topk(std::vector<std::pair<uint32_t, float> >& out) {
  heap_.items(out);
  for (auto &i : out) {
    i.second *= scale_;
  }
  std::sort(out.begin(), out.end(),
      [](auto& a, auto& b) { return fabs(a.second) > fabs(b.second); });
}

float TruncatedLogisticTopK::get_weight(uint32_t key) {
  if (heap_.contains(key)) {
    return heap_.get(key);
  }
  return 0.f;
}

float TruncatedLogisticTopK::dot(const std::vector<std::pair<uint32_t, float> >& x) {
  float z = 0.f;
  for (auto& pair : x) {
    uint32_t key = pair.first;
    float val = pair.second;
    z += get_weight(key) * val;
  }
  z *= scale_;
  return z;
}

bool TruncatedLogisticTopK::predict(const std::vector<std::pair<uint32_t, float> >& x) {
  float z = dot(x) + bias_;
  return z >= 0;
}

bool TruncatedLogisticTopK::update(const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  int y = label ? +1 : -1;
  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);
  float z = dot(x) + bias_;
  scale_ *= (1 - lr * l2_reg_);
  float g = logistic_grad(y * z);
  for (auto& pair : x) {
    uint32_t key = pair.first;
    float val = pair.second;
    float new_w = get_weight(key) - lr * y * g * val / scale_;
    heap_.insert_or_change(key, new_w);
  }

  bias_ -= lr * y * g;
  t_++;
  return z >= 0;
}

float TruncatedLogisticTopK::bias() {
  return bias_;
}

///////////////////////////////////////////////////////////////////////////////

ProbTruncatedLogisticTopK::ProbTruncatedLogisticTopK(
    uint32_t k,
    int32_t seed,
    float lr_init,
    float l2_reg,
    float pow)
 : TopKFeatures(k, seed, pow),
   bias_{0.f},
   lr_init_{lr_init},
   l2_reg_{l2_reg},
   scale_{1.f},
   t_{0} { }

ProbTruncatedLogisticTopK::~ProbTruncatedLogisticTopK() = default;

void ProbTruncatedLogisticTopK::topk(std::vector<std::pair<uint32_t, float> > &out) {
  res_.items(out);
  for (auto &i : out) {
    i.second *= scale_;
  }
  std::sort(out.begin(), out.end(),
      [](auto& a, auto& b) { return fabs(a.second) > fabs(b.second); });
}

float ProbTruncatedLogisticTopK::dot(const std::vector<std::pair<uint32_t, float> > &x) {
  float z = 0.f;
  for (auto& pair : x) {
    uint32_t key = pair.first;
    float val = pair.second;
    z += get_weight(key) * val;
  }
  z *= scale_;
  return z;
}

bool ProbTruncatedLogisticTopK::predict(const std::vector<std::pair<uint32_t, float> > &x) {
  float z = dot(x) + bias_;
  return z >= 0;
}

bool ProbTruncatedLogisticTopK::update(const std::vector<std::pair<uint32_t, float> > &x, bool label) {
  int y = label ? +1 : -1;
  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);
  float z = dot(x) + bias_;
  scale_ *= (1 - lr * l2_reg_);
  float g = logistic_grad(y * z);
  for (auto &pair : x) {
    uint32_t key = pair.first;
    float val = pair.second;
    float new_w = get_weight(key) - lr * y * g * val / scale_;
    res_.insert_or_change(key, new_w);
  }

  bias_ -= lr * y * g;
  t_++;
  return z >= 0;
}

float ProbTruncatedLogisticTopK::bias() {
  return bias_;
}

float ProbTruncatedLogisticTopK::get_weight(uint32_t key) {
  if (res_.contains(key)) {
    return res_.get(key);
  }
  return 0.f;
}

///////////////////////////////////////////////////////////////////////////////

SpaceSavingLogisticTopK::SpaceSavingLogisticTopK(
    uint32_t k,
    int32_t seed,
    float lr_init,
    float l2_reg)
 : TopKFeatures(k),
   cheap_(k),
   bias_{0.f},
   lr_init_{lr_init},
   l2_reg_{l2_reg},
   scale_{1.f},
   t_{0},
   gen_(seed),
   rand_(0, 1) { }

float SpaceSavingLogisticTopK::get_weight(uint32_t key) {
  if (cheap_.contains(key)) {
    return cheap_.get(key);
  }
  return 0.f;
}

void SpaceSavingLogisticTopK::topk(std::vector<std::pair<uint32_t, float> >& out) {
  cheap_.items(out);
  for (auto &i : out) {
    i.second *= scale_;
  }
  std::sort(out.begin(), out.end(),
      [](auto& a, auto& b) { return fabs(a.second) > fabs(b.second); });
}

float SpaceSavingLogisticTopK::dot(const std::vector<std::pair<uint32_t, float> >& x) {
  float z = 0.f;
  for (auto& pair : x) {
    uint32_t key = pair.first;
    float val = pair.second;
    z += get_weight(key) * val;
  }
  z *= scale_;
  return z;
}

bool SpaceSavingLogisticTopK::predict(const std::vector<std::pair<uint32_t, float> >& x) {
  float z = dot(x) + bias_;
  return z >= 0;
}

bool SpaceSavingLogisticTopK::update(const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  int y = label ? +1 : -1;
  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);
  float z = dot(x) + bias_;
  scale_ *= (1 - lr * l2_reg_);
  float g = logistic_grad(y * z);

  int32_t replace = -1;
  uint32_t count = 0;
  for (auto& pair : x) {
    uint32_t key = pair.first;
    if (cheap_.contains(key)) {
      cheap_.increment_count(key);
    } else if (!cheap_.is_full()) {
      cheap_.insert(key, 1, 0.f);
    } else {
      count++;
      if (rand_(gen_) < 1. / count) replace = key;
    }
  }

  if (replace >= 0) {
    uint32_t min_count = cheap_.min_val();
    cheap_.del_min();
    cheap_.insert((uint32_t) replace, min_count + 1, 0.f);
  }

  for (auto& pair : x) {
    uint32_t key = pair.first;
    float val = pair.second;
    if (cheap_.contains(key)) {
      float new_w = get_weight(key) - lr * y * g * val / scale_;
      cheap_.change_val(key, cheap_.get_count(key), new_w);
    }
  }

  bias_ -= lr * y * g;
  t_++;
  return z >= 0;
}

float SpaceSavingLogisticTopK::bias() {
  return bias_;
}

///////////////////////////////////////////////////////////////////////////////

CountMinLogisticTopK::CountMinLogisticTopK(
    uint32_t k,
    uint32_t log2_width,
    uint32_t depth,
    int32_t seed,
    float lr_init,
    float l2_reg,
    bool consv_update)
 : TopKFeatures(k),
   cheap_(k),
   sk_(log2_width, depth, seed, consv_update),
   bias_{0.f},
   lr_init_{lr_init},
   l2_reg_{l2_reg},
   scale_{1.f},
   t_{0} { }

float CountMinLogisticTopK::get_weight(uint32_t key) {
  if (cheap_.contains(key)) {
    return cheap_.get(key);
  }
  return 0.f;
}

void CountMinLogisticTopK::topk(std::vector<std::pair<uint32_t, float> >& out) {
  cheap_.items(out);
  for (auto &i : out) {
    i.second *= scale_;
  }
  std::sort(out.begin(), out.end(),
      [](auto& a, auto& b) { return fabs(a.second) > fabs(b.second); });
}

float CountMinLogisticTopK::dot(const std::vector<std::pair<uint32_t, float> >& x) {
  float z = 0.f;
  for (auto& pair : x) {
    uint32_t key = pair.first;
    float val = pair.second;
    z += get_weight(key) * val;
  }
  z *= scale_;
  return z;
}

bool CountMinLogisticTopK::predict(const std::vector<std::pair<uint32_t, float> >& x) {
  float z = dot(x) + bias_;
  return z >= 0;
}

bool CountMinLogisticTopK::update(const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  int y = label ? +1 : -1;
  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);
  float z = dot(x) + bias_;
  scale_ *= (1 - lr * l2_reg_);
  float g = logistic_grad(y * z);
  for (auto& pair : x) {
    uint32_t key = pair.first;
    if (cheap_.contains(key)) cheap_.increment_count(key);
    sk_.update(key);
  }

  for (auto& pair : x) {
    uint32_t key = pair.first;
    float val = pair.second;
    float new_w = get_weight(key) - lr * y * g * val / scale_;
    uint32_t count = (cheap_.contains(key)) ? cheap_.get_count(key) : sk_.get(key);
    cheap_.insert_or_change(key, count, new_w);
  }

  bias_ -= lr * y * g;
  t_++;
  return z >= 0;
}

float CountMinLogisticTopK::bias() {
  return bias_;
}

///////////////////////////////////////////////////////////////////////////////

PairedCountMinTopK::PairedCountMinTopK(
    uint32_t k,
    uint32_t log2_width,
    uint32_t depth,
    int32_t seed,
    float smooth,
    bool consv_update)
 : TopKFeatures(k),
   sk_(log2_width, depth, seed + 1, smooth, consv_update),
   t_{0} { }

PairedCountMinTopK::~PairedCountMinTopK() = default;

void PairedCountMinTopK::topk(std::vector<std::pair<uint32_t, float> >& out) {
  refresh_heap();
  TopKFeatures::topk(out);
}

bool PairedCountMinTopK::predict(const std::vector<std::pair<uint32_t, float> >& x) {
  // TODO
  return true;
}

bool PairedCountMinTopK::update(const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  sk_.update(new_weights_, x, label);
  for (int i = 0; i < x.size(); i++) {
    uint32_t key = x[i].first;
    heap_.insert_or_change(key, log(new_weights_[i]));
  }

  // TODO
  return true;
}

void PairedCountMinTopK::refresh_heap() {
  heap_.keys(idxs_);
  for (uint32_t idx : idxs_) {
    heap_.change_val(idx, log(sk_.get(idx)));
  }
}

float PairedCountMinTopK::bias() {
  return sk_.bias();
}

///////////////////////////////////////////////////////////////////////////////

LogisticSketchTopK::LogisticSketchTopK(
    uint32_t k,
    uint32_t log2_width,
    uint32_t depth,
    int32_t seed,
    float lr_init,
    float l2_reg,
    bool median_update)
 : TopKFeatures(k),
   sk_(log2_width, depth, seed, lr_init, l2_reg, median_update),
   t_{0} { }

LogisticSketchTopK::~LogisticSketchTopK() = default;

void LogisticSketchTopK::topk(std::vector<std::pair<uint32_t, float> >& out) {
  refresh_heap();
  TopKFeatures::topk(out);
  float s = sk_.scale();
  for (auto& i : out) {
    i.second *= s;
  }
}

bool LogisticSketchTopK::predict(const std::vector<std::pair<uint32_t, float> >& x) {
  return sk_.predict(x);
}

bool LogisticSketchTopK::update(const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  bool yhat = sk_.update(new_weights_, x, label);
  for (int i = 0; i < x.size(); i++) {
    uint32_t key = x[i].first;
    heap_.insert_or_change(key, new_weights_[i]);
  }
  t_++;
  return yhat;
}

float LogisticSketchTopK::bias() {
  return sk_.bias();
}

void LogisticSketchTopK::refresh_heap() {
  heap_.keys(idxs_);
  for (uint32_t idx : idxs_) {
    heap_.change_val(idx, sk_.get(idx));
  }
}

///////////////////////////////////////////////////////////////////////////////

ActiveSetLogisticTopK::ActiveSetLogisticTopK(
    uint32_t k,
    uint32_t log2_width,
    uint32_t depth,
    int32_t seed,
    float lr_init,
    float l2_reg)
 : TopKFeatures(k),
   sk_(log2_width, depth, seed),
   bias_{0.f},
   lr_init_{lr_init},
   l2_reg_{l2_reg},
   scale_{1.f},
   t_{0} { }

ActiveSetLogisticTopK::~ActiveSetLogisticTopK() = default;

void ActiveSetLogisticTopK::topk(std::vector<std::pair<uint32_t, float> >& out) {
  heap_.items(out);
  for (auto& i : out) {
    i.second *= scale_;
  }
  std::sort(out.begin(), out.end(),
      [](auto& a, auto& b) { return fabs(a.second) > fabs(b.second); });
}

float ActiveSetLogisticTopK::dot(const std::vector<std::pair<uint32_t, float> >& x) {
  float z = 0.f;
  heap_feats_.clear();
  sk_feats_.clear();
  weight_buf_.clear();

  uint32_t idx;
  float val, w;
  if (x.empty()) return z;
  for (const auto &i : x) {
    std::tie(idx, val) = i;
    if (heap_.contains(idx)) {
      w = heap_.get(idx);
      heap_feats_.push_back(std::make_tuple(idx, val, w));
    } else {
      w = sk_.get(idx);
      sk_feats_.push_back(std::make_tuple(idx, val, w));
    }
    z += w * val;
    weight_buf_.push_back(w);
  }
  z *= scale_;
  return z;
}

bool ActiveSetLogisticTopK::predict(const std::vector<std::pair<uint32_t, float> >& x) {
  float z = dot(x) + bias_;
  return z >= 0.;
}

bool ActiveSetLogisticTopK::update(const std::vector<std::pair<uint32_t, float> >& x, bool label) {
  if (x.empty()) return bias_ >= 0;
  int y = label ? +1 : -1;
  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);
  float z = dot(x) + bias_;
  bool yhat = z >= 0;
  float g = logistic_grad(y * z);
  scale_ *= (1 - lr * l2_reg_);
  float u = lr * y * g / scale_;

  uint32_t idx;
  float val, w;
  for (auto& tup : heap_feats_) {
    std::tie(idx, val, w) = tup;
    heap_.change_val(idx, w - u * val);
  }

  for (auto& tup : sk_feats_) {
    std::tie(idx, val, w) = tup;
    float new_w = w - u * val;
    std::get<2>(tup) = new_w;
  }

  std::sort(sk_feats_.begin(), sk_feats_.end(),
      [](auto& a, auto& b) { return fabs(std::get<2>(a)) > fabs(std::get<2>(b)); });

  for (auto& tup : sk_feats_) {
    uint32_t popped_idx;
    float popped_w;
    std::tie(idx, val, w) = tup;

    auto opt = heap_.insert(idx, w);
    if (!opt) continue;
    std::tie(popped_idx, popped_w) = *opt;

    if (idx == popped_idx) {
      sk_.update(idx, -u * val);
    } else {
      sk_.update(popped_idx, popped_w - sk_.get(popped_idx));
    }
  }

  bias_ -= lr * y * g;
  t_++;
  return yhat;
}

float ActiveSetLogisticTopK::bias() {
  return bias_;
}

} // namespace wmsketch
