#include "sgns.h"

namespace wmsketch {

StreamingSGNS::StreamingSGNS(
    uint32_t k,
    uint32_t log2_width,
    uint32_t depth,
    uint32_t neg_samples,
    uint32_t window_size,
    uint32_t reservoir_size,
    int32_t seed,
    float lr_init,
    float l2_reg)
 : heap_(k),
   reservoir_(reservoir_size, seed),
   sk_(log2_width, depth, seed),
   window_size_{window_size},
   neg_samples_{neg_samples},
   seed_{seed},
   bias_{0.f},
   lr_init_{lr_init},
   l2_reg_{l2_reg},
   scale_{1.f},
   t_{0},
   gen_(seed),
   rand_(0, 1) { }

void StreamingSGNS::topk(std::vector<std::pair<StringPair, float> >& out) {
  heap_.items(out);
  for (auto& i : out) i.second *= scale_;
  std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return fabs(a.second) > fabs(b.second); });
}

void StreamingSGNS::update(const std::string& token) {
  if (token.empty()) {
    return;
  }

  reservoir_.update(token);
  if (window_.size() == window_size_ + 1) {
    window_.pop_front();
  }
  window_.push_back(token);
  if (window_.size() < window_size_ + 1) {
    return;
  }
  std::string& w = window_[0];
  for (int i = 0; i < window_size_; i++) {
    std::string& v = window_[i + 1];
    update(w, v);
  }
}

void StreamingSGNS::update(const std::string& a, const std::string& b) {
  update(a, b, true);
  for (int j = 0; j < neg_samples_; j++) {
    if (rand_(gen_) < 0.5) {
      update(a, reservoir_.sample(), false);
    } else {
      update(reservoir_.sample(), b, false);
    }
  }
}

void StreamingSGNS::update(const std::string& a, const std::string& b, bool real) {
  int y = real ? +1 : -1;
  StringPair s(a, b);
  bool in_heap = heap_.contains(s);

  float w;
  uint32_t h = 0;
  if (in_heap) {
    w = heap_.get(s);
  } else {
    h = strings_to_key(a, b);
    w = sk_.get(h);
  }

  float lr = lr_init_ / (1.f + lr_init_ * l2_reg_ * t_);
  float z = w * scale_ + bias_;
  float g = logistic_grad(y * z);
  scale_ *= (1 - lr * l2_reg_);
  float u = lr * y * g / scale_;

  if (in_heap) {
    heap_.change_val(s, w - u);
  } else {
    auto opt = heap_.insert(s, w - u);
    if (opt) {
      if (s == opt->first) {
        sk_.update(h, -u);
      } else {
        uint32_t popped_h = strings_to_key(opt->first.first, opt->first.second);
        sk_.update(popped_h, opt->second - sk_.get(popped_h));
      }
    }
  }

  bias_ -= lr * y * g;
  t_++;
}

void StreamingSGNS::flush() {
  if (window_.size() == window_size_ + 1) {
    window_.pop_front();
  }

  while (!window_.empty()) {
    const auto& w = window_[0];
    for (int i = 0; i < window_.size() - 1; i++) {
      const auto& v = window_[i + 1];
      update(w, v);
    }
    window_.pop_front();
  }
}

uint32_t StreamingSGNS::strings_to_key(const std::string& a, const std::string& b) {
  uint32_t h1 = hash::murmurhash3_32(a.data(), (int) a.length(), (uint32_t) seed_);
  uint32_t h2 = hash::murmurhash3_32(b.data(), (int) b.length(), (uint32_t) seed_);
  uint32_t h = 101 * h1 + h2;
  return h;
}

} // namespace wmsketch
