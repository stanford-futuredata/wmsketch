#ifndef SRC_TOPK_H_
#define SRC_TOPK_H_

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <tuple>
#include <random>
#include "countmin.h"
#include "countsketch.h"
#include "paired_countmin.h"
#include "logistic.h"
#include "logistic_sketch.h"
#include "heap.h"

namespace wmsketch {

class TopKFeatures {
 protected:
  uint32_t k_;
  TopKHeap<uint32_t> heap_;
  WeightedReservoir res_;  // weighted reservoir sampler for probabilistic truncation baseline

  explicit TopKFeatures(uint32_t k): k_{k}, heap_(k), res_(k) { }
  TopKFeatures(uint32_t k, int32_t seed, float pow = 1.f): k_{k}, heap_(k), res_(k, seed, pow) { }

 public:
  virtual ~TopKFeatures() = default;
  virtual void topk(std::vector<std::pair<uint32_t, float> >& out) {
    heap_.items(out);
    std::sort(out.begin(), out.end(),
        [](auto& a, auto& b) { return fabs(a.second) > fabs(b.second); });
  }
  virtual bool predict(const std::vector<std::pair<uint32_t, float> >& x) = 0;
  virtual bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label) = 0;
  virtual float bias() {
    return 0.f;
  }
};

class LogisticTopK : public TopKFeatures {
 private:
  LogisticRegression lr_;
  std::vector<float> new_weights_;

 public:
  LogisticTopK(uint32_t k, uint32_t dim, float lr_init, float l2_reg, bool no_bias);
  ~LogisticTopK() override;
  bool predict(const std::vector<std::pair<uint32_t, float> >& x) override;
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label) override;
  float bias() override;
};

class TruncatedLogisticTopK : public TopKFeatures {
 private:
  float bias_;
  float lr_init_;
  float l2_reg_;
  float scale_;
  uint64_t t_;

 public:
  TruncatedLogisticTopK(
      uint32_t k,
      float lr_init,
      float l2_reg);
  ~TruncatedLogisticTopK() override;
  void topk(std::vector<std::pair<uint32_t, float> >& out) override;
  float dot(const std::vector<std::pair<uint32_t, float> >& x);
  bool predict(const std::vector<std::pair<uint32_t, float> >& x) override;
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label) override;
  float bias() override;

 private:
  float get_weight(uint32_t key);
};

class ProbTruncatedLogisticTopK : public TopKFeatures {
 private:
  float bias_;
  float lr_init_;
  float l2_reg_;
  float scale_;
  uint64_t t_;
  std::mt19937 gen_;
  std::uniform_real_distribution<> rand_;

 public:
  ProbTruncatedLogisticTopK(
      uint32_t k,
      int32_t seed,
      float lr_init,
      float l2_reg,
      float pow = 1.0);
  ~ProbTruncatedLogisticTopK();
  void topk(std::vector<std::pair<uint32_t, float> >& out);
  float dot(const std::vector<std::pair<uint32_t, float> >& x);
  bool predict(const std::vector<std::pair<uint32_t, float> >& x);
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label);
  float bias();

 private:
  float get_weight(uint32_t key);
};

class SpaceSavingLogisticTopK : public TopKFeatures {
 private:
  TopKCountHeap cheap_;
  float bias_;
  float lr_init_;
  float l2_reg_;
  float scale_;
  uint64_t t_;
  std::mt19937 gen_;
  std::uniform_real_distribution<> rand_;

 public:
  explicit SpaceSavingLogisticTopK(
      uint32_t k,
      int32_t seed,
      float lr_init = 0.1,
      float l2_reg = 1e-3
  );
  ~SpaceSavingLogisticTopK() override = default;
  void topk(std::vector<std::pair<uint32_t, float> >& out) override;
  float dot(const std::vector<std::pair<uint32_t, float> >& x);
  bool predict(const std::vector<std::pair<uint32_t, float> >& x) override;
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label) override;
  float bias() override;

 private:
  float get_weight(uint32_t key);
};

class CountMinLogisticTopK : public TopKFeatures {
 private:
  TopKCountHeap cheap_;
  CountMinSketch sk_;
  float bias_;
  float lr_init_;
  float l2_reg_;
  float scale_;
  uint64_t t_;

 public:
  CountMinLogisticTopK(
      uint32_t k,
      uint32_t log2_width,
      uint32_t depth,
      int32_t seed,
      float lr_init = 0.1,
      float l2_reg = 1e-3,
      bool consv_update = true
  );
  ~CountMinLogisticTopK() override = default;
  void topk(std::vector<std::pair<uint32_t, float> >& out) override;
  float dot(const std::vector<std::pair<uint32_t, float> >& x);
  bool predict(const std::vector<std::pair<uint32_t, float> >& x) override;
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label) override;
  float bias() override;

 private:
  float get_weight(uint32_t key);
};

class PairedCountMinTopK : public TopKFeatures {
 private:
  PairedCountMin sk_;
  std::vector<float> new_weights_;
  std::vector<uint32_t> idxs_;
  uint64_t t_;

 public:
  PairedCountMinTopK(
      uint32_t k,
      uint32_t log2_width,
      uint32_t depth,
      int32_t seed,
      float smooth = 1.f,
      bool consv_update = false);
  ~PairedCountMinTopK();
  void topk(std::vector<std::pair<uint32_t, float> >& out) override;
  bool predict(const std::vector<std::pair<uint32_t, float> >& x) override;
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label) override;
  float bias() override;

 private:
  void refresh_heap();
};

class LogisticSketchTopK : public TopKFeatures {
 private:
  LogisticSketch sk_;
  std::vector<float> new_weights_;
  std::vector<uint32_t> idxs_;
  uint64_t t_;

 public:
  LogisticSketchTopK(
      uint32_t k,
      uint32_t log2_width,
      uint32_t depth,
      int32_t seed,
      float lr_init = 0.1,
      float l2_reg = 1e-3,
      bool median_update = false);
  ~LogisticSketchTopK();
  void topk(std::vector<std::pair<uint32_t, float> >& out);
  bool predict(const std::vector<std::pair<uint32_t, float> >& x);
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label);
  float bias();

 private:
  void refresh_heap();
};

class ActiveSetLogisticTopK : public TopKFeatures {
 private:
  CountSketch sk_;
  float bias_;
  float lr_init_;
  float l2_reg_;
  float scale_;
  uint64_t t_;
  std::vector<float> weight_buf_;
  std::vector<std::tuple<uint32_t, float, float> > heap_feats_, sk_feats_;

 public:
  ActiveSetLogisticTopK(
      uint32_t k,
      uint32_t log2_width,
      uint32_t depth,
      int32_t seed,
      float lr_init = 0.1,
      float l2_reg = 1e-3);
  ~ActiveSetLogisticTopK();
  void topk(std::vector<std::pair<uint32_t, float> >& out);
  float dot(const std::vector<std::pair<uint32_t, float> >& x);
  bool predict(const std::vector<std::pair<uint32_t, float> >& x);
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label);
  float bias();
};

} // namespace wmsketch

#endif /* SRC_TOPK_H_ */
