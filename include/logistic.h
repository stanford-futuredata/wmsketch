/*
 * Uncompressed logistic regression with L2 regularization.
 */

#ifndef LOGISTIC_H_
#define LOGISTIC_H_

#include <vector>
#include "binary_estimator.h"

namespace wmsketch {

class LogisticRegression : public BinaryEstimator {

 private:
  std::vector<float> weights_;
  float bias_;
  float lr_init_;
  float l2_reg_;
  float scale_;
  uint32_t dim_;
  uint64_t t_;
  bool no_bias_;

 public:
  explicit LogisticRegression(uint32_t dim, float lr_init = 0.1, float l2_reg = 1e-3, bool no_bias = false);
  ~LogisticRegression() override = default;
  float get(uint32_t key) override;
  float dot(const std::vector<std::pair<uint32_t, float> >& x);
  bool predict(uint32_t key);
  bool predict(const std::vector<std::pair<uint32_t, float> >& x);
  bool update(uint32_t key, bool label) override;
  bool update(const std::vector<std::pair<uint32_t, float> >& x, bool label) override;
  bool update(std::vector<float>& new_weights, const std::vector<std::pair<uint32_t, float> >& x, bool label) override;
  float bias() override;
};

} // namespace wmsketch

#endif /* LOGISTIC_H_ */
