#ifndef BINARY_ESTIMATOR_H_
#define BINARY_ESTIMATOR_H_

#include <cstdlib>
#include <cstdint>

namespace wmsketch {

class BinaryEstimator {
 public:
  virtual ~BinaryEstimator() = default;
  virtual float get(uint32_t key) = 0;
  virtual bool update(uint32_t key, bool pos) = 0;
  virtual bool update(const std::vector<std::pair<uint32_t, float> >& x, bool pos) = 0;
  virtual bool update(std::vector<float>& new_weights, const std::vector<std::pair<uint32_t, float> >& x, bool pos) = 0;
  virtual float bias() = 0;
};

} // namespace wmsketch

#endif /* BINARY_ESTIMATOR_H_ */
