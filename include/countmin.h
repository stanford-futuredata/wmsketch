/*
 * Count-Min Sketch
 */

#ifndef COUNTMIN_H_
#define COUNTMIN_H_

#include "hash.h"

namespace wmsketch {

class CountMinSketch {

 public:
  static const uint32_t MAX_LOG2_WIDTH = 30;

 private:
  const uint32_t depth_;
  const bool consv_update_;
  uint32_t width_mask_;
  uint32_t **counts_;
  hash::PolynomialHash hash_fn_;
  std::vector<uint32_t> hash_buf_;

 public:
  /**
   * Count-Min Sketch
   *
   * @param log2_width Base-2 logarithm of sketch width.
   * @param depth Sketch depth.
   * @param seed Random seed.
   * @param consv_update Flag to enable conservative update heuristic.
   */
  CountMinSketch(uint32_t log2_width, uint32_t depth, int32_t seed, bool consv_update = false);
  ~CountMinSketch();
  uint32_t get(uint32_t key);
  uint32_t update(uint32_t key);

};

} // namespace wmsketch

#endif /* COUNTMIN_H_ */
