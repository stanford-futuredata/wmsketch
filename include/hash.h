/*
 * Hash functions
 */

#ifndef HASH_H_
#define HASH_H_

#include <cstdlib>
#include <cstdint>
#include <vector>

namespace wmsketch {
namespace hash {

class HashFunction {
 public:
  virtual ~HashFunction() = default;
  virtual void hash(uint32_t* out, uint32_t x) = 0;
};

class PolynomialHash : public HashFunction {
 private:
  uint32_t** table_;
  uint32_t copies_;

 public:
  PolynomialHash(uint32_t copies, int32_t seed);
  ~PolynomialHash() override;
  void hash(uint32_t* out, uint32_t x) override;
};

// tabulation hashing
static const size_t THASH_CHUNK_BITS = 8;
static const size_t THASH_NUM_CHUNKS = 32 / THASH_CHUNK_BITS;
static const size_t THASH_CHUNK_CARD = 1 << THASH_CHUNK_BITS;

class TabulationHash : public HashFunction {
 private:
  uint32_t** table_;
  uint32_t copies_;

 public:
  TabulationHash(uint32_t copies, int32_t seed);
  ~TabulationHash() override;
  void hash(uint32_t* out, uint32_t x) override;
};

// 32-bit MurmurHash3
uint32_t murmurhash3_32(const void* key, int len, uint32_t seed);

} // namespace hash
} // namespace wmsketch

#endif /* HASH_H_ */
