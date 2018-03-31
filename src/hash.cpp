#include <cstring>
#include <random>
#include "hash.h"

#define MOD 2147483647  // 2^31 - 1
#define HL 31

namespace wmsketch {
namespace hash {

// 2-independent polynomial hash
PolynomialHash::PolynomialHash(uint32_t copies, int32_t seed)
 : copies_{copies} {
  table_ = (uint32_t**) calloc(copies_, sizeof(uint32_t*));
  table_[0] = (uint32_t*) calloc(2 * copies_, sizeof(uint32_t));
  std::mt19937 prng(seed);
  for (int i = 0; i < copies_; i++) {
    table_[i] = table_[0] + 2 * i;
    table_[i][0] = prng();
    table_[i][1] = prng();
  }
}

PolynomialHash::~PolynomialHash() {
  free(table_[0]);
  free(table_);
}

void PolynomialHash::hash(uint32_t* out, uint32_t x) {
  for (int i = 0; i < copies_; i++) {
    uint64_t res = ((uint64_t) table_[i][0] * x) + table_[i][1];
    res = ((res >> HL) + res) & MOD;
    out[i] = res;
  }
}

// tabulation hashing
TabulationHash::TabulationHash(uint32_t copies, int32_t seed)
 : copies_{copies} {
  table_ = (uint32_t**) calloc(THASH_NUM_CHUNKS, sizeof(uint32_t*));
  table_[0] = (uint32_t*) calloc(THASH_NUM_CHUNKS * THASH_CHUNK_CARD * copies_, sizeof(uint32_t));
  std::mt19937 prng(seed);
  for (int i = 0; i < THASH_NUM_CHUNKS; i++) {
    table_[i] = table_[0] + i * THASH_CHUNK_CARD * copies_;
    for (int j = 0; j < THASH_CHUNK_CARD * copies_; j++) {
      table_[i][j] = prng();
    }
  }
}

TabulationHash::~TabulationHash() {
  free(table_[0]);
  free(table_);
}

void TabulationHash::hash(uint32_t* out, uint32_t x) {
  memset(out, 0, copies_ * sizeof(uint32_t));
  for (int i = 0; i < THASH_NUM_CHUNKS; i++) {
    uint32_t c = (x >> (i * THASH_CHUNK_BITS)) & (THASH_CHUNK_CARD - 1);
    uint32_t *hashes = table_[i] + c * copies_;
    for (int j = 0; j < copies_; j++) {
      out[j] ^= hashes[j];
    }
  }
}

inline __attribute__((always_inline)) uint32_t fmix32 ( uint32_t h )
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

inline uint32_t rotl32 (uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}

// MurmurHash3: https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
uint32_t murmurhash3_32(const void* key, int len, uint32_t seed) {
  const uint8_t * data = (const uint8_t*)key;
  const int nblocks = len / 4;

  uint32_t h1 = seed;

  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;

  //----------
  // body

  const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);

  for(int i = -nblocks; i; i++) {
    uint32_t k1 = blocks[i];

    k1 *= c1;
    k1 = rotl32(k1,15);
    k1 *= c2;

    h1 ^= k1;
    h1 = rotl32(h1,13);
    h1 = h1*5+0xe6546b64;
  }

  //----------
  // tail

  const uint8_t * tail = (const uint8_t*)(data + nblocks*4);

  uint32_t k1 = 0;

  switch(len & 3) {
  case 3: k1 ^= tail[2] << 16;
  case 2: k1 ^= tail[1] << 8;
  case 1: k1 ^= tail[0];
          k1 *= c1; k1 = rotl32(k1,15); k1 *= c2; h1 ^= k1;
  };

  //----------
  // finalization

  h1 ^= len;
  h1 = fmix32(h1);
  return h1;
}

} // namespace hash
} // namespace wmsketch
