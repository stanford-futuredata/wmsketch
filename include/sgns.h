/*
 * Streaming skip-gram with negative sampling.
 */

#ifndef GIST_SGNS_H
#define GIST_SGNS_H

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <tuple>
#include <stack>
#include <deque>
#include <unordered_map>
#include <random>
#include <string>
#include "util.h"
#include "heap.h"
#include "countsketch.h"

#include <iostream>

namespace wmsketch {

class TokenReservoir {
 private:
  struct TokenInfo {
    std::string token;
    uint32_t count;
    TokenInfo(const std::string& t, uint32_t c) {
      token = t;
      count = c;
    }
  };

  uint32_t capacity_;
  uint32_t n_;
  std::vector<uint32_t> reservoir_;
  std::vector<TokenInfo> tokens_;
  std::stack<uint32_t> free_;
  std::unordered_map<std::string, uint32_t> token_idx_map_;
  std::mt19937 gen_;
  std::uniform_real_distribution<> rand_;

 public:
  /**
   * Reservoir sampler for unigrams.
   * @param capacity Size of token reservoir
   * @param seed Random seed
   */
  TokenReservoir(uint32_t capacity, int32_t seed)
   : capacity_{capacity},
     n_{0},
     gen_(seed),
     rand_(0, 1) {
    reservoir_.reserve(capacity);
    for (int i = 0; i < capacity_; i++) {
      free_.push(capacity_ - i - 1);
      tokens_.emplace_back(std::string(), 0);
    }
  }

  ~TokenReservoir() = default;

  /**
   * Add a token to the reservoir.
   * @param token Token to be added.
   */
  void update(const std::string& token) {
    n_++;
    if (n_ <= capacity_) {
      uint32_t idx = add(token);
      reservoir_.emplace_back(idx);
    } else {
      auto r = (int) (rand_(gen_) * n_);
      if (r >= capacity_) return;
      uint32_t ri = reservoir_[r];
      auto& ti = tokens_[ri];
      if (--ti.count == 0) {
        token_idx_map_.erase(ti.token);
        free_.push(ri);
      }
      uint32_t idx = add(token);
      reservoir_[r] = idx;
    }
  }

  /**
   * Sample a token from the reservoir.
   * @return Sampled token.
   */
  const std::string& sample() {
    auto r = (int) (rand_(gen_) * reservoir_.size());
    return tokens_[reservoir_[r]].token;
  }

 private:
  uint32_t add(const std::string& token) {
    auto it = token_idx_map_.find(token);
    uint32_t idx;
    if (it == token_idx_map_.end()) {
      idx = free_.top();
      free_.pop();
      tokens_[idx].token = token;
      tokens_[idx].count = 1;
      token_idx_map_[token] = idx;
    } else {
      idx = it->second;
      tokens_[idx].count++;
    }
    return idx;
  }
};

class StreamingSGNS {
 public:
  typedef std::pair<std::string, std::string> StringPair;
  struct StringPairHash {
    std::size_t operator()(StringPair const& s) const noexcept {
      std::size_t h1 = std::hash<std::string>{}(s.first);
      std::size_t h2 = std::hash<std::string>{}(s.second);
      return h1 * 101 + h2;
    }
  };

 private:
  TopKHeap<StringPair, StringPairHash> heap_;
  TokenReservoir reservoir_;
  CountSketch sk_;
  std::deque<std::string> window_;
  uint32_t window_size_;
  uint32_t neg_samples_;
  int32_t seed_;
  float bias_;
  float lr_init_;
  float l2_reg_;
  float scale_;
  uint64_t t_;
  std::mt19937 gen_;
  std::uniform_real_distribution<> rand_;

 public:
  /**
   * Streaming skip-gram with negative sampling.
   *
   * @param k Number of high-magnitude PMI bigrams to track.
   * @param log2_width Base-2 logarithm of the sketch width.
   * @param depth Sketch depth.
   * @param neg_samples Number of samples to draw from the unigram product distribution for each bigram sample.
   * @param window_size Radius of context window. For word at index i, the context is [i - window_size, i + window_size].
   * @param reservoir_size Size of the reservoir sample of unigrams.
   * @param seed Random seed.
   * @param lr_init Initial learning rate.
   * @param l2_reg L2 regularization parameter.
   */
  StreamingSGNS(
      uint32_t k,
      uint32_t log2_width,
      uint32_t depth,
      uint32_t neg_samples,
      uint32_t window_size,
      uint32_t reservoir_size,
      int32_t seed,
      float lr_init,
      float l2_reg);

  ~StreamingSGNS() = default;

  /**
   * Fill the given output vector with k (StringPair, float) pairs denoting the token pairs with the highest-magnitude
   * estimated PMIs, where the first item is the token pair and the second item is the estimated PMI.
   *
   * @param out Target output vector for token pairs with high-magnitude estimated PMI. Overwrites any existing
   *   contents of the vector.
   */
  void topk(std::vector<std::pair<StringPair, float> >& out);

  /**
   * Update the model with a new token.
   *
   * @param token Token to be used to update the model.
   */
  void update(const std::string& token);

  /**
   * Flush the current context window.
   */
  void flush();

 private:
  void update(const std::string& a, const std::string& b);
  void update(const std::string& a, const std::string& b, bool real);
  uint32_t strings_to_key(const std::string& a, const std::string& b);
};

} // namespace wmsketch

#endif //GIST_SGNS_H
