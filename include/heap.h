/*
 * Heap data structures.
 *
 * Heap implementation adapted from Sedgewick & Wayne, Algorithms 4th Ed.
 * http://algs4.cs.princeton.edu/24pq/IndexMinPQ.java.html
 */

#ifndef HEAP_H_
#define HEAP_H_

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <experimental/optional>
#include <random>
#include <algorithm>
#include <functional>

namespace wmsketch {

template <class T, class H = std::hash<T>>
class TopKHeap {
 private:
  uint32_t capacity_;
  uint32_t n_;
  std::vector<T> pq_;  // [1, capacity+1) -> idx
  std::unordered_map<T, std::pair<uint32_t, float>, H> qp_;  // idx -> ([1, capacity+1), val)

 public:
  /**
   * Min-heap for tracking top-k items ordered by the magnitude of a floating point value associated with each item.
   * When an item is added to a heap that already contains k items, the item with the lowest-magnitude value is evicted.
   *
   * @param capacity Heap capacity.
   */
  explicit TopKHeap(uint32_t capacity)
   : capacity_{capacity},
     n_{0} {
    pq_.resize(capacity + 1);
    qp_.reserve(capacity + 1);
  }

  ~TopKHeap() = default;
  uint32_t size() {
    return n_;
  }

  bool is_empty() {
    return n_ == 0;
  }

  bool is_full() {
    return n_ == capacity_;
  }

  bool contains(const T& key) {
    return qp_.find(key) != qp_.end();
  }

  float get(const T& key) {
    return qp_.at(key).second;
  }

  void keys(std::vector<T>& out) {
    out.clear();
    for (int i = 1; i <= n_; i++) {
      out.push_back(pq_[i]);
    }
  }

  void items(std::vector<std::pair<T, float> >& out) {
    out.clear();
    for (auto& it : qp_) {
      out.push_back(std::make_pair(it.first, it.second.second));
    }
  }

  void change_val(const T& key, float val) {
    if (!contains(key)) throw std::invalid_argument("Key does not exist");
    qp_.at(key).second = val;
    swim(qp_.at(key).first);
    sink(qp_.at(key).first);
  }

  /**
   * Attempt to insert an item with key \p key and value \p val. Throws an exception if an item with key \p key already
   * exists. If the heap is full, returns the evicted item (this can be the item that the caller just tried to insert).
   *
   * @param key The key.
   * @param val The value (e.g. a weight).
   * @return The evicted item, if any.
   */
  std::experimental::optional<std::pair<T, float> >
  insert(const T& key, float val) {
    if (contains(key)) throw std::invalid_argument("Key already exists");
    bool opt = false;
    std::pair<T, float> evicted;
    if (n_ == capacity_) {
      opt = true;
      if (fabs(min_val()) > fabs(val)) {
        return std::make_pair(key, val);
      } else {
        evicted = del_min();
      }
    }
    n_++;
    qp_[key] = std::make_pair(n_, val);
    pq_[n_] = key;
    swim(n_);
    if (opt) return evicted;
    else return {};
  }

  /**
   * Attempt to insert an item with key \p key and value \p val. If an item with key \p key already exists, updates the
   * associated value. If the heap is full, returns the evicted item (this can be the item that the caller just tried
   * to insert).
   *
   * @param key The key.
   * @param val The value (e.g. a weight).
   * @return The evicted item, if any.
   */
  std::experimental::optional<std::pair<T, float> >
  insert_or_change(const T& key, float val) {
    if (contains(key)) {
      change_val(key, val);
      return {};
    } else {
      return insert(key, val);
    }
  }

  float min_val() {
    if (n_ == 0) throw std::runtime_error("Priority queue underflow");
    const T& idx = pq_[1];
    return qp_.at(idx).second;
  }

  std::pair<T, float> min() {
    if (n_ == 0) throw std::runtime_error("Priority queue underflow");
    const T& idx = pq_[1];
    return std::make_pair(idx, qp_.at(idx).second);
  }

  std::pair<T, float> del_min() {
    if (n_ == 0) throw std::runtime_error("Priority queue underflow");
    auto pair = std::make_pair(pq_[1], qp_.at(pq_[1]).second);
    exch(1, n_--);
    sink(1);
    qp_.erase(pair.first);
    return pair;
  }

 private:
  bool greater(uint32_t i, uint32_t j) {
    return fabs(qp_.at(pq_[i]).second) > fabs(qp_.at(pq_[j]).second);
  }

  void exch(uint32_t i, uint32_t j) {
    std::iter_swap(pq_.begin() + i, pq_.begin() + j);
    qp_.at(pq_[i]).first = i;
    qp_.at(pq_[j]).first = j;
  }

  void swim(uint32_t k) {
    while (k > 1 && greater(k/2, k)) {
      exch(k, k/2);
      k = k/2;
    }
  }

  void sink(uint32_t k) {
    while (2*k <= n_) {
      uint32_t j = 2*k;
      if (j < n_ && greater(j, j+1)) j++;
      if (!greater(k, j)) break;
      exch(k, j);
      k = j;
    }
  }
};

class TopKCountHeap {
 private:
  uint32_t capacity_;
  uint32_t n_;
  std::vector<uint32_t> pq_;
  std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t, float> > qp_;  // idx -> ([1, capacity+1], count, val)

 public:
  /**
   * Min-heap for tracking top-k items ordered by integer count. When an item is added to a heap that already contains
   * k items, the item with the lowest count is evicted.
   *
   * @param capacity Heap capacity.
   */
  explicit TopKCountHeap(uint32_t capacity);
  ~TopKCountHeap();
  uint32_t size();
  bool is_empty();
  bool is_full();
  bool contains(uint32_t key);
  float get(uint32_t key);
  void keys(std::vector<uint32_t>& out);
  void items(std::vector<std::pair<uint32_t, float> >& out);

  /**
   * Change the count and auxiliary value for key \p key.
   * @param key The key.
   * @param count The new count.
   * @param val The new value.
   */
  void change_val(uint32_t key, uint32_t count, float val);

  uint32_t get_count(uint32_t key);
  void increment_count(uint32_t key);

  /**
   * Attempt to insert an item with key \p key, count \p count, and auxiliary value \p val. Throws an exception if an
   * item with key \p key already exists. If the heap is full, returns the evicted item (this can be the item that the
   * caller just tried to insert).
   *
   * @param key The key.
   * @param count The count.
   * @param val An auxiliary value (e.g. a weight). Has no effect on heap ordering.
   * @return The evicted item, if any.
   */
  std::experimental::optional<std::tuple<uint32_t, uint32_t, float> >
  insert(uint32_t key, uint32_t count, float val);

  /**
   * Attempt to insert an item with key \p key, count \p count, and auxiliary value \p val. If an item with key \p key
   * already exists, updates the associated count and value. If the heap is full, returns the evicted item (this can be
   * the item that the caller just tried to insert).
   *
   * @param key The key.
   * @param count The count.
   * @param val An auxiliary value (e.g. a weight). Has no effect on heap ordering.
   * @return The evicted item, if any.
   */
  std::experimental::optional<std::tuple<uint32_t, uint32_t, float> >
  insert_or_change(uint32_t key, uint32_t count, float val);

  /**
   * Return the minimum count in the heap.
   * @return The minimum count.
   */
  uint32_t min_val();
  std::tuple<uint32_t, uint32_t, float> min();
  std::tuple<uint32_t, uint32_t, float> del_min();

 private:
  bool greater(uint32_t i, uint32_t j);
  void exch(uint32_t i, uint32_t j);
  void swim(uint32_t k);
  void sink(uint32_t k);
};

class WeightedReservoir {
 private:
  uint32_t capacity_;
  uint32_t n_;
  std::vector<uint32_t> pq_;
  std::unordered_map<uint32_t, std::tuple<uint32_t, float, float> > qp_;  // idx -> ([1, capacity+1], rand_key, val)
  std::mt19937 gen_;
  std::uniform_real_distribution<> rand_;
  float pow_;

 public:
  /**
   * Weighted reservoir sampler where the weight of each entry is given by the absolute value of its associated value.
   *
   * @param capacity Capacity of the reservoir.
   */
  WeightedReservoir(uint32_t capacity);

  /**
   * Weighted reservoir sampler where the weight of each entry is given by the absolute value of its associated value.
   *
   * @param capacity Capacity of the reservoir.
   * @param seed Random seed.
   * @param pow Power to which weight is raised.
   */
  WeightedReservoir(uint32_t capacity, int32_t seed, float pow = 1.);
  ~WeightedReservoir();
  uint32_t size();
  bool is_empty();
  bool is_full();
  bool contains(uint32_t key);
  float get(uint32_t key);
  void keys(std::vector<uint32_t>& out);
  void items(std::vector<std::pair<uint32_t, float> >& out);
  void change_val(uint32_t key, float val);
  std::experimental::optional<std::pair<uint32_t, float> > insert(uint32_t key, float val);
  std::experimental::optional<std::pair<uint32_t, float> > insert_or_change(uint32_t key, float val);

 private:
  float max_val();
  std::pair<uint32_t, float> del_max();
  bool greater(uint32_t i, uint32_t j);
  void exch(uint32_t i, uint32_t j);
  void swim(uint32_t k);
  void sink(uint32_t k);
};

} // namespace wmsketch

#endif
