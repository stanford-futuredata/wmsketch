#include <math.h>
#include "heap.h"

namespace wmsketch {

TopKCountHeap::TopKCountHeap(uint32_t capacity)
 : capacity_{capacity},
   n_{0} {
  pq_.resize(capacity + 1);
  qp_.reserve(capacity + 1);
}

TopKCountHeap::~TopKCountHeap() = default;

uint32_t TopKCountHeap::size() {
  return n_;
}

bool TopKCountHeap::is_empty() {
  return n_ == 0;
}

bool TopKCountHeap::is_full() {
  return n_ == capacity_;
}

bool TopKCountHeap::contains(uint32_t key) {
  return qp_.find(key) != qp_.end();
}

float TopKCountHeap::get(uint32_t key) {
  return std::get<2>(qp_[key]);
}

void TopKCountHeap::keys(std::vector<uint32_t>& out) {
  out.clear();
  for (int i = 1; i <= n_; i++) {
    out.push_back(pq_[i]);
  }
}

void TopKCountHeap::items(std::vector<std::pair<uint32_t, float> >& out) {
  out.clear();
  for (auto& it : qp_) {
    out.emplace_back(std::make_pair(it.first, std::get<2>(it.second)));
  }
}

uint32_t TopKCountHeap::get_count(uint32_t key) {
  return std::get<1>(qp_[key]);
}

void TopKCountHeap::increment_count(uint32_t key) {
  std::get<1>(qp_[key])++;
}

void TopKCountHeap::change_val(uint32_t key, uint32_t count, float val) {
  if (!contains(key)) throw std::invalid_argument("Key does not exist");
  std::get<1>(qp_[key]) = count;
  std::get<2>(qp_[key]) = val;
  swim(std::get<0>(qp_[key]));
  sink(std::get<0>(qp_[key]));
}

std::experimental::optional<std::tuple<uint32_t, uint32_t, float> >
TopKCountHeap::insert(uint32_t key, uint32_t count, float val) {
  if (contains(key)) throw std::invalid_argument("Key already exists");
  bool opt = false;
  std::tuple<uint32_t, uint32_t, float> evicted;
  if (n_ == capacity_) {
    opt = true;
    if (min_val() > count) {
      return std::make_tuple(key, count, val);
    } else {
      evicted = del_min();
    }
  }
  n_++;
  qp_[key] = std::make_tuple(n_, count, val);
  pq_[n_] = key;
  swim(n_);
  if (opt) return evicted;
  else return {};
}

std::experimental::optional<std::tuple<uint32_t, uint32_t, float> >
TopKCountHeap::insert_or_change(uint32_t key, uint32_t count, float val) {
  if (contains(key)) {
    change_val(key, count, val);
    return {};
  } else  {
    return insert(key, count, val);
  }
}

uint32_t TopKCountHeap::min_val() {
  if (n_ == 0) throw std::runtime_error("Priority queue underflow");
  uint32_t idx = pq_[1];
  return std::get<1>(qp_[idx]);
}

std::tuple<uint32_t, uint32_t, float> TopKCountHeap::min() {
  if (n_ == 0) throw std::runtime_error("Priority queue underflow");
  uint32_t idx = pq_[1];
  return std::make_tuple(idx, std::get<1>(qp_[idx]), std::get<2>(qp_[idx]));
}

std::tuple<uint32_t, uint32_t, float> TopKCountHeap::del_min() {
  if (n_ == 0) throw std::runtime_error("Priority queue underflow");
  uint32_t idx = pq_[1];
  auto tup = std::make_tuple(idx, std::get<1>(qp_[idx]), std::get<2>(qp_[idx]));
  exch(1, n_--);
  sink(1);
  qp_.erase(idx);
  return tup;
}

bool TopKCountHeap::greater(uint32_t i, uint32_t j) {
  return std::get<1>(qp_[pq_[i]]) > std::get<1>(qp_[pq_[j]]);
}

void TopKCountHeap::exch(uint32_t i, uint32_t j) {
  uint32_t swap = pq_[i];
  pq_[i] = pq_[j];
  pq_[j] = swap;
  std::get<0>(qp_[pq_[i]]) = i;
  std::get<0>(qp_[pq_[j]]) = j;
}

void TopKCountHeap::swim(uint32_t k) {
  while (k > 1 && greater(k/2, k)) {
    exch(k, k/2);
    k = k/2;
  }
}

void TopKCountHeap::sink(uint32_t k) {
  while (2*k <= n_) {
    int j = 2*k;
    if (j < n_ && greater(j, j+1)) j++;
    if (!greater(k, j)) break;
    exch(k, j);
    k = j;
  }
}

///////////////////////////////////////////////////////////////////////////////

WeightedReservoir::WeightedReservoir(uint32_t capacity)
 : capacity_{capacity},
   n_{0},
   rand_(0, 1),
   pow_{1.} {
  pq_.resize(capacity + 1);
  qp_.reserve(capacity + 1);
}

WeightedReservoir::WeightedReservoir(uint32_t capacity, int32_t seed, float pow)
 : capacity_{capacity},
   n_{0},
   gen_(seed),
   rand_(0, 1),
   pow_{pow} {
  pq_.reserve(capacity + 1);
  qp_.reserve(capacity + 1);
}

WeightedReservoir::~WeightedReservoir() { }

uint32_t WeightedReservoir::size() {
  return n_;
}

bool WeightedReservoir::is_empty() {
  return n_ == 0;
}

bool WeightedReservoir::is_full() {
  return n_ == capacity_;
}

bool WeightedReservoir::contains(uint32_t key) {
  return qp_.find(key) != qp_.end();
}

float WeightedReservoir::get(uint32_t key) {
  return std::get<2>(qp_[key]);
}

void WeightedReservoir::keys(std::vector<uint32_t>& out) {
  out.clear();
  for (int i = 1; i <= n_; i++) {
    out.push_back(pq_[i]);
  }
}

void WeightedReservoir::items(std::vector<std::pair<uint32_t, float> >& out) {
  out.clear();
  for (auto& it : qp_) {
    out.emplace_back(it.first, std::get<2>(it.second));
  }
}

void WeightedReservoir::change_val(uint32_t key, float val) {
  if (!contains(key)) throw std::invalid_argument("Key does not exist");
  float old_val = std::get<2>(qp_[key]);
  if (pow_ == 1.) {
    std::get<1>(qp_[key]) *= fabs(val / old_val);
  } else {
    std::get<1>(qp_[key]) *= pow(fabs(val / old_val), pow_);
  }
  std::get<2>(qp_[key]) = val;
  swim(std::get<0>(qp_[key]));
  sink(std::get<0>(qp_[key]));
}

std::experimental::optional<std::pair<uint32_t, float> >
WeightedReservoir::insert(uint32_t key, float val) {
  if (contains(key)) throw std::invalid_argument("Key already exists");
  bool opt = false;

  // assign item a random weight propotional to |value|
  // heap holds up to k items with smallest random weights
  // see Efraimidis and Spirakis (2006) for more details on this method of implementing weighted reservoir sampling
  float r = pow(fabs(val), pow_) * log(rand_(gen_));
  std::pair<uint32_t, float> evicted;
  if (n_ == capacity_) {
    opt = true;
    if (r > max_val()) {
      return std::make_pair(key, val);
    } else {
      evicted = del_max();
    }
  }
  n_++;
  qp_[key] = std::make_tuple(n_, r, val);
  pq_[n_] = key;
  swim(n_);
  if (opt) return evicted;
  else return {};
}

std::experimental::optional<std::pair<uint32_t, float> >
WeightedReservoir::insert_or_change(uint32_t key, float val) {
  if (contains(key)) {
    change_val(key, val);
    return {};
  } else  {
    return insert(key, val);
  }
}

float WeightedReservoir::max_val() {
  if (n_ == 0) throw std::runtime_error("Priority queue underflow");
  uint32_t idx = pq_[1];
  return std::get<1>(qp_[idx]);
}

std::pair<uint32_t, float> WeightedReservoir::del_max() {
  if (n_ == 0) throw std::runtime_error("Priority queue underflow");
  uint32_t idx = pq_[1];
  auto pair = std::make_pair(idx, std::get<2>(qp_[idx]));
  exch(1, n_--);
  sink(1);
  qp_.erase(idx);
  return pair;
}

bool WeightedReservoir::greater(uint32_t i, uint32_t j) {
  return std::get<1>(qp_[pq_[i]]) > std::get<1>(qp_[pq_[j]]);
}

void WeightedReservoir::exch(uint32_t i, uint32_t j) {
  uint32_t swap = pq_[i];
  pq_[i] = pq_[j];
  pq_[j] = swap;
  std::get<0>(qp_[pq_[i]]) = i;
  std::get<0>(qp_[pq_[j]]) = j;
}

void WeightedReservoir::swim(uint32_t k) {
  while (k > 1 && greater(k, k/2)) {
    exch(k, k/2);
    k = k/2;
  }
}

void WeightedReservoir::sink(uint32_t k) {
  while (2*k <= n_) {
    int j = 2*k;
    if (j < n_ && greater(j+1, j)) j++;
    if (!greater(j, k)) break;
    exch(k, j);
    k = j;
  }
}

} // namespace wmsketch
