#include "util.h"
#include <algorithm>
#include <math.h>
#include <numeric>
#include <sys/time.h>
#include <time.h>

namespace wmsketch {

void tic(uint64_t& s) {
  struct timeval tv;
  gettimeofday(&tv, 0);
  s = (1000 * tv.tv_sec) + (tv.tv_usec / 1000);
}

uint64_t toc(uint64_t s) {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (1000 * tv.tv_sec) + (tv.tv_usec / 1000) - s;
}

float mean(const std::vector<float>& buf) {
  size_t n = buf.size();
  return std::accumulate(buf.begin(), buf.end(), 0.) / n;
}

float median(std::vector<float>& buf) {
  size_t n = buf.size();
  std::nth_element(buf.begin(), buf.begin() + n/2, buf.end());
  if (n % 2 == 1) return buf[n/2];
  std::nth_element(buf.begin(), buf.begin() + n/2 - 1, buf.begin() + n/2);
  return (buf[n/2 - 1] + buf[n/2]) / 2;
}

float sigmoid(float x) {
  return 1.f / (1.f + exp(-x));
}

float logistic_loss(float x) {
  return log(1.f + exp(-x));
}

float logistic_grad(float x) {
  return -sigmoid(-x);
}

} // namespace wmsketch
