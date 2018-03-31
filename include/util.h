/*
 * Utility functions.
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <cstdlib>
#include <cstdint>
#include <climits>
#include <stdexcept>
#include <vector>

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

namespace wmsketch {

void tic(uint64_t& s);
uint64_t toc(uint64_t s);

float mean(const std::vector<float>& buf);
float median(std::vector<float>& buf);

float sigmoid(float x);
float logistic_loss(float x);
float logistic_grad(float x);

} // namespace wmsketch

#endif /* UTIL_H_ */
