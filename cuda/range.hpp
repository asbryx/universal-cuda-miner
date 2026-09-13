#pragma once
#include <cstdint>
#include <stdexcept>
namespace nonce_range {
inline uint64_t high_product(uint64_t a, uint64_t b) {
  const uint64_t mask = 0xffffffffULL;
  uint64_t t = (a & mask) * (b & mask);
  uint64_t carry = t >> 32;
  t = (a >> 32) * (b & mask) + carry;
  uint64_t mid = t & mask, high = t >> 32;
  t = (a & mask) * (b >> 32) + mid;
  return (a >> 32) * (b >> 32) + high + (t >> 32);
}
inline void advance(const uint64_t start[4], uint64_t index, uint64_t step, uint64_t out[4]) {
  uint64_t add[4] = {index * step, high_product(index, step), 0, 0};
  uint64_t carry = 0;
  for (int i = 0; i < 4; ++i) {
    uint64_t t = start[i] + add[i];
    uint64_t c = t < start[i];
    out[i] = t + carry;
    carry = c | (out[i] < t);
  }
  if (carry) throw std::invalid_argument("nonce range overflows uint256");
}
inline void validate(const uint64_t start[4], uint64_t count, uint64_t step, unsigned width) {
  if (!width || width > 32 || !step) throw std::invalid_argument("invalid nonce width or step");
  uint64_t end[4];
  advance(start, count ? count - 1 : 0, step, end);
  for (unsigned byte = width; byte < 32; ++byte)
    if ((end[byte / 8] >> (8 * (byte % 8))) & 255)
      throw std::invalid_argument("nonce range exceeds field width");
}
}
