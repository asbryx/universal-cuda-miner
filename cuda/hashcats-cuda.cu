#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using u64 = unsigned long long;

// Hashcats work is keccak256(address[20] || nonce[32] || prevWork[32] || anchor[32]).
// The 116-byte message is padded with Keccak's 0x01 ... 0x80 suffix.
// U256 limbs are little-endian words on the host/device, while the wire fields
// and the nonce bytes inserted into the message are big-endian.
struct Work {
  u64 base[17];       // padded Keccak block, with the 32 nonce bytes zeroed
  u64 start[4];       // little-endian uint256 start nonce
  u64 count;           // practical batch size; start + count is performed in U256
  u64 target_be[4];   // target as four big-endian comparison words
};

struct Result {
  unsigned int found;
  u64 nonce[4];
  u64 digest[4];       // Keccak lanes; serialized as little-endian bytes
};

static void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

__device__ __forceinline__ u64 rotl64(u64 x, int n) {
  return n == 0 ? x : ((x << n) | (x >> (64 - n)));
}

__device__ __forceinline__ u64 bswap64(u64 x) {
  return ((x & 0x00000000000000ffULL) << 56) |
         ((x & 0x000000000000ff00ULL) << 40) |
         ((x & 0x0000000000ff0000ULL) << 24) |
         ((x & 0x00000000ff000000ULL) << 8) |
         ((x & 0x000000ff00000000ULL) >> 8) |
         ((x & 0x0000ff0000000000ULL) >> 24) |
         ((x & 0x00ff000000000000ULL) >> 40) |
         ((x & 0xff00000000000000ULL) >> 56);
}

__device__ __forceinline__ void add_offset(const u64 start[4], u64 offset, u64 out[4]) {
  out[0] = start[0] + offset;
  u64 carry = out[0] < start[0] ? 1ULL : 0ULL;
  out[1] = start[1] + carry;
  carry = carry && out[1] == 0ULL ? 1ULL : 0ULL;
  out[2] = start[2] + carry;
  carry = carry && out[2] == 0ULL ? 1ULL : 0ULL;
  out[3] = start[3] + carry;
}

__device__ __forceinline__ void keccakf(u64 a[25]) {
  constexpr u64 rc[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL,
      0x800000000000808aULL, 0x8000000080008000ULL,
      0x000000000000808bULL, 0x0000000080000001ULL,
      0x8000000080008081ULL, 0x8000000000008009ULL,
      0x000000000000008aULL, 0x0000000000000088ULL,
      0x0000000080008009ULL, 0x000000008000000aULL,
      0x000000008000808bULL, 0x800000000000008bULL,
      0x8000000000008089ULL, 0x8000000000008003ULL,
      0x8000000000008002ULL, 0x8000000000000080ULL,
      0x000000000000800aULL, 0x800000008000000aULL,
      0x8000000080008081ULL, 0x8000000000008080ULL,
      0x0000000080000001ULL, 0x8000000080008008ULL};
  constexpr int rotc[24] = {1,  3,  6,  10, 15, 21, 28, 36,
                             45, 55, 2,  14, 27, 41, 56, 8,
                             25, 43, 62, 18, 39, 61, 20, 44};
  constexpr int piln[24] = {10, 7,  11, 17, 18, 3,  5,  16,
                             8,  21, 24, 4,  15, 23, 19, 13,
                             12, 2,  20, 14, 22, 9,  6,  1};

#pragma unroll
  for (int round = 0; round < 24; ++round) {
    u64 c[5];
#pragma unroll
    for (int x = 0; x < 5; ++x) {
      c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
    }
#pragma unroll
    for (int x = 0; x < 5; ++x) {
      u64 d = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
#pragma unroll
      for (int y = 0; y < 5; ++y) a[x + 5 * y] ^= d;
    }

    u64 t = a[1];
#pragma unroll
    for (int i = 0; i < 24; ++i) {
      u64 next = a[piln[i]];
      a[piln[i]] = rotl64(t, rotc[i]);
      t = next;
    }
#pragma unroll
    for (int y = 0; y < 5; ++y) {
      const int row = 5 * y;
      u64 r0 = a[row], r1 = a[row + 1], r2 = a[row + 2];
      u64 r3 = a[row + 3], r4 = a[row + 4];
      a[row] = r0 ^ ((~r1) & r2);
      a[row + 1] = r1 ^ ((~r2) & r3);
      a[row + 2] = r2 ^ ((~r3) & r4);
      a[row + 3] = r3 ^ ((~r4) & r0);
      a[row + 4] = r4 ^ ((~r0) & r1);
    }
    a[0] ^= rc[round];
  }
}

// Strict numeric comparison: digest < target, never digest <= target.
__device__ __forceinline__ bool under_target(const u64 a[25], const u64 target_be[4]) {
  const u64 h0 = bswap64(a[0]);
  if (h0 != target_be[0]) return h0 < target_be[0];
  const u64 h1 = bswap64(a[1]);
  if (h1 != target_be[1]) return h1 < target_be[1];
  const u64 h2 = bswap64(a[2]);
  if (h2 != target_be[2]) return h2 < target_be[2];
  return bswap64(a[3]) < target_be[3];
}

__global__ void mine_kernel(Work work, Result* result) {
  const u64 tid = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
  const u64 stride = static_cast<u64>(gridDim.x) * blockDim.x;

  for (u64 offset = tid; offset < work.count; offset += stride) {
    if (result->found) return;

    u64 nonce[4];
    add_offset(work.start, offset, nonce);

    u64 a[25] = {};
#pragma unroll
    for (int i = 0; i < 17; ++i) a[i] = work.base[i];

    // The nonce is a full uint256 big-endian field at message bytes 20..51.
    // Keccak lanes absorb little-endian bytes, hence the byte swaps and reverse
    // limb order. No high nonce bytes are discarded.
    const u64 n3 = bswap64(nonce[3]);
    const u64 n2 = bswap64(nonce[2]);
    const u64 n1 = bswap64(nonce[1]);
    const u64 n0 = bswap64(nonce[0]);
    // The 32-byte field starts four bytes into lane 2 and ends four
    // bytes into lane 6, so the address/prevWork halves must be preserved.
    a[2] = (a[2] & 0x00000000ffffffffULL) | ((n3 & 0xffffffffULL) << 32);
    a[3] = (n3 >> 32) | ((n2 & 0xffffffffULL) << 32);
    a[4] = (n2 >> 32) | ((n1 & 0xffffffffULL) << 32);
    a[5] = (n1 >> 32) | ((n0 & 0xffffffffULL) << 32);
    a[6] = (a[6] & 0xffffffff00000000ULL) | (n0 >> 32);

    keccakf(a);
    if (under_target(a, work.target_be) &&
        atomicCAS(&result->found, 0U, 1U) == 0U) {
      for (int i = 0; i < 4; ++i) {
        result->nonce[i] = nonce[i];
        result->digest[i] = a[i];
      }
    }
  }
}

struct U256 {
  u64 limb[4]{};
};

static int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static std::string strip_0x(std::string s) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s.erase(0, 2);
  return s;
}

static U256 parse_u256(std::string input) {
  if (input.empty()) throw std::runtime_error("empty uint256");
  const bool explicit_hex = input.rfind("0x", 0) == 0 || input.rfind("0X", 0) == 0;
  std::string s = strip_0x(std::move(input));
  if (s.empty()) throw std::runtime_error("empty uint256");

  U256 out{};
  if (explicit_hex) {
    if (s.size() > 64) throw std::runtime_error("uint256 overflow");
    for (char c : s) {
      int d = hex_digit(c);
      if (d < 0) throw std::runtime_error("bad uint256 hex");
      u64 carry = static_cast<u64>(d);
      for (int i = 0; i < 4; ++i) {
        const u64 old = out.limb[i];
        out.limb[i] = (old << 4) | carry;
        carry = old >> 60;
      }
      if (carry) throw std::runtime_error("uint256 overflow");
    }
    return out;
  }

  std::array<uint32_t, 8> words{};
  for (char c : s) {
    if (c < '0' || c > '9') throw std::runtime_error("bad uint256 decimal");
    uint64_t carry = static_cast<unsigned>(c - '0');
    for (int i = 0; i < 8; ++i) {
      uint64_t v = static_cast<uint64_t>(words[i]) * 10ULL + carry;
      words[i] = static_cast<uint32_t>(v);
      carry = v >> 32;
    }
    if (carry) throw std::runtime_error("uint256 overflow");
  }
  for (int i = 0; i < 4; ++i) {
    out.limb[i] = static_cast<u64>(words[2 * i]) |
                  (static_cast<u64>(words[2 * i + 1]) << 32);
  }
  return out;
}

static std::vector<uint8_t> parse_bytes(const std::string& input, size_t bytes) {
  const std::string s = strip_0x(input);
  if (s.size() != bytes * 2) throw std::runtime_error("bad hex length");
  std::vector<uint8_t> out(bytes);
  for (size_t i = 0; i < bytes; ++i) {
    int hi = hex_digit(s[2 * i]);
    int lo = hex_digit(s[2 * i + 1]);
    if (hi < 0 || lo < 0) throw std::runtime_error("bad hex");
    out[i] = static_cast<uint8_t>((hi << 4) | lo);
  }
  return out;
}

static u64 load_le(const uint8_t* p) {
  u64 x = 0;
  for (int i = 7; i >= 0; --i) x = (x << 8) | p[i];
  return x;
}

static u64 load_be64(const uint8_t* p) {
  u64 x = 0;
  for (int i = 0; i < 8; ++i) x = (x << 8) | p[i];
  return x;
}

static Work make_work(const std::string& address, const std::string& prev,
                      const std::string& anchor, const std::string& target,
                      const U256& start, const U256& count) {
  const auto address_bytes = parse_bytes(address, 20);
  const auto prev_bytes = parse_bytes(prev, 32);
  const auto anchor_bytes = parse_bytes(anchor, 32);
  const auto target_bytes = parse_bytes(target, 32);

  uint8_t message[136] = {};
  std::memcpy(message, address_bytes.data(), 20);
  std::memcpy(message + 52, prev_bytes.data(), 32);
  std::memcpy(message + 84, anchor_bytes.data(), 32);
  message[116] = 0x01;
  message[135] = 0x80;

  Work work{};
  for (int i = 0; i < 17; ++i) work.base[i] = load_le(message + 8 * i);
  for (int i = 0; i < 4; ++i) work.start[i] = start.limb[i];
  for (int i = 0; i < 4; ++i) work.target_be[i] = load_be64(target_bytes.data() + 8 * i);

  if (count.limb[1] || count.limb[2] || count.limb[3]) {
    throw std::runtime_error("count must fit uint64");
  }
  work.count = count.limb[0];
  return work;
}

static std::string u256_decimal(const u64 limbs[4]) {
  std::array<uint32_t, 8> words{};
  for (int i = 0; i < 4; ++i) {
    words[2 * i] = static_cast<uint32_t>(limbs[i]);
    words[2 * i + 1] = static_cast<uint32_t>(limbs[i] >> 32);
  }
  std::string out;
  bool started = false;
  while (!started) {
    uint64_t remainder = 0;
    bool any = false;
    for (int i = 7; i >= 0; --i) {
      uint64_t value = (remainder << 32) | words[i];
      words[i] = static_cast<uint32_t>(value / 10ULL);
      remainder = value % 10ULL;
      any = any || words[i] != 0;
    }
    out.push_back(static_cast<char>('0' + remainder));
    started = !any;
  }
  std::reverse(out.begin(), out.end());
  return out;
}

static std::string digest_hex(const u64 digest[4]) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (int i = 0; i < 4; ++i) {
    u64 x = digest[i];
    for (int j = 0; j < 8; ++j) {
      const uint8_t b = static_cast<uint8_t>(x & 0xffU);
      out.push_back(hex[b >> 4]);
      out.push_back(hex[b & 0x0f]);
      x >>= 8;
    }
  }
  return out;
}

static int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  const bool allow_zero = std::string(name) == "CUDA_DEVICE";
  if (*end || parsed < 0 || (!allow_zero && parsed == 0) ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("bad ") + name);
  }
  return static_cast<int>(parsed);
}

class Miner {
 public:
  Miner() {
    const int device = env_int("CUDA_DEVICE", 0);
    check(cudaSetDevice(device), "cudaSetDevice");
    check(cudaGetDeviceProperties(&properties_, device), "cudaGetDeviceProperties");
    check(cudaMalloc(&device_result_, sizeof(Result)), "cudaMalloc");
    check(cudaEventCreate(&begin_), "cudaEventCreate(begin)");
    check(cudaEventCreate(&end_), "cudaEventCreate(end)");
    threads_ = env_int("CUDA_THREADS", 256);
    if (threads_ < 32 || threads_ > 1024 || (threads_ % 32) != 0) {
      throw std::runtime_error("CUDA_THREADS must be a warp-aligned value from 32 to 1024");
    }
    // The Keccak state is register-heavy; four resident blocks per SM is a
    // measured starting point on RTX 5090. The environment override keeps
    // tuning possible without recompiling.
    const int default_blocks = properties_.multiProcessorCount * 4;
    const int blocks_per_sm = env_int("CUDA_BLOCKS_PER_SM", 4);
    blocks_ = env_int("CUDA_BLOCKS", properties_.multiProcessorCount * blocks_per_sm);
    if (blocks_ <= 0) blocks_ = default_blocks;
  }

  ~Miner() {
    if (begin_) cudaEventDestroy(begin_);
    if (end_) cudaEventDestroy(end_);
    if (device_result_) cudaFree(device_result_);
  }

  Result run(const Work& work, float* milliseconds) {
    check(cudaMemset(device_result_, 0, sizeof(Result)), "cudaMemset(result)");
    check(cudaEventRecord(begin_), "cudaEventRecord(begin)");
    mine_kernel<<<blocks_, threads_>>>(work, device_result_);
    check(cudaGetLastError(), "mine_kernel launch");
    check(cudaEventRecord(end_), "cudaEventRecord(end)");
    check(cudaEventSynchronize(end_), "cudaEventSynchronize");
    check(cudaEventElapsedTime(milliseconds, begin_, end_), "cudaEventElapsedTime");
    Result result{};
    check(cudaMemcpy(&result, device_result_, sizeof(result), cudaMemcpyDeviceToHost),
          "cudaMemcpy(result)");
    return result;
  }

 private:
  cudaDeviceProp properties_{};
  Result* device_result_ = nullptr;
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t end_ = nullptr;
  int threads_ = 256;
  int blocks_ = 1;
};

int main() {
  try {
    Miner miner;
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      // One line is exactly six whitespace-separated fields:
      // address prevWork anchor target start count
      std::istringstream input(line);
      std::string address, prev, anchor, target, start_text, count_text, extra;
      input >> address >> prev >> anchor >> target >> start_text >> count_text;
      if (!input || (input >> extra) || address.empty() || prev.empty() ||
          anchor.empty() || target.empty()) {
        std::cout << "error expected: address prev anchor target start count"
                  << std::endl;
        continue;
      }

      try {
        const U256 start = parse_u256(start_text);
        const U256 count = parse_u256(count_text);
        const Work work = make_work(address, prev, anchor, target, start, count);
        float milliseconds = 0.0f;
        const Result result = miner.run(work, &milliseconds);
        std::cout << "result " << result.found << ' ';
        if (result.found) {
          std::cout << u256_decimal(result.nonce) << ' ' << digest_hex(result.digest);
        } else {
          std::cout << "- -";
        }
        std::cout << ' ' << std::fixed << std::setprecision(3) << milliseconds
                  << std::endl;
      } catch (const std::exception& error) {
        std::cout << "error " << error.what() << std::endl;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "hashcats-cuda: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
