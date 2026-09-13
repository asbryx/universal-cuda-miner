// Universal CUDA proof-of-work worker.
//
// One persistent process supports the protocol layouts documented by the
// Python adapter. A job supplies a zeroed message template plus nonce offset
// and width at runtime; the kernel is compiled once and never regenerated per
// protocol or job. The worker contains no RPC, wallet, signing, or network
// code.
//
// Line protocol:
//   job ID PROTOCOL ALGORITHM NONCE_OFFSET NONCE_WIDTH TEMPLATE_HEX TARGET_HEX START_HEX COUNT STEP
//   stop
//   quit
//
// ALGORITHM is sha256 or keccak256. TARGET is strict digest < target.

#include <cuda_runtime.h>

#include <atomic>
#include <future>
#include <mutex>
#include <algorithm>
#include <array>
#include <chrono>
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

namespace {

constexpr int kMaxMessage = 116;
constexpr int kDigestBytes = 32;
constexpr int kKeccakRate = 136;
constexpr int kShaBlock = 64;
__device__ __constant__ uint32_t kShaInitial[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
__device__ __constant__ uint32_t kShaK[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

__device__ __constant__ uint64_t kKeccakRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
__device__ __constant__ int kKeccakRot[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
                                27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
__device__ __constant__ int kKeccakPiln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
                                 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

struct Job {
  uint8_t templ[kMaxMessage]{};
  uint32_t message_width = 0;
  uint32_t nonce_offset = 0;
  uint32_t nonce_width = 0;
  uint32_t algorithm = 0;  // 0 = SHA-256, 1 = Ethereum Keccak-256
  uint64_t start[4]{};
  uint64_t count = 0;
  uint64_t step = 1;
  uint64_t target[4]{};  // big-endian comparison words
};

struct DeviceResult {
  uint32_t found;
  uint64_t nonce[4];
  uint8_t digest[kDigestBytes];
};

__host__ __device__ inline uint32_t rotr32(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}
__host__ __device__ inline uint32_t ch32(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}
__host__ __device__ inline uint32_t maj32(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}
__host__ __device__ inline uint32_t big0(uint32_t x) {
  return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}
__host__ __device__ inline uint32_t big1(uint32_t x) {
  return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}
__host__ __device__ inline uint32_t small0(uint32_t x) {
  return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}
__host__ __device__ inline uint32_t small1(uint32_t x) {
  return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

__host__ __device__ void sha256(const uint8_t* input, uint32_t length, uint8_t out[32]) {
  uint8_t padded[128] = {};
  for (uint32_t i = 0; i < length; ++i) padded[i] = input[i];
  padded[length] = 0x80u;
  const uint64_t bits = static_cast<uint64_t>(length) * 8u;
  const uint32_t padded_length = length < 56 ? 64 : 128;
  for (int i = 0; i < 8; ++i) padded[padded_length - 1 - i] = static_cast<uint8_t>(bits >> (8 * i));
  uint32_t state[8];
  for (int i = 0; i < 8; ++i) state[i] = kShaInitial[i];
  for (uint32_t block = 0; block < padded_length; block += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      const uint32_t p = block + 4u * static_cast<uint32_t>(i);
      w[i] = (static_cast<uint32_t>(padded[p]) << 24) |
             (static_cast<uint32_t>(padded[p + 1]) << 16) |
             (static_cast<uint32_t>(padded[p + 2]) << 8) | padded[p + 3];
    }
    for (int i = 16; i < 64; ++i) w[i] = small1(w[i - 2]) + w[i - 7] + small0(w[i - 15]) + w[i - 16];
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = h + big1(e) + ch32(e, f, g) + kShaK[i] + w[i];
      const uint32_t t2 = big0(a) + maj32(a, b, c);
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
  }
  for (int i = 0; i < 8; ++i) {
    out[4 * i] = static_cast<uint8_t>(state[i] >> 24);
    out[4 * i + 1] = static_cast<uint8_t>(state[i] >> 16);
    out[4 * i + 2] = static_cast<uint8_t>(state[i] >> 8);
    out[4 * i + 3] = static_cast<uint8_t>(state[i]);
  }
}

__host__ __device__ inline uint64_t rol64(uint64_t x, int n) {
  return n == 0 ? x : (x << n) | (x >> (64 - n));
}
__host__ __device__ inline uint64_t load64le(const uint8_t* p) {
  uint64_t x = 0;
  for (int i = 7; i >= 0; --i) x = (x << 8) | p[i];
  return x;
}
__host__ __device__ inline void store64le(uint8_t* p, uint64_t x) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(x >> (8 * i));
}

__host__ __device__ void keccak_f(uint64_t a[25]) {
  for (int round = 0; round < 24; ++round) {
    uint64_t c[5], d[5];
    for (int x = 0; x < 5; ++x) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
    for (int x = 0; x < 5; ++x) d[x] = c[(x + 4) % 5] ^ rol64(c[(x + 1) % 5], 1);
    for (int x = 0; x < 5; ++x) for (int y = 0; y < 5; ++y) a[x + 5 * y] ^= d[x];
    uint64_t t = a[1];
    for (int i = 0; i < 24; ++i) {
      const uint64_t next = a[kKeccakPiln[i]];
      a[kKeccakPiln[i]] = rol64(t, kKeccakRot[i]);
      t = next;
    }
    for (int y = 0; y < 5; ++y) {
      const int row = 5 * y;
      const uint64_t r0 = a[row], r1 = a[row + 1], r2 = a[row + 2];
      const uint64_t r3 = a[row + 3], r4 = a[row + 4];
      a[row] = r0 ^ ((~r1) & r2); a[row + 1] = r1 ^ ((~r2) & r3);
      a[row + 2] = r2 ^ ((~r3) & r4); a[row + 3] = r3 ^ ((~r4) & r0);
      a[row + 4] = r4 ^ ((~r0) & r1);
    }
    a[0] ^= kKeccakRC[round];
  }
}

__host__ __device__ void keccak256(const uint8_t* input, uint32_t length, uint8_t out[32]) {
  uint64_t state[25] = {};
  uint8_t block[kKeccakRate] = {};
  for (uint32_t i = 0; i < length; ++i) block[i] = input[i];
  block[length] ^= 0x01u;
  block[kKeccakRate - 1] ^= 0x80u;
  for (int i = 0; i < 17; ++i) state[i] ^= load64le(block + 8 * i);
  keccak_f(state);
  for (int i = 0; i < 4; ++i) store64le(out + 8 * i, state[i]);
}

__device__ void add_offset(const uint64_t start[4], uint64_t index, uint64_t step, uint64_t out[4]) {
  const uint64_t low = index * step;
  const uint64_t high = __umul64hi(index, step);
  out[0] = start[0] + low;
  uint64_t carry = out[0] < start[0] ? 1ULL : 0ULL;
  const uint64_t old1 = start[1];
  const uint64_t sum1 = old1 + high;
  carry = sum1 < old1 ? 1ULL : 0ULL;
  out[1] = sum1 + (out[0] < start[0] ? 1ULL : 0ULL);
  carry |= out[1] < sum1 ? 1ULL : 0ULL;
  out[2] = start[2] + carry;
  carry = out[2] < start[2] ? 1ULL : 0ULL;
  out[3] = start[3] + carry;
}

__device__ void write_nonce(uint8_t* message, uint32_t offset, uint32_t width, const uint64_t nonce[4]) {
  for (uint32_t i = 0; i < width; ++i) message[offset + i] = 0;
  for (uint32_t i = 0; i < width && i < 32; ++i) {
    const uint32_t from_right = width - 1 - i;
    const uint32_t limb = i / 8;
    const uint32_t byte = i % 8;
    // nonce limbs are little-endian; field is big-endian.
    message[offset + from_right] = static_cast<uint8_t>(nonce[limb] >> (8 * byte));
  }
}

__device__ bool less_than_target(const uint8_t digest[32], const uint64_t target[4]) {
  for (int word = 0; word < 4; ++word) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value = (value << 8) | digest[word * 8 + i];
    if (value < target[word]) return true;
    if (value > target[word]) return false;
  }
  return false;
}

__global__ void search_kernel(Job job, DeviceResult* result) {
  const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = tid; index < job.count; index += stride) {
    if (result->found) return;
    uint64_t nonce[4];
    add_offset(job.start, index, job.step, nonce);
    uint8_t message[kMaxMessage] = {};
    for (uint32_t i = 0; i < job.message_width; ++i) message[i] = job.templ[i];
    write_nonce(message, job.nonce_offset, job.nonce_width, nonce);
    uint8_t digest[32];
    if (job.algorithm == 0) sha256(message, job.message_width, digest);
    else keccak256(message, job.message_width, digest);
    if (less_than_target(digest, job.target) && atomicCAS(&result->found, 0u, 1u) == 0u) {
      for (int i = 0; i < 4; ++i) result->nonce[i] = nonce[i];
      for (int i = 0; i < 32; ++i) result->digest[i] = digest[i];
    }
  }
}

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

std::string strip_0x(std::string text) {
  if (text.rfind("0x", 0) == 0 || text.rfind("0X", 0) == 0) text.erase(0, 2);
  return text;
}

std::vector<uint8_t> parse_bytes(const std::string& input, size_t width, const char* name) {
  const std::string text = strip_0x(input);
  if (text.size() != width * 2) throw std::invalid_argument(std::string(name) + " has wrong width");
  std::vector<uint8_t> out(width);
  for (size_t i = 0; i < width; ++i) {
    const int high = hex_value(text[2 * i]);
    const int low = hex_value(text[2 * i + 1]);
    if (high < 0 || low < 0) throw std::invalid_argument(std::string("invalid hex in ") + name);
    out[i] = static_cast<uint8_t>((high << 4) | low);
  }
  return out;
}

uint64_t parse_u64(const std::string& text, const char* name) {
  if (text.empty() || text[0] == '-' || text[0] == '+') throw std::invalid_argument("unsigned integer required");
  size_t used = 0;
  try {
    const int base = (text.rfind("0x", 0) == 0 || text.rfind("0X", 0) == 0) ? 16 : 10;
    const auto value = std::stoull(text, &used, base);
    if (used != text.size()) throw std::invalid_argument("trailing characters");
    return static_cast<uint64_t>(value);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
}

void parse_u256(const std::string& input, uint64_t out[4]) {
  const std::string text = strip_0x(input);
  if (text.empty() || text.size() > 64) throw std::invalid_argument("start must fit uint256");
  for (char c : text) if (hex_value(c) < 0) throw std::invalid_argument("start must be hexadecimal");
  for (size_t pos = 0; pos < text.size(); ++pos) {
    const int digit = hex_value(text[pos]);
    uint64_t carry = static_cast<uint64_t>(digit);
    for (int limb = 0; limb < 4; ++limb) {
      const uint64_t old = out[limb];
      out[limb] = (old << 4) | carry;
      carry = old >> 60;
    }
    if (carry) throw std::invalid_argument("start overflows uint256");
  }
}

std::string hex_digest(const uint8_t digest[32]) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0');
  for (int i = 0; i < 32; ++i) out << std::setw(2) << static_cast<unsigned>(digest[i]);
  return out.str();
}

std::string hex_u256(const uint64_t value[4]) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0');
  for (int limb = 3; limb >= 0; --limb) out << std::setw(16) << value[limb];
  return out.str();
}

void cuda_check(cudaError_t error, const char* expression) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(error));
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr)

class PersistentWorker {
 public:
  explicit PersistentWorker(int gpu, int blocks, int threads)
      : gpu_(gpu), blocks_(blocks), threads_(threads) {
    CUDA_CHECK(cudaSetDevice(gpu_));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_));
    if (blocks_ <= 0) blocks_ = prop.multiProcessorCount * 4;
    if (threads_ < 32 || threads_ > prop.maxThreadsPerBlock || threads_ % 32 != 0) {
      throw std::invalid_argument("threads must be a warp-aligned device-supported value");
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&device_result_, sizeof(DeviceResult)));
    std::cout << "ready gpu=" << gpu_ << " device=\"" << prop.name << "\" blocks=" << blocks_
              << " threads=" << threads_ << "\nprotocol_ready commands=job,stop,quit\n" << std::flush;
  }

  ~PersistentWorker() {
    cudaSetDevice(gpu_);
    if (device_result_) cudaFree(device_result_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  void run(const Job& job, const std::atomic<bool>& cancelled) {
    CUDA_CHECK(cudaSetDevice(gpu_));
    DeviceResult result{};
    // Bounded launches let the stdin thread cancel without waiting for an
    // arbitrarily large user range. No CPU/GPU shared-memory data race.
    for (uint64_t done = 0; done < job.count;) {
      if (cancelled.load()) return;
      Job chunk = job;
      chunk.count = std::min<uint64_t>(job.count - done, 262144);
      nonce_range::advance(job.start, done, job.step, chunk.start);
      CUDA_CHECK(cudaMemsetAsync(device_result_, 0, sizeof(DeviceResult), stream_));
      search_kernel<<<blocks_, threads_, 0, stream_>>>(chunk, device_result_);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      CUDA_CHECK(cudaMemcpy(&result, device_result_, sizeof(result), cudaMemcpyDeviceToHost));
      if (cancelled.load()) return;
      if (result.found) break;
      done += chunk.count;
    }
    if (!result.found) {
      std::cout << "result job=none found=0\n" << std::flush;
      return;
    }
    uint8_t message[kMaxMessage] = {};
    for (uint32_t i = 0; i < job.message_width; ++i) message[i] = job.templ[i];
    // Host copy of the runtime nonce substitution, independent from the kernel.
    for (uint32_t i = 0; i < job.nonce_width; ++i) message[job.nonce_offset + i] = 0;
    for (uint32_t i = 0; i < job.nonce_width && i < 32; ++i) {
      const uint32_t from_right = job.nonce_width - 1 - i;
      message[job.nonce_offset + from_right] = static_cast<uint8_t>(result.nonce[i / 8] >> (8 * (i % 8)));
    }
    uint8_t expected[32];
    if (job.algorithm == 0) sha256(message, job.message_width, expected);
    else keccak256(message, job.message_width, expected);
    if (std::memcmp(expected, result.digest, 32) != 0) throw std::runtime_error("GPU digest failed CPU parity check");
    bool under = false;
    for (int word = 0; word < 4; ++word) {
      uint64_t value = 0;
      for (int i = 0; i < 8; ++i) value = (value << 8) | expected[word * 8 + i];
      if (value < job.target[word]) { under = true; break; }
      if (value > job.target[word]) break;
    }
    if (!under) throw std::runtime_error("GPU candidate failed strict CPU target check");
    std::cout << "solution nonce=" << hex_u256(result.nonce) << " digest=" << hex_digest(expected)
              << " verified=true\n" << std::flush;
  }

 private:
  int gpu_;
  int blocks_;
  int threads_;
  cudaStream_t stream_ = nullptr;
  DeviceResult* device_result_ = nullptr;
};

Job parse_job(const std::string& line) {
  std::istringstream in(line);
  std::string command, id, protocol, algorithm, offset, width, templ, target, start, count, step, extra;
  in >> command >> id >> protocol >> algorithm >> offset >> width >> templ >> target >> start >> count >> step;
  if (!in || (in >> extra) || command != "job") throw std::invalid_argument("invalid job line");
  Job job;
  const std::string template_text = strip_0x(templ);
  if (template_text.size() % 2 != 0 || template_text.size() > 2 * kMaxMessage) throw std::invalid_argument("template width");
  job.message_width = static_cast<uint32_t>(template_text.size() / 2);
  const auto template_bytes = parse_bytes(templ, job.message_width, "template");
  for (uint32_t i = 0; i < job.message_width; ++i) job.templ[i] = template_bytes[i];
  const auto offset_value = parse_u64(offset, "nonce_offset");
  const auto width_value = parse_u64(width, "nonce_width");
  if (!width_value || width_value > 32 || offset_value > job.message_width || width_value > job.message_width - offset_value)
    throw std::invalid_argument("nonce field is outside template");
  job.nonce_offset = static_cast<uint32_t>(offset_value);
  job.nonce_width = static_cast<uint32_t>(width_value);
  if (algorithm == "sha256") job.algorithm = 0;
  else if (algorithm == "keccak256") job.algorithm = 1;
  else throw std::invalid_argument("algorithm must be sha256 or keccak256");
  const auto target_bytes = parse_bytes(target, 32, "target");
  for (int word = 0; word < 4; ++word) for (int i = 0; i < 8; ++i)
    job.target[word] = (job.target[word] << 8) | target_bytes[word * 8 + i];
  parse_u256(start, job.start);
  job.count = parse_u64(count, "count");
  job.step = parse_u64(step, "step");
  if (job.step == 0) throw std::invalid_argument("step must be nonzero");
  if (job.count != 0 && job.nonce_width < 32) {
    const unsigned bits = job.nonce_width * 8;
    if (bits < 64 && (job.start[1] != 0 || job.start[0] >= (1ULL << bits)))
      throw std::invalid_argument("start exceeds nonce width");
  }
  std::cout << "job_accepted protocol=" << protocol << " message_bytes=" << job.message_width
            << " nonce_offset=" << job.nonce_offset << " nonce_width=" << job.nonce_width << "\n" << std::flush;
  return job;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int gpu = 0, blocks = 0, threads = 256;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--gpu" && i + 1 < argc) gpu = static_cast<int>(parse_u64(argv[++i], "gpu"));
      else if (arg == "--blocks" && i + 1 < argc) blocks = static_cast<int>(parse_u64(argv[++i], "blocks"));
      else if (arg == "--threads" && i + 1 < argc) threads = static_cast<int>(parse_u64(argv[++i], "threads"));
      else if (arg == "--help") {
        std::cout << "universal_cuda_miner --gpu ID --blocks N --threads N\n";
        return 0;
      } else throw std::invalid_argument("unknown option: " + arg);
    }
    PersistentWorker worker(gpu, blocks, threads);
    std::atomic<bool> cancelled{false};
    std::future<void> active;
    auto finish = [&]() {
      if (active.valid()) active.get();
    };
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      try {
        if (line == "stop" || line == "quit") {
          cancelled.store(true);
          finish();
          std::cout << "stopped\n" << std::flush;
          if (line == "quit") break;
          continue;
        }
        if (line.rfind("job ", 0) != 0) throw std::invalid_argument("expected job, stop, or quit");
        // Superseding work cancels the old range before accepting a new one.
        cancelled.store(true);
        finish();
        const Job job = parse_job(line);
        cancelled.store(false);
        active = std::async(std::launch::async, [&worker, &cancelled, job]() { worker.run(job, cancelled); });
      } catch (const std::exception& error) {
        std::cout << "error " << error.what() << "\n" << std::flush;
      }
    }
    cancelled.store(true);
    finish();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}

#undef CUDA_CHECK
