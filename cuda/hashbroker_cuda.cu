// HashBroker CUDA SHA-256 worker.
// Message layout intentionally matches the public WebGPU reference implementation:
//   address[20] || zero[24] || nonce[8] || challenge[32] = 84 bytes.
// The nonce is encoded as a big-endian uint64 (hi word followed by lo word).
// This worker does not contain wallet signing, RPC, or transaction code.

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kShaWords = 8;
constexpr int kShaRounds = 64;
constexpr uint32_t kShaInitial[kShaWords] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
constexpr uint32_t kShaK[kShaRounds] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

// Keep job words in constant memory. All lanes in a launch read the same 13
// words; no per-hash global-memory message construction is needed.
struct DeviceJob {
  uint32_t address[5];
  uint32_t challenge[8];
  uint32_t difficulty;
};
__constant__ DeviceJob c_job;
__constant__ uint32_t c_k[kShaRounds];
__constant__ uint32_t c_initial[kShaWords];

struct BlockResult {
  uint32_t found;
  uint32_t nonce_lo;
  uint32_t nonce_hi;
  uint32_t hash[kShaWords];
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}
__device__ __forceinline__ uint32_t ch32(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}
__device__ __forceinline__ uint32_t maj32(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}
__device__ __forceinline__ uint32_t big0(uint32_t x) {
  return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}
__device__ __forceinline__ uint32_t big1(uint32_t x) {
  return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}
__device__ __forceinline__ uint32_t small0(uint32_t x) {
  return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}
__device__ __forceinline__ uint32_t small1(uint32_t x) {
  return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

// One SHA-256 compression. The 16-word ring keeps register pressure below a
// 64-word schedule while retaining a fixed, fully deterministic two-block path.
__device__ __forceinline__ void compress_block(uint32_t w[16], uint32_t s[8]) {
  uint32_t a = s[0], b = s[1], c = s[2], d = s[3];
  uint32_t e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll 64
  for (int t = 0; t < 64; ++t) {
    uint32_t wt;
    if (t < 16) {
      wt = w[t];
    } else {
      const int idx = t & 15;
      wt = small1(w[(t - 2) & 15]) + w[(t - 7) & 15] +
           small0(w[(t - 15) & 15]) + w[idx];
      w[idx] = wt;
    }
    const uint32_t t1 = h + big1(e) + ch32(e, f, g) + c_k[t] + wt;
    const uint32_t t2 = big0(a) + maj32(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  s[0] += a;
  s[1] += b;
  s[2] += c;
  s[3] += d;
  s[4] += e;
  s[5] += f;
  s[6] += g;
  s[7] += h;
}

__device__ __forceinline__ uint32_t leading_zero_bits(const uint32_t hash[8]) {
  uint32_t count = 0;
#pragma unroll 8
  for (int i = 0; i < 8; ++i) {
    if (hash[i] == 0u) {
      count += 32u;
    } else {
      count += static_cast<uint32_t>(__clz(hash[i]));
      return count;
    }
  }
  return count;
}

__device__ __forceinline__ void hash_fixed_message(uint64_t nonce,
                                                    uint32_t out[8]) {
  uint32_t state[8];
#pragma unroll 8
  for (int i = 0; i < 8; ++i) state[i] = c_initial[i];

  const uint32_t nonce_hi = static_cast<uint32_t>(nonce >> 32);
  const uint32_t nonce_lo = static_cast<uint32_t>(nonce);
  uint32_t w[16];
  w[0] = c_job.address[0];
  w[1] = c_job.address[1];
  w[2] = c_job.address[2];
  w[3] = c_job.address[3];
  w[4] = c_job.address[4];
  w[5] = 0u;
  w[6] = 0u;
  w[7] = 0u;
  w[8] = 0u;
  w[9] = 0u;
  w[10] = 0u;
  w[11] = nonce_hi;
  w[12] = nonce_lo;
  w[13] = c_job.challenge[0];
  w[14] = c_job.challenge[1];
  w[15] = c_job.challenge[2];
  compress_block(w, state);

  w[0] = c_job.challenge[3];
  w[1] = c_job.challenge[4];
  w[2] = c_job.challenge[5];
  w[3] = c_job.challenge[6];
  w[4] = c_job.challenge[7];
  w[5] = 0x80000000u;
  w[6] = 0u;
  w[7] = 0u;
  w[8] = 0u;
  w[9] = 0u;
  w[10] = 0u;
  w[11] = 0u;
  w[12] = 0u;
  w[13] = 0u;
  w[14] = 0u;
  w[15] = 672u;
  compress_block(w, state);

#pragma unroll 8
  for (int i = 0; i < 8; ++i) out[i] = state[i];
}

// Each thread owns a contiguous sequence of nonce indexes. A worker's step
// interleaves those sequences across GPUs, making identical job starts safe:
// gpu 0 tests start+0, start+step, ...; gpu 1 tests start+1, start+1+step, ...
// There are no atomics in the normal per-hash path. Atomics occur only when a
// qualifying solution is actually found (one CAS per block at most).
__global__ void mine_kernel(uint64_t worker_start, uint64_t worker_step,
                            uint32_t hashes_per_thread, BlockResult* results,
                            uint32_t* stop) {
  const uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                             threadIdx.x;
  const uint64_t first_index = thread_id * hashes_per_thread;
  for (uint32_t i = 0; i < hashes_per_thread; ++i) {
    if ((i & 31u) == 0u && stop[0] != 0u) return;
    const uint64_t sequence = first_index + i;
    const uint64_t nonce = worker_start + sequence * worker_step;
    uint32_t digest[8];
    hash_fixed_message(nonce, digest);
    if (leading_zero_bits(digest) >= c_job.difficulty) {
      if (atomicCAS(&results[blockIdx.x].found, 0u, 1u) == 0u) {
        results[blockIdx.x].nonce_lo = static_cast<uint32_t>(nonce);
        results[blockIdx.x].nonce_hi = static_cast<uint32_t>(nonce >> 32);
#pragma unroll 8
        for (int w = 0; w < 8; ++w) results[blockIdx.x].hash[w] = digest[w];
        atomicExch(stop, 1u);
      }
      return;
    }
  }
}

struct CudaFailure : std::runtime_error {
  explicit CudaFailure(const std::string& message) : std::runtime_error(message) {}
};

void cuda_check(cudaError_t error, const char* expression) {
  if (error != cudaSuccess) {
    std::ostringstream out;
    out << expression << " failed: " << cudaGetErrorString(error);
    throw CudaFailure(out.str());
  }
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr)

struct Job {
  std::string id;
  std::array<uint32_t, 5> address{};
  std::array<uint32_t, 8> challenge{};
  uint32_t difficulty = 0;
  uint64_t start_nonce = 0;
  uint64_t range_step = 0;  // zero means use --workers.
};

struct Config {
  int gpu_id = 0;
  int workers = 1;
  int blocks = 0;       // zero means 8 blocks per SM.
  int threads = 256;
  uint32_t iters = 256; // hashes per CUDA thread per launch.
};

std::mutex g_output_mutex;

void output_line(const std::string& line) {
  std::lock_guard<std::mutex> lock(g_output_mutex);
  std::cout << line << '\n' << std::flush;
}

std::string hex_u64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0') << std::setw(16) << value;
  return out.str();
}

std::string hex_u32(uint32_t value) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(8) << value;
  return out.str();
}

std::string hex_words(const uint32_t* words, size_t count) {
  std::ostringstream out;
  for (size_t i = 0; i < count; ++i) out << hex_u32(words[i]);
  return out.str();
}

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

template <size_t N>
std::array<uint32_t, N> parse_hex_words(std::string text, const char* name) {
  if (text.rfind("0x", 0) == 0 || text.rfind("0X", 0) == 0) text.erase(0, 2);
  if (text.size() != N * 8) {
    std::ostringstream out;
    out << name << " must contain exactly " << (N * 8) << " hex digits";
    throw std::invalid_argument(out.str());
  }
  std::array<uint32_t, N> words{};
  for (size_t i = 0; i < N; ++i) {
    uint32_t word = 0;
    for (size_t j = 0; j < 8; ++j) {
      const int nibble = hex_value(text[i * 8 + j]);
      if (nibble < 0) throw std::invalid_argument(std::string("invalid hex in ") + name);
      word = (word << 4) | static_cast<uint32_t>(nibble);
    }
    words[i] = word;
  }
  return words;
}

uint64_t parse_u64(const std::string& text, const char* name) {
  size_t consumed = 0;
  int base = 10;
  if (text.rfind("0x", 0) == 0 || text.rfind("0X", 0) == 0) base = 16;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &consumed, base);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
  }
  if (consumed != text.size()) throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
  return static_cast<uint64_t>(value);
}

uint32_t parse_u32(const std::string& text, const char* name) {
  uint64_t value = parse_u64(text, name);
  if (value > std::numeric_limits<uint32_t>::max()) throw std::invalid_argument(std::string(name) + " is too large");
  return static_cast<uint32_t>(value);
}

// Parse the documented form without relying on fragile positional references.
Job parse_job(const std::string& line, int default_workers) {
  std::istringstream in(line);
  std::vector<std::string> fields;
  std::string field;
  while (in >> field) fields.push_back(field);
  if (fields.size() < 6 || fields.size() > 7 || fields[0] != "job") {
    throw std::invalid_argument("job <id> <address40hex> <challenge64hex> <difficulty> <start_nonce> [range_step]");
  }
  Job job;
  job.id = fields[1];
  if (job.id.empty()) throw std::invalid_argument("job id is required");
  job.address = parse_hex_words<5>(fields[2], "address");
  job.challenge = parse_hex_words<8>(fields[3], "challenge");
  job.difficulty = parse_u32(fields[4], "difficulty");
  if (job.difficulty > 256) throw std::invalid_argument("difficulty must be between 0 and 256");
  job.start_nonce = parse_u64(fields[5], "start_nonce");
  job.range_step = fields.size() == 7 ? parse_u64(fields[6], "range_step") : static_cast<uint64_t>(default_workers);
  if (job.range_step == 0) throw std::invalid_argument("range_step must be nonzero");
  return job;
}

// Independent host SHA-256 implementation used to verify every GPU solution.
uint32_t cpu_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
uint32_t cpu_ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
uint32_t cpu_maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
uint32_t cpu_big0(uint32_t x) { return cpu_rotr(x, 2) ^ cpu_rotr(x, 13) ^ cpu_rotr(x, 22); }
uint32_t cpu_big1(uint32_t x) { return cpu_rotr(x, 6) ^ cpu_rotr(x, 11) ^ cpu_rotr(x, 25); }
uint32_t cpu_small0(uint32_t x) { return cpu_rotr(x, 7) ^ cpu_rotr(x, 18) ^ (x >> 3); }
uint32_t cpu_small1(uint32_t x) { return cpu_rotr(x, 17) ^ cpu_rotr(x, 19) ^ (x >> 10); }

std::array<uint32_t, 8> cpu_sha256(const std::vector<uint8_t>& input) {
  std::vector<uint8_t> padded = input;
  padded.push_back(0x80u);
  while ((padded.size() % 64) != 56) padded.push_back(0u);
  const uint64_t bit_length = static_cast<uint64_t>(input.size()) * 8u;
  for (int shift = 56; shift >= 0; shift -= 8) padded.push_back(static_cast<uint8_t>(bit_length >> shift));

  std::array<uint32_t, 8> state{};
  for (int i = 0; i < 8; ++i) state[i] = kShaInitial[i];
  for (size_t offset = 0; offset < padded.size(); offset += 64) {
    uint32_t w[64]{};
    for (int i = 0; i < 16; ++i) {
      const size_t p = offset + static_cast<size_t>(i) * 4;
      w[i] = (static_cast<uint32_t>(padded[p]) << 24) |
             (static_cast<uint32_t>(padded[p + 1]) << 16) |
             (static_cast<uint32_t>(padded[p + 2]) << 8) |
             static_cast<uint32_t>(padded[p + 3]);
    }
    for (int i = 16; i < 64; ++i) w[i] = cpu_small1(w[i - 2]) + w[i - 7] + cpu_small0(w[i - 15]) + w[i - 16];
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = h + cpu_big1(e) + cpu_ch(e, f, g) + kShaK[i] + w[i];
      const uint32_t t2 = cpu_big0(a) + cpu_maj(a, b, c);
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
  }
  return state;
}

std::vector<uint8_t> fixed_message(const Job& job, uint64_t nonce) {
  std::vector<uint8_t> message;
  message.reserve(84);
  auto append_word = [&message](uint32_t word) {
    message.push_back(static_cast<uint8_t>(word >> 24));
    message.push_back(static_cast<uint8_t>(word >> 16));
    message.push_back(static_cast<uint8_t>(word >> 8));
    message.push_back(static_cast<uint8_t>(word));
  };
  for (uint32_t word : job.address) append_word(word);
  for (int i = 0; i < 6; ++i) append_word(0u);
  append_word(static_cast<uint32_t>(nonce >> 32));
  append_word(static_cast<uint32_t>(nonce));
  for (uint32_t word : job.challenge) append_word(word);
  return message;
}

std::array<uint32_t, 8> cpu_fixed_hash(const Job& job, uint64_t nonce) {
  return cpu_sha256(fixed_message(job, nonce));
}

uint32_t cpu_leading_zero_bits(const std::array<uint32_t, 8>& hash) {
  uint32_t count = 0;
  for (uint32_t word : hash) {
    if (word == 0u) count += 32u;
    else {
      count += static_cast<uint32_t>(__builtin_clz(word));
      break;
    }
  }
  return count;
}

std::string hash_hex(const std::array<uint32_t, 8>& hash) {
  return hex_words(hash.data(), hash.size());
}

void run_self_test() {
  const auto empty = cpu_sha256({});
  const std::vector<uint8_t> abc = {'a', 'b', 'c'};
  const auto abc_hash = cpu_sha256(abc);
  const std::string expected_empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
  const std::string expected_abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  if (hash_hex(empty) != expected_empty || hash_hex(abc_hash) != expected_abc) {
    throw std::runtime_error("CPU SHA-256 self-test failed against standard vectors");
  }
  Job job;
  job.address.fill(0u);
  job.challenge.fill(0u);
  const auto fixed = cpu_fixed_hash(job, 0);
  if (fixed_message(job, 0).size() != 84u) {
    throw std::runtime_error("fixed HashBroker message must be exactly 84 bytes");
  }
  std::cout << "cpu_sha256_empty=" << hash_hex(empty) << '\n';
  std::cout << "cpu_sha256_abc=" << hash_hex(abc_hash) << '\n';
  std::cout << "fixed_message_len=" << fixed_message(job, 0).size() << '\n';
  std::cout << "fixed_message_hash=" << hash_hex(fixed) << '\n';
  std::cout << "fixed_message_bits=" << cpu_leading_zero_bits(fixed) << '\n';
}

struct BatchResult {
  bool found = false;
  uint64_t nonce = 0;
  std::array<uint32_t, 8> hash{};
};

class Worker {
 public:
  explicit Worker(Config config) : config_(config) {}
  ~Worker() {
    cudaSetDevice(config_.gpu_id);
    if (stream_) cudaStreamDestroy(stream_);
    if (d_results_) cudaFree(d_results_);
    if (d_stop_) cudaFree(d_stop_);
  }

  void initialize() {
    if (config_.workers <= 0 || config_.gpu_id < 0 || config_.gpu_id >= config_.workers) {
      throw std::invalid_argument("require 0 <= gpu_id < workers");
    }
    CUDA_CHECK(cudaSetDevice(config_.gpu_id));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, config_.gpu_id));
    if (config_.blocks == 0) config_.blocks = prop.multiProcessorCount * 8;
    if (config_.blocks <= 0 || config_.blocks > 2147483647) throw std::invalid_argument("invalid block count");
    if (config_.threads < 32 || config_.threads > prop.maxThreadsPerBlock || (config_.threads % 32) != 0) {
      throw std::invalid_argument("threads must be a multiple of 32 within the device block limit");
    }
    if (config_.iters == 0) throw std::invalid_argument("iters must be nonzero");
    CUDA_CHECK(cudaMemcpyToSymbol(c_k, kShaK, sizeof(kShaK)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_initial, kShaInitial, sizeof(kShaInitial)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_stop_), sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_results_), sizeof(BlockResult) * static_cast<size_t>(config_.blocks)));
    device_name_ = prop.name;
    std::ostringstream line;
    line << "ready gpu=" << config_.gpu_id << " workers=" << config_.workers
         << " device=\"" << device_name_ << "\" blocks=" << config_.blocks
         << " threads=" << config_.threads << " iters=" << config_.iters;
    output_line(line.str());
  }

  uint64_t hashes_per_batch() const {
    const uint64_t lanes = static_cast<uint64_t>(config_.blocks) * static_cast<uint64_t>(config_.threads);
    return lanes * static_cast<uint64_t>(config_.iters);
  }

  uint64_t nonce_span(uint64_t range_step) const {
    return hashes_per_batch() * range_step;
  }

  BatchResult run_batch(const Job& job, uint64_t worker_start, uint64_t range_step) {
    DeviceJob device_job{};
    for (int i = 0; i < 5; ++i) device_job.address[i] = job.address[i];
    for (int i = 0; i < 8; ++i) device_job.challenge[i] = job.challenge[i];
    device_job.difficulty = job.difficulty;
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_job, &device_job, sizeof(device_job), 0,
                                       cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemsetAsync(d_stop_, 0, sizeof(uint32_t), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_results_, 0, sizeof(BlockResult) * static_cast<size_t>(config_.blocks), stream_));
    mine_kernel<<<config_.blocks, config_.threads, 0, stream_>>>(
        worker_start, range_step, config_.iters, d_results_, d_stop_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<BlockResult> host_results(static_cast<size_t>(config_.blocks));
    CUDA_CHECK(cudaMemcpy(host_results.data(), d_results_,
                          sizeof(BlockResult) * host_results.size(), cudaMemcpyDeviceToHost));
    for (const BlockResult& result : host_results) {
      if (result.found != 0u) {
        BatchResult out;
        out.found = true;
        out.nonce = (static_cast<uint64_t>(result.nonce_hi) << 32) | result.nonce_lo;
        for (int i = 0; i < 8; ++i) out.hash[i] = result.hash[i];
        return out;
      }
    }
    return {};
  }

  void benchmark(uint64_t requested_hashes) {
    Job job;
    job.id = "benchmark";
    job.address.fill(0u);
    job.challenge.fill(0u);
    job.difficulty = 256u;  // practically impossible; benchmark does not accept a result.
    job.start_nonce = 0;
    const uint64_t step = static_cast<uint64_t>(config_.workers);
    const uint64_t hashes = hashes_per_batch();
    const uint64_t nonce_step = nonce_span(step);
    uint64_t batches = (requested_hashes + hashes - 1) / hashes;
    if (batches == 0) batches = 1;
    const auto begin = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < batches; ++i) {
      (void)run_batch(job, job.start_nonce + static_cast<uint64_t>(config_.gpu_id) + i * nonce_step, step);
    }
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    const uint64_t launched = batches * hashes;
    std::ostringstream out;
    out << "benchmark gpu=" << config_.gpu_id << " device=\"" << device_name_
        << "\" hashes=" << launched << " seconds=" << std::fixed << std::setprecision(6)
        << seconds << " hashrate=" << std::setprecision(3)
        << (seconds > 0.0 ? static_cast<double>(launched) / seconds : 0.0);
    output_line(out.str());
  }

  void mine(const Job& job, uint64_t generation, const std::atomic<uint64_t>& current_generation) {
    // CUDA's per-thread device selection is independent of the host thread
    // that created the stream and allocations. Explicitly select this worker's
    // device before issuing any work, especially for GPU 1..N-1 processes.
    CUDA_CHECK(cudaSetDevice(config_.gpu_id));
    if (job.range_step != 0 && static_cast<uint64_t>(config_.gpu_id) >= job.range_step) {
      output_line("error job=" + job.id + " gpu id is outside range_step");
      return;
    }
    const uint64_t range_step = job.range_step == 0 ? static_cast<uint64_t>(config_.workers) : job.range_step;
    uint64_t next_nonce = job.start_nonce + static_cast<uint64_t>(config_.gpu_id);
    const uint64_t hashes = hashes_per_batch();
    const uint64_t nonce_step = nonce_span(range_step);
    uint64_t launched = 0;
    const auto begin = std::chrono::steady_clock::now();
    auto last_stats = begin;
    {
      std::ostringstream out;
      out << "job_started job=" << job.id << " gpu=" << config_.gpu_id
          << " start_nonce=" << hex_u64(next_nonce) << " range_step=" << range_step;
      output_line(out.str());
    }
    while (current_generation.load(std::memory_order_acquire) == generation) {
      const BatchResult result = run_batch(job, next_nonce, range_step);
      launched += hashes;
      if (result.found) {
        const auto expected = cpu_fixed_hash(job, result.nonce);
        const uint32_t bits = cpu_leading_zero_bits(expected);
        bool same_hash = true;
        for (int i = 0; i < 8; ++i) same_hash = same_hash && (expected[i] == result.hash[i]);
        if (!same_hash || bits < job.difficulty) {
          output_line("error job=" + job.id + " gpu solution failed independent CPU verification");
        } else if (current_generation.load(std::memory_order_acquire) == generation) {
          std::ostringstream out;
          out << "solution job=" << job.id << " gpu=" << config_.gpu_id
              << " nonce=" << hex_u64(result.nonce) << " hash=" << hash_hex(expected)
              << " bits=" << bits;
          output_line(out.str());
          output_stats(job, launched, begin, true);
        }
        return;
      }
      next_nonce += nonce_step;
      const auto now = std::chrono::steady_clock::now();
      if (now - last_stats >= std::chrono::seconds(1)) {
        output_stats(job, launched, begin, false);
        last_stats = now;
      }
    }
    output_line("job_stopped job=" + job.id + " gpu=" + std::to_string(config_.gpu_id));
  }

 private:
  void output_stats(const Job& job, uint64_t launched,
                    const std::chrono::steady_clock::time_point& begin, bool final) {
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::ostringstream out;
    out << "stats job=" << job.id << " gpu=" << config_.gpu_id
        << " hashes_launched=" << launched << " seconds=" << std::fixed << std::setprecision(3)
        << seconds << " hashrate=" << std::setprecision(3)
        << (seconds > 0.0 ? static_cast<double>(launched) / seconds : 0.0);
    if (final) out << " final=1";
    output_line(out.str());
  }

  Config config_;
  std::string device_name_;
  cudaStream_t stream_ = nullptr;
  uint32_t* d_stop_ = nullptr;
  BlockResult* d_results_ = nullptr;
};

void print_help(const char* program) {
  std::cout << "HashBroker CUDA SHA-256 worker\n"
            << "Usage:\n"
            << "  " << program << " --self-test\n"
            << "  " << program << " --gpu ID --workers N [options] < stdin\n"
            << "  " << program << " --benchmark HASHES --gpu ID --workers N [options]\n\n"
            << "Options: --gpu ID (default 0), --workers N (default 1),\n"
            << "         --blocks N (default 8*SM), --threads N (default 256),\n"
            << "         --iters N (default 256), --benchmark HASHES, --self-test\n\n"
            << "Line protocol:\n"
            << "  job ID ADDRESS40HEX CHALLENGE64HEX DIFFICULTY START_NONCE [RANGE_STEP]\n"
            << "  stop\n"
            << "  quit\n"
            << "START_NONCE is decimal or 0x-prefixed uint64. With four processes\n"
            << "started as --gpu 0..3 --workers 4 and the same START_NONCE, lanes are\n"
            << "interleaved by range_step=4, so GPU ranges are unique.\n";
}

int run_worker(const Config& config) {
  Worker worker(config);
  worker.initialize();
  std::atomic<uint64_t> generation{0};
  std::mutex job_mutex;
  std::condition_variable job_cv;
  std::optional<Job> pending;
  bool quitting = false;

  std::thread miner([&]() {
    for (;;) {
      Job job;
      uint64_t mine_generation = 0;
      {
        std::unique_lock<std::mutex> lock(job_mutex);
        job_cv.wait(lock, [&] { return quitting || pending.has_value(); });
        if (quitting && !pending.has_value()) return;
        job = *pending;
        pending.reset();
        mine_generation = generation.load(std::memory_order_acquire);
      }
      worker.mine(job, mine_generation, generation);
    }
  });

  output_line("protocol_ready commands=job,stop,quit");
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    try {
      std::istringstream first(line);
      std::string command;
      first >> command;
      if (command == "job") {
        Job job = parse_job(line, config.workers);
        {
          std::lock_guard<std::mutex> lock(job_mutex);
          pending = std::move(job);
          generation.fetch_add(1, std::memory_order_acq_rel);
        }
        job_cv.notify_one();
      } else if (command == "stop") {
        generation.fetch_add(1, std::memory_order_acq_rel);
        std::lock_guard<std::mutex> lock(job_mutex);
        pending.reset();
        output_line("stopping gpu=" + std::to_string(config.gpu_id));
      } else if (command == "quit") {
        {
          std::lock_guard<std::mutex> lock(job_mutex);
          quitting = true;
          pending.reset();
          generation.fetch_add(1, std::memory_order_acq_rel);
        }
        job_cv.notify_one();
        break;
      } else if (command == "help") {
        print_help("hashbroker_cuda");
      } else {
        output_line("error unknown command; use job, stop, quit, or help");
      }
    } catch (const std::exception& error) {
      output_line(std::string("error ") + error.what());
    }
  }
  {
    std::lock_guard<std::mutex> lock(job_mutex);
    quitting = true;
    pending.reset();
    generation.fetch_add(1, std::memory_order_acq_rel);
  }
  job_cv.notify_one();
  miner.join();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Config config;
    bool self_test = false;
    bool benchmark = false;
    uint64_t benchmark_hashes = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto require_value = [&](const char* option) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string(option) + " requires a value");
        return argv[++i];
      };
      if (arg == "--help" || arg == "-h") {
        print_help(argv[0]);
        return 0;
      } else if (arg == "--self-test") {
        self_test = true;
      } else if (arg == "--gpu") {
        config.gpu_id = static_cast<int>(parse_u32(require_value("--gpu"), "gpu"));
      } else if (arg == "--workers") {
        config.workers = static_cast<int>(parse_u32(require_value("--workers"), "workers"));
      } else if (arg == "--blocks") {
        config.blocks = static_cast<int>(parse_u32(require_value("--blocks"), "blocks"));
      } else if (arg == "--threads") {
        config.threads = static_cast<int>(parse_u32(require_value("--threads"), "threads"));
      } else if (arg == "--iters") {
        config.iters = parse_u32(require_value("--iters"), "iters");
      } else if (arg == "--benchmark") {
        benchmark = true;
        benchmark_hashes = parse_u64(require_value("--benchmark"), "benchmark_hashes");
      } else if (arg.rfind("--benchmark=", 0) == 0) {
        benchmark = true;
        benchmark_hashes = parse_u64(arg.substr(12), "benchmark_hashes");
      } else {
        throw std::invalid_argument("unknown option: " + arg);
      }
    }
    if (self_test) {
      run_self_test();
      return 0;
    }
    if (benchmark) {
      Worker worker(config);
      worker.initialize();
      worker.benchmark(benchmark_hashes);
      return 0;
    }
    return run_worker(config);
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}

#undef CUDA_CHECK
