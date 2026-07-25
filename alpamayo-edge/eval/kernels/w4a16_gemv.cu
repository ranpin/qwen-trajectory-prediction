// W4A16 dequant-fused GEMV microbenchmark for Jetson Orin (sm_87).
//
// Why: Nsight showed TRT-Edge-LLM's gemv_kernel is 72% of INT4 decode time and
// reaches 147.9 GB/s = 72.2% of the 204.8 GB/s theoretical peak. But "72% of
// theoretical" only means "slow" if the device can actually do better, so this
// benchmark FIRST measures a pure streaming-read ceiling on the same device,
// then compares three hand-written GEMV variants against both numbers.
//
// Layout (AWQ-style, weight-only): W[N][K/8] as packed uint32 (8 x int4,
// low nibble first), per-group fp16 scale and fp16 zero with GROUP=128 along K.
// y[n] = sum_k (dequant(W[n][k]) * x[k]),  dequant = (q - zero) * scale.
//
// Variants:
//   v1  scalar uint32 loads            (4 B per load)
//   v2  vectorized uint4 loads         (16 B per load)
//   v3  vectorized + 2 loads in flight (more memory-level parallelism)
// One warp per output row, 8 rows per block, warp-shuffle reduction.
//
// Build:  nvcc -O3 -arch=sm_87 -o w4a16_gemv w4a16_gemv.cu
// Run:    ./w4a16_gemv            (prints CSV to stdout)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); \
  exit(1);} } while (0)

static constexpr int GROUP = 128;
static constexpr int WARP = 32;
static constexpr int ROWS_PER_BLOCK = 8;      // 8 warps = 256 threads
static constexpr int THREADS = WARP * ROWS_PER_BLOCK;

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

// ---------------------------------------------------------------- ceiling ---
// Pure streaming read with 128-bit loads: what this DRAM can actually deliver.
__global__ void bw_read(const uint4 *__restrict__ p, size_t n4, float *sink) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (; i < n4; i += stride) {
    uint4 v = p[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x | acc.y | acc.z | acc.w) == 0xffffffffu) sink[0] = 1.f;  // keep alive
}

// --------------------------------------------------------------------- v1 ---
__global__ void gemv_v1(const uint32_t *__restrict__ W, const __half *__restrict__ x,
                        const __half *__restrict__ scale, const __half *__restrict__ zero,
                        __half *__restrict__ y, int N, int K) {
  const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WARP);
  if (row >= N) return;
  const int lane = threadIdx.x % WARP;
  const int packs = K / 8;                       // uint32 per row
  const uint32_t *wrow = W + (size_t)row * packs;
  const int ngroup = K / GROUP;
  float acc = 0.f;
  for (int p = lane; p < packs; p += WARP) {
    uint32_t q = wrow[p];
    const int k0 = p * 8;
    const int g = k0 / GROUP;
    const float s = __half2float(scale[(size_t)row * ngroup + g]);
    const float z = __half2float(zero[(size_t)row * ngroup + g]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float w = (float)((q >> (4 * j)) & 0xF);
      acc += (w - z) * s * __half2float(x[k0 + j]);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = __float2half(acc);
}

// --------------------------------------------------------------------- v2 ---
__global__ void gemv_v2(const uint32_t *__restrict__ W, const __half *__restrict__ x,
                        const __half *__restrict__ scale, const __half *__restrict__ zero,
                        __half *__restrict__ y, int N, int K) {
  const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WARP);
  if (row >= N) return;
  const int lane = threadIdx.x % WARP;
  const int vecs = K / 32;                       // uint4 per row (32 weights each)
  const uint4 *wrow = reinterpret_cast<const uint4 *>(W + (size_t)row * (K / 8));
  const int ngroup = K / GROUP;
  float acc = 0.f;
  for (int v = lane; v < vecs; v += WARP) {
    uint4 q4 = wrow[v];
    const int k0 = v * 32;
    const int g = k0 / GROUP;
    const float s = __half2float(scale[(size_t)row * ngroup + g]);
    const float z = __half2float(zero[(size_t)row * ngroup + g]);
    const uint32_t qs[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float w = (float)((qs[c] >> (4 * j)) & 0xF);
        acc += (w - z) * s * __half2float(x[k0 + c * 8 + j]);
      }
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = __float2half(acc);
}

// --------------------------------------------------------------------- v4 ---
// v2 + x staged in shared memory (8 warps/block reuse it instead of hitting L1
// per element) + half2 dequant/FMA to halve the conversion count. This is the
// variant that actually competes: v1-v3 are bottlenecked by re-reading x and by
// scalar float conversions, not by DRAM.
extern "C" __global__ void gemv_v4(const uint32_t *__restrict__ W, const __half *__restrict__ x,
                                   const __half *__restrict__ scale, const __half *__restrict__ zero,
                                   __half *__restrict__ y, int N, int K) {
  extern __shared__ __half xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();

  const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WARP);
  const int lane = threadIdx.x % WARP;
  if (row >= N) return;
  const int vecs = K / 32;
  const uint4 *wrow = reinterpret_cast<const uint4 *>(W + (size_t)row * (K / 8));
  const int ngroup = K / GROUP;
  float acc = 0.f;
  for (int v = lane; v < vecs; v += WARP) {
    const uint4 q4 = wrow[v];
    const int k0 = v * 32;
    const int g = k0 / GROUP;
    const __half s = scale[(size_t)row * ngroup + g];
    const __half z = zero[(size_t)row * ngroup + g];
    const half2 s2 = __half2half2(s), z2 = __half2half2(z);
    const uint32_t qs[4] = {q4.x, q4.y, q4.z, q4.w};
    half2 pacc = __floats2half2_rn(0.f, 0.f);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t q = qs[c];
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        // two int4 weights -> half2, dequant (q - z) * s, then FMA with x
        half2 w2 = __floats2half2_rn((float)((q >> (4 * j)) & 0xF),
                                     (float)((q >> (4 * (j + 1))) & 0xF));
        w2 = __hmul2(__hsub2(w2, z2), s2);
        const half2 xv = *reinterpret_cast<const half2 *>(&xs[k0 + c * 8 + j]);
        pacc = __hfma2(w2, xv, pacc);
      }
    }
    const float2 pf = __half22float2(pacc);
    acc += pf.x + pf.y;
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = __float2half(acc);
}

// --------------------------------------------------------------------- v3 ---
// Same as v2 but issues two independent 128-bit loads before consuming either,
// so each thread keeps two DRAM requests in flight (memory-level parallelism).
__global__ void gemv_v3(const uint32_t *__restrict__ W, const __half *__restrict__ x,
                        const __half *__restrict__ scale, const __half *__restrict__ zero,
                        __half *__restrict__ y, int N, int K) {
  const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WARP);
  if (row >= N) return;
  const int lane = threadIdx.x % WARP;
  const int vecs = K / 32;
  const uint4 *wrow = reinterpret_cast<const uint4 *>(W + (size_t)row * (K / 8));
  const int ngroup = K / GROUP;
  float acc = 0.f;
  const int step = WARP * 2;
  for (int base = lane; base < vecs; base += step) {
    uint4 a = wrow[base];
    const bool has_b = (base + WARP) < vecs;
    uint4 b = has_b ? wrow[base + WARP] : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int half_i = 0; half_i < 2; ++half_i) {
      if (half_i == 1 && !has_b) break;
      const uint4 q4 = (half_i == 0) ? a : b;
      const int k0 = (base + half_i * WARP) * 32;
      const int g = k0 / GROUP;
      const float s = __half2float(scale[(size_t)row * ngroup + g]);
      const float z = __half2float(zero[(size_t)row * ngroup + g]);
      const uint32_t qs[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
      for (int c = 0; c < 4; ++c) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float w = (float)((qs[c] >> (4 * j)) & 0xF);
          acc += (w - z) * s * __half2float(x[k0 + c * 8 + j]);
        }
      }
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = __float2half(acc);
}

// ---------------------------------------------------------------- harness ---
struct Shape { int K, N; const char *name; };

// Jetson scales GPU *and* EMC (DRAM) clocks with sustained load, and
// `jetson_clocks` needs root (unavailable). So warm up for ~1.5 s of continuous
// work before timing, and report the BEST iteration: that is the closest we can
// get to pinned-clock behaviour without sudo.
static double time_kernel(void (*launch)(), int iters) {
  cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
  for (int i = 0; i < 400; ++i) launch();             // sustained clock ramp
  CHECK(cudaDeviceSynchronize());
  std::vector<float> ts;
  for (int i = 0; i < iters; ++i) {
    CHECK(cudaEventRecord(a));
    launch();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
    ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
  return ts[0];                                      // best ms (clocks ramped)
}

static uint32_t *g_W; static __half *g_x, *g_s, *g_z, *g_y; static int g_N, g_K;
static uint4 *g_buf; static size_t g_n4; static float *g_sink;
static void L_v1() { gemv_v1<<<(g_N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, THREADS>>>(g_W, g_x, g_s, g_z, g_y, g_N, g_K); }
static void L_v2() { gemv_v2<<<(g_N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, THREADS>>>(g_W, g_x, g_s, g_z, g_y, g_N, g_K); }
static void L_v3() { gemv_v3<<<(g_N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, THREADS>>>(g_W, g_x, g_s, g_z, g_y, g_N, g_K); }
static void L_v4() { gemv_v4<<<(g_N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, THREADS,
                              g_K * sizeof(__half)>>>(g_W, g_x, g_s, g_z, g_y, g_N, g_K); }
static void L_bw() { bw_read<<<2048, 256>>>(g_buf, g_n4, g_sink); }

int main() {
  // measured Cosmos-Reason2-8B decode projections (K -> N), 36 layers each
  const Shape shapes[] = {
      {4096, 4096,  "q_proj/o_proj (4096x4096)"},
      {4096, 1024,  "k_proj/v_proj (4096x1024)"},
      {4096, 12288, "gate/up      (4096x12288)"},
      {12288, 4096, "down         (12288x4096)"},
  };
  const int ITERS = 100;

  // ---- achievable read-bandwidth ceiling on this device ----
  g_n4 = (size_t)1024 * 1024 * 1024 / sizeof(uint4);  // 1 GB
  CHECK(cudaMalloc(&g_buf, g_n4 * sizeof(uint4)));
  CHECK(cudaMemset(g_buf, 0x5a, g_n4 * sizeof(uint4)));
  CHECK(cudaMalloc(&g_sink, sizeof(float)));
  double ms = time_kernel(L_bw, 60);
  double ceil_gbs = (g_n4 * sizeof(uint4)) / (ms / 1e3) / 1e9;
  printf("kind,shape,variant,ms,GB_per_s,pct_of_peak,pct_of_ceiling,max_abs_err\n");
  printf("ceiling,1GB streaming read (uint4),bw_read,%.4f,%.1f,%.1f,100.0,0\n",
         ms, ceil_gbs, 100.0 * ceil_gbs / 204.8);
  CHECK(cudaFree(g_buf)); CHECK(cudaFree(g_sink));

  for (const Shape &sh : shapes) {
    g_K = sh.K; g_N = sh.N;
    const size_t packs = (size_t)g_N * g_K / 8;
    const size_t ngroup = (size_t)g_N * (g_K / GROUP);
    CHECK(cudaMallocManaged(&g_W, packs * sizeof(uint32_t)));
    CHECK(cudaMallocManaged(&g_x, (size_t)g_K * sizeof(__half)));
    CHECK(cudaMallocManaged(&g_s, ngroup * sizeof(__half)));
    CHECK(cudaMallocManaged(&g_z, ngroup * sizeof(__half)));
    CHECK(cudaMallocManaged(&g_y, (size_t)g_N * sizeof(__half)));
    srand(1234);
    for (size_t i = 0; i < packs; ++i) g_W[i] = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
    for (int i = 0; i < g_K; ++i) g_x[i] = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.1f);
    for (size_t i = 0; i < ngroup; ++i) {
      g_s[i] = __float2half(0.01f + 0.001f * (i % 7));
      g_z[i] = __float2half(8.0f);
    }
    CHECK(cudaDeviceSynchronize());

    // bytes actually read: packed weights + per-group scale&zero + x, plus y write
    const double bytes = packs * 4.0 + ngroup * 2.0 * 2.0 + g_K * 2.0 + g_N * 2.0;

    // CPU reference on the first 8 rows to validate correctness
    std::vector<float> ref(8, 0.f);
    for (int r = 0; r < 8 && r < g_N; ++r) {
      double a = 0;
      for (int p = 0; p < g_K / 8; ++p) {
        uint32_t q = g_W[(size_t)r * (g_K / 8) + p];
        int g = (p * 8) / GROUP;
        double s = __half2float(g_s[(size_t)r * (g_K / GROUP) + g]);
        double z = __half2float(g_z[(size_t)r * (g_K / GROUP) + g]);
        for (int j = 0; j < 8; ++j)
          a += (((q >> (4 * j)) & 0xF) - z) * s * (double)__half2float(g_x[p * 8 + j]);
      }
      ref[r] = (float)a;
    }

    struct V { const char *n; void (*f)(); } vs[] = {{"v1_scalar", L_v1}, {"v2_vec128", L_v2}, {"v3_vec128_mlp2", L_v3}, {"v4_shmem_half2", L_v4}};
    for (const V &v : vs) {
      CHECK(cudaMemset(g_y, 0, (size_t)g_N * sizeof(__half)));
      double t = time_kernel(v.f, ITERS);
      CHECK(cudaDeviceSynchronize());
      double err = 0;
      for (int r = 0; r < 8 && r < g_N; ++r) {
        double got = (double)__half2float(g_y[r]);
        double want = (double)ref[r];
        double rel = std::fabs(got - want) / std::max(1e-3, std::fabs(want));
        err = std::max(err, rel);
      }
      double gbs = bytes / (t / 1e3) / 1e9;
      printf("gemv,%s,%s,%.4f,%.1f,%.1f,%.1f,%.4f\n", sh.name, v.n, t, gbs,
             100.0 * gbs / 204.8, 100.0 * gbs / ceil_gbs, err);
    }
    CHECK(cudaFree(g_W)); CHECK(cudaFree(g_x)); CHECK(cudaFree(g_s));
    CHECK(cudaFree(g_z)); CHECK(cudaFree(g_y));
  }
  return 0;
}
