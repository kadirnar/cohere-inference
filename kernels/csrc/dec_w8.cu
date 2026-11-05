// Decoder projections with int8 weights (W8A16): Y (M, N) = A (M, K) bf16 x dequant(W)^T + bias (+ ReLU), M <= 64.
//
// A decoding step feeds a few dozen rows through 50 projections, so it is bound by reading the weights. Here they are
// int8 with one scale per output channel (half the bytes of bf16), widened to bf16 in registers right before the
// tensor cores: the products and the fp32 sums are exactly those of the dequantized bf16 weights.
// Layout, prepared once (prepare_w8): uint8 = int8 + 128, and inside every 16 values of K the bytes reordered to
// (2t, 2t+1, 2t+8, 2t+9) at 4t..4t+3, so one ldmatrix row of 16 bytes gives each lane exactly the four weights of an
// mma B fragment (b0 = k 2t..2t+1, b1 = k 2t+8..2t+9).
//
// A CTA computes BN columns (16, 32 or 64) for all M rows over a K slice: its 4 warps are BN/16 column groups of 16
// by 4/(BN/16) K groups (each warp a share of every 128-wide K tile), the K groups summed in shared memory at the end.
// Small N gets more CTAs from smaller BN; K can also be split over gridDim.y CTAs, the last CTA of a column tile
// adding the slices up (serial split-K: partials in a workspace, a counter per tile that this CTA resets, so the
// kernel replays inside CUDA graphs). The partials' order is fixed, so results do not depend on timing.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.cuh"

namespace {

constexpr int BK = 128;

struct W8Args {
  const __nv_bfloat16* A;
  const uint8_t* W;
  const float* sw;
  const __nv_bfloat16* bias;
  __nv_bfloat16* Y;
  float* ws;      // (N / BN, splits, BM, BN) partial sums
  int* counters;  // (N / BN)
  int lda, ldy, M, N, K, k_split;
};

// 4 offset-uint8 weights -> two bf16x2 (exact: int8 values are bf16 numbers)
DEVI void widen(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const float f0 = __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440)) - 8388736.f;  // 2^23 + 128
  const float f1 = __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7441)) - 8388736.f;
  const float f2 = __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7442)) - 8388736.f;
  const float f3 = __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7443)) - 8388736.f;
  lo = pack_bf16(f0, f1);
  hi = pack_bf16(f2, f3);
}

template <int BM, int BN, int STAGES, bool RELU>
__global__ void __launch_bounds__(128) dec_w8_kernel(const W8Args p) {
  constexpr int MI = BM / 16, WN = BN / 16, WK = 4 / WN, KS = BK / 16 / WK;  // k16 steps per warp per tile
  constexpr int A_ROW = BK * 2, A_CH = A_ROW / 16, W_CH = BK / 16;
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* sA = smem;                        // STAGES x BM x 256 B
  unsigned char* sW = smem + STAGES * BM * A_ROW;  // STAGES x BN x 128 B
  const int tile = blockIdx.x, split = blockIdx.y, n0 = tile * BN;
  const int k_lo = split * p.k_split, KT = p.k_split / BK;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int wn = warp % WN, wk = warp / WN;

  auto swz = [](int row, int c) { return (c & ~7) | ((c ^ row) & 7); };  // 16-byte chunks, XOR within 8
  auto load = [&](int stage, int k0) {
    unsigned char* a = sA + stage * BM * A_ROW;
    unsigned char* w = sW + stage * BN * BK;
    for (int i = tid; i < BM * A_CH; i += 128) {
      const int r = i / A_CH, c = i % A_CH;
      const bool ok = r < p.M;
      cp_async16(a + r * A_ROW + swz(r, c) * 16, p.A + (size_t)(ok ? r : 0) * p.lda + k0 + c * 8, ok);
    }
    for (int i = tid; i < BN * W_CH; i += 128) {
      const int r = i / W_CH, c = i % W_CH;
      cp_async16(w + r * BK + swz(r, c) * 16, p.W + (size_t)(n0 + r) * p.K + k0 + c * 16, true);
    }
  };

  float acc[MI][2][4] = {};
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load(s, k_lo + s * BK);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, k_lo + (kt + STAGES - 1) * BK);
    cp_async_commit();
    const unsigned char* a = sA + (kt % STAGES) * BM * A_ROW;
    const unsigned char* w = sW + (kt % STAGES) * BN * BK;
#pragma unroll
    for (int j = 0; j < KS; ++j) {
      const int ks = wk * KS + j;
      uint32_t w0, w1;  // rows 16 wn + 0..7 and + 8..15, this k16 block's 16 bytes
      const int wr = wn * 16 + (lane & 15);
      ldmatrix_x2(w0, w1, w + wr * BK + swz(wr, ks) * 16);
      uint32_t b[2][2];
      widen(w0, b[0][0], b[0][1]);
      widen(w1, b[1][0], b[1][1]);
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        uint32_t af[4];
        const int ar = mi * 16 + (lane & 15), ac = ks * 2 + (lane >> 4);
        ldmatrix_x4(af[0], af[1], af[2], af[3], a + ar * A_ROW + swz(ar, ac) * 16);
        mma_bf16(acc[mi][0], af, b[0][0], b[0][1]);
        mma_bf16(acc[mi][1], af, b[1][0], b[1][1]);
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // the K groups' partial tiles summed in shared memory (fixed order): red[m][c] for the CTA's BM x BN
  float* red = reinterpret_cast<float*>(smem);
  constexpr int LDR = BN + 4;
  if (WK > 1) {
    for (int k = 0; k < WK; ++k) {
      if (wk == k) {
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
          for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int m = mi * 16 + g + (e >> 1) * 8, c = wn * 16 + nt * 8 + 2 * t + (e & 1);
              red[m * LDR + c] = (k ? red[m * LDR + c] : 0.f) + acc[mi][nt][e];
            }
      }
      __syncthreads();
    }
  } else {
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int m = mi * 16 + g + (e >> 1) * 8, c = wn * 16 + nt * 8 + 2 * t + (e & 1);
          red[m * LDR + c] = acc[mi][nt][e];
        }
    __syncthreads();
  }

  auto finish = [&](int m, int c, float v) {
    const int n = n0 + c;
    v = v * p.sw[n] + (p.bias ? bf(p.bias[n]) : 0.f);
    if (RELU) v = fmaxf(v, 0.f);
    p.Y[(size_t)m * p.ldy + n] = tobf(v);
  };
  const int nsplit = gridDim.y;
  if (nsplit == 1) {
    for (int i = tid; i < p.M * BN; i += 128) finish(i / BN, i % BN, red[(i / BN) * LDR + i % BN]);
    return;
  }
  float* part = p.ws + ((size_t)tile * nsplit + split) * BM * BN;
  for (int i = tid; i < p.M * BN; i += 128) part[i] = red[(i / BN) * LDR + i % BN];
  __threadfence();
  __shared__ int last;
  __syncthreads();
  if (tid == 0) last = atomicAdd(p.counters + tile, 1) == nsplit - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const float* base = p.ws + (size_t)tile * nsplit * BM * BN;
  for (int i = tid * 4; i < p.M * BN; i += 128 * 4) {  // 4 columns per thread, every split's loads in flight
    float4 v[16];
#pragma unroll
    for (int s = 0; s < 16; ++s)
      if (s < nsplit) v[s] = __ldcg(reinterpret_cast<const float4*>(base + (size_t)s * BM * BN + i));
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 16; ++s)
      if (s < nsplit) {
        sum.x += v[s].x;
        sum.y += v[s].y;
        sum.z += v[s].z;
        sum.w += v[s].w;
      }
    finish(i / BN, i % BN, sum.x);
    finish(i / BN, i % BN + 1, sum.y);
    finish(i / BN, i % BN + 2, sum.z);
    finish(i / BN, i % BN + 3, sum.w);
  }
  if (tid == 0) p.counters[tile] = 0;
}

template <int BM, int BN, bool RELU>
void launch(const W8Args& p, int splits, cudaStream_t s) {
  constexpr int STAGES = 3, pipe = STAGES * (BM * BK * 2 + BN * BK), red = BM * (BN + 4) * 4;
  constexpr int smem = pipe > red ? pipe : red;
  auto kernel = dec_w8_kernel<BM, BN, STAGES, RELU>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  kernel<<<dim3(p.N / BN, splits), 128, smem, s>>>(p);
}

template <int BN, bool RELU>
void by_rows(const W8Args& p, int splits, cudaStream_t s) {
  if (p.M <= 16) launch<16, BN, RELU>(p, splits, s);
  else if (p.M <= 32) launch<32, BN, RELU>(p, splits, s);
  else launch<64, BN, RELU>(p, splits, s);
}

}  // namespace

// y (M, N) bf16 = x (M, K) bf16 @ dequant(w)^T + bias (relu), M <= 64; w (N, K) uint8 in prepare_w8's layout, sw (N)
// fp32 scales; bn = 16 | 32 | 64 columns per CTA; ws (>= N/bn * splits * 64 * bn) fp32 and counters (N/bn, zeros)
// int32 when splits > 1 (at most 16)
void dec_w8_linear(const at::Tensor& x, const at::Tensor& w, const at::Tensor& sw, const c10::optional<at::Tensor>& bias,
                   at::Tensor& y, bool relu, int64_t bn, int64_t splits, at::Tensor& ws, at::Tensor& counters) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M <= 64 && x.stride(1) == 1 && w.is_contiguous() && w.size(1) == K && y.stride(1) == 1);
  TORCH_CHECK((bn == 16 || bn == 32 || bn == 64) && N % bn == 0 && splits <= 16 && K % (BK * splits) == 0,
              "dec_w8_linear: shape (", N, ", ", K, "), bn ", bn, ", splits ", splits);
  TORCH_CHECK(splits == 1 || (ws.numel() >= (int64_t)(N / bn) * splits * 64 * bn && counters.numel() >= N / bn));
  W8Args p;
  p.A = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.W = w.data_ptr<uint8_t>();
  p.sw = sw.data_ptr<float>();
  p.bias = bias ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr;
  p.Y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  p.ws = splits > 1 ? ws.data_ptr<float>() : nullptr;
  p.counters = splits > 1 ? counters.data_ptr<int>() : nullptr;
  p.lda = x.stride(0);
  p.ldy = y.stride(0);
  p.M = M;
  p.N = N;
  p.K = K;
  p.k_split = K / splits;
  if (M == 0) return;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  if (bn == 16) relu ? by_rows<16, true>(p, splits, s) : by_rows<16, false>(p, splits, s);
  else if (bn == 32) relu ? by_rows<32, true>(p, splits, s) : by_rows<32, false>(p, splits, s);
  else relu ? by_rows<64, true>(p, splits, s) : by_rows<64, false>(p, splits, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
