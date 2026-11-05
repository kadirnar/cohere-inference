// bf16 GEMM with fused epilogues for the encoder and the decoder's prompt pass: C = epilogue(A W^T + bias).
//
// A (M, K) and W (N, K) are both K-major, as nn.Linear stores them. A CTA computes a BM x BN tile in BK = 32 steps
// through a STAGES-deep cp.async pipeline; shared memory rows are XOR-swizzled so ldmatrix reads are conflict-free;
// warps multiply with mma.sync m16n8k16 (fp32 accumulation). Tiles are rastered in groups of 8 M-tiles per N column
// so the A and W blocks a wave reads stay in L2. The epilogue adds the bias in fp32, rounds once to bf16 where
// PyTorch rounds the linear's output, then applies:
//   EPI_BIAS   y = acc + b
//   EPI_SILU   y = silu(bf16(acc + b))                      (Linear -> SiLU)
//   EPI_RELU   y = relu(acc + b)                            (Linear -> ReLU)
//   EPI_RESID  y = r + alpha * bf16(acc + b)                (residual + alpha * Linear), r may alias y
//   EPI_GLU    y[:, j] = a * sigmoid(g), a = bf16(acc + b)[:, 2j], g = ...[:, 2j + 1]   (W rows interleaved)
//   EPI_CONV_RELU  y = relu(bf16(acc) + b)                  (cuDNN convolution, its bias added after, ReLU)
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.cuh"

namespace {

enum { EPI_BIAS = 0, EPI_SILU = 1, EPI_RELU = 2, EPI_RESID = 3, EPI_GLU = 4, EPI_CONV_RELU = 6 };

struct GemmArgs {
  const __nv_bfloat16* A;
  const __nv_bfloat16* W;
  const __nv_bfloat16* bias;
  __nv_bfloat16* C;
  const __nv_bfloat16* R;
  int lda, ldw, ldc, ldr;
  float alpha;
  int M, N, K;
};

template <int BM, int BN, int WM, int WN, int STAGES, int EPI>
__global__ void __launch_bounds__(WM* WN * 32) gemm_kernel(const GemmArgs p) {
  constexpr int BK = 32;
  constexpr int THREADS = WM * WN * 32;
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int MI = TM / 16, NI = TN / 8;
  static_assert(NI % 2 == 0 && (BM * 4) % THREADS == 0 && (BN * 4) % THREADS == 0, "tile shape");
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sB = sA + STAGES * BM * BK;

  // grouped raster: 8 consecutive M tiles share each N column
  const int num_n = p.N / BN, num_m = (p.M + BM - 1) / BM;
  const int pid = blockIdx.x, width = 8 * num_n;
  const int first_m = (pid / width) * 8, gsize = min(num_m - first_m, 8);
  const int m0 = (first_m + (pid % width) % gsize) * BM, n0 = ((pid % width) / gsize) * BN;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;

  // per-thread source pointers and swizzled destinations, advanced by BK each tile
  constexpr int LA = BM * 4 / THREADS, LB = BN * 4 / THREADS;
  const __nv_bfloat16* gA[LA];
  const __nv_bfloat16* gB[LB];
  int dA[LA], dB[LB];
  bool okA[LA];
#pragma unroll
  for (int i = 0; i < LA; ++i) {
    const int id = tid + i * THREADS, row = id >> 2, ch = id & 3, gm = m0 + row;
    gA[i] = p.A + (size_t)min(gm, p.M - 1) * p.lda + ch * 8;
    dA[i] = swz32(row, ch);
    okA[i] = gm < p.M;
  }
#pragma unroll
  for (int i = 0; i < LB; ++i) {
    const int id = tid + i * THREADS, row = id >> 2, ch = id & 3;
    gB[i] = p.W + (size_t)(n0 + row) * p.ldw + ch * 8;
    dB[i] = swz32(row, ch);
  }
  auto load_stage = [&](int stage, int k0) {
    __nv_bfloat16* a = sA + stage * BM * BK;
    __nv_bfloat16* b = sB + stage * BN * BK;
#pragma unroll
    for (int i = 0; i < LA; ++i) cp_async16(a + dA[i], gA[i] + k0, okA[i]);
#pragma unroll
    for (int i = 0; i < LB; ++i) cp_async16(b + dB[i], gB[i] + k0, true);
  };
  // ldmatrix offsets of this lane for k16 step ks
  int oA[2][MI], oB[2][NI / 2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) oA[ks][mi] = swz32(wm * TM + mi * 16 + (lane & 15), ks * 2 + (lane >> 4));
#pragma unroll
    for (int nj = 0; nj < NI / 2; ++nj)
      oB[ks][nj] = swz32(wn * TN + nj * 16 + ((lane >> 4) << 3) + (lane & 7), ks * 2 + ((lane >> 3) & 1));
  }
  uint32_t af[2][MI][4], bfr[2][NI][2];
  auto load_frags = [&](int buf, int stage, int ks) {
    const __nv_bfloat16* a = sA + stage * BM * BK;
    const __nv_bfloat16* b = sB + stage * BN * BK;
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
      ldmatrix_x4(af[buf][mi][0], af[buf][mi][1], af[buf][mi][2], af[buf][mi][3], a + oA[ks][mi]);
#pragma unroll
    for (int nj = 0; nj < NI / 2; ++nj)
      ldmatrix_x4(bfr[buf][2 * nj][0], bfr[buf][2 * nj][1], bfr[buf][2 * nj + 1][0], bfr[buf][2 * nj + 1][1],
                  b + oB[ks][nj]);
  };

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;

  // multistage pipeline (as CUTLASS's): tile kt+STAGES-1 is requested while tile kt is multiplied, and the fragments
  // of the next k16 step are read from shared memory while the current one's mma run
  const int KT = p.K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_stage(s, s * BK);
    cp_async_commit();
  }
  cp_async_wait<STAGES - 2>();
  __syncthreads();
  load_frags(0, 0, 0);
  int stage = 0;
  for (int kt = 0; kt < KT; ++kt) {
    // ks = 0
    {
      const int nk = kt + STAGES - 1;
      if (nk < KT) load_stage(nk % STAGES, nk * BK);
      cp_async_commit();
      load_frags(1, stage, 1);
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_bf16(acc[mi][ni], af[0][mi], bfr[0][ni][0], bfr[0][ni][1]);
    }
    // ks = 1
    {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      stage = stage + 1 == STAGES ? 0 : stage + 1;
      if (kt + 1 < KT) load_frags(0, stage, 0);
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_bf16(acc[mi][ni], af[1][mi], bfr[1][ni][0], bfr[1][ni][1]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();  // the pipeline's shared memory is reused for the output tile

  // epilogue: bias and activation in registers, the rounded tile staged in shared memory (rows padded by 8 so the
  // fragment writes do not conflict), then written out (and the residual read) in coalesced 16-byte chunks
  constexpr int OUT_N = EPI == EPI_GLU ? BN / 2 : BN;
  constexpr int LDS = OUT_N + 8;
  __nv_bfloat16* sC = sA;
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = wm * TM + mi * 16 + g + half * 8;
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int c = wn * TN + ni * 8 + 2 * t;
        float v0 = acc[mi][ni][half * 2], v1 = acc[mi][ni][half * 2 + 1];
        if constexpr (EPI == EPI_CONV_RELU) {
          v0 = round_bf(v0);
          v1 = round_bf(v1);
        }
        if (p.bias) {
          const __nv_bfloat162 bb = *reinterpret_cast<const __nv_bfloat162*>(p.bias + n0 + c);
          v0 += __low2float(bb);
          v1 += __high2float(bb);
        }
        if constexpr (EPI == EPI_GLU) {
          sC[r * LDS + (c >> 1)] = tobf(round_bf(v0) * sigmoidf_(round_bf(v1)));
        } else {
          if constexpr (EPI == EPI_SILU) {
            v0 = round_bf(v0);
            v1 = round_bf(v1);
            v0 = v0 * sigmoidf_(v0);
            v1 = v1 * sigmoidf_(v1);
          } else if constexpr (EPI == EPI_RELU || EPI == EPI_CONV_RELU) {
            v0 = fmaxf(v0, 0.f);
            v1 = fmaxf(v1, 0.f);
          }
          *reinterpret_cast<uint32_t*>(sC + r * LDS + c) = pack_bf16(v0, v1);
        }
      }
    }
  }
  __syncthreads();
  constexpr int CH = OUT_N / 8;  // 16-byte chunks per row
  const int out_col0 = EPI == EPI_GLU ? n0 / 2 : n0;
#pragma unroll 4
  for (int id = tid; id < BM * CH; id += THREADS) {
    const int r = id / CH, c = (id % CH) * 8, row = m0 + r;
    if (row >= p.M) continue;
    uint4 v = *reinterpret_cast<const uint4*>(sC + r * LDS + c);
    if constexpr (EPI == EPI_RESID) {
      const uint4 rv = *reinterpret_cast<const uint4*>(p.R + (size_t)row * p.ldr + out_col0 + c);
      const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(&v);
      const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&rv);
      uint32_t o[4];
#pragma unroll
      for (int k = 0; k < 4; ++k)
        o[k] = pack_bf16(__low2float(r2[k]) + p.alpha * __low2float(y2[k]),
                         __high2float(r2[k]) + p.alpha * __high2float(y2[k]));
      v = make_uint4(o[0], o[1], o[2], o[3]);
    }
    *reinterpret_cast<uint4*>(p.C + (size_t)row * p.ldc + out_col0 + c) = v;
  }
}

template <int BM, int BN, int WM, int WN, int STAGES, int EPI>
void launch(const GemmArgs& p, cudaStream_t stream) {
  constexpr int smem = STAGES * (BM + BN) * 32 * 2 > BM * (BN + 8) * 2 ? STAGES * (BM + BN) * 32 * 2
                                                                    : BM * (BN + 8) * 2;
  auto kernel = gemm_kernel<BM, BN, WM, WN, STAGES, EPI>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  TORCH_CHECK(p.N % BN == 0, "gemm: N=", p.N, " not a multiple of the tile width ", BN);
  const int grid = ((p.M + BM - 1) / BM) * (p.N / BN);
  kernel<<<grid, WM * WN * 32, smem, stream>>>(p);
}

template <int EPI>
void dispatch(int cfg, const GemmArgs& p, cudaStream_t s) {
  switch (cfg) {
    // tiles (M x N), warps (M x N), stages; 8 is the fastest on an A10 for the encoder's shapes (bench numbers in the
    // repository history); 3 and 5 for smaller M
    case 3: launch<64, 128, 2, 2, 4, EPI>(p, s); break;
    case 5: launch<64, 64, 2, 2, 4, EPI>(p, s); break;
    case 8: launch<128, 128, 2, 2, 3, EPI>(p, s); break;
    default: TORCH_CHECK(false, "gemm: unknown config ", cfg);
  }
}

}  // namespace

// out = epilogue(a @ w^T + bias); a (M, K) with unit stride along K, w (N, K) contiguous. For EPI_GLU out is (M, N/2).
void gemm(const at::Tensor& a, const at::Tensor& w, const c10::optional<at::Tensor>& bias, at::Tensor& out,
          int64_t epi, const c10::optional<at::Tensor>& resid, double alpha, int64_t cfg) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "gemm: bf16 only");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.stride(1) == 1 && w.is_contiguous() && out.stride(1) == 1);
  TORCH_CHECK(a.size(1) == w.size(1) && a.size(1) % 32 == 0, "gemm: K must match and be a multiple of 32");
  GemmArgs p;
  p.A = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr());
  p.W = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  p.bias = bias ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr;
  p.C = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.R = resid ? reinterpret_cast<const __nv_bfloat16*>(resid->data_ptr()) : nullptr;
  p.lda = a.stride(0);
  p.ldw = w.stride(0);
  p.ldc = out.stride(0);
  p.ldr = resid ? resid->stride(0) : 0;
  p.alpha = alpha;
  p.M = a.size(0);
  p.N = w.size(0);
  p.K = a.size(1);
  if (p.M == 0) return;
  TORCH_CHECK(epi != EPI_RESID || (resid && resid->stride(1) == 1), "gemm: residual epilogue needs resid");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  switch (epi) {
    case EPI_BIAS: dispatch<EPI_BIAS>(cfg, p, s); break;
    case EPI_SILU: dispatch<EPI_SILU>(cfg, p, s); break;
    case EPI_RELU: dispatch<EPI_RELU>(cfg, p, s); break;
    case EPI_RESID: dispatch<EPI_RESID>(cfg, p, s); break;
    case EPI_GLU: dispatch<EPI_GLU>(cfg, p, s); break;
    case EPI_CONV_RELU: dispatch<EPI_CONV_RELU>(cfg, p, s); break;
    default: TORCH_CHECK(false, "gemm: unknown epilogue ", epi);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
