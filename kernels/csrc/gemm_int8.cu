// int8 GEMM (W8A8) for the encoder: C = epilogue(dequant(A_q W_q^T) + bias).
//
// A10 tensor cores run int8 at twice the bf16 rate, and under the card's 150 W cap the measured gain is ~1.5x. A is
// int8 with fp32 scales per row and K-group (sa[m, k / group]; group = K: one scale per row), W int8 with one fp32
// scale per output channel (sw[n]). The pipeline is the bf16 kernel's (gemm.cu) with the same byte layout: a BK = 64
// int8 tile row is 64 bytes, so the swizzle and the ldmatrix addresses are the same; the warps multiply with
// mma.sync m16n8k32 s8 (int32 accumulation, exact), converted to fp32 with the row's scale at the end of each K-group.
// Few rows (a request or two: M of a few hundred frames) leave most of the 72 SMs idle with 128-row tiles, each CTA
// bound by its own tile's arithmetic; configs 5-9 have 32 or 64-row tiles for them (the same sums in the same order:
// a clip's output does not depend on what it is batched with).
// Epilogues (value v = acc * sa * sw + bias, rounded to bf16 where the bf16 module rounds):
//   EPI_BIAS      bf16 y = v
//   EPI_GLU       bf16 y[:, j] = a * sigmoid(g) of interleaved columns (2j, 2j+1)
//   EPI_RESID     bf16 y = r + alpha * bf16(v)
//   EPI_SILU_Q8   int8 y = quant(silu(bf16(v))) with one scale per row and tile of BN output columns (128 or 256),
//                 written to qs[m, n / BN]: the input of the next GEMM with group = BN
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.cuh"

namespace {

enum { EPI_BIAS = 0, EPI_SILU = 1, EPI_RELU = 2, EPI_RESID = 3, EPI_GLU = 4, EPI_SILU_Q8 = 5 };

struct I8Args {
  const int8_t* A;
  const float* sa;  // (M, K / group)
  const int8_t* W;
  const float* sw;  // (N)
  const __nv_bfloat16* bias;
  void* C;
  float* qs;  // EPI_SILU_Q8: (M, N / 128) output scales
  const __nv_bfloat16* R;
  int lda, ldw, ldc, ldr, ldsa, ldqs;
  int group_tiles;  // K-tiles of 64 per A scale group
  float alpha;
  int M, N, K;
};

DEVI void mma_s8(int* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// int32 -> fp32 for |x| < 2^22 without the quarter-rate I2F: the bits of 1.5 * 2^23 + x, minus 1.5 * 2^23 (a K-group of
// up to 256 int8 products is under 256 * 127 * 127 < 2^22)
DEVI float i2f_small(int x) { return __int_as_float(x + 0x4B400000) - 12582912.f; }

// sigmoid for the int8 path's epilogues: their outputs are rounded to bf16 (or quantized to int8) right after, far
// coarser than __expf's error
DEVI float fast_sigmoid(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }

// byte offset of 16-byte chunk ch (of 4) in a 64-byte row, swizzled as gemm.cu's swz32
DEVI int sw64b(int row, int ch) { return row * 64 + ((ch ^ ((row >> 1) & 3)) << 4); }

// GROUPED: A has several scale groups along K (fp32 accumulators besides the int32 ones); otherwise one scale per row,
// applied in the epilogue (int32 accumulators only: room for 64 x 64 warp tiles)
template <int BM, int BN, int WM, int WN, int STAGES, int EPI, bool GROUPED>
__global__ void __launch_bounds__(WM* WN * 32) gemm_i8_kernel(const I8Args p) {
  constexpr int BK = 64;
  constexpr int THREADS = WM * WN * 32;
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int MI = TM / 16, NI = TN / 8;
  static_assert(NI % 2 == 0 && (BM * 4) % THREADS == 0 && (BN * 4) % THREADS == 0, "tile shape");
  extern __shared__ __align__(128) unsigned char smem_raw[];
  int8_t* sA = reinterpret_cast<int8_t*>(smem_raw);
  int8_t* sB = sA + STAGES * BM * BK;

  const int num_n = p.N / BN, num_m = (p.M + BM - 1) / BM;
  const int pid = blockIdx.x, width = 8 * num_n;
  const int first_m = (pid / width) * 8, gsize = min(num_m - first_m, 8);
  const int m0 = (first_m + (pid % width) % gsize) * BM, n0 = ((pid % width) / gsize) * BN;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int g = lane >> 2, t = lane & 3;

  constexpr int LA = BM * 4 / THREADS, LB = BN * 4 / THREADS;
  const int8_t* gA[LA];
  const int8_t* gB[LB];
  int dA[LA], dB[LB];
  bool okA[LA];
#pragma unroll
  for (int i = 0; i < LA; ++i) {
    const int id = tid + i * THREADS, row = id >> 2, ch = id & 3, gm = m0 + row;
    gA[i] = p.A + (size_t)min(gm, p.M - 1) * p.lda + ch * 16;
    dA[i] = sw64b(row, ch);
    okA[i] = gm < p.M;
  }
#pragma unroll
  for (int i = 0; i < LB; ++i) {
    const int id = tid + i * THREADS, row = id >> 2, ch = id & 3;
    gB[i] = p.W + (size_t)(n0 + row) * p.ldw + ch * 16;
    dB[i] = sw64b(row, ch);
  }
  auto load_stage = [&](int stage, int k0) {
    int8_t* a = sA + stage * BM * BK;
    int8_t* b = sB + stage * BN * BK;
#pragma unroll
    for (int i = 0; i < LA; ++i) cp_async16(a + dA[i], gA[i] + k0, okA[i]);
#pragma unroll
    for (int i = 0; i < LB; ++i) cp_async16(b + dB[i], gB[i] + k0, true);
  };
  int oA[2][MI], oB[2][NI / 2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) oA[ks][mi] = sw64b(wm * TM + mi * 16 + (lane & 15), ks * 2 + (lane >> 4));
#pragma unroll
    for (int nj = 0; nj < NI / 2; ++nj)
      oB[ks][nj] = sw64b(wn * TN + nj * 16 + ((lane >> 4) << 3) + (lane & 7), ks * 2 + ((lane >> 3) & 1));
  }
  uint32_t af[2][MI][4], bfr[2][NI][2];
  auto load_frags = [&](int buf, int stage, int ks) {
    const int8_t* a = sA + stage * BM * BK;
    const int8_t* b = sB + stage * BN * BK;
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
      ldmatrix_x4(af[buf][mi][0], af[buf][mi][1], af[buf][mi][2], af[buf][mi][3], a + oA[ks][mi]);
#pragma unroll
    for (int nj = 0; nj < NI / 2; ++nj)
      ldmatrix_x4(bfr[buf][2 * nj][0], bfr[buf][2 * nj][1], bfr[buf][2 * nj + 1][0], bfr[buf][2 * nj + 1][1],
                  b + oB[ks][nj]);
  };

  int acc[MI][NI][4];
  float accf[GROUPED ? MI : 1][GROUPED ? NI : 1][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0;
  if constexpr (GROUPED) {
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NI; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) accf[i][j][k] = 0.f;
  }
  // fold the int32 sums of the finished K-group into fp32 with each row's scale of that group
  auto fold = [&](int grp) {
    if constexpr (!GROUPED) return;
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = min(m0 + wm * TM + mi * 16 + g + half * 8, p.M - 1);
        const float s = p.sa[(size_t)row * p.ldsa + grp];
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          if constexpr (GROUPED) {
            accf[mi][ni][half * 2] = __fmaf_rn(i2f_small(acc[mi][ni][half * 2]), s, accf[mi][ni][half * 2]);
            accf[mi][ni][half * 2 + 1] = __fmaf_rn(i2f_small(acc[mi][ni][half * 2 + 1]), s, accf[mi][ni][half * 2 + 1]);
          }
          acc[mi][ni][half * 2] = acc[mi][ni][half * 2 + 1] = 0;
        }
      }
  };

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
    {
      const int nk = kt + STAGES - 1;
      if (nk < KT) load_stage(nk % STAGES, nk * BK);
      cp_async_commit();
      load_frags(1, stage, 1);
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_s8(acc[mi][ni], af[0][mi], bfr[0][ni][0], bfr[0][ni][1]);
    }
    {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      stage = stage + 1 == STAGES ? 0 : stage + 1;
      if (kt + 1 < KT) load_frags(0, stage, 0);
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_s8(acc[mi][ni], af[1][mi], bfr[1][ni][0], bfr[1][ni][1]);
    }
    if (GROUPED && (kt + 1) % p.group_tiles == 0) fold(kt / p.group_tiles);
  }
  cp_async_wait<0>();
  __syncthreads();

  // epilogue: dequantize, bias, activation in registers; the bf16 tile staged in shared memory; written out in
  // 16-byte chunks (int8 output: 8 values and a per-row, per-tile scale)
  constexpr int OUT_N = EPI == EPI_GLU ? BN / 2 : BN;
  constexpr int LDS = OUT_N + 8;
  __nv_bfloat16* sC = reinterpret_cast<__nv_bfloat16*>(smem_raw);
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = wm * TM + mi * 16 + g + half * 8;
      const float srow = GROUPED ? 1.f : p.sa[(size_t)min(m0 + r, p.M - 1) * p.ldsa];
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int c = wn * TN + ni * 8 + 2 * t;
        const float2 w2 = *reinterpret_cast<const float2*>(p.sw + n0 + c);
        // explicit roundings (the bias added in the product's FMA): every config computes the same values
        float x0, x1;
        if constexpr (GROUPED) {
          x0 = accf[mi][ni][half * 2];
          x1 = accf[mi][ni][half * 2 + 1];
        } else {
          x0 = __fmul_rn((float)acc[mi][ni][half * 2], srow);
          x1 = __fmul_rn((float)acc[mi][ni][half * 2 + 1], srow);
        }
        float v0, v1;
        if (p.bias) {
          const __nv_bfloat162 bb = *reinterpret_cast<const __nv_bfloat162*>(p.bias + n0 + c);
          v0 = __fmaf_rn(x0, w2.x, __low2float(bb));
          v1 = __fmaf_rn(x1, w2.y, __high2float(bb));
        } else {
          v0 = __fmul_rn(x0, w2.x);
          v1 = __fmul_rn(x1, w2.y);
        }
        if constexpr (EPI == EPI_GLU) {
          sC[r * LDS + (c >> 1)] = tobf(round_bf(v0) * fast_sigmoid(round_bf(v1)));
        } else {
          if constexpr (EPI == EPI_SILU_Q8) {
            v0 = round_bf(v0);
            v1 = round_bf(v1);
            v0 = v0 * fast_sigmoid(v0);
            v1 = v1 * fast_sigmoid(v1);
          }
          *reinterpret_cast<uint32_t*>(sC + r * LDS + c) = pack_bf16(v0, v1);
        }
      }
    }
  }
  __syncthreads();
  constexpr int CH = OUT_N / 8;
  const int out_col0 = EPI == EPI_GLU ? n0 / 2 : n0;
#pragma unroll 4
  for (int id = tid; id < BM * CH; id += THREADS) {
    const int r = id / CH, c = (id % CH) * 8, row = m0 + r;
    uint4 v = *reinterpret_cast<const uint4*>(sC + r * LDS + c);
    if constexpr (EPI == EPI_SILU_Q8) {
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&v);
      float mx = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) mx = fmaxf(mx, fabsf(bf(e[k])));
#pragma unroll
      for (int o = 1; o < CH; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      const float scale = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / scale;
      uint32_t q[2] = {0, 0};
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int qi = __float2int_rn(fminf(fmaxf(bf(e[k]) * inv, -127.f), 127.f));
        q[k >> 2] |= (uint32_t)(qi & 0xff) << ((k & 3) * 8);
      }
      if (row < p.M) {
        *reinterpret_cast<uint2*>(reinterpret_cast<int8_t*>(p.C) + (size_t)row * p.ldc + n0 + c) = make_uint2(q[0], q[1]);
        if (c == 0) p.qs[(size_t)row * p.ldqs + n0 / BN] = scale;
      }
    } else {
      if (row >= p.M) continue;
      if constexpr (EPI == EPI_RESID) {
        const uint4 rv = *reinterpret_cast<const uint4*>(p.R + (size_t)row * p.ldr + out_col0 + c);
        const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(&v);
        const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&rv);
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
          o[k] = pack_bf16(__fmaf_rn(p.alpha, __low2float(y2[k]), __low2float(r2[k])),
                           __fmaf_rn(p.alpha, __high2float(y2[k]), __high2float(r2[k])));
        v = make_uint4(o[0], o[1], o[2], o[3]);
      }
      *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(p.C) + (size_t)row * p.ldc + out_col0 + c) = v;
    }
  }
}

template <int BM, int BN, int WM, int WN, int STAGES, int EPI, bool GROUPED>
void launch_one(const I8Args& p, cudaStream_t stream) {
  constexpr int pipe = STAGES * (BM + BN) * 64, out = BM * (BN + 8) * 2;
  constexpr int smem = pipe > out ? pipe : out;
  auto kernel = gemm_i8_kernel<BM, BN, WM, WN, STAGES, EPI, GROUPED>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  TORCH_CHECK(p.N % BN == 0, "gemm_i8: N=", p.N, " not a multiple of ", BN);
  kernel<<<((p.M + BM - 1) / BM) * (p.N / BN), WM * WN * 32, smem, stream>>>(p);
}

template <int BM, int BN, int WM, int WN, int STAGES, int EPI>
void launch(const I8Args& p, cudaStream_t stream) {
  if (p.group_tiles * 64 < p.K) {
    // grouped scales keep fp32 accumulators too: warp tiles of 64 x 64 would spill
    if constexpr ((BM / WM) * (BN / WN) <= 64 * 32) launch_one<BM, BN, WM, WN, STAGES, EPI, true>(p, stream);
    else TORCH_CHECK(false, "gemm_i8: this config cannot take grouped A scales");
  } else {
    launch_one<BM, BN, WM, WN, STAGES, EPI, false>(p, stream);
  }
}

template <int EPI, int BN>
constexpr bool valid() {  // the int8 output's scales cover one tile's columns: at most a warp's 32 lanes x 8
  return EPI != EPI_SILU_Q8 || BN <= 256;
}

template <int EPI>
void dispatch(int cfg, const I8Args& p, cudaStream_t s) {
  switch (cfg) {
    case 0: launch<128, 128, 2, 2, 3, EPI>(p, s); break;  // warp tile 64 x 64: per-row A scales
    case 1: launch<128, 128, 2, 4, 3, EPI>(p, s); break;  // warp tile 64 x 32
    case 3: launch<64, 128, 2, 2, 4, EPI>(p, s); break;   // warp tile 32 x 64: grouped A scales
    case 5: launch<32, 64, 1, 2, 4, EPI>(p, s); break;    // warp tile 32 x 32: few rows (grouped too)
    case 6:                                               // the feed-forwards' first GEMM, few rows
      if constexpr (valid<EPI, 256>()) launch<32, 256, 1, 4, 3, EPI>(p, s);
      else TORCH_CHECK(false, "gemm_i8: config 6 has 256-column tiles");
      break;
    case 7: launch<32, 128, 1, 4, 4, EPI>(p, s); break;   // warp tile 32 x 32: few rows
    case 8:
      if constexpr (valid<EPI, 256>()) launch<32, 256, 1, 4, 2, EPI>(p, s);
      else TORCH_CHECK(false, "gemm_i8: config 8 has 256-column tiles");
      break;
    case 9: launch<64, 64, 2, 2, 4, EPI>(p, s); break;    // warp tile 32 x 32
    case 10: launch<128, 128, 4, 2, 3, EPI>(p, s); break;  // warp tile 32 x 64, 8 warps: grouped A scales, many rows
    case 4:                                               // 256-column tiles: the feed-forwards' first GEMM
      if constexpr (valid<EPI, 256>()) launch<128, 256, 2, 4, 3, EPI>(p, s);
      else TORCH_CHECK(false, "gemm_i8: config 4 has 256-column tiles");
      break;
    default: TORCH_CHECK(false, "gemm_i8: unknown config ", cfg);
  }
}

}  // namespace

// out = epilogue(dequant(a @ w^T) + bias): a int8 (M, K) with sa (M, K / group) fp32, w int8 (N, K) with sw (N) fp32.
// EPI_SILU_Q8 writes int8 out (M, N) and qs (M, N / 128).
void gemm_i8(const at::Tensor& a, const at::Tensor& sa, int64_t group, const at::Tensor& w, const at::Tensor& sw,
             const c10::optional<at::Tensor>& bias, at::Tensor& out, int64_t epi, const c10::optional<at::Tensor>& resid,
             double alpha, const c10::optional<at::Tensor>& qs, int64_t cfg) {
  TORCH_CHECK(a.scalar_type() == at::kChar && w.scalar_type() == at::kChar && sa.scalar_type() == at::kFloat &&
              sw.scalar_type() == at::kFloat);
  TORCH_CHECK(a.dim() == 2 && a.stride(1) == 1 && w.is_contiguous() && out.stride(1) == 1 && sa.stride(1) == 1);
  TORCH_CHECK(a.size(1) == w.size(1) && a.size(1) % 64 == 0 && group % 64 == 0 && a.size(1) % group == 0);
  I8Args p;
  p.A = a.data_ptr<int8_t>();
  p.sa = sa.data_ptr<float>();
  p.W = w.data_ptr<int8_t>();
  p.sw = sw.data_ptr<float>();
  p.bias = bias ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr;
  p.C = out.data_ptr();
  p.qs = qs ? qs->data_ptr<float>() : nullptr;
  p.R = resid ? reinterpret_cast<const __nv_bfloat16*>(resid->data_ptr()) : nullptr;
  p.lda = a.stride(0);
  p.ldw = w.stride(0);
  p.ldc = out.stride(0);
  p.ldr = resid ? resid->stride(0) : 0;
  p.ldsa = sa.stride(0);
  p.ldqs = qs ? qs->stride(0) : 0;
  p.group_tiles = group / 64;
  p.alpha = alpha;
  p.M = a.size(0);
  p.N = w.size(0);
  p.K = a.size(1);
  if (p.M == 0) return;
  TORCH_CHECK(epi != EPI_SILU_Q8 || (qs && out.scalar_type() == at::kChar), "gemm_i8: SILU_Q8 needs int8 out and qs");
  TORCH_CHECK(epi != EPI_RESID || resid, "gemm_i8: residual epilogue needs resid");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  switch (epi) {
    case EPI_BIAS: dispatch<EPI_BIAS>(cfg, p, s); break;
    case EPI_RESID: dispatch<EPI_RESID>(cfg, p, s); break;
    case EPI_GLU: dispatch<EPI_GLU>(cfg, p, s); break;
    case EPI_SILU_Q8: dispatch<EPI_SILU_Q8>(cfg, p, s); break;
    default: TORCH_CHECK(false, "gemm_i8: unsupported epilogue ", epi);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
