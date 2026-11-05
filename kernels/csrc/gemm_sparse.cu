// 2:4 sparse int8 GEMM for the encoder feed-forwards' second linear layers (FFN_SPARSE): out = r + alpha * bf16(
// dequant(X W^T) + bias), W pruned 2:4 along K (kernels/sparse.py) and stored compressed, X int8 with fp32 scales per
// row and K-group (the feed-forwards' first GEMM writes them), as gemm_int8.cu's grouped kernel with EPI_RESID.
//
// mma.sp takes the sparse operand as A, so the kernel computes (W X^T)^T: A = W (channels x K, half its bytes, a
// 4-bit selector per 4 values: the indices of the two kept), B = X^T (X's rows are B's columns, contiguous in K). A
// m16n8k64 does twice a m16n8k32's work. Layouts on an A10, measured with a one-warp probe: the compressed 16 x 32 A
// fragment is a dense m16n8k32 A fragment; B: lane (g, t) holds K 16 i + 4 t .. + 3 of column g in register i; the
// metadata register of lane (g, t) holds the selectors of row g + 8 (t & 1), groups 8 (t >> 1) .. + 7 of the 16 in
// the k64 step, low nibble first, each the index of the first kept value in its low 2 bits and of the second in the
// high 2.
//
// The int32 sums of each K-group are exact and folded into fp32 in order, the epilogue rounds as gemm_int8.cu's: the
// output equals the dense kernel's on the same (pruned) weights.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.cuh"

namespace {

struct SpArgs {
  const int8_t* X;      // (M, K)
  const float* sx;      // (M, K / group)
  const int8_t* W;      // (N, K / 2) the kept values, in order
  const uint32_t* E;    // (N, K / 32) selectors, 8 groups of 4 per word
  const float* sw;      // (N)
  const __nv_bfloat16* bias;
  const __nv_bfloat16* R;  // (M, N) residual
  __nv_bfloat16* C;        // (M, N)
  int ldx, ldsx, ldr, ldc;
  int group_tiles;  // K-tiles of 128 per X scale group
  float alpha;
  int M, N, K;
};

DEVI void mma_sp(int* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
  asm volatile(
      "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
      "{%0,%1,%2,%3}, %12, 0x0;\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(e));
}

DEVI void cp_async4(void* smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_u32(smem)), "l"(gmem));
}

DEVI float i2f_small(int x) { return __int_as_float(x + 0x4B400000) - 12582912.f; }  // |x| < 2^22

DEVI int sw64(int row, int ch) { return row * 64 + ((ch ^ ((row >> 1) & 3)) << 4); }    // 4 chunks of 16 B a row
DEVI int sw128(int row, int ch) { return row * 128 + ((ch ^ (row & 7)) << 4); }        // 8 chunks of 16 B a row

// a CTA: BC channels x BT tokens, WC x WT warps; a stage: 128 K (64 compressed bytes and 4 selector words a channel,
// 128 bytes a token)
template <int BC, int BT, int WC, int WT, int STAGES>
__global__ void __launch_bounds__(WC* WT * 32) gemm_sp_kernel(const SpArgs p) {
  constexpr int BK = 128, THREADS = WC * WT * 32;
  constexpr int TC = BC / WC, TT = BT / WT, MI = TC / 16, NI = TT / 8;
  constexpr int W_BYTES = BC * 64, E_BYTES = BC * 16, X_BYTES = BT * 128, STAGE = W_BYTES + E_BYTES + X_BYTES;
  static_assert(MI >= 1 && NI >= 1 && (BC * 4) % THREADS == 0 && (BT * 8) % THREADS == 0, "tile shape");
  extern __shared__ __align__(128) unsigned char smem[];
  // the tokens' scales of a K-group, loaded with the group's first stage, two groups' worth (after the stages)
  float* s_scale = reinterpret_cast<float*>(smem + STAGES * STAGE);

  const int num_c = p.N / BC, num_t = (p.M + BT - 1) / BT;
  const int pid = blockIdx.x, width = 8 * num_c;
  const int first_t = (pid / width) * 8, gsize = min(num_t - first_t, 8);
  const int t0 = (first_t + (pid % width) % gsize) * BT, c0 = ((pid % width) / gsize) * BC;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wc = warp / WT, wt = warp % WT;
  const int g = lane >> 2, t = lane & 3;

  auto load_stage = [&](int stage, int kt) {
    unsigned char* s = smem + stage * STAGE;
    for (int i = tid; i < BC * 4; i += THREADS) {  // compressed weights: 64 bytes a channel
      const int r = i >> 2, ch = i & 3;
      cp_async16(s + sw64(r, ch), p.W + (size_t)(c0 + r) * (p.K / 2) + kt * 64 + ch * 16, true);
    }
    for (int i = tid; i < BC; i += THREADS)  // selectors: 4 words a channel
      cp_async16(s + W_BYTES + i * 16, p.E + (size_t)(c0 + i) * (p.K / 32) + kt * 4, true);
    for (int i = tid; i < BT * 8; i += THREADS) {  // activations: 128 bytes a token
      const int r = i >> 3, ch = i & 7, row = t0 + r;
      const bool ok = row < p.M;
      cp_async16(s + W_BYTES + E_BYTES + sw128(r, ch), p.X + (size_t)(ok ? row : 0) * p.ldx + kt * BK + ch * 16, ok);
    }
    if (kt % p.group_tiles == 0) {
      const int grp = kt / p.group_tiles;
      for (int i = tid; i < BT; i += THREADS)
        cp_async4(s_scale + (grp & 1) * BT + i, p.sx + (size_t)min(t0 + i, p.M - 1) * p.ldsx + grp);
    }
  };

  int acc[MI][NI][4];
  float accf[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0, accf[i][j][k] = 0.f;

  auto fold = [&](int grp) {  // each token's (column's) scale of the group
    const float* sc = s_scale + (grp & 1) * BT;
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int tok = wt * TT + ni * 8 + 2 * t;
      const float s0 = sc[tok], s1 = sc[tok + 1];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        accf[mi][ni][0] = __fmaf_rn(i2f_small(acc[mi][ni][0]), s0, accf[mi][ni][0]);
        accf[mi][ni][1] = __fmaf_rn(i2f_small(acc[mi][ni][1]), s1, accf[mi][ni][1]);
        accf[mi][ni][2] = __fmaf_rn(i2f_small(acc[mi][ni][2]), s0, accf[mi][ni][2]);
        accf[mi][ni][3] = __fmaf_rn(i2f_small(acc[mi][ni][3]), s1, accf[mi][ni][3]);
        acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0;
      }
    }
  };

  const int KT = p.K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_stage(s, s);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (kt + STAGES - 1 < KT) load_stage((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
    cp_async_commit();
    const unsigned char* s = smem + (kt % STAGES) * STAGE;
    const uint32_t* se = reinterpret_cast<const uint32_t*>(s + W_BYTES);
    const unsigned char* sx = s + W_BYTES + E_BYTES;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {  // two k64 steps a stage
      uint32_t a[MI][4], e[MI], b[NI][4];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r = wc * TC + mi * 16;
        ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], s + sw64(r + (lane & 15), kk * 2 + (lane >> 4)));
        e[mi] = se[(r + g + 8 * (t & 1)) * 4 + kk * 2 + (t >> 1)];
      }
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
        ldmatrix_x4(b[ni][0], b[ni][1], b[ni][2], b[ni][3],
                    sx + sw128(wt * TT + ni * 8 + (lane & 7), kk * 4 + (lane >> 3)));
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_sp(acc[mi][ni], a[mi], b[ni], e[mi]);
    }
    if ((kt + 1) % p.group_tiles == 0) fold(kt / p.group_tiles);
  }
  cp_async_wait<0>();
  __syncthreads();

  // epilogue: v = accf * sw + bias (one FMA), bf16, staged per token in shared memory; out = r + alpha * v
  constexpr int LDS = BC + 8;
  __nv_bfloat16* sC = reinterpret_cast<__nv_bfloat16*>(smem);  // BT x LDS
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int c = wc * TC + mi * 16 + g + half * 8;
      const float w = p.sw[c0 + c], bb = p.bias ? bf(p.bias[c0 + c]) : 0.f;
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int tok = wt * TT + ni * 8 + 2 * t + j;
          const float x = accf[mi][ni][half * 2 + j];
          const float v = p.bias ? __fmaf_rn(x, w, bb) : __fmul_rn(x, w);
          sC[tok * LDS + c] = tobf(v);
        }
    }
  __syncthreads();
  constexpr int CH = BC / 8;
  for (int id = tid; id < BT * CH; id += THREADS) {
    const int r = id / CH, c = (id % CH) * 8, row = t0 + r;
    if (row >= p.M) continue;
    const uint4 yv = *reinterpret_cast<const uint4*>(sC + r * LDS + c);
    const uint4 rv = *reinterpret_cast<const uint4*>(p.R + (size_t)row * p.ldr + c0 + c);
    const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(&yv);
    const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&rv);
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      o[k] = pack_bf16(__fmaf_rn(p.alpha, __low2float(y2[k]), __low2float(r2[k])),
                       __fmaf_rn(p.alpha, __high2float(y2[k]), __high2float(r2[k])));
    *reinterpret_cast<uint4*>(p.C + (size_t)row * p.ldc + c0 + c) = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

template <int BC, int BT, int WC, int WT, int STAGES>
void launch(const SpArgs& p, cudaStream_t stream) {
  constexpr int stage = BC * 64 + BC * 16 + BT * 128, pipe = STAGES * stage + 2 * BT * 4, out = BT * (BC + 8) * 2;
  constexpr int smem = pipe > out ? pipe : out;
  auto kernel = gemm_sp_kernel<BC, BT, WC, WT, STAGES>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  TORCH_CHECK(p.N % BC == 0, "gemm_sp: N=", p.N, " not a multiple of ", BC);
  kernel<<<((p.M + BT - 1) / BT) * (p.N / BC), WC * WT * 32, smem, stream>>>(p);
}

}  // namespace

// out = r + alpha * bf16(dequant(x @ w^T) + bias): x int8 (M, K) with sx (M, K / group) fp32 (group a multiple of
// 128), w 2:4 sparse as wc (N, K / 2) int8 kept values and e (N, K / 32) int32 selectors (kernels/sparse.py), sw (N)
// fp32. cfg 0: 128 x 128 tiles of 8 warps; 1: 128 channels x 64 tokens; 2: 64 x 32 (few rows)
void gemm_sp(const at::Tensor& x, const at::Tensor& sx, int64_t group, const at::Tensor& wc, const at::Tensor& e,
             const at::Tensor& sw, const c10::optional<at::Tensor>& bias, const at::Tensor& r, at::Tensor& out,
             double alpha, int64_t cfg) {
  TORCH_CHECK(x.scalar_type() == at::kChar && wc.scalar_type() == at::kChar && e.scalar_type() == at::kInt &&
              sx.scalar_type() == at::kFloat && sw.scalar_type() == at::kFloat);
  TORCH_CHECK(x.stride(1) == 1 && wc.is_contiguous() && e.is_contiguous() && sx.stride(1) == 1 && r.stride(1) == 1 &&
              out.stride(1) == 1);
  const int64_t K = x.size(1);
  TORCH_CHECK(wc.size(1) * 2 == K && e.size(1) * 32 == K && K % 128 == 0 && group % 128 == 0 && K % group == 0);
  SpArgs p;
  p.X = x.data_ptr<int8_t>();
  p.sx = sx.data_ptr<float>();
  p.W = wc.data_ptr<int8_t>();
  p.E = reinterpret_cast<const uint32_t*>(e.data_ptr<int>());
  p.sw = sw.data_ptr<float>();
  p.bias = bias ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr;
  p.R = reinterpret_cast<const __nv_bfloat16*>(r.data_ptr());
  p.C = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ldx = x.stride(0);
  p.ldsx = sx.stride(0);
  p.ldr = r.stride(0);
  p.ldc = out.stride(0);
  p.group_tiles = group / 128;
  p.alpha = alpha;
  p.M = x.size(0);
  p.N = wc.size(0);
  p.K = K;
  if (p.M == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  switch (cfg) {
    case 0: launch<128, 128, 4, 2, 3>(p, s); break;
    case 1: launch<128, 64, 4, 2, 3>(p, s); break;
    case 2: launch<64, 32, 2, 2, 3>(p, s); break;
    case 3: launch<128, 128, 2, 4, 3>(p, s); break;
    case 4: launch<256, 64, 4, 2, 3>(p, s); break;
    default: TORCH_CHECK(false, "gemm_sp: unknown config ", cfg);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
