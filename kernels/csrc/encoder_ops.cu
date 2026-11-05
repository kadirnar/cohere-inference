// The FastConformer encoder's non-GEMM operations on packed variable-length batches (all sequences' frames
// concatenated, no padding): LayerNorm (single and two in a row), the convolution module's depthwise convolution with
// BatchNorm and SiLU, and relative-position self-attention.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.cuh"
#include "layernorm.cuh"

namespace {

// ---- LayerNorm ------------------------------------------------------------------------------------------------------
// One warp per row of N = 32 * 8 * VEC features, 16-byte loads; fp32 statistics, output rounded to bf16 as
// nn.LayerNorm in bf16 does.

// the bf16 row out (VEC chunks per lane) as int8 with one scale per row (absmax / 127), as the next int8 GEMM reads it
template <int VEC>
DEVI void store_q8(const uint4 (&out)[VEC], int lane, int8_t* q, float* scale) {
  float mx = 0.f;
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&out[i]);
#pragma unroll
    for (int k = 0; k < 8; ++k) mx = fmaxf(mx, fabsf(bf(e[k])));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  const float s = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / s;
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&out[i]);
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int v = __float2int_rn(fminf(fmaxf(bf(e[k]) * inv, -127.f), 127.f));
      w[k >> 2] |= (uint32_t)(v & 0xff) << ((k & 3) * 8);
    }
    *reinterpret_cast<uint2*>(q + (i * 32 + lane) * 8) = make_uint2(w[0], w[1]);
  }
  if (lane == 0) *scale = s;
}

// x (rows, 1280) bf16 as int8 with one scale per row, one warp per row (the input of an int8 GEMM)
__global__ void quantize_rows_kernel(const __nv_bfloat16* __restrict__ x, int ldx, int8_t* __restrict__ q,
                                     float* __restrict__ s, int rows) {
  constexpr int VEC = 5;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  uint4 v[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) v[i] = *reinterpret_cast<const uint4*>(x + (size_t)row * ldx + (i * 32 + lane) * 8);
  store_q8<VEC>(v, lane, q + (size_t)row * 1280, s + row);
}

// h = LN(x); with TWO: x = LN1(x) (in place), h = LN2(x); with Q8, h is written as int8 hq with scale hs per row
template <int VEC, bool TWO, bool Q8 = false>
__global__ void layernorm_kernel(__nv_bfloat16* x, int ldx, const __nv_bfloat16* w1, const __nv_bfloat16* b1,
                                 float eps1, const __nv_bfloat16* w2, const __nv_bfloat16* b2, float eps2,
                                 __nv_bfloat16* h, int ldh, int rows, int8_t* hq = nullptr, float* hs = nullptr) {
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  float v[VEC][8];
  uint4 out[VEC];
  load_row<VEC>(x + (size_t)row * ldx, lane, v);
  ln_row<VEC>(v, w1, b1, eps1, lane, out);
  if constexpr (TWO) {
#pragma unroll
    for (int i = 0; i < VEC; ++i) *reinterpret_cast<uint4*>(x + (size_t)row * ldx + (i * 32 + lane) * 8) = out[i];
    unpack_row<VEC>(out, v);
    ln_row<VEC>(v, w2, b2, eps2, lane, out);
  }
  if constexpr (Q8) {
    store_q8<VEC>(out, lane, hq + (size_t)row * ldh, hs + row);
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) *reinterpret_cast<uint4*>(h + (size_t)row * ldh + (i * 32 + lane) * 8) = out[i];
  }
}

// ---- relative-position self-attention --------------------------------------------------------------------------------
// Transformer-XL attention of the FastConformer (ParakeetEncoderAttention) for one head and 64 query rows of one
// sequence per CTA (4 warps x 16 rows), head size 160:
//   score[i, j] = ((q_i + u) . k_j) * scale + bf16(bf16((q_i + v) . R[i - j]) * scale)
// with R[d] the projected sinusoid of relative distance d (a per-layer table, rows d + rel_off). Keys and values
// stream through in tiles of 32 with an online softmax (fp32), the probabilities rounded to bf16 for the value product,
// as PyTorch's attention kernels do; the position term (bf16, rounded as the module rounds it) is computed 32
// distances at a time as the key tiles need it, into a ring in shared memory. (Computed for all distances up front it
// took (len + 63) x 64 x 2 bytes: one CTA an SM for sequences over 150 frames, a third slower.)
constexpr int AD = 160;       // head size
constexpr int ADP = AD + 8;   // padded shared-memory row: ldmatrix rows fall in distinct banks
constexpr int ABM = 64;       // query rows per CTA
constexpr int ABN = 32;       // keys per tile
constexpr int AKS = AD / 16;  // k16 steps over the head size
constexpr int RING_LD = 4 * ABN + 8;  // a row of the position-term ring (4 chunks), padded off bank alignment

struct AttnArgs {
  const __nv_bfloat16* qkv;  // (rows, 3 * H * AD): q | k | v
  const __nv_bfloat16* rk;   // (table rows, H * AD)
  const __nv_bfloat16* bu;   // (H, AD)
  const __nv_bfloat16* bv;   // (H, AD)
  __nv_bfloat16* out;        // (rows, H * AD)
  const int4* tiles;         // (seq start, seq length, first query row, 0)
  int H, rel_off, rel_rows;
  float scale;
};

// load 64 rows x 160 of q (+ bias) into s as bf16(q + bias); rows past the sequence are zero
DEVI void load_q_plus(const AttnArgs& p, int start, int len, int q0, int h, const __nv_bfloat16* bias,
                      __nv_bfloat16* s) {
  const int ld = 3 * p.H * AD;
  for (int id = threadIdx.x; id < ABM * (AD / 8); id += blockDim.x) {
    const int r = id / (AD / 8), c = (id % (AD / 8)) * 8;
    uint4 u = make_uint4(0, 0, 0, 0);
    if (q0 + r < len) {
      const uint4 q = *reinterpret_cast<const uint4*>(p.qkv + (size_t)(start + q0 + r) * ld + h * AD + c);
      const uint4 b = *reinterpret_cast<const uint4*>(bias + h * AD + c);
      const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&q);
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
      uint32_t o[4];
#pragma unroll
      for (int k = 0; k < 4; ++k)
        o[k] = pack_bf16(__low2float(q2[k]) + __low2float(b2[k]), __high2float(q2[k]) + __high2float(b2[k]));
      u = make_uint4(o[0], o[1], o[2], o[3]);
    }
    *reinterpret_cast<uint4*>(s + r * ADP + c) = u;
  }
}

DEVI void load_a_frags(const __nv_bfloat16* s, int warp, int lane, uint32_t (&a)[AKS][4]) {
#pragma unroll
  for (int ks = 0; ks < AKS; ++ks)
    ldmatrix_x4(a[ks][0], a[ks][1], a[ks][2], a[ks][3],
                s + (warp * 16 + (lane & 15)) * ADP + ks * 16 + (lane >> 4) * 8);
}

// rows [r0, r0 + 32) of a (rows, ld) bf16 matrix, columns col0..col0+159, into s (zero past n_valid rows)
DEVI void cp_tile32(const __nv_bfloat16* src, int ld, int col0, int r0, int n_valid, __nv_bfloat16* s) {
  for (int id = threadIdx.x; id < ABN * (AD / 8); id += blockDim.x) {
    const int r = id / (AD / 8), c = (id % (AD / 8)) * 8;
    const bool ok = r0 + r < n_valid;
    cp_async16(s + r * ADP + c, src + (size_t)(ok ? r0 + r : 0) * ld + col0 + c, ok);
  }
}

// the position term of the tile's 64 rows for 32 distances (the chunk whose 32 table rows sR holds) into ring slot
// `slot` of sP, each warp its own 16 rows: bf16(bf16((q + v) . R[d]) * scale), as the module rounds it
DEVI void position_chunk(const uint32_t (&a)[AKS][4], const __nv_bfloat16* sR, __nv_bfloat16* sP, int warp, int lane,
                         float scale, int slot) {
  const int g = lane >> 2, t = lane & 3;
  float acc[4][4] = {};
#pragma unroll
  for (int ks = 0; ks < AKS; ++ks) {
#pragma unroll
    for (int nj = 0; nj < 2; ++nj) {
      uint32_t b0, b1, b2, b3;
      ldmatrix_x4(b0, b1, b2, b3, sR + (nj * 16 + ((lane >> 4) << 3) + (lane & 7)) * ADP + ks * 16 + ((lane >> 3) & 1) * 8);
      mma_bf16(acc[2 * nj], a[ks], b0, b1);
      mma_bf16(acc[2 * nj + 1], a[ks], b2, b3);
    }
  }
#pragma unroll
  for (int ni = 0; ni < 4; ++ni)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = warp * 16 + g + half * 8, c = slot * ABN + ni * 8 + 2 * t;
      *reinterpret_cast<uint32_t*>(sP + r * RING_LD + c) =
          pack_bf16(round_bf(acc[ni][half * 2]) * scale, round_bf(acc[ni][half * 2 + 1]) * scale);
    }
}

__global__ void __launch_bounds__(128, 2) relpos_attention_kernel(const AttnArgs p) {
  // the query tile is staged where the key, value and distance-table tiles go afterwards; the position term lives in a
  // ring of 4 chunks of 32 distances (a key tile's scores read 95 consecutive distances, one chunk further down per
  // tile), so the shared memory does not grow with the sequence and two CTAs fit an SM
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sQ = sA;                        // 64 x 168 (staging)
  __nv_bfloat16* sK = sA;                        // 32 x 168
  __nv_bfloat16* sV = sA + ABN * ADP;            // 32 x 168
  __nv_bfloat16* sR = sA + 2 * ABN * ADP;        // 32 x 168
  __nv_bfloat16* sP = sA + 3 * ABN * ADP;        // 64 x RING_LD

  const int4 tile = p.tiles[blockIdx.x];
  const int start = tile.x, len = tile.y, q0 = tile.z, h = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int ld = 3 * p.H * AD;
  // distances i - j of this tile: q0 - (len - 1) ... q0 + 63, column c = d - dmin of the position term
  const int dmin = q0 - len + 1;
  const int n_chunks = (len + ABM - 1 + ABN - 1) / ABN;
  auto load_r = [&](int chunk, __nv_bfloat16* dst) {
    for (int id = tid; id < ABN * (AD / 8); id += blockDim.x) {
      const int r = id / (AD / 8), c = (id % (AD / 8)) * 8;
      const int row = min(max(dmin + chunk * ABN + r + p.rel_off, 0), p.rel_rows - 1);
      cp_async16(dst + r * ADP + c, p.rk + (size_t)row * (p.H * AD) + h * AD + c, true);
    }
  };

  uint32_t pa[AKS][4], qa[AKS][4];  // (q + v) for the position term, (q + u) for the content term
  load_q_plus(p, start, len, q0, h, p.bv, sQ);
  __syncthreads();
  load_a_frags(sQ, warp, lane, pa);
  __syncthreads();
  load_q_plus(p, start, len, q0, h, p.bu, sQ);
  __syncthreads();
  load_a_frags(sQ, warp, lane, qa);
  __syncthreads();

  // the first key tile reads columns len - 32 ... len + 62: chunks lo0 ... n_chunks - 1 (at most 4), up to 3 at a time
  // (the key and value tiles' space is free until the key loop)
  const int lo0 = max(0, (len - ABN) / ABN);
  for (int c = n_chunks - 1; c >= lo0; c -= 3) {
    __nv_bfloat16* buf[3] = {sR, sK, sV};
    const int n = min(3, c - lo0 + 1);
    for (int k = 0; k < n; ++k) load_r(c - k, buf[k]);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    for (int k = 0; k < n; ++k) position_chunk(pa, buf[k], sP, warp, lane, p.scale, (c - k) & 3);
    __syncthreads();
  }

  float o[AD / 8][4];
#pragma unroll
  for (int i = 0; i < AD / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

  const __nv_bfloat16* kbase = p.qkv + (size_t)start * ld;
  // cp.async groups K0, V0, (K1 + the chunk tile 1 adds), V1, ...: K(j+1) loads during the softmax and the value
  // product of tile j, V(j+1) during the scores of tile j+1
  cp_tile32(kbase, ld, p.H * AD + h * AD, 0, len, sK);
  cp_async_commit();
  cp_tile32(kbase, ld, 2 * p.H * AD + h * AD, 0, len, sV);
  cp_async_commit();
  for (int jt = 0, j0 = 0; j0 < len; ++jt, j0 += ABN) {
    const bool more = j0 + ABN < len;
    cp_async_wait<1>();
    __syncthreads();
    if (jt > 0 && lo0 - jt >= 0) position_chunk(pa, sR, sP, warp, lane, p.scale, (lo0 - jt) & 3);
    float s[4][4] = {};
#pragma unroll
    for (int ks = 0; ks < AKS; ++ks) {
#pragma unroll
      for (int nj = 0; nj < 2; ++nj) {
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4(b0, b1, b2, b3,
                    sK + (nj * 16 + ((lane >> 4) << 3) + (lane & 7)) * ADP + ks * 16 + ((lane >> 3) & 1) * 8);
        mma_bf16(s[2 * nj], qa[ks], b0, b1);
        mma_bf16(s[2 * nj + 1], qa[ks], b2, b3);
      }
    }
    __syncthreads();  // every warp is done with K(j) and the table chunk
    if (more) {
      cp_tile32(kbase, ld, p.H * AD + h * AD, j0 + ABN, len, sK);
      if (lo0 - jt - 1 >= 0) load_r(lo0 - jt - 1, sR);
    }
    cp_async_commit();
    __syncwarp();  // the warp's ring rows, written by its lanes, read by the others
    // scores: content * scale + position term (column i - j - dmin), keys past the sequence masked
    float mx[2] = {m[0], m[1]};
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int half = e >> 1, r = warp * 16 + g + half * 8, j = j0 + ni * 8 + 2 * t + (e & 1);
        float v = -INFINITY;
        if (j < len) {
          const int c = q0 + r - j - dmin;
          v = s[ni][e] * p.scale + bf(sP[r * RING_LD + (((c / ABN) & 3) * ABN) + (c % ABN)]);
        }
        s[ni][e] = v;
        mx[half] = fmaxf(mx[half], v);
      }
    float alpha[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      mx[half] = fmaxf(mx[half], __shfl_xor_sync(0xffffffff, mx[half], 1));
      mx[half] = fmaxf(mx[half], __shfl_xor_sync(0xffffffff, mx[half], 2));
      alpha[half] = expf(m[half] - mx[half]);
      m[half] = mx[half];
    }
    float rs[2] = {0.f, 0.f};
    uint32_t pr[2][4];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const float p0 = expf(s[ni][0] - m[0]), p1 = expf(s[ni][1] - m[0]);
      const float p2 = expf(s[ni][2] - m[1]), p3 = expf(s[ni][3] - m[1]);
      rs[0] += p0 + p1;
      rs[1] += p2 + p3;
      pr[ni >> 1][(ni & 1) * 2] = pack_bf16(p0, p1);
      pr[ni >> 1][(ni & 1) * 2 + 1] = pack_bf16(p2, p3);
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      rs[half] += __shfl_xor_sync(0xffffffff, rs[half], 1);
      rs[half] += __shfl_xor_sync(0xffffffff, rs[half], 2);
      l[half] = l[half] * alpha[half] + rs[half];
    }
#pragma unroll
    for (int i = 0; i < AD / 8; ++i) {
      o[i][0] *= alpha[0];
      o[i][1] *= alpha[0];
      o[i][2] *= alpha[1];
      o[i][3] *= alpha[1];
    }
    cp_async_wait<1>();
    __syncthreads();  // V(j) landed
    // o += p v: v rows are keys, read transposed as the col operand
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
      for (int nj = 0; nj < AD / 16; ++nj) {
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4_trans(b0, b1, b2, b3,
                          sV + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * ADP + nj * 16 + (lane >> 4) * 8);
        mma_bf16(o[2 * nj], pr[kk], b0, b1);
        mma_bf16(o[2 * nj + 1], pr[kk], b2, b3);
      }
    }
    __syncthreads();  // every warp is done with V(j)
    if (more) cp_tile32(kbase, ld, 2 * p.H * AD + h * AD, j0 + ABN, len, sV);
    cp_async_commit();
  }
  cp_async_wait<0>();

  const float inv[2] = {1.f / l[0], 1.f / l[1]};
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int r = q0 + warp * 16 + g + half * 8;
    if (r >= len) continue;
    __nv_bfloat16* dst = p.out + (size_t)(start + r) * (p.H * AD) + h * AD;
#pragma unroll
    for (int i = 0; i < AD / 8; ++i)
      *reinterpret_cast<uint32_t*>(dst + i * 8 + 2 * t) =
          pack_bf16(o[i][half * 2] * inv[half], o[i][half * 2 + 1] * inv[half]);
  }
}

// ---- subsampling (dw_striding, 8x) on unpadded sequences ------------------------------------------------------------
// ParakeetEncoderSubsamplingConv2D runs on the padded batch and masks the padded frames after every convolution;
// here each sequence is convolved on its own (zero padding at its ends is what the masks amount to), channels-last.
// Roundings as PyTorch's: cuDNN convolutions round before their bias is added (conv0, the pointwise ones), ATen's
// depthwise kernel adds the bias in fp32 before its single rounding.

// conv0 (1 -> C, 3x3, stride 2, pad 1) + bias + ReLU, then the depthwise 3x3 stride-2 convolution of it + bias, for
// one output row t2 of one sequence per CTA, one channel per thread: out (rows2, F2, C) channels-last.
__global__ void sub_conv0_dw_kernel(const __nv_bfloat16* __restrict__ feats, int ld_seq, const int* __restrict__ mel_len,
                                    const int2* __restrict__ rows, const float* __restrict__ w0,
                                    const float* __restrict__ b0, const float* __restrict__ w2,
                                    const float* __restrict__ b2, __nv_bfloat16* __restrict__ out, int F) {
  constexpr int MELS = 128, F1 = 64, F2 = 32, PADL = 4, LDM = PADL + MELS + 4;
  // rows zero-padded by 4 on both sides: a 9-tap sum reads its 3 x 3 window with float4 loads (a zero term adds
  // nothing to the fp32 sum, so the padded taps change no value)
  __shared__ __align__(16) float mel[7][LDM];
  const int2 rw = rows[blockIdx.x];  // (sequence, t2)
  const int seq = rw.x, t2 = rw.y, c = threadIdx.x, C = blockDim.x;
  const int L = mel_len[seq], T1 = (L - 1) / 2 + 1;
  const __nv_bfloat16* x = feats + (size_t)seq * ld_seq;
  for (int i = threadIdx.x; i < 7 * LDM; i += C) {
    const int r = 4 * t2 - 3 + i / LDM, m = i % LDM - PADL;
    mel[i / LDM][i % LDM] = (r >= 0 && r < L && m >= 0 && m < MELS) ? bf(x[(size_t)r * MELS + m]) : 0.f;
  }
  __syncthreads();
  float wa[9], wd[9];
#pragma unroll
  for (int k = 0; k < 9; ++k) {
    wa[k] = w0[c * 9 + k];
    wd[k] = w2[c * 9 + k];
  }
  const float ba = b0[c];
  float acc[F2];
#pragma unroll
  for (int f = 0; f < F2; ++f) acc[f] = b2[c];
#pragma unroll
  for (int r = 0; r < 3; ++r) {  // conv0 row t1 = 2 t2 - 1 + r feeds tap row r of the depthwise convolution
    const int t1 = 2 * t2 - 1 + r;
    if (t1 < 0 || t1 >= T1) continue;
#pragma unroll
    for (int fc = 0; fc < F1; fc += 8) {  // conv0 columns fc .. fc + 7 read mel columns 2 fc - 1 .. 2 fc + 15
      float m[3][20];
#pragma unroll
      for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int q4 = 0; q4 < 5; ++q4) {
          const float4 v4 = *reinterpret_cast<const float4*>(&mel[2 * r + i][PADL + 2 * fc - 4 + 4 * q4]);
          m[i][4 * q4] = v4.x;
          m[i][4 * q4 + 1] = v4.y;
          m[i][4 * q4 + 2] = v4.z;
          m[i][4 * q4 + 3] = v4.w;
        }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int f1 = fc + u;
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
          for (int j = 0; j < 3; ++j) v = __fmaf_rn(wa[i * 3 + j], m[i][2 * u + 3 + j], v);  // column 2 f1 - 1 + j
        v = fmaxf(round_bf(round_bf(v) + ba), 0.f);
        // depthwise output f2 reads conv0 column 2 f2 - 1 + j: f1 even -> (f2 = f1/2, j = 1); odd -> j = 0 and j = 2
        if (f1 & 1) {
          const int fa = (f1 + 1) / 2, fb = (f1 - 1) / 2;
          if (fa < F2) acc[fa] = __fmaf_rn(wd[r * 3 + 0], v, acc[fa]);
          acc[fb] = __fmaf_rn(wd[r * 3 + 2], v, acc[fb]);
        } else {
          acc[f1 / 2] = __fmaf_rn(wd[r * 3 + 1], v, acc[f1 / 2]);
        }
      }
    }
  }
  __nv_bfloat16* o = out + (size_t)blockIdx.x * F2 * C + c;
#pragma unroll
  for (int f = 0; f < F2; ++f) o[f * C] = tobf(acc[f]);
}

// depthwise 3x3 stride-2 convolution + bias on channels-last (T, F, C) sequences: out row (t, f) of rows[i] =
// (sequence, t); 8 channels per thread
__global__ void sub_dw_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ in_start,
                              const int* __restrict__ in_len, const int2* __restrict__ rows, const float* __restrict__ w,
                              const float* __restrict__ b, __nv_bfloat16* __restrict__ out, int n_rows, int F_in, int C) {
  const int F_out = F_in / 2, chunks = C / 8;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)n_rows * F_out * chunks) return;
  const int c = (idx % chunks) * 8, f = (idx / chunks) % F_out, i = idx / chunks / F_out;
  const int2 rw = rows[i];
  const int start = in_start[rw.x], len = in_len[rw.x], t = rw.y;
  float acc[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = b[c + k];
#pragma unroll
  for (int di = 0; di < 3; ++di) {
    const int ti = 2 * t - 1 + di;
    if (ti < 0 || ti >= len) continue;
#pragma unroll
    for (int dj = 0; dj < 3; ++dj) {
      const int fi = 2 * f - 1 + dj;
      if (fi < 0 || fi >= F_in) continue;
      const uint4 u = *reinterpret_cast<const uint4*>(x + ((size_t)(start + ti) * F_in + fi) * C + c);
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[k] += w[(c + k) * 9 + di * 3 + dj] * bf(e[k]);
    }
  }
  uint32_t o[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) o[k] = pack_bf16(acc[2 * k], acc[2 * k + 1]);
  *reinterpret_cast<uint4*>(out + ((size_t)i * F_out + f) * C + c) = make_uint4(o[0], o[1], o[2], o[3]);
}

// ---- convolution module's depthwise convolution, tiled -------------------------------------------------------------
// 32 frames x 64 channels per CTA: the 40 input frames it needs and the tile's weights staged in shared memory, each
// thread then computes 8 channels of one frame
constexpr int DW_T = 32, DW_C = 64;
constexpr int DWQ_ROWS = 4;  // rows per CTA of dwconv_q8_kernel
__global__ void __launch_bounds__(256) dwconv_tile_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
                                                          const float* __restrict__ cb, const float* __restrict__ bn_scale,
                                                          const float* __restrict__ bn_shift,
                                                          const int* __restrict__ seq_lo, const int* __restrict__ seq_hi,
                                                          __nv_bfloat16* __restrict__ y, int rows, int C) {
  constexpr int K = 9, PAD = 4;
  __shared__ __align__(16) __nv_bfloat16 sx[DW_T + K - 1][DW_C];
  __shared__ float sw[DW_C][K + 1];
  const int r0 = blockIdx.x * DW_T, c0 = blockIdx.y * DW_C, tid = threadIdx.x;
  for (int i = tid; i < (DW_T + K - 1) * (DW_C / 8); i += blockDim.x) {
    const int r = i / (DW_C / 8), ch = (i % (DW_C / 8)) * 8, gr = r0 + r - PAD;
    uint4 u = make_uint4(0, 0, 0, 0);
    if (gr >= 0 && gr < rows) u = *reinterpret_cast<const uint4*>(x + (size_t)gr * C + c0 + ch);
    *reinterpret_cast<uint4*>(&sx[r][ch]) = u;
  }
  for (int i = tid; i < DW_C * K; i += blockDim.x) sw[i / K][i % K] = w[(c0 + i / K) * K + i % K];
  __syncthreads();
  const int r = tid / (DW_C / 8), ch = (tid % (DW_C / 8)) * 8, row = r0 + r;
  if (row >= rows) return;
  const int lo = seq_lo[row], hi = seq_hi[row];
  float acc[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = cb[c0 + ch + k];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int src = row + k - PAD;
    if (src < lo || src >= hi) continue;
    const uint4 u = *reinterpret_cast<const uint4*>(&sx[r + k][ch]);
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] += bf(e[j]) * sw[ch + j][k];
  }
  uint32_t o[4];
#pragma unroll
  for (int j = 0; j < 8; j += 2) {
    const int c = c0 + ch + j;
    float a = round_bf(acc[j]), b = round_bf(acc[j + 1]);
    a = round_bf(a * bn_scale[c] + bn_shift[c]);
    b = round_bf(b * bn_scale[c + 1] + bn_shift[c + 1]);
    o[j / 2] = pack_bf16(a * sigmoidf_(a), b * sigmoidf_(b));
  }
  *reinterpret_cast<uint4*>(y + (size_t)row * C + c0 + ch) = make_uint4(o[0], o[1], o[2], o[3]);
}

}  // namespace

void layernorm(at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1, double eps1,
               const c10::optional<at::Tensor>& w2, const c10::optional<at::Tensor>& b2, double eps2, at::Tensor& h) {
  const int rows = x.size(0), n = x.size(1);
  TORCH_CHECK(x.stride(1) == 1 && h.stride(1) == 1 && n == 1280, "layernorm: (rows, 1280) bf16");
  if (rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  auto P = [](const at::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  const int warps = 4, grid = (rows + warps - 1) / warps;
  auto xp = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
  auto hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr());
  if (w2)
    layernorm_kernel<5, true><<<grid, warps * 32, 0, s>>>(xp, x.stride(0), P(w1), P(b1), eps1, P(*w2), P(*b2), eps2,
                                                          hp, h.stride(0), rows);
  else
    layernorm_kernel<5, false><<<grid, warps * 32, 0, s>>>(xp, x.stride(0), P(w1), P(b1), eps1, nullptr, nullptr, 0.f,
                                                           hp, h.stride(0), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// as layernorm, h written as int8 hq (rows, 1280) with fp32 scales hs (rows)
void layernorm_q8(at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1, double eps1,
                  const c10::optional<at::Tensor>& w2, const c10::optional<at::Tensor>& b2, double eps2, at::Tensor& hq,
                  at::Tensor& hs) {
  const int rows = x.size(0), n = x.size(1);
  TORCH_CHECK(x.stride(1) == 1 && hq.is_contiguous() && hq.scalar_type() == at::kChar && n == 1280 &&
              hs.scalar_type() == at::kFloat);
  if (rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  auto P = [](const at::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  const int warps = 4, grid = (rows + warps - 1) / warps;
  auto xp = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
  if (w2)
    layernorm_kernel<5, true, true><<<grid, warps * 32, 0, s>>>(xp, x.stride(0), P(w1), P(b1), eps1, P(*w2), P(*b2),
                                                                eps2, nullptr, n, rows, hq.data_ptr<int8_t>(),
                                                                hs.data_ptr<float>());
  else
    layernorm_kernel<5, false, true><<<grid, warps * 32, 0, s>>>(xp, x.stride(0), P(w1), P(b1), eps1, nullptr, nullptr,
                                                                 0.f, nullptr, n, rows, hq.data_ptr<int8_t>(),
                                                                 hs.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quantize_rows(const at::Tensor& x, at::Tensor& q, at::Tensor& s) {
  const int rows = x.size(0);
  TORCH_CHECK(x.size(1) == 1280 && x.stride(1) == 1 && q.is_contiguous() && q.scalar_type() == at::kChar);
  if (rows == 0) return;
  quantize_rows_kernel<<<(rows + 3) / 4, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), q.data_ptr<int8_t>(), s.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// int8 path: the depthwise convolution, BatchNorm and SiLU of whole rows, quantized at once for the next GEMM (one
// scale per row, store_q8's arithmetic): a CTA of 160 threads holds 4 rows, each thread 8 channels with their 9 taps
// in registers and the 12 input rows it needs loaded up front. The bf16 output dwconv_tile_kernel wrote and
// quantize_rows read back is never stored (6,800 rows: 124 us against 180; more rows per CTA spill or serialize).
__global__ void __launch_bounds__(160) dwconv_q8_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
                                                        const float* __restrict__ cb, const float* __restrict__ bn_scale,
                                                        const float* __restrict__ bn_shift,
                                                        const int* __restrict__ seq_lo, const int* __restrict__ seq_hi,
                                                        int8_t* __restrict__ q, float* __restrict__ qs, int rows) {
  constexpr int R = DWQ_ROWS, K = 9, PAD = 4, C = 1280, WARPS = 5;
  __shared__ float red[WARPS][R];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = tid * 8, r0 = blockIdx.x * R;
  uint4 in[R + K - 1];
#pragma unroll
  for (int i = 0; i < R + K - 1; ++i) {
    const int gr = r0 - PAD + i;
    in[i] = gr >= 0 && gr < rows ? __ldg(reinterpret_cast<const uint4*>(x + (size_t)gr * C + c)) : make_uint4(0, 0, 0, 0);
  }
  float wk[8][K], bias[8], sc[8], sh[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#pragma unroll
    for (int k = 0; k < K; ++k) wk[j][k] = __ldg(w + (c + j) * K + k);
    bias[j] = __ldg(cb + c + j);
    sc[j] = __ldg(bn_scale + c + j);
    sh[j] = __ldg(bn_shift + c + j);
  }
  uint4 out[R];
  float mx[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int row = r0 + r;
    mx[r] = 0.f;
    out[r] = make_uint4(0, 0, 0, 0);
    if (row >= rows) continue;
    const int lo = seq_lo[row], hi = seq_hi[row];
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = bias[j];
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int src = row + k - PAD;
      if (src < lo || src >= hi) continue;
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&in[r + k]);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = __fmaf_rn(bf(e[j]), wk[j][k], acc[j]);
    }
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      float a = round_bf(acc[j]), b = round_bf(acc[j + 1]);
      a = round_bf(__fmaf_rn(a, sc[j], sh[j]));
      b = round_bf(__fmaf_rn(b, sc[j + 1], sh[j + 1]));
      o[j / 2] = pack_bf16(a * sigmoidf_(a), b * sigmoidf_(b));
    }
    out[r] = make_uint4(o[0], o[1], o[2], o[3]);
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&out[r]);
#pragma unroll
    for (int j = 0; j < 8; ++j) mx[r] = fmaxf(mx[r], fabsf(bf(e[j])));
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float m = mx[r];
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) red[warp][r] = m;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int row = r0 + r;
    if (row >= rows) break;
    float m = red[0][r];
#pragma unroll
    for (int k = 1; k < WARPS; ++k) m = fmaxf(m, red[k][r]);
    const float sv = m > 0.f ? m / 127.f : 1.f, inv = 1.f / sv;
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&out[r]);
    uint32_t v[2] = {0, 0};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int qi = __float2int_rn(fminf(fmaxf(bf(e[k]) * inv, -127.f), 127.f));
      v[k >> 2] |= (uint32_t)(qi & 0xff) << ((k & 3) * 8);
    }
    *reinterpret_cast<uint2*>(q + (size_t)row * C + c) = make_uint2(v[0], v[1]);
    if (tid == 0) qs[row] = sv;
  }
}

void dwconv_bn_silu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& cb, const at::Tensor& bn_scale,
                    const at::Tensor& bn_shift, const at::Tensor& seq_lo, const at::Tensor& seq_hi, at::Tensor& y) {
  const int rows = x.size(0), C = x.size(1), K = w.size(1);
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && C % 8 == 0 && w.scalar_type() == at::kFloat);
  if (rows == 0) return;
  TORCH_CHECK(K == 9 && C % DW_C == 0);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  dwconv_tile_kernel<<<dim3((rows + DW_T - 1) / DW_T, C / DW_C), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<float>(), cb.data_ptr<float>(),
      bn_scale.data_ptr<float>(), bn_shift.data_ptr<float>(), seq_lo.data_ptr<int>(), seq_hi.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), rows, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// hq (rows, 1280) int8, hs (rows) = quantize_rows(dwconv_bn_silu(x)) without the bf16 intermediate
void dwconv_bn_silu_q8(const at::Tensor& x, const at::Tensor& w, const at::Tensor& cb, const at::Tensor& bn_scale,
                       const at::Tensor& bn_shift, const at::Tensor& seq_lo, const at::Tensor& seq_hi, at::Tensor& hq,
                       at::Tensor& hs) {
  const int rows = x.size(0);
  TORCH_CHECK(x.is_contiguous() && x.size(1) == 1280 && w.size(1) == 9 && w.scalar_type() == at::kFloat &&
              hq.is_contiguous() && hq.scalar_type() == at::kChar);
  if (rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  dwconv_q8_kernel<<<(rows + DWQ_ROWS - 1) / DWQ_ROWS, 160, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<float>(), cb.data_ptr<float>(),
      bn_scale.data_ptr<float>(), bn_shift.data_ptr<float>(), seq_lo.data_ptr<int>(), seq_hi.data_ptr<int>(),
      hq.data_ptr<int8_t>(), hs.data_ptr<float>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out (rows, H*160) = relative-position attention of qkv (rows, 3*H*160) per sequence; tiles (n, 4) int32 lists
// (seq start, seq length, first query row, 0) for every 64-row query tile; rk the layer's distance table, row
// d + rel_off for distance d; max_len the longest sequence (checked against the table).
void relpos_attention(const at::Tensor& qkv, const at::Tensor& rk, const at::Tensor& bu, const at::Tensor& bv,
                      const at::Tensor& tiles, int64_t rel_off, int64_t max_len, double scale, at::Tensor& out) {
  TORCH_CHECK(qkv.is_contiguous() && out.is_contiguous() && rk.is_contiguous() && tiles.is_contiguous());
  const int H = bu.size(0);
  TORCH_CHECK(bu.size(1) == AD && qkv.size(1) == 3 * H * AD && rk.size(1) == H * AD);
  const int n_tiles = tiles.size(0);
  if (n_tiles == 0) return;
  AttnArgs p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.rk = reinterpret_cast<const __nv_bfloat16*>(rk.data_ptr());
  p.bu = reinterpret_cast<const __nv_bfloat16*>(bu.data_ptr());
  p.bv = reinterpret_cast<const __nv_bfloat16*>(bv.data_ptr());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.tiles = reinterpret_cast<const int4*>(tiles.data_ptr<int>());
  p.H = H;
  p.rel_off = rel_off;
  p.rel_rows = rk.size(0);
  p.scale = scale;
  constexpr int smem = (3 * ABN * ADP + ABM * RING_LD) * 2;  // two CTAs an SM
  TORCH_CHECK(max_len - 1 <= rel_off && rel_off + max_len + ABM - 2 < p.rel_rows,
              "relpos_attention: sequence too long for the distance table (", max_len, ")");
  const at::cuda::OptionalCUDAGuard guard(qkv.device());
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(relpos_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  relpos_attention_kernel<<<dim3(n_tiles, H), 128, smem, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// conv0 + ReLU + depthwise of the subsampling: feats (B, T, 128) bf16 padded, mel_len (B) frames, rows (n, 2) int32
// (sequence, t2) of every output row; w0, w2 (C, 9) fp32, b0, b2 (C); out (n, 32, C) bf16
void sub_conv0_dw(const at::Tensor& feats, const at::Tensor& mel_len, const at::Tensor& rows, const at::Tensor& w0,
                  const at::Tensor& b0, const at::Tensor& w2, const at::Tensor& b2, at::Tensor& out) {
  TORCH_CHECK(feats.is_contiguous() && feats.size(2) == 128 && out.is_contiguous() && w0.size(0) == 256);
  const int n = rows.size(0);
  if (n == 0) return;
  sub_conv0_dw_kernel<<<n, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(feats.data_ptr()), feats.stride(0), mel_len.data_ptr<int>(),
      reinterpret_cast<const int2*>(rows.data_ptr<int>()), w0.data_ptr<float>(), b0.data_ptr<float>(),
      w2.data_ptr<float>(), b2.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), feats.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// depthwise 3x3 stride-2 + bias on channels-last sequences: x (sum T_in, F_in, C) with per-sequence start / length
// (in rows of x); rows (n, 2) (sequence, t_out); out (n, F_in / 2, C)
void sub_dw(const at::Tensor& x, const at::Tensor& in_start, const at::Tensor& in_len, const at::Tensor& rows,
            const at::Tensor& w, const at::Tensor& b, at::Tensor& out) {
  const int n = rows.size(0), F_in = x.size(1), C = x.size(2);
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && C % 8 == 0);
  if (n == 0) return;
  const size_t total = (size_t)n * (F_in / 2) * (C / 8);
  sub_dw_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), in_start.data_ptr<int>(), in_len.data_ptr<int>(),
      reinterpret_cast<const int2*>(rows.data_ptr<int>()), w.data_ptr<float>(), b.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, F_in, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
