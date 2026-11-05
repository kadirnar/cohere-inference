// LayerNorm of one row held by a warp: VEC 16-byte chunks per lane (N = 256 * VEC features), fp32 statistics (two
// passes over the registers), output rounded to bf16 as nn.LayerNorm in bf16 does.
#pragma once
#include "common.cuh"

template <int VEC>
DEVI void ln_row(float (&v)[VEC][8], const __nv_bfloat16* w, const __nv_bfloat16* b, float eps, int lane,
                 uint4 (&out)[VEC]) {
  constexpr int N = 32 * 8 * VEC;
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < VEC; ++i)
#pragma unroll
    for (int k = 0; k < 8; ++k) s += v[i][k];
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
  const float mean = s / N;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < VEC; ++i)
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float d = v[i][k] - mean;
      q += d * d;
    }
#pragma unroll
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffff, q, o);
  const float rstd = rsqrtf(q / N + eps);
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    const int c = (i * 32 + lane) * 8;
    const uint4 wv = *reinterpret_cast<const uint4*>(w + c);
    const uint4 bv = *reinterpret_cast<const uint4*>(b + c);
    const __nv_bfloat16* w8 = reinterpret_cast<const __nv_bfloat16*>(&wv);
    const __nv_bfloat16* b8 = reinterpret_cast<const __nv_bfloat16*>(&bv);
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      o[k] = pack_bf16((v[i][2 * k] - mean) * rstd * bf(w8[2 * k]) + bf(b8[2 * k]),
                       (v[i][2 * k + 1] - mean) * rstd * bf(w8[2 * k + 1]) + bf(b8[2 * k + 1]));
    out[i] = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

template <int VEC>
DEVI void load_row(const __nv_bfloat16* x, int lane, float (&v)[VEC][8]) {
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    const uint4 u = *reinterpret_cast<const uint4*>(x + (i * 32 + lane) * 8);
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
    for (int k = 0; k < 8; ++k) v[i][k] = bf(e[k]);
  }
}

template <int VEC>
DEVI void unpack_row(const uint4 (&u)[VEC], float (&v)[VEC][8]) {
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&u[i]);
#pragma unroll
    for (int k = 0; k < 8; ++k) v[i][k] = bf(e[k]);
  }
}

