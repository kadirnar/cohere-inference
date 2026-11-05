// The decoder's attention of one new token per row over its slot's cache, shared by the step kernels
// (decoder_ops.cu) and the persistent step kernel (dec_mega.cu).
#pragma once
#include "common.cuh"

constexpr int HD = 128;  // head size

struct DecAttnArgs {
  const int* active;
  const int8_t* k8;   // int8 cross-attention cache (slots, H, T, 128) with one scale per 128-vector: ks, vs
  const int8_t* v8;
  const float* ks;
  const float* vs;
  const __nv_bfloat16* q;  // (rows, ldq): head h at column h * 128 (self-attention: q | k | v of the new token)
  int ldq;
  __nv_bfloat16* k;        // cache (slots, H, T, 128)
  __nv_bfloat16* v;
  const int* n;            // self: position of the new token per slot; cross: audio frames per slot
  int H, T;
  float scale;
  __nv_bfloat16* out;      // (rows, H * 128)
  float* part;             // (rows, H, splits, 130) partial results when split
};

// loads through L2 (ld.global.cg): in the persistent decoder kernel the rows were just written by other CTAs
DEVI void load16(const __nv_bfloat16* p, float (&f)[16]) {
  const uint4 a = __ldcg(reinterpret_cast<const uint4*>(p));
  const uint4 b = __ldcg(reinterpret_cast<const uint4*>(p + 8));
  const __nv_bfloat16* a8 = reinterpret_cast<const __nv_bfloat16*>(&a);
  const __nv_bfloat16* b8 = reinterpret_cast<const __nv_bfloat16*>(&b);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    f[k] = bf(a8[k]);
    f[k + 8] = bf(b8[k]);
  }
}

DEVI void load16_q8(const int8_t* p, float s, float (&f)[16]) {
  const uint4 u = __ldcg(reinterpret_cast<const uint4*>(p));
  const int8_t* e = reinterpret_cast<const int8_t*>(&u);
#pragma unroll
  for (int k = 0; k < 16; ++k) f[k] = (float)e[k] * s;
}

// 16 values of a 128-vector held by the 8 lanes of a group (gl = lane in group) as int8 with the vector's scale
DEVI void store16_q8(const float (&f)[16], int8_t* dst, float* scale, int gl) {
  const unsigned gmask = 0xffu << (threadIdx.x & 24);
  float mx = 0.f;
#pragma unroll
  for (int k = 0; k < 16; ++k) mx = fmaxf(mx, fabsf(f[k]));
  mx = fmaxf(mx, __shfl_xor_sync(gmask, mx, 1));
  mx = fmaxf(mx, __shfl_xor_sync(gmask, mx, 2));
  mx = fmaxf(mx, __shfl_xor_sync(gmask, mx, 4));
  const float sc = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / sc;
  uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const int q = __float2int_rn(fminf(fmaxf(f[k] * inv, -127.f), 127.f));
    w[k >> 2] |= (uint32_t)(q & 0xff) << ((k & 3) * 8);
  }
  *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
  if (gl == 0) *scale = sc;
}

DEVI void attend_one(const float (&q)[16], const float (&kk)[16], const float (&vv)[16], float scale, float& m,
                     float& l, float (&acc)[16]) {
  // the 8 lanes of this key's group reduce among themselves: groups of a warp may run different numbers of keys
  const unsigned gmask = 0xffu << (threadIdx.x & 24);
  float d = 0.f;
#pragma unroll
  for (int k = 0; k < 16; ++k) d += q[k] * kk[k];
  d += __shfl_xor_sync(gmask, d, 1);
  d += __shfl_xor_sync(gmask, d, 2);
  d += __shfl_xor_sync(gmask, d, 4);
  const float s = d * scale, mn = fmaxf(m, s), a = expf(m - mn), e = expf(s - mn);
  l = l * a + e;
#pragma unroll
  for (int k = 0; k < 16; ++k) acc[k] = acc[k] * a + e * vv[k];
  m = mn;
}

// attention of row r (slot active[r]), head h, key split sp of nsp, by a CTA of NG * 8 threads (NG groups of 8 lanes,
// each lane 16 of the 128 dims, a key per group at a time); scratch: NG * (HD + 6) floats of shared memory
template <bool SELF, bool Q8 = false, int NG = 16>
DEVI void attend_item(const DecAttnArgs& p, int r, int h, int sp, int nsp, float* scratch) {
  const int s = p.active[r];
  const int tid = threadIdx.x, grp = tid >> 3, gl = tid & 7;
  float q[16];
  load16(p.q + (size_t)r * p.ldq + h * HD + gl * 16, q);
  const size_t base = ((size_t)s * p.H + h) * p.T * HD + gl * 16;
  const __nv_bfloat16* kb = Q8 ? nullptr : p.k + base;
  const __nv_bfloat16* vb = Q8 ? nullptr : p.v + base;
  const int n = p.n[s];  // keys in the cache: self 0..pos-1 (pos is the new one), cross 0..frames-1
  float m = -INFINITY, l = 0.f, acc[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) acc[k] = 0.f;
  if (SELF && sp == 0 && grp == 0) {  // the new key and value: into the cache at pos, and attended
    float kn[16], vn[16];
    const __nv_bfloat16* row = p.q + (size_t)r * p.ldq + p.H * HD + h * HD + gl * 16;
    load16(row, kn);
    load16(row + p.H * HD, vn);
    if constexpr (Q8) {  // int8 with one scale per 128-vector: the absmax over the group's 8 lanes
      const size_t vec = ((size_t)s * p.H + h) * p.T + n;
      store16_q8(kn, const_cast<int8_t*>(p.k8) + vec * HD + gl * 16, const_cast<float*>(p.ks) + vec, gl);
      store16_q8(vn, const_cast<int8_t*>(p.v8) + vec * HD + gl * 16, const_cast<float*>(p.vs) + vec, gl);
    } else {
      *reinterpret_cast<uint4*>(p.k + base + (size_t)n * HD) = __ldcg(reinterpret_cast<const uint4*>(row));
      *reinterpret_cast<uint4*>(p.k + base + (size_t)n * HD + 8) = __ldcg(reinterpret_cast<const uint4*>(row + 8));
      *reinterpret_cast<uint4*>(p.v + base + (size_t)n * HD) = __ldcg(reinterpret_cast<const uint4*>(row + p.H * HD));
      *reinterpret_cast<uint4*>(p.v + base + (size_t)n * HD + 8) =
          __ldcg(reinterpret_cast<const uint4*>(row + p.H * HD + 8));
    }
    attend_one(q, kn, vn, p.scale, m, l, acc);
  }
  const int chunk = (n + nsp - 1) / nsp, start = sp * chunk, end = min(n, start + chunk);
  int j = start + grp;
  if constexpr (Q8) {
    const size_t vec = ((size_t)s * p.H + h) * p.T;  // scale index of key 0
    const int8_t* k8 = p.k8 + vec * HD + gl * 16;
    const int8_t* v8 = p.v8 + vec * HD + gl * 16;
    for (; j + NG < end; j += 2 * NG) {
      float k0[16], v0[16], k1[16], v1[16];
      load16_q8(k8 + (size_t)j * HD, p.ks[vec + j], k0);
      load16_q8(k8 + (size_t)(j + NG) * HD, p.ks[vec + j + NG], k1);
      load16_q8(v8 + (size_t)j * HD, p.vs[vec + j], v0);
      load16_q8(v8 + (size_t)(j + NG) * HD, p.vs[vec + j + NG], v1);
      attend_one(q, k0, v0, p.scale, m, l, acc);
      attend_one(q, k1, v1, p.scale, m, l, acc);
    }
    if (j < end) {
      float k0[16], v0[16];
      load16_q8(k8 + (size_t)j * HD, p.ks[vec + j], k0);
      load16_q8(v8 + (size_t)j * HD, p.vs[vec + j], v0);
      attend_one(q, k0, v0, p.scale, m, l, acc);
    }
  } else {
    for (; j + NG < end; j += 2 * NG) {  // two keys per group in flight
      float k0[16], v0[16], k1[16], v1[16];
      load16(kb + (size_t)j * HD, k0);
      load16(kb + (size_t)(j + NG) * HD, k1);
      load16(vb + (size_t)j * HD, v0);
      load16(vb + (size_t)(j + NG) * HD, v1);
      attend_one(q, k0, v0, p.scale, m, l, acc);
      attend_one(q, k1, v1, p.scale, m, l, acc);
    }
    if (j < end) {
      float k0[16], v0[16];
      load16(kb + (size_t)j * HD, k0);
      load16(vb + (size_t)j * HD, v0);
      attend_one(q, k0, v0, p.scale, m, l, acc);
    }
  }
  // combine the NG groups
  float* sm = scratch;
  float* sl = scratch + NG;
  float(*sacc)[HD + 4] = reinterpret_cast<float(*)[HD + 4]>(scratch + 2 * NG);
  if (gl == 0) {
    sm[grp] = m;
    sl[grp] = l;
  }
#pragma unroll
  for (int k = 0; k < 16; ++k) sacc[grp][gl * 16 + k] = acc[k];
  __syncthreads();
  float M = -INFINITY;
#pragma unroll
  for (int g = 0; g < NG; ++g) M = fmaxf(M, sm[g]);
  float L = 0.f, O = 0.f;
  if (M != -INFINITY && tid < HD) {
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      const float w = expf(sm[g] - M);
      L += w * sl[g];
      O += w * sacc[g][tid];
    }
  }
  if (tid >= HD) {
  } else if (nsp == 1) {
    p.out[(size_t)r * p.H * HD + h * HD + tid] = tobf(O / L);
  } else {
    float* part = p.part + (((size_t)r * p.H + h) * nsp + sp) * (HD + 2);
    part[tid] = O;
    if (tid == 0) {
      part[HD] = M;
      part[HD + 1] = L;
    }
  }
  __syncthreads();  // scratch is free for the next item
}

