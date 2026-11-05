// One decoding step of the Cohere Transcribe decoder for continuous batching. Row r of a step works on slot active[r]
// of the per-slot state (token, position, prompt, limit, end flag) and of the KV caches, so rows join and leave the
// batch between steps without moving any cache: the host only rewrites the active list. Rows past the batch's real
// rows point to a dummy slot whose state is never advanced.
//
//   dec_embed      x = LN0(embed[token] + pos_table[position]), h = LN1(x)
//   dec_add_ln     x += y (bf16), h = LN(x)
//   dec_attention  softmax(q k^T * scale) v of one new token per row over its slot's cache (self-attention: positions
//                  0..pos, the new key and value written at pos first; cross-attention: the slot's audio frames),
//                  flash-decoding: a CTA per (row, head, split), 16 groups of 8 lanes each streaming one key at a time
//   dec_pick       greedy choice (first index of the maximum, as torch.argmax), the prompt's tokens while inside it,
//                  pad once finished, end at the end-of-text token or the slot's token limit; advances the position
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.cuh"
#include "dec_attn.cuh"
#include "layernorm.cuh"

namespace {


__global__ void dec_embed_kernel(const int* __restrict__ active, const int64_t* __restrict__ tok,
                                 const int* __restrict__ pos, const __nv_bfloat16* __restrict__ emb,
                                 const __nv_bfloat16* __restrict__ ptab, const __nv_bfloat16* w0,
                                 const __nv_bfloat16* b0, float eps0, const __nv_bfloat16* w1, const __nv_bfloat16* b1,
                                 float eps1, __nv_bfloat16* x, __nv_bfloat16* h, int rows) {
  constexpr int VEC = 4, N = 1024;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  const int s = active[row];
  const int64_t t = tok[s];
  const int p = pos[s];
  float v[VEC][8];
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    const int c = (i * 32 + lane) * 8;
    const uint4 e = *reinterpret_cast<const uint4*>(emb + t * N + c);
    const uint4 q = *reinterpret_cast<const uint4*>(ptab + (size_t)p * N + c);
    const __nv_bfloat16* e8 = reinterpret_cast<const __nv_bfloat16*>(&e);
    const __nv_bfloat16* q8 = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
    for (int k = 0; k < 8; ++k) v[i][k] = round_bf(bf(e8[k]) + bf(q8[k]));
  }
  uint4 out[VEC];
  ln_row<VEC>(v, w0, b0, eps0, lane, out);
#pragma unroll
  for (int i = 0; i < VEC; ++i) *reinterpret_cast<uint4*>(x + (size_t)row * N + (i * 32 + lane) * 8) = out[i];
  unpack_row<VEC>(out, v);
  ln_row<VEC>(v, w1, b1, eps1, lane, out);
#pragma unroll
  for (int i = 0; i < VEC; ++i) *reinterpret_cast<uint4*>(h + (size_t)row * N + (i * 32 + lane) * 8) = out[i];
}

__global__ void dec_add_ln_kernel(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y, int ldy,
                                  const __nv_bfloat16* w, const __nv_bfloat16* b, float eps, __nv_bfloat16* h,
                                  int rows) {
  constexpr int VEC = 4, N = 1024;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  float v[VEC][8];
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    const int c = (i * 32 + lane) * 8;
    const uint4 a = *reinterpret_cast<const uint4*>(x + (size_t)row * N + c);
    const uint4 d = *reinterpret_cast<const uint4*>(y + (size_t)row * ldy + c);
    const __nv_bfloat16* a8 = reinterpret_cast<const __nv_bfloat16*>(&a);
    const __nv_bfloat16* d8 = reinterpret_cast<const __nv_bfloat16*>(&d);
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 8; k += 2) {
      v[i][k] = round_bf(bf(a8[k]) + bf(d8[k]));
      v[i][k + 1] = round_bf(bf(a8[k + 1]) + bf(d8[k + 1]));
      o[k / 2] = pack_bf16(v[i][k], v[i][k + 1]);
    }
    *reinterpret_cast<uint4*>(x + (size_t)row * N + c) = make_uint4(o[0], o[1], o[2], o[3]);
  }
  uint4 out[VEC];
  ln_row<VEC>(v, w, b, eps, lane, out);
#pragma unroll
  for (int i = 0; i < VEC; ++i) *reinterpret_cast<uint4*>(h + (size_t)row * N + (i * 32 + lane) * 8) = out[i];
}

template <bool SELF, bool Q8 = false>
__global__ void __launch_bounds__(128) dec_attention_kernel(const DecAttnArgs p) {
  __shared__ float scratch[16 * (HD + 6)];
  attend_item<SELF, Q8>(p, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z, scratch);
}

__global__ void dec_combine_kernel(const float* __restrict__ part, int nsp, __nv_bfloat16* out, int H) {
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const float* base = part + ((size_t)r * H + h) * nsp * (HD + 2);
  float M = -INFINITY;
  for (int s = 0; s < nsp; ++s) M = fmaxf(M, base[s * (HD + 2) + HD]);
  float L = 0.f, O = 0.f;
  for (int s = 0; s < nsp; ++s) {
    const float* q = base + s * (HD + 2);
    const float w = q[HD] == -INFINITY ? 0.f : expf(q[HD] - M);
    L += w * q[HD + 1];
    O += w * q[d];
  }
  out[(size_t)r * H * HD + h * HD + d] = tobf(O / L);
}

__global__ void __launch_bounds__(256) dec_pick_kernel(const __nv_bfloat16* __restrict__ logits, int V,
                                                       const int* __restrict__ active, int dummy, int64_t* tok,
                                                       int* pos, uint8_t* finished, const int64_t* forced, int FP,
                                                       const int* plen, const int* limit, int64_t* out, int OUT,
                                                       int pad, int eos) {
  const int r = blockIdx.x, tid = threadIdx.x;
  const int s = active[r];
  if (s == dummy) return;
  const __nv_bfloat16* row = logits + (size_t)r * V;
  float best = -INFINITY;
  int arg = 0x7fffffff;
  for (int i = tid * 8; i < V; i += 256 * 8) {  // 8 consecutive logits per thread per pass
    const uint4 u = *reinterpret_cast<const uint4*>(row + i);
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float v = bf(e[k]);
      if (v > best) {  // strictly greater: the first index of a tie stays
        best = v;
        arg = i + k;
      }
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float ob = __shfl_xor_sync(0xffffffff, best, o);
    const int oa = __shfl_xor_sync(0xffffffff, arg, o);
    if (ob > best || (ob == best && oa < arg)) {
      best = ob;
      arg = oa;
    }
  }
  __shared__ float sb[8];
  __shared__ int sa[8];
  if ((tid & 31) == 0) {
    sb[tid >> 5] = best;
    sa[tid >> 5] = arg;
  }
  __syncthreads();
  if (tid != 0) return;
  for (int w = 1; w < 8; ++w)
    if (sb[w] > best || (sb[w] == best && sa[w] < arg)) {
      best = sb[w];
      arg = sa[w];
    }
  if (finished[s]) return;  // ended at an earlier step: nothing moves until the host retires the slot
  const int at = pos[s] + 1;
  const bool in_prompt = at < plen[s];
  const int64_t nt = in_prompt ? forced[(size_t)s * FP + at] : (int64_t)arg;
  out[(size_t)s * OUT + at] = nt;
  tok[s] = nt;
  pos[s] = at;
  finished[s] = (!in_prompt && (nt == eos || at >= limit[s])) ? 1 : 0;
}

// cross-attention keys and values of encoded frames into their slots: kv (rows, L * 2 * H * 128) of packed frames,
// row i = frame frame_of[i] of slot slot_of[i]; cache (L, slots, H, T, 128)
__global__ void dec_scatter_kv_kernel(const __nv_bfloat16* __restrict__ kv, const int* __restrict__ slot_of,
                                      const int* __restrict__ frame_of, __nv_bfloat16* ck, __nv_bfloat16* cv, int rows,
                                      int L, int S, int H, int T) {
  const int chunks = L * 2 * H * (HD / 8);
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)rows * chunks) return;
  const int row = idx / chunks, c = idx % chunks;
  const int d8 = c % (HD / 8), h = (c / (HD / 8)) % H, kvi = (c / (HD / 8 * H)) % 2, l = c / (HD / 8 * H * 2);
  const uint4 u = *reinterpret_cast<const uint4*>(kv + (size_t)row * chunks * 8 + (size_t)c * 8);
  __nv_bfloat16* dst = (kvi ? cv : ck) + ((((size_t)l * S + slot_of[row]) * H + h) * T + frame_of[row]) * HD + d8 * 8;
  *reinterpret_cast<uint4*>(dst) = u;
}

// the same into int8 caches: 16 threads per 128-vector (one key or value of one head and frame), its absmax scale
__global__ void dec_scatter_kv_q8_kernel(const __nv_bfloat16* __restrict__ kv, const int* __restrict__ slot_of,
                                         const int* __restrict__ frame_of, int8_t* ck, int8_t* cv, float* cks,
                                         float* cvs, int rows, int L, int S, int H, int T) {
  const int chunks = L * 2 * H * (HD / 8);
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // rows * chunks is a multiple of 16
  const bool live = idx < (size_t)rows * chunks;
  const int row = live ? idx / chunks : 0, c = live ? idx % chunks : 0;
  const int d8 = c % (HD / 8), h = (c / (HD / 8)) % H, kvi = (c / (HD / 8 * H)) % 2, l = c / (HD / 8 * H * 2);
  uint4 u = make_uint4(0, 0, 0, 0);
  if (live) u = *reinterpret_cast<const uint4*>(kv + (size_t)row * chunks * 8 + (size_t)c * 8);
  const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&u);
  float mx = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) mx = fmaxf(mx, fabsf(bf(e[k])));
#pragma unroll
  for (int o = 1; o < 16; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if (!live) return;
  const float sc = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / sc;
  uint32_t w[2] = {0, 0};
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int q = __float2int_rn(fminf(fmaxf(bf(e[k]) * inv, -127.f), 127.f));
    w[k >> 2] |= (uint32_t)(q & 0xff) << ((k & 3) * 8);
  }
  const size_t vec = (((size_t)l * S + slot_of[row]) * H + h) * T + frame_of[row];
  *reinterpret_cast<uint2*>((kvi ? cv : ck) + vec * HD + d8 * 8) = make_uint2(w[0], w[1]);
  if (d8 == 0) (kvi ? cvs : cks)[vec] = sc;
}

}  // namespace

void dec_embed(const at::Tensor& active, const at::Tensor& tok, const at::Tensor& pos, const at::Tensor& emb,
               const at::Tensor& ptab, const at::Tensor& w0, const at::Tensor& b0, double eps0, const at::Tensor& w1,
               const at::Tensor& b1, double eps1, at::Tensor& x, at::Tensor& h) {
  const int rows = x.size(0);
  TORCH_CHECK(x.size(1) == 1024 && x.is_contiguous() && h.is_contiguous());
  auto P = [](const at::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  dec_embed_kernel<<<(rows + 3) / 4, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      active.data_ptr<int>(), tok.data_ptr<int64_t>(), pos.data_ptr<int>(), P(emb), P(ptab), P(w0), P(b0), eps0,
      P(w1), P(b1), eps1, reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(h.data_ptr()),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dec_add_ln(at::Tensor& x, const at::Tensor& y, const at::Tensor& w, const at::Tensor& b, double eps,
                at::Tensor& h) {
  const int rows = x.size(0);
  TORCH_CHECK(x.size(1) == 1024 && x.is_contiguous() && h.is_contiguous() && y.stride(1) == 1);
  auto P = [](const at::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  const int grid = (rows + 3) / 4;
  dec_add_ln_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), P(y), y.stride(0), P(w), P(b), eps,
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr()), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// self (q = the qkv projection, n = positions) or cross attention (q = query projection, n = audio frames) of the
// step's rows; k, v (slots, H, T, 128) caches of one layer; out (rows, H*128); splits > 1 needs part
void dec_attention(bool self, const at::Tensor& active, const at::Tensor& q, at::Tensor& k, at::Tensor& v,
                   const at::Tensor& n, double scale, int64_t splits, at::Tensor& part, at::Tensor& out) {
  const int rows = q.size(0), H = k.size(1);
  TORCH_CHECK(k.size(3) == HD && k.is_contiguous() && v.is_contiguous() && q.stride(1) == 1 && out.is_contiguous());
  DecAttnArgs p;
  p.active = active.data_ptr<int>();
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.ldq = q.stride(0);
  p.k = reinterpret_cast<__nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<__nv_bfloat16*>(v.data_ptr());
  p.n = n.data_ptr<int>();
  p.H = H;
  p.T = k.size(2);
  p.scale = scale;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.part = splits > 1 ? part.data_ptr<float>() : nullptr;
  TORCH_CHECK(splits == 1 || part.numel() >= (int64_t)rows * H * splits * (HD + 2));
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const dim3 grid(rows, H, splits);
  if (self)
    dec_attention_kernel<true><<<grid, 128, 0, s>>>(p);
  else
    dec_attention_kernel<false><<<grid, 128, 0, s>>>(p);
  if (splits > 1) dec_combine_kernel<<<dim3(rows, H), HD, 0, s>>>(p.part, splits, p.out, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// self (q = the qkv projection, n = positions; the new key and value quantized into the caches) or cross attention
// over int8 caches k8, v8 (slots, H, T, 128) with scales ks, vs (slots, H, T)
void dec_attention_q8(bool self, const at::Tensor& active, const at::Tensor& q, at::Tensor& k8, at::Tensor& v8,
                      at::Tensor& ks, at::Tensor& vs, const at::Tensor& n, double scale, int64_t splits,
                      at::Tensor& part, at::Tensor& out) {
  const int rows = q.size(0), H = k8.size(1);
  TORCH_CHECK(k8.size(3) == HD && k8.is_contiguous() && v8.is_contiguous() && ks.is_contiguous() && q.stride(1) == 1);
  DecAttnArgs p{};
  p.active = active.data_ptr<int>();
  p.k8 = k8.data_ptr<int8_t>();
  p.v8 = v8.data_ptr<int8_t>();
  p.ks = ks.data_ptr<float>();
  p.vs = vs.data_ptr<float>();
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.ldq = q.stride(0);
  p.n = n.data_ptr<int>();
  p.H = H;
  p.T = k8.size(2);
  p.scale = scale;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.part = splits > 1 ? part.data_ptr<float>() : nullptr;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  if (self)
    dec_attention_kernel<true, true><<<dim3(rows, H, splits), 128, 0, s>>>(p);
  else
    dec_attention_kernel<false, true><<<dim3(rows, H, splits), 128, 0, s>>>(p);
  if (splits > 1) dec_combine_kernel<<<dim3(rows, H), HD, 0, s>>>(p.part, splits, p.out, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dec_scatter_kv_q8(const at::Tensor& kv, const at::Tensor& slot_of, const at::Tensor& frame_of, at::Tensor& ck,
                       at::Tensor& cv, at::Tensor& cks, at::Tensor& cvs) {
  const int rows = kv.size(0), L = ck.size(0), S = ck.size(1), H = ck.size(2), T = ck.size(3);
  TORCH_CHECK(kv.is_contiguous() && kv.size(1) == L * 2 * H * HD && ck.scalar_type() == at::kChar);
  if (rows == 0) return;
  const size_t n = (size_t)rows * L * 2 * H * (HD / 8);
  dec_scatter_kv_q8_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()), slot_of.data_ptr<int>(), frame_of.data_ptr<int>(),
      ck.data_ptr<int8_t>(), cv.data_ptr<int8_t>(), cks.data_ptr<float>(), cvs.data_ptr<float>(), rows, L, S, H, T);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dec_pick(const at::Tensor& logits, const at::Tensor& active, int64_t dummy, at::Tensor& tok, at::Tensor& pos,
              at::Tensor& finished, const at::Tensor& forced, const at::Tensor& plen, const at::Tensor& limit,
              at::Tensor& out, int64_t pad, int64_t eos) {
  const int rows = logits.size(0), V = logits.size(1);
  TORCH_CHECK(V % 8 == 0 && logits.is_contiguous());
  dec_pick_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), V, active.data_ptr<int>(), dummy,
      tok.data_ptr<int64_t>(), pos.data_ptr<int>(), finished.data_ptr<uint8_t>(), forced.data_ptr<int64_t>(),
      forced.size(1), plen.data_ptr<int>(), limit.data_ptr<int>(), out.data_ptr<int64_t>(), out.size(1), pad, eos);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dec_scatter_kv(const at::Tensor& kv, const at::Tensor& slot_of, const at::Tensor& frame_of, at::Tensor& ck,
                    at::Tensor& cv) {
  const int rows = kv.size(0), L = ck.size(0), S = ck.size(1), H = ck.size(2), T = ck.size(3);
  TORCH_CHECK(kv.is_contiguous() && kv.size(1) == L * 2 * H * HD && ck.is_contiguous() && cv.is_contiguous());
  if (rows == 0) return;
  const size_t n = (size_t)rows * L * 2 * H * (HD / 8);
  dec_scatter_kv_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()), slot_of.data_ptr<int>(), frame_of.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(ck.data_ptr()), reinterpret_cast<__nv_bfloat16*>(cv.data_ptr()), rows, L, S, H,
      T);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
