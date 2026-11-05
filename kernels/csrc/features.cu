// Log-mel features of a batch of pieces (kernels/features.py) on their valid frames only: the batch's padded layout
// spent ~20 elementwise passes over (pieces x longest piece) samples and a (pieces x 257 x frames) spectrum, 22 ms for
// 50 FLEURS clips, most of it on padding. Here:
//   frames     every valid frame of every piece (frame t of a piece of n samples, t < n // 160): its 512 samples
//              centred at 160 t, pre-emphasised (x[s] - p x[s-1], as the extractor: product then difference) and zero
//              outside the piece, times the 400-sample Hann window centred in 512
//   rfft       cuFFT, as torch.stft does on the same frames
//   mel        |X| squared again (the extractor's sqrt then square), the mel filter bank (fp32, bins in order),
//              log(mel + 2^-24)
//   normalize  per piece and feature: mean and unbiased variance over its valid frames (fp64 sums: with fp32 ones
//              twice as many values came out a bf16 step away from the CPU extractor's), (v - mean) / (std + 1e-5),
//              bf16, into the padded (pieces, T, 128) output whose other frames stay zero
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.cuh"

namespace {

constexpr int NFFT = 512, HOP = 160, BINS = NFFT / 2 + 1, MELS = 128;

// piece of flat frame f: the last p with frame_start[p] <= f (frame_start has pieces + 1 entries)
DEVI int piece_of(const int64_t* frame_start, int pieces, int f) {
  int lo = 0, hi = pieces - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (frame_start[mid] <= f) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

__global__ void frames_kernel(const float* __restrict__ x, const int64_t* __restrict__ sample_start,
                              const int64_t* __restrict__ lens, const int64_t* __restrict__ frame_start, int pieces,
                              const float* __restrict__ window, float preemph, float* __restrict__ out) {
  const int f = blockIdx.x, p = piece_of(frame_start, pieces, f), t = f - (int)frame_start[p];
  const float* xp = x + sample_start[p];
  const int n = (int)lens[p];
  for (int j = threadIdx.x; j < NFFT; j += blockDim.x) {
    const int s = t * HOP - NFFT / 2 + j;
    float v = 0.f;
    if (s >= 0 && s < n) v = s == 0 ? xp[0] : __fsub_rn(xp[s], __fmul_rn(preemph, xp[s - 1]));
    out[(size_t)f * NFFT + j] = __fmul_rn(v, window[j]);
  }
}

// one frame per CTA: power of its 257 bins in shared memory, then a thread per mel band over the band's nonzero bins
// (band[m] = first, end): the zero weights would add nothing to the fp32 sum
__global__ void mel_kernel(const float2* __restrict__ spec, const float* __restrict__ mel,
                           const int2* __restrict__ band, float* __restrict__ out) {
  __shared__ float power[BINS];
  const int f = blockIdx.x;
  for (int k = threadIdx.x; k < BINS; k += blockDim.x) {
    const float2 c = spec[(size_t)f * BINS + k];
    const float a = __fsqrt_rn(__fadd_rn(__fmul_rn(c.x, c.x), __fmul_rn(c.y, c.y)));
    power[k] = __fmul_rn(a, a);
  }
  __syncthreads();
  const int m = threadIdx.x;  // blockDim.x == MELS
  const float* w = mel + (size_t)m * BINS;
  const int2 r = band[m];
  float acc = 0.f;
  for (int k = r.x; k < r.y; ++k) acc = __fmaf_rn(w[k], power[k], acc);
  out[(size_t)f * MELS + m] = logf(__fadd_rn(acc, 5.9604644775390625e-8f));  // 2^-24
}

// a CTA per piece: NG groups of 128 threads (a thread per mel band), group g summing frames g, g + NG, ... and the
// groups' sums added in order (the same for every batch); mean, then the unbiased variance the same way, then the
// output
constexpr int NG = 4;
__global__ void __launch_bounds__(NG * MELS) normalize_kernel(const float* __restrict__ logmel,
                                                              const int64_t* __restrict__ frame_start, int T,
                                                              __nv_bfloat16* __restrict__ out) {
  __shared__ double part[NG][MELS];
  const int p = blockIdx.x, m = threadIdx.x % MELS, g = threadIdx.x / MELS;
  const int f0 = (int)frame_start[p], nv = (int)(frame_start[p + 1] - f0);
  const float* v = logmel + (size_t)f0 * MELS + m;
  __nv_bfloat16* o = out + (size_t)p * T * MELS + m;
  if (nv == 0) return;
  auto total = [&](double x) {
    part[g][m] = x;
    __syncthreads();
    double s = 0.0;
#pragma unroll
    for (int k = 0; k < NG; ++k) s += part[k][m];
    __syncthreads();
    return s;
  };
  double s = 0.0;
  for (int t = g; t < nv; t += NG) s += v[(size_t)t * MELS];
  const float mean = (float)(total(s) / nv);
  double q = 0.0;
  for (int t = g; t < nv; t += NG) {
    const double d = (double)v[(size_t)t * MELS] - mean;
    q += d * d;
  }
  const float den = __fadd_rn((float)sqrt(total(q) / (nv - 1)), 1e-5f);
  for (int t = g; t < nv; t += NG) o[(size_t)t * MELS] = tobf(__fdiv_rn(v[(size_t)t * MELS] - mean, den));
}

}  // namespace

// out (pieces, T, 128) bf16 zeros: the normalized log-mel of each piece's valid frames (lens[p] // 160 of them) at
// its first frames. x: the pieces' samples back to back (fp32, dithered); meta int64: each piece's first sample,
// its length, then the cumulative valid frames (pieces + 1); window (512) the centred Hann window; mel (128, 257) the
// filter bank, band (128, 2) int32 each band's first and end nonzero bins
void logmel_features(const at::Tensor& x, const at::Tensor& meta, int64_t total_frames, const at::Tensor& window,
                     const at::Tensor& mel, const at::Tensor& band, double preemph, at::Tensor& out) {
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kFloat && window.numel() == NFFT && band.is_contiguous() &&
              band.numel() == 2 * MELS && band.scalar_type() == at::kInt &&
              mel.size(0) == MELS && mel.size(1) == BINS && mel.is_contiguous() && out.is_contiguous() &&
              out.size(2) == MELS && out.scalar_type() == at::kBFloat16 && meta.scalar_type() == at::kLong &&
              meta.numel() == 3 * out.size(0) + 1);
  const int pieces = out.size(0);
  const int64_t* sample_start = meta.data_ptr<int64_t>();
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  if (total_frames == 0) return;
  at::Tensor frames = at::empty({total_frames, NFFT}, x.options());
  frames_kernel<<<total_frames, 128, 0, s>>>(x.data_ptr<float>(), sample_start, sample_start + pieces,
                                             sample_start + 2 * pieces, pieces, window.data_ptr<float>(),
                                             (float)preemph, frames.data_ptr<float>());
  at::Tensor spec = at::fft_rfft(frames, NFFT, -1);
  at::Tensor logmel = at::empty({total_frames, MELS}, x.options());
  mel_kernel<<<total_frames, MELS, 0, s>>>(reinterpret_cast<const float2*>(spec.data_ptr()), mel.data_ptr<float>(),
                                           reinterpret_cast<const int2*>(band.data_ptr<int>()),
                                           logmel.data_ptr<float>());
  normalize_kernel<<<pieces, NG * MELS, 0, s>>>(logmel.data_ptr<float>(), sample_start + 2 * pieces, out.size(1),
                                           reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
