// Shared helpers for the CUDA kernels: bf16 conversions, cp.async, ldmatrix and mma.sync wrappers (sm_80+).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

DEVI float bf(const __nv_bfloat16 x) { return __bfloat162float(x); }
DEVI __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }
DEVI float round_bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// 16-byte global -> shared copy; src_bytes 0 fills the destination with zeros
DEVI void cp_async16(void* smem, const void* gmem, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(smem)), "l"(gmem),
               "r"(pred ? 16 : 0));
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

DEVI void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_u32(p)));
}
DEVI void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_u32(p)));
}
DEVI void ldmatrix_x2(uint32_t& r0, uint32_t& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
DEVI void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(smem_u32(p)));
}

// D (16x8 fp32) += A (16x16 bf16, row) * B (16x8 bf16, col)
DEVI void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

DEVI uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

DEVI float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

// element offset of 16-byte chunk `ch` (of 4 per 64-byte row) in row `row`, XOR-swizzled so that the 8 rows an
// ldmatrix phase reads fall in 8 different bank groups
DEVI int swz32(int row, int ch) { return row * 32 + ((ch ^ ((row >> 1) & 3)) << 3); }
// the same for 128-byte rows (64 bf16, 8 chunks)
DEVI int swz64(int row, int ch) { return row * 64 + ((ch ^ (row & 7)) << 3); }

