#include <torch/extension.h>

void gemm(const at::Tensor& a, const at::Tensor& w, const c10::optional<at::Tensor>& bias, at::Tensor& out,
          int64_t epi, const c10::optional<at::Tensor>& resid, double alpha, int64_t cfg);

void layernorm(at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1, double eps1,
               const c10::optional<at::Tensor>& w2, const c10::optional<at::Tensor>& b2, double eps2, at::Tensor& h);
void dwconv_bn_silu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& cb, const at::Tensor& bn_scale,
                    const at::Tensor& bn_shift, const at::Tensor& seq_lo, const at::Tensor& seq_hi, at::Tensor& y);
void relpos_attention(const at::Tensor& qkv, const at::Tensor& rk, const at::Tensor& bu, const at::Tensor& bv,
                      const at::Tensor& tiles, int64_t rel_off, int64_t max_len, double scale, at::Tensor& out);

void dec_embed(const at::Tensor& active, const at::Tensor& tok, const at::Tensor& pos, const at::Tensor& emb,
               const at::Tensor& ptab, const at::Tensor& w0, const at::Tensor& b0, double eps0, const at::Tensor& w1,
               const at::Tensor& b1, double eps1, at::Tensor& x, at::Tensor& h);
void dec_add_ln(at::Tensor& x, const at::Tensor& y, const at::Tensor& w, const at::Tensor& b, double eps,
                at::Tensor& h);
void dec_attention(bool self, const at::Tensor& active, const at::Tensor& q, at::Tensor& k, at::Tensor& v,
                   const at::Tensor& n, double scale, int64_t splits, at::Tensor& part, at::Tensor& out);
void dec_pick(const at::Tensor& logits, const at::Tensor& active, int64_t dummy, at::Tensor& tok, at::Tensor& pos,
              at::Tensor& finished, const at::Tensor& forced, const at::Tensor& plen, const at::Tensor& limit,
              at::Tensor& out, int64_t pad, int64_t eos);
void dec_scatter_kv(const at::Tensor& kv, const at::Tensor& slot_of, const at::Tensor& frame_of, at::Tensor& ck,
                    at::Tensor& cv);

void gemm_i8(const at::Tensor& a, const at::Tensor& sa, int64_t group, const at::Tensor& w, const at::Tensor& sw,
             const c10::optional<at::Tensor>& bias, at::Tensor& out, int64_t epi, const c10::optional<at::Tensor>& resid,
             double alpha, const c10::optional<at::Tensor>& qs, int64_t cfg);
void layernorm_q8(at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1, double eps1,
                  const c10::optional<at::Tensor>& w2, const c10::optional<at::Tensor>& b2, double eps2, at::Tensor& hq,
                  at::Tensor& hs);

void sub_conv0_dw(const at::Tensor& feats, const at::Tensor& mel_len, const at::Tensor& rows, const at::Tensor& w0,
                  const at::Tensor& b0, const at::Tensor& w2, const at::Tensor& b2, at::Tensor& out);
void sub_dw(const at::Tensor& x, const at::Tensor& in_start, const at::Tensor& in_len, const at::Tensor& rows,
            const at::Tensor& w, const at::Tensor& b, at::Tensor& out);

void dec_w8_linear(const at::Tensor& x, const at::Tensor& w, const at::Tensor& sw, const c10::optional<at::Tensor>& bias,
                   at::Tensor& y, bool relu, int64_t bn, int64_t splits, at::Tensor& ws, at::Tensor& counters);

void dec_attention_q8(bool self, const at::Tensor& active, const at::Tensor& q, at::Tensor& k8, at::Tensor& v8,
                      at::Tensor& ks, at::Tensor& vs, const at::Tensor& n, double scale, int64_t splits,
                      at::Tensor& part, at::Tensor& out);
void dec_scatter_kv_q8(const at::Tensor& kv, const at::Tensor& slot_of, const at::Tensor& frame_of, at::Tensor& ck,
                       at::Tensor& cv, at::Tensor& cks, at::Tensor& cvs);

void encoder_layers(at::Tensor& x, const std::vector<std::vector<at::Tensor>>& layers, const std::vector<double>& eps,
                    const at::Tensor& tiles, const at::Tensor& lo, const at::Tensor& hi, int64_t rel_off,
                    int64_t max_len, double scale, bool int8, const std::vector<int64_t>& cfg);

void quantize_rows(const at::Tensor& x, at::Tensor& q, at::Tensor& s);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm, "bf16 GEMM with fused epilogue: out = epi(a @ w^T + bias)");
  m.def("encoder_layers", &encoder_layers, "the encoder's layers on packed frames, launched from C++");
  m.def("gemm_i8", &gemm_i8, "int8 GEMM (W8A8) with fused dequantization and epilogue");
  m.def("quantize_rows", &quantize_rows, "bf16 rows (1280) to int8 with per-row scales");
  m.def("layernorm_q8", &layernorm_q8, "LayerNorm(s) with int8 output and per-row scale");
  m.def("sub_conv0_dw", &sub_conv0_dw, "subsampling: conv0 + ReLU + depthwise, per sequence, channels-last");
  m.def("sub_dw", &sub_dw, "subsampling: depthwise 3x3 stride 2 per sequence, channels-last");
  m.def("layernorm", &layernorm, "h = LN(x), or x = LN1(x) and h = LN2(x)");
  m.def("dwconv_bn_silu", &dwconv_bn_silu, "depthwise conv + BatchNorm + SiLU per packed sequence");
  m.def("dec_w8_linear", &dec_w8_linear, "decoder projection with int8 weights (W8A16), split-K");
  m.def("dec_embed", &dec_embed, "decoder step: embedding + LayerNorms per active slot");
  m.def("dec_add_ln", &dec_add_ln, "x += y; h = LN(x)");
  m.def("dec_attention", &dec_attention, "decoder self/cross attention of one token per active slot");
  m.def("dec_attention_q8", &dec_attention_q8, "decoder self/cross attention over int8 caches");
  m.def("dec_scatter_kv_q8", &dec_scatter_kv_q8, "cross-attention keys/values into int8 slots with scales");
  m.def("dec_pick", &dec_pick, "greedy choice, prompt forcing and end bookkeeping per active slot");
  m.def("dec_scatter_kv", &dec_scatter_kv, "cross-attention keys/values of packed frames into their slots");
  m.def("relpos_attention", &relpos_attention, "relative-position self-attention over packed sequences");
}
