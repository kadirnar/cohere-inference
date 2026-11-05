// The encoder's 48 layers launched from C++ in one call: from Python each of the ~700 launches of a forward cost
// ~25 us of interpreter, argument parsing and GIL (beside the decoder thread), about 20 ms per encoder call however few
// frames it had. The sequence is fast_encoder.py's, kernel for kernel.
#include <torch/extension.h>

#include <vector>

void gemm(const at::Tensor& a, const at::Tensor& w, const c10::optional<at::Tensor>& bias, at::Tensor& out,
          int64_t epi, const c10::optional<at::Tensor>& resid, double alpha, int64_t cfg);
void gemm_i8(const at::Tensor& a, const at::Tensor& sa, int64_t group, const at::Tensor& w, const at::Tensor& sw,
             const c10::optional<at::Tensor>& bias, at::Tensor& out, int64_t epi, const c10::optional<at::Tensor>& resid,
             double alpha, const c10::optional<at::Tensor>& qs, int64_t cfg);
void layernorm(at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1, double eps1,
               const c10::optional<at::Tensor>& w2, const c10::optional<at::Tensor>& b2, double eps2, at::Tensor& h);
void layernorm_q8(at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1, double eps1,
                  const c10::optional<at::Tensor>& w2, const c10::optional<at::Tensor>& b2, double eps2, at::Tensor& hq,
                  at::Tensor& hs);
void dwconv_bn_silu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& cb, const at::Tensor& bn_scale,
                    const at::Tensor& bn_shift, const at::Tensor& seq_lo, const at::Tensor& seq_hi, at::Tensor& y);
void quantize_rows(const at::Tensor& x, at::Tensor& q, at::Tensor& s);
void gemm_sp(const at::Tensor& x, const at::Tensor& sx, int64_t group, const at::Tensor& wc, const at::Tensor& e,
             const at::Tensor& sw, const c10::optional<at::Tensor>& bias, const at::Tensor& r, at::Tensor& out,
             double alpha, int64_t cfg);
void dwconv_bn_silu_q8(const at::Tensor& x, const at::Tensor& w, const at::Tensor& cb, const at::Tensor& bn_scale,
                       const at::Tensor& bn_shift, const at::Tensor& seq_lo, const at::Tensor& seq_hi, at::Tensor& hq,
                       at::Tensor& hs);
void relpos_attention(const at::Tensor& qkv, const at::Tensor& rk, const at::Tensor& bu, const at::Tensor& bv,
                      const at::Tensor& tiles, int64_t rel_off, int64_t max_len, double scale, at::Tensor& out);

namespace {
enum { EPI_BIAS = 0, EPI_SILU = 1, EPI_RESID = 3, EPI_GLU = 4, EPI_SILU_Q8 = 5 };

// a layer's tensors, in fast_encoder.py's LAYER_FIELDS order
enum F {
  LN_FF1_W, LN_FF1_B, LN_ATT_W, LN_ATT_B, LN_CONV_W, LN_CONV_B, LN_FF2_W, LN_FF2_B, LN_OUT_W, LN_OUT_B,
  FF1_W1, FF1_B1, FF1_W2, FF1_B2, FF2_W1, FF2_B1, FF2_W2, FF2_B2,
  WQKV, BQKV, WO, BO, BU, BV, RK, WPW1, BPW1, WDW, BDW, BN_SCALE, BN_SHIFT, WPW2, BPW2,
  // int8 only: (weight, scale) of FF1 W1, W2, FF2 W1, W2, QKV, PW1, O, PW2
  Q_FF1_W1, Q_FF1_S1, Q_FF1_W2, Q_FF1_S2, Q_FF2_W1, Q_FF2_S1, Q_FF2_W2, Q_FF2_S2, Q_QKV_W, Q_QKV_S, Q_PW1_W, Q_PW1_S,
  Q_O_W, Q_O_S, Q_PW2_W, Q_PW2_S,
  // int8, FFN_SPARSE: the feed-forwards' second GEMMs 2:4 sparse (kept values, selectors), empty when dense
  SP_FF1_W, SP_FF1_E, SP_FF2_W, SP_FF2_E,
  N_FIELDS
};
enum E { EPS_FF1, EPS_ATT, EPS_CONV, EPS_FF2, EPS_OUT, N_EPS };
}  // namespace

// x (N, 1280) bf16 packed frames, in place. cfg: (bf16 GEMM config, int8 config of QKV, of the feed-forwards' second
// GEMM (grouped), of their first, FF group, of PW1, of O and PW2, of the sparse second GEMM)
void encoder_layers(at::Tensor& x, const std::vector<std::vector<at::Tensor>>& layers, const std::vector<double>& eps,
                    const at::Tensor& tiles, const at::Tensor& lo, const at::Tensor& hi, int64_t rel_off,
                    int64_t max_len, double scale, bool int8, const std::vector<int64_t>& cfg) {
  const int64_t n = x.size(0);
  const auto bf = x.options(), i8 = x.options().dtype(at::kChar), f32 = x.options().dtype(at::kFloat);
  const int64_t c_bf = cfg[0], c_i8 = cfg[1], c_grp = cfg[2], c_ff1 = cfg[3], group = cfg[4];
  const int64_t c_pw1 = cfg.size() > 5 ? cfg[5] : c_i8, c_o = cfg.size() > 6 ? cfg[6] : c_i8;
  const int64_t c_sp = cfg.size() > 7 ? cfg[7] : 0;
  at::Tensor qkv = at::empty({n, 3840}, bf), a = at::empty({n, 1280}, bf), z = at::empty({n, 1280}, bf);
  const c10::optional<at::Tensor> none;
  const size_t L = layers.size();
  auto ep = [&](size_t i, int e) { return eps[i * N_EPS + e]; };

  if (int8) {
    at::Tensor hq = at::empty({n, 1280}, i8), hs = at::empty({n, 1}, f32);
    at::Tensor tq = at::empty({n, 5120}, i8), ts = at::empty({n, 5120 / group}, f32);
    auto ffn = [&](const std::vector<at::Tensor>& ly, int w1, int s1, int b1, int w2, int s2, int b2, int spw, int spe) {
      gemm_i8(hq, hs, 1280, ly[w1], ly[s1], ly[b1], tq, EPI_SILU_Q8, none, 1.0, ts, c_ff1);
      if (ly[spw].numel() > 0)
        gemm_sp(tq, ts, group, ly[spw], ly[spe], ly[s2], ly[b2], x, x, 0.5, c_sp);
      else
        gemm_i8(tq, ts, group, ly[w2], ly[s2], ly[b2], x, EPI_RESID, x, 0.5, none, c_grp);
    };
    layernorm_q8(x, layers[0][LN_FF1_W], layers[0][LN_FF1_B], ep(0, EPS_FF1), none, none, 0.0, hq, hs);
    for (size_t i = 0; i < L; ++i) {
      const auto& ly = layers[i];
      ffn(ly, Q_FF1_W1, Q_FF1_S1, FF1_B1, Q_FF1_W2, Q_FF1_S2, FF1_B2, SP_FF1_W, SP_FF1_E);
      layernorm_q8(x, ly[LN_ATT_W], ly[LN_ATT_B], ep(i, EPS_ATT), none, none, 0.0, hq, hs);
      gemm_i8(hq, hs, 1280, ly[Q_QKV_W], ly[Q_QKV_S], ly[BQKV], qkv, EPI_BIAS, none, 1.0, none, c_i8);
      relpos_attention(qkv, ly[RK], ly[BU], ly[BV], tiles, rel_off, max_len, scale, a);
      quantize_rows(a, hq, hs);
      gemm_i8(hq, hs, 1280, ly[Q_O_W], ly[Q_O_S], ly[BO], x, EPI_RESID, x, 1.0, none, c_o);
      layernorm_q8(x, ly[LN_CONV_W], ly[LN_CONV_B], ep(i, EPS_CONV), none, none, 0.0, hq, hs);
      gemm_i8(hq, hs, 1280, ly[Q_PW1_W], ly[Q_PW1_S], ly[BPW1], a, EPI_GLU, none, 1.0, none, c_pw1);
      if (n >= 512) {  // the same int8 rows and scales; with fewer rows the 64-channel tiles are faster
        dwconv_bn_silu_q8(a, ly[WDW], ly[BDW], ly[BN_SCALE], ly[BN_SHIFT], lo, hi, hq, hs);
      } else {
        dwconv_bn_silu(a, ly[WDW], ly[BDW], ly[BN_SCALE], ly[BN_SHIFT], lo, hi, z);
        quantize_rows(z, hq, hs);
      }
      gemm_i8(hq, hs, 1280, ly[Q_PW2_W], ly[Q_PW2_S], ly[BPW2], x, EPI_RESID, x, 1.0, none, c_o);
      layernorm_q8(x, ly[LN_FF2_W], ly[LN_FF2_B], ep(i, EPS_FF2), none, none, 0.0, hq, hs);
      ffn(ly, Q_FF2_W1, Q_FF2_S1, FF2_B1, Q_FF2_W2, Q_FF2_S2, FF2_B2, SP_FF2_W, SP_FF2_E);
      if (i + 1 < L)
        layernorm_q8(x, ly[LN_OUT_W], ly[LN_OUT_B], ep(i, EPS_OUT), layers[i + 1][LN_FF1_W], layers[i + 1][LN_FF1_B],
                     ep(i + 1, EPS_FF1), hq, hs);
      else
        layernorm(x, ly[LN_OUT_W], ly[LN_OUT_B], ep(i, EPS_OUT), none, none, 0.0, x);
    }
    return;
  }
  at::Tensor h = at::empty({n, 1280}, bf), t = at::empty({n, 5120}, bf);
  layernorm(x, layers[0][LN_FF1_W], layers[0][LN_FF1_B], ep(0, EPS_FF1), none, none, 0.0, h);
  for (size_t i = 0; i < L; ++i) {
    const auto& ly = layers[i];
    gemm(h, ly[FF1_W1], ly[FF1_B1], t, EPI_SILU, none, 1.0, c_bf);
    gemm(t, ly[FF1_W2], ly[FF1_B2], x, EPI_RESID, x, 0.5, c_bf);
    layernorm(x, ly[LN_ATT_W], ly[LN_ATT_B], ep(i, EPS_ATT), none, none, 0.0, h);
    gemm(h, ly[WQKV], ly[BQKV], qkv, EPI_BIAS, none, 1.0, c_bf);
    relpos_attention(qkv, ly[RK], ly[BU], ly[BV], tiles, rel_off, max_len, scale, a);
    gemm(a, ly[WO], ly[BO], x, EPI_RESID, x, 1.0, c_bf);
    layernorm(x, ly[LN_CONV_W], ly[LN_CONV_B], ep(i, EPS_CONV), none, none, 0.0, h);
    gemm(h, ly[WPW1], ly[BPW1], a, EPI_GLU, none, 1.0, c_bf);
    dwconv_bn_silu(a, ly[WDW], ly[BDW], ly[BN_SCALE], ly[BN_SHIFT], lo, hi, z);
    gemm(z, ly[WPW2], ly[BPW2], x, EPI_RESID, x, 1.0, c_bf);
    layernorm(x, ly[LN_FF2_W], ly[LN_FF2_B], ep(i, EPS_FF2), none, none, 0.0, h);
    gemm(h, ly[FF2_W1], ly[FF2_B1], t, EPI_SILU, none, 1.0, c_bf);
    gemm(t, ly[FF2_W2], ly[FF2_B2], x, EPI_RESID, x, 0.5, c_bf);
    if (i + 1 < L)
      layernorm(x, ly[LN_OUT_W], ly[LN_OUT_B], ep(i, EPS_OUT), layers[i + 1][LN_FF1_W], layers[i + 1][LN_FF1_B],
                ep(i + 1, EPS_FF1), h);
    else
      layernorm(x, ly[LN_OUT_W], ly[LN_OUT_B], ep(i, EPS_OUT), none, none, 0.0, x);
  }
}
