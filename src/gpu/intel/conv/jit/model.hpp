/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_INTEL_CONV_JIT_MODEL_HPP
#define GPU_INTEL_CONV_JIT_MODEL_HPP

#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "common/float16.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {
namespace model {

template <typename Type>
using vec1d = std::vector<Type>;

template <typename Type>
using vec2d = std::vector<std::vector<Type>>;

enum class hw_t {
    undef,
    xehpg,
    xehpc,
};

inline hw_t to_hw(const std::string &s) {
    if (s == "xehpg") return hw_t::xehpg;
    if (s == "xehpc") return hw_t::xehpc;
    gpu_assert(false);
    return hw_t::undef;
}

enum class fma_t {
    undef,
    dpas,
    mad,
};

inline fma_t to_fma(const std::string &s) {
    if (s == "mad") return fma_t::mad;
    if (s == "dpas") return fma_t::dpas;
    gpu_assert(false);
    return fma_t::undef;
}

enum class prop_t {
    undef,
    fwd,
    bwd_d,
    bwd_w,
};

inline prop_t to_prop(const std::string &s) {
    if (s == "fwd") return prop_t::fwd;
    if (s == "bwd_d") return prop_t::bwd_d;
    if (s == "bwd_w") return prop_t::bwd_w;
    gpu_assert(false);
    return prop_t::undef;
}

enum class type_t { undef, d8, d16, d32, d64 };

inline void get_types(const std::string &type_cfg, prop_t prop,
        type_t &src_type, type_t &dst_type) {
    const std::pair<const char *, type_t> all_types[] = {
            std::make_pair("bf16", type_t::d16),
            std::make_pair("f16", type_t::d16),
            std::make_pair("f32", type_t::d32),
            std::make_pair("f64", type_t::d64),
            std::make_pair("s32", type_t::d32),
            std::make_pair("hf8", type_t::d8),
            std::make_pair("bf8", type_t::d8),
            std::make_pair("s8", type_t::d8),
            std::make_pair("tf32", type_t::d32),
            std::make_pair("u8", type_t::d8),
    };
    const int ntypes = 3;
    type_t types[ntypes] = {type_t::undef, type_t::undef, type_t::undef};
    size_t pos = 0;
    int idx = 0;
    while (pos < type_cfg.length() && idx < ntypes) {
        for (auto &p : all_types) {
            size_t len = std::strlen(p.first);
            if (type_cfg.compare(pos, len, p.first) == 0) {
                types[idx++] = p.second;
                pos += std::strlen(p.first);
                break;
            }
        }
    }
    if (pos == type_cfg.length() && idx == 1) {
        while (idx < ntypes)
            types[idx++] = types[0];
    }
    if (pos != type_cfg.length() || idx != ntypes) {
        std::cout << type_cfg << std::endl;
        gpu_assert(false);
    }
    switch (prop) {
        case prop_t::fwd: break;
        case prop_t::bwd_d: std::swap(types[0], types[2]); break;
        case prop_t::bwd_w: std::swap(types[1], types[2]); break;
        default: gpu_assert(false);
    }
    src_type = types[0];
    dst_type = types[2];
}

inline type_t to_src_type(
        const std::string &type_cfg, prop_t prop = prop_t::fwd) {
    type_t src_type;
    type_t dst_type;
    get_types(type_cfg, prop, src_type, dst_type);
    return src_type;
}

inline type_t to_dst_type(
        const std::string &type_cfg, prop_t prop = prop_t::fwd) {
    type_t src_type;
    type_t dst_type;
    get_types(type_cfg, prop, src_type, dst_type);
    return dst_type;
}

struct hw_config_t {
    static constexpr int default_eus = 512;

    hw_t hw = hw_t::undef;
    fma_t fma = fma_t::undef;
    int eus = 0;
    float freq = 0;
    int eus_per_sublice = 0;
    int ops_per_clock = 0;
    int threads_per_eu = 0;

    hw_config_t() = default;

    hw_config_t(hw_t hw, fma_t fma, type_t src_type, int eus = 0)
        : hw(hw), fma(fma), eus(eus) {
        if (eus == 0) this->eus = default_eus;
        int s8_dpas_ops_per_clock = 0;
        int f32_mad_ops_per_clock = 0;
        switch (hw) {
            case hw_t::xehpg:
                freq = 2.05e9;
                eus_per_sublice = 16;
                s8_dpas_ops_per_clock = 512;
                f32_mad_ops_per_clock = 16;
                break;
            case hw_t::xehpc:
                freq = 1.6e9;
                eus_per_sublice = 8;
                s8_dpas_ops_per_clock = 1024;
                f32_mad_ops_per_clock = 32;
                break;
            default: gpu_assert(false); break;
        }
        bool is_dpas = (fma == fma_t::dpas);
        switch (src_type) {
            case type_t::d8: ops_per_clock = s8_dpas_ops_per_clock; break;
            case type_t::d16: ops_per_clock = s8_dpas_ops_per_clock / 2; break;
            case type_t::d32:
                ops_per_clock = (is_dpas ? s8_dpas_ops_per_clock / 4
                                         : f32_mad_ops_per_clock);
                break;
            case type_t::d64:
                ops_per_clock = (is_dpas ? s8_dpas_ops_per_clock / 8
                                         : f32_mad_ops_per_clock / 2);
                break;
            default: gpu_assert(false); break;
        }
        threads_per_eu = (is_dpas ? 4 : 8);
    }

    int max_tgs() const {
        int subslices_per_tile = eus / eus_per_sublice;
        return subslices_per_tile * threads_per_eu;
    }

    int max_threads() const { return eus * threads_per_eu; }

    float max_gops_per_sec() const {
        float max_ops_per_sec = freq * eus * ops_per_clock;
        return max_ops_per_sec / 1e9f;
    }
};

// Metric function to use for model training.
enum class metric_t {
    undef,
    mse, // Mean squared error.
    msre, // Mean squared relative error.
};

inline metric_t to_metric(const std::string &s) {
    if (s == "mse") return metric_t::mse;
    if (s == "msre") return metric_t::msre;
    gpu_assert(false);
    return metric_t::undef;
}

// Score function to use to assess prediction error.
enum class score_t {
    undef,
    r2, // Coefficient of determination.
    mae, // Mean absolute error.
    mape, // Mean absolute percentage error.
};

inline score_t to_score(const std::string &s) {
    if (s == "r2") return score_t::r2;
    if (s == "mae") return score_t::mae;
    if (s == "mape") return score_t::mape;
    gpu_assert(false);
    return score_t::undef;
}

// Convolution training sample converted to BMNK-notation.
struct bmnk_conv_sample_t {
    prop_t prop;
    type_t src_type;
    type_t dst_type;
    hw_config_t hw_cfg;
    dim_t b, m, n, k;
    dim_t bt, mt, nt, kt;
    dim_t bl, ml, nl, kl;
    dim_t bi, mi, ni, ki;
    float sec = 0;
    float gops_sec = 0;
    float weight = 1;

    bmnk_conv_sample_t() = default;

    float ops() const { return 2.0f * b * m * n * k; }

    float thr_util() const {
        return std::min(1.0f, threads() / (float)hw_cfg.max_threads());
    }

    float wave_util() const {
        int64_t waves = utils::div_up(threads(), hw_cfg.max_threads());
        return (float)(threads() / (waves * hw_cfg.max_threads()));
    }

    float tg_util() const {
        float ntgs = 1.0f;
        ntgs *= utils::div_up(b, bl * bt * bi);
        ntgs *= utils::div_up(m, ml * mt * mi);
        ntgs *= utils::div_up(n, nl * nt * ni);
        ntgs *= utils::div_up(k, kl * kt * ki);
        return std::min(1.0f, ntgs / hw_cfg.max_tgs());
    }

    int64_t threads() const {
        int64_t ret = 1;
        ret *= utils::div_up(b, bl * bi);
        ret *= utils::div_up(m, ml * mi);
        ret *= utils::div_up(n, nl * ni);
        ret *= utils::div_up(k, kl * ki);
        return ret;
    }

    float is_dpas() const { return (hw_cfg.fma == fma_t::dpas) ? 1.0f : 0.0f; }

    float is_dpasw_hint() const {
        if (hw_cfg.fma != fma_t::dpas) return 0.0f;
        if (hw_cfg.hw != hw_t::xehpg) return 0.0f;
        return mi % 2 == 0 && nt % 2 == 0;
    }

    float with_atomic() const {
        dim_t k_tg = kl * kt * ki;
        dim_t k_rounded = utils::rnd_up(k, k_tg);
        return k_rounded > k_tg ? 1.0f : 0.0f;
    }

    float eff() const { return ops() / 1e9f / sec / hw_cfg.max_gops_per_sec(); }

    static std::vector<const char *> feature_names() {
        std::vector<const char *> ret;
        ret.push_back("hw");
        ret.push_back("thr_util");
        ret.push_back("wave_util");
        ret.push_back("tg_util");
        ret.push_back("ops");
        ret.push_back("bmnk_g");
        ret.push_back("k_g");
        ret.push_back("mt");
        ret.push_back("nt");
        ret.push_back("kt");
        ret.push_back("bi");
        ret.push_back("mi");
        ret.push_back("ni");
        ret.push_back("ki");
        ret.push_back("kl");
        ret.push_back("is_dpas");
        ret.push_back("is_dpasw_hint");
        ret.push_back("src_type");
        ret.push_back("dst_type");
        return ret;
    }

    std::vector<float> to_x() const {
        std::vector<float> ret;
        ret.push_back((float)hw_cfg.hw);
        ret.push_back(thr_util());
        ret.push_back(wave_util());
        ret.push_back(tg_util());
        ret.push_back(ops());
        dim_t bg = b / (bl * bt * bi);
        dim_t mg = m / (ml * mt * mi);
        dim_t ng = n / (nl * nt * ni);
        ret.push_back((float)bg * mg * ng);
        ret.push_back((float)(k / (kl * kt * ki)));
        ret.push_back((float)mt);
        ret.push_back((float)nt);
        ret.push_back((float)kt);
        ret.push_back((float)bi);
        ret.push_back((float)mi);
        ret.push_back((float)ni);
        ret.push_back((float)ki);
        ret.push_back((float)kl);
        ret.push_back(is_dpas());
        ret.push_back(is_dpasw_hint());
        ret.push_back((float)((int)src_type));
        ret.push_back((float)((int)dst_type));
        return ret;
    }

    float to_y() const { return eff(); }

    float to_w() const { return weight; }

    std::string str() const {
        ostringstream_t oss;
        oss << "shape: b" << b << "m" << m << "n" << n << "k" << k;
        oss << " loop: b" << bl << "m" << ml << "n" << nl << "k" << kl;
        oss << " tg: b" << bt << "m" << mt << "n" << nt << "k" << kt;
        oss << " iter: b" << bi << "m" << mi << "n" << ni << "k" << ki;
        return oss.str();
    }
};

// Convolution training sample.
struct conv_sample_t {
    struct tile_t {
        dim_t g, mb;
        dim_t oc, ic;
        dim_t id, ih, iw;
        dim_t od, oh, ow;
        dim_t kd, kh, kw;
    };

    prop_t prop;
    type_t src_type;
    type_t dst_type;
    hw_config_t hw_cfg;
    tile_t shape;
    tile_t loop;
    tile_t tg;
    tile_t iter;
    float sec = 0;
    float gops_sec = 0;
    bool transpose;

    conv_sample_t() = default;
    conv_sample_t(const std::string &hw, const std::string &fma,
            const std::string &prop, const std::string &type_cfg,
            const std::string &desc, const std::string &loop,
            const std::string &tg, const std::string &iter, float sec,
            float gops_sec, bool transpose = false)
        : prop(to_prop(prop))
        , src_type(to_src_type(type_cfg, this->prop))
        , dst_type(to_dst_type(type_cfg, this->prop))
        , hw_cfg(to_hw(hw), to_fma(fma), src_type)
        , shape(parse_tile(desc, /*do_promote=*/true))
        , loop(parse_tile(loop))
        , tg(parse_tile(tg))
        , iter(parse_tile(iter))
        , sec(sec)
        , gops_sec(gops_sec)
        , transpose(transpose) {
        pad();
    }

    void pad() {
        auto pad_dim = [](dim_t &dim, dim_t loop, dim_t tg, dim_t iter) {
            if (iter == -1) return;
            dim = utils::rnd_up(dim, loop * tg * iter);
        };
#define PAD_DIM(name) pad_dim(shape.name, loop.name, tg.name, iter.name)
        PAD_DIM(g);
        PAD_DIM(mb);
        PAD_DIM(oc);
        PAD_DIM(ic);
        PAD_DIM(id);
        PAD_DIM(ih);
        PAD_DIM(iw);
        PAD_DIM(od);
        PAD_DIM(oh);
        PAD_DIM(ow);
        PAD_DIM(kd);
        PAD_DIM(kh);
        PAD_DIM(kw);
#undef PAD_DIM
    }

    float eff() const {
        dim_t b, m, n, k;
        to_gemm_tile(shape, b, m, n, k);
        float ops = 2.0f * b * m * n * k;
        return ops / 1e9f / sec / hw_cfg.max_gops_per_sec();
    }

    bmnk_conv_sample_t to_bmnk_conv_sample() const {
        bmnk_conv_sample_t s;
        s.prop = prop;
        s.src_type = src_type;
        s.dst_type = dst_type;
        s.hw_cfg = hw_cfg;
        to_gemm_tile(shape, s.b, s.m, s.n, s.k);
        to_gemm_tile(loop, s.bl, s.ml, s.nl, s.kl);
        to_gemm_tile(tg, s.bt, s.mt, s.nt, s.kt);
        to_gemm_tile(iter, s.bi, s.mi, s.ni, s.ki);
        s.sec = sec;
        s.gops_sec = gops_sec;
        return s;
    }

    static int parse_dim(
            const std::string &s, const char *name, int default_value = -1) {
        int ret = default_value;
        auto pos = s.find(name);
        if (pos == std::string::npos) return ret;
        size_t i0 = pos + std::strlen(name);
        size_t i = i0;
        while (i < s.length() && std::isdigit(s[i]))
            i++;
        ret = std::stoi(s.substr(i0, i - i0));
        return ret;
    }

    tile_t parse_tile(const std::string &s, bool do_promote = false) const {
        tile_t ret;
        ret.g = parse_dim(s, "g");
        ret.mb = parse_dim(s, "mb");
        ret.oc = parse_dim(s, "oc");
        ret.ic = parse_dim(s, "ic");
        ret.id = parse_dim(s, "id");
        ret.ih = parse_dim(s, "ih");
        ret.iw = parse_dim(s, "iw");
        ret.od = parse_dim(s, "od");
        ret.oh = parse_dim(s, "oh");
        ret.ow = parse_dim(s, "ow");
        ret.kd = parse_dim(s, "kd");
        ret.kh = parse_dim(s, "kh");
        ret.kw = parse_dim(s, "kw");
        // Promote missing spatial dimensions based on others.
        auto promote = [](dim_t &d, dim_t &h, dim_t &w) {
            if (d != -1 && h == -1 && w == -1) {
                h = w = d;
            } else if (d == -1 && h != -1 && w == -1) {
                w = h;
            }
        };
        if (do_promote) {
            promote(ret.id, ret.ih, ret.iw);
            promote(ret.od, ret.oh, ret.ow);
            promote(ret.kd, ret.kh, ret.kw);
        }
        return normalize_tile(ret);
    }

    void to_gemm_tile(
            const tile_t &t, dim_t &b, dim_t &m, dim_t &n, dim_t &k) const {
        b = t.g;
        switch (prop) {
            case prop_t::fwd:
                m = t.mb * t.od * t.oh * t.ow;
                n = t.oc;
                k = t.ic * t.kd * t.kh * t.kw;
                break;
            case prop_t::bwd_d:
                m = t.mb * t.id * t.ih * t.iw;
                n = t.ic;
                k = t.oc * t.kd * t.kh * t.kw;
                break;
            case prop_t::bwd_w:
                m = t.ic * t.kd * t.kh * t.kw;
                n = t.oc;
                k = t.mb * t.od * t.oh * t.ow;
                break;
            default: gpu_assert(false);
        }
        if (transpose) std::swap(m, n);
    }

    // Initializes missing dimensions to one.
    tile_t normalize_tile(const tile_t &t) const {
        tile_t ret = t;
        std::vector<dim_t *> dims = {
                &ret.g, &ret.mb, &ret.oc, &ret.ic, &ret.kd, &ret.kh, &ret.kw};
        switch (prop) {
            case prop_t::fwd:
            case prop_t::bwd_w:
                dims.push_back(&ret.od);
                dims.push_back(&ret.oh);
                dims.push_back(&ret.ow);
                break;
            case prop_t::bwd_d:
                dims.push_back(&ret.id);
                dims.push_back(&ret.ih);
                dims.push_back(&ret.iw);
                break;
            default: gpu_assert(false);
        }
        for (auto *d : dims)
            if (*d == -1) *d = 1;
        return ret;
    }
};

// Histogram class, responsible for bucketing of feature values. Input float
// values are bucketed into up to 128 sub-ranges. Model training is then done
// for 8-bit input values.
class histogram_t {
public:
    static const int bucket_count = 256;

    histogram_t() = default;

    histogram_t(const vec2d<float> &X) {
        auto &x0 = X[0];
        int np = (int)X.size();
        int nf = (int)x0.size();
        std::vector<std::map<float, int>> stats(nf);
        for (auto &x : X) {
            for (int i = 0; i < nf; i++) {
                stats[i][x[i]]++;
            }
        }
        buckets_.resize(nf);
        for (int i = 0; i < nf; i++) {
            int per_bucket = std::max(1, np / bucket_count);
            int cur = 0;
            for (auto &kv : stats[i]) {
                cur += kv.second;
                if (cur > per_bucket) {
                    cur = 0;
                    buckets_[i].push_back(kv.first);
                }
            }
            gpu_assert((int)buckets_[i].size() <= bucket_count);
        }
    }

    template <typename T>
    T to_bucket(float value, int idx) const {
        auto &b = buckets_[idx];
        int n = (int)b.size();
        for (int i = 0; i < n; i++)
            if (b[i] >= value) return into<T>(i);
        return into<T>(n);
    }

    template <typename T>
    vec1d<T> to(const vec1d<float> &x) const {
        gpu_assert(x.size() == buckets_.size());
        vec1d<T> ret(x.size());
        for (int i = 0; i < (int)x.size(); i++)
            ret[i] = to_bucket<T>(x[i], i);
        return ret;
    }

    template <typename T>
    vec2d<T> to(const vec2d<float> &X) const {
        vec2d<T> ret;
        for (auto &x : X)
            ret.push_back(to<T>(x));
        return ret;
    }

    void serialize(serialization_stream_t &s) const { s.append(buckets_); }

    static histogram_t deserialize(deserializer_t &d) {
        histogram_t h;
        d.pop(h.buckets_);
        return h;
    }

private:
    vec2d<float> buckets_;
};

inline float r2_score(
        const std::vector<float> &y, const std::vector<float> &y_pred) {
    float u = 0;
    float v = 0;
    float y_mean = 0;
    int n = (int)y.size();
    for (int i = 0; i < n; i++)
        y_mean += y[i];
    y_mean /= n;
    for (int i = 0; i < n; i++) {
        u += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
        v += (y[i] - y_mean) * (y[i] - y_mean);
    }
    return 1 - u / v;
}

inline float mae_score(
        const std::vector<float> &y, const std::vector<float> &y_pred) {
    int n = (int)y.size();
    float err = 0;
    for (int i = 0; i < n; i++) {
        err += std::abs(y[i] - y_pred[i]);
    }
    return -err / n;
}

inline float mape_score(
        const std::vector<float> &y, const std::vector<float> &y_pred) {
    float eps = std::numeric_limits<float>::epsilon();
    int n = (int)y.size();
    float err = 0;
    for (int i = 0; i < n; i++) {
        err += std::abs(y[i] - y_pred[i]) / std::max(eps, std::abs(y[i]));
    }
    return -err / n;
}

inline float score(const std::vector<float> &y,
        const std::vector<float> &y_pred, score_t score) {
    switch (score) {
        case score_t::r2: return r2_score(y, y_pred);
        case score_t::mae: return mae_score(y, y_pred);
        case score_t::mape: return mape_score(y, y_pred);
        default: gpu_assert(false);
    }
    return 0;
}

struct tree_node_t {
    int feature_idx = -1;
    float value;
    int left = -1;
    int right = -1;
};

// Decision tree.
template <typename x_type>
class tree_t {
public:
    tree_t() = default;

    tree_t(int max_depth, int subsamples, metric_t metric = metric_t::mse)
        : max_depth_(max_depth), subsamples_(subsamples), metric_(metric) {}

    void fit(const vec2d<x_type> &X, const vec1d<float> &y,
            const vec1d<float> &w = {}) {
        // Use an array of indices to track features going to the left/right
        // sub-trees to avoid shuffling the original data.
        vec1d<int> idxs(X.size());
        std::iota(idxs.begin(), idxs.end(), 0);
        nfeatures_ = (int)X[0].size();

        int root_idx = create_node();
        build_tree(root_idx, X, y, w, idxs, 0, (int)idxs.size(), 0);
    }

    int node_count() const { return (int)nodes_.size(); }

    int feature_count() const { return nfeatures_; }

    float predict(const vec1d<x_type> &x) const {
        gpu_assert((int)x.size() == (int)nfeatures_);
        return predict_impl(x, 0);
    }

    vec1d<float> predict(const vec2d<x_type> &X) const {
        vec1d<float> y;
        for (auto &x : X) {
            y.push_back(predict(x));
        }
        return y;
    }

    // Feature importance is computed based on how often this feature is used
    // in comparisons.
    vec1d<float> feature_importances() const {
        vec1d<float> count(feature_count());
        std::function<void(int)> walk;
        int non_leaf_nodes = 0;
        walk = [&](int idx) {
            auto &node = get_node(idx);
            if (node.feature_idx == -1) return;
            count[node.feature_idx]++;
            non_leaf_nodes++;
            walk(node.left);
            walk(node.right);
        };
        walk(0);
        gpu_assert(non_leaf_nodes * 2 + 1 == node_count());
        for (auto &c : count)
            c /= non_leaf_nodes;
        return count;
    }

    void print_info() const {
        std::cout << "Tree" << std::endl;
        std::cout << "  Features:   " << feature_count() << std::endl;
        std::cout << "  Max depth:  " << max_depth_ << std::endl;
        std::cout << "  Nodes:      " << node_count() << std::endl;
    }

    void serialize(serialization_stream_t &s) const {
        s.append(nfeatures_);
        s.append(max_depth_);
        s.append(subsamples_);
        s.append(metric_);

        std::vector<uint8_t> node_data;
        size_t off = 0;
        serialize_node(node_data, off);

        s.append(node_data);
    }

    static tree_t deserialize(deserializer_t &d) {
        tree_t t;
        d.pop(t.nfeatures_);
        d.pop(t.max_depth_);
        d.pop(t.subsamples_);
        d.pop(t.metric_);
        std::vector<uint8_t> node_data;
        d.pop(node_data);
        size_t off = 0;
        t.deserialize_node(node_data, off);
        return t;
    }

private:
    int create_node() { return reserved_nodes_++; }

    tree_node_t &get_node(int idx) {
        if (idx >= (int)nodes_.size()) { nodes_.resize(idx + 1); }
        return nodes_[idx];
    }

    const tree_node_t &get_node(int idx) const {
        gpu_assert(idx < reserved_nodes_);
        return nodes_[idx];
    }

    void build_tree(int node_idx, const vec2d<x_type> &X, const vec1d<float> &y,
            const vec1d<float> &w, vec1d<int> &idxs, int beg, int end,
            int depth) {
        auto &node = get_node(node_idx);
        if (end == beg + 1 || depth > max_depth_) {
            node.value = get_mean(y, idxs, beg, end);
            return;
        }
        int feature_idx = -1;
        x_type threshold = 0;
        find_best_split(X, y, w, idxs, beg, end, feature_idx, threshold);
        int nleft = 0;
        for (int i = beg; i < end; i++)
            if (X[idxs[i]][feature_idx] <= threshold) nleft++;
        if (nleft == 0 || nleft == end - beg) {
            node.feature_idx = -1;
            node.value = get_mean(y, idxs, beg, end);
            return;
        }
        int left = create_node();
        int right = create_node();
        node.feature_idx = feature_idx;
        node.value = threshold;
        node.left = left;
        node.right = right;
        std::nth_element(idxs.begin() + beg, idxs.begin() + beg + nleft,
                idxs.begin() + end, [&](int a, int b) {
            return X[a][feature_idx] < X[b][feature_idx];
        });
        if (subsamples_ > 0) {
            intel::jit::ir_utils::fast_random_t r;
            r.shuffle(idxs.begin() + beg, idxs.begin() + beg + nleft);
            r.shuffle(idxs.begin() + beg + nleft, idxs.begin() + end);
        }

        build_tree(left, X, y, w, idxs, beg, beg + nleft, depth + 1);
        build_tree(right, X, y, w, idxs, beg + nleft, end, depth + 1);
    }

    void find_best_split(const vec2d<x_type> &X, const vec1d<float> &y,
            const vec1d<float> &w, const vec1d<int> &idxs, int beg, int end,
            int &feature_idx, x_type &threshold) const {
        // Use first subsamples samples to find the best split. Assume the
        // range is randomly pre-shuffled.
        if (subsamples_ > 0) end = beg + std::min(subsamples_, end - beg);
        int n = end - beg;
        int nsplits = std::numeric_limits<x_type>::max() + 1;
        // Use compact local vectors to speed up processing.
        vec2d<x_type> X_local(n);
        vec1d<float> y_local(n);
        vec1d<float> w_local(w.empty() ? 0 : n);
        // Check the feature values to avoid unnecessary split checks.
        std::vector<std::vector<bool>> splits(
                nfeatures_, std::vector<bool>(nsplits));
#ifdef DNNL_GPU_MODEL_USE_OMP
#pragma omp parallel for
#endif
        for (int i = beg; i < end; i++) {
            X_local[i - beg] = X[idxs[i]];
            y_local[i - beg] = y[idxs[i]];
            if (!w.empty()) w_local[i - beg] = w[idxs[i]];
            for (int j = 0; j < nfeatures_; j++)
                splits[j][X_local[i - beg][j]] = true;
        }
        float min_err = std::numeric_limits<float>::max();
#ifdef DNNL_GPU_MODEL_USE_OMP
#pragma omp parallel for
#endif
        for (int ij = 0; ij < nfeatures_ * nsplits; ij++) {
            int f = ij / nsplits;
            int v = ij % nsplits;
            if (!splits[f][v]) continue;
            float err = get_err(X_local, y_local, w_local, f, (float)v);
#ifdef DNNL_GPU_MODEL_USE_OMP
#pragma omp critical
#endif
            if (err < min_err) {
                min_err = err;
                feature_idx = f;
                threshold = into<x_type>(v);
            }
        }
    }

    float get_mean(const vec1d<float> &y, const vec1d<int> &idxs, int beg,
            int end) const {
        float mean = 0;
        for (int i = beg; i < end; i++) {
            int idx = idxs[i];
            mean += y[idx];
        }
        return mean / (end - beg);
    }

    float get_err(const vec2d<x_type> &X, const vec1d<float> &y,
            const vec1d<float> &w, int f, float threshold) const {
        float left_mean = 0;
        float right_mean = 0;
        int total = (int)X.size();
        int nleft = 0;
        std::vector<bool> lr(total);
        for (int i = 0; i < total; i++) {
            if (X[i][f] <= threshold) {
                left_mean += y[i];
                lr[i] = true;
                nleft++;
            } else {
                right_mean += y[i];
            }
        }
        left_mean /= nleft;
        right_mean /= (total - nleft);
        float err = 0;
        switch (metric_) {
            case metric_t::mse:
                for (int i = 0; i < total; i++) {
                    float mean = lr[i] ? left_mean : right_mean;
                    float weight = w.empty() ? 1 : w[i];
                    float val = (y[i] - mean);
                    err += weight * val * val;
                }
                break;
            case metric_t::msre:
                for (int i = 0; i < total; i++) {
                    float mean = lr[i] ? left_mean : right_mean;
                    float weight = w.empty() ? 1 : w[i];
                    float val = (y[i] - mean) / y[i];
                    err += weight * val * val;
                }
                break;
            default: gpu_assert(false);
        }
        return err / total;
    }

    float predict_impl(const vec1d<x_type> &x, int node_idx) const {
        auto &node = get_node(node_idx);
        if (node.feature_idx == -1) return node.value;
        if (x[node.feature_idx] <= node.value)
            return predict_impl(x, node.left);
        return predict_impl(x, node.right);
    }

    static void write_bits(
            std::vector<uint8_t> &data, size_t &off, uint32_t v, int nbits) {
        for (int i = 0; i < nbits; i++, off++) {
            if (off % 8 == 0) data.push_back(0);
            if ((v >> i) & 1) data.back() |= (uint8_t)(1 << (off % 8));
        }
    }

    static uint32_t read_bits(
            const std::vector<uint8_t> &data, size_t &off, int nbits) {
        gpu_assert(off + nbits <= data.size() * 8);
        uint32_t v = 0;
        for (int i = 0; i < nbits; i++, off++)
            v |= (uint32_t)((data[off / 8] >> (off % 8)) & 1) << i;
        return v;
    }

    // Bit-packed pre-order: leaf flag (omitted at depth > max_depth_, always a
    // leaf), then f16 leaf value or feature/threshold.
    void serialize_node(std::vector<uint8_t> &data, size_t &off, int idx = 0,
            int depth = 0) const {
        auto &node = get_node(idx);
        bool is_leaf = (node.feature_idx == -1);
        if (depth <= max_depth_) write_bits(data, off, is_leaf, 1);
        if (is_leaf) {
            write_bits(data, off, float16_t(node.value).raw, 16);
            return;
        }
        gpu_assert(depth <= max_depth_);
        gpu_assert(node.feature_idx < (1 << feature_bits));
        write_bits(data, off, node.feature_idx, feature_bits);
        write_bits(data, off, (uint32_t)node.value, threshold_bits);
        serialize_node(data, off, node.left, depth + 1);
        serialize_node(data, off, node.right, depth + 1);
    }

    int deserialize_node(
            const std::vector<uint8_t> &data, size_t &off, int depth = 0) {
        int idx = create_node();
        tree_node_t node;
        if (depth > max_depth_ || read_bits(data, off, 1)) {
            node.value = float16_t((uint16_t)read_bits(data, off, 16), true);
        } else {
            node.feature_idx = (int)read_bits(data, off, feature_bits);
            node.value = (float)read_bits(data, off, threshold_bits);
            node.left = deserialize_node(data, off, depth + 1);
            node.right = deserialize_node(data, off, depth + 1);
        }
        get_node(idx) = node;
        return idx;
    }

    static constexpr int feature_bits = 5;
    static constexpr int threshold_bits = 8 * sizeof(x_type);

    int nfeatures_ = 0;
    int max_depth_ = 0;
    int subsamples_ = 0;
    metric_t metric_ = metric_t::undef;

    int reserved_nodes_ = 0;
    std::vector<tree_node_t> nodes_;
};

// Gradient boosting regression.
class gradient_boost_regressor_t {
public:
    gradient_boost_regressor_t(int ntrees = 100, int max_depth = 5,
            float learning_rate = 0.1, int subsamples = 1000,
            metric_t metric = metric_t::mse)
        : learning_rate_(learning_rate) {
        for (int i = 0; i < ntrees; i++)
            trees_.emplace_back(max_depth, subsamples, metric);
    }

    void fit(const vec2d<float> &_X, const vec1d<float> &y,
            const vec1d<float> &w = {}) {
        hist_ = histogram_t(_X);
        for (auto v : y)
            f0_ += v;
        f0_ /= y.size();

        auto X = hist_.to<tree_type_t>(_X);
        auto fi = vec1d<float>(y.size(), f0_);
        for (auto &tree : trees_) {
            auto y_fi = sub(y, fi);
            tree.fit(X, y_fi, w);
            add(fi, tree.predict(X), learning_rate_);
        }
    }

    int tree_count() const { return (int)trees_.size(); }

    int feature_count() const {
        gpu_assert(!trees_.empty());
        return trees_[0].feature_count();
    }

    float predict(const vec1d<float> &_x,
            int max_trees = std::numeric_limits<int>::max()) const {
        auto x = hist_.to<tree_type_t>(_x);
        float y = f0_;
        max_trees = std::min(max_trees, tree_count());
        for (int i = 0; i < max_trees; i++) {
            y += trees_[i].predict(x) * learning_rate_;
        }
        return y;
    }

    vec1d<float> predict(const vec2d<float> &X,
            int max_trees = std::numeric_limits<int>::max()) const {
        vec1d<float> ret;
        for (auto &x : X)
            ret.push_back(predict(x, max_trees));
        return ret;
    }

    float score(const vec2d<float> &X, const vec1d<float> &y, score_t score,
            int max_trees = std::numeric_limits<int>::max()) const {
        auto y_pred = predict(X, max_trees);
        return model::score(y, y_pred, score);
    }

    std::vector<std::pair<std::string, float>> feature_importances(
            const std::vector<const char *> &feature_names) const {
        gpu_assert((int)feature_names.size() == feature_count());
        vec1d<float> fi(feature_count());
        for (auto &tree : trees_) {
            auto tree_fi = tree.feature_importances();
            for (int i = 0; i < feature_count(); i++) {
                fi[i] += tree_fi[i];
            }
        }
        for (int i = 0; i < feature_count(); i++) {
            fi[i] /= tree_count();
        }
        using entry_t = std::pair<std::string, float>;
        std::vector<entry_t> ret;
        ret.reserve(feature_count());
        for (int i = 0; i < feature_count(); i++) {
            ret.emplace_back(feature_names[i], fi[i]);
        }
        std::sort(ret.begin(), ret.end(),
                [&](const entry_t &a, const entry_t &b) {
            return a.second > b.second;
        });
        return ret;
    }

    void print_info(const std::vector<const char *> &feature_names,
            const std::string &prefix = "") const {
        std::cout << prefix << "Gradient boost regressor" << std::endl;
        std::cout << prefix << "  Features:   " << feature_count() << std::endl;
        std::cout << prefix << "  Trees:      " << tree_count() << std::endl;
        std::cout << prefix << "  Feature importances:" << std::endl;
        for (auto &kv : feature_importances(feature_names)) {
            std::cout << prefix << "    " << std::left << std::setw(20)
                      << kv.first << ": ";
            std::cout << std::fixed << std::setprecision(3) << kv.second
                      << std::endl;
        }
    }

    int serialized_size() const {
        serialization_stream_t s;
        serialize(s);
        return (int)s.get_data().size();
    }

    void serialize(serialization_stream_t &s) const {
        s.append(learning_rate_);
        s.append(hist_);
        s.append(f0_);
        s.append(trees_);
    }

    static gradient_boost_regressor_t deserialize(deserializer_t &d) {
        gradient_boost_regressor_t r;
        d.pop(r.learning_rate_);
        d.pop(r.hist_);
        d.pop(r.f0_);
        d.pop(r.trees_);
        return r;
    }

private:
    using tree_type_t = uint8_t;

    void add(vec1d<float> &c, const vec1d<float> &a, float b) {
        for (int i = 0; i < (int)c.size(); i++) {
            c[i] += a[i] * b;
        }
    }

    vec1d<float> sub(const vec1d<float> &a, const vec1d<float> &b) {
        vec1d<float> c(a.size());
        for (int i = 0; i < (int)c.size(); i++) {
            c[i] = a[i] - b[i];
        }
        return c;
    }

    float learning_rate_ = 0.1f;
    histogram_t hist_;

    float f0_ = 0.f;
    std::vector<tree_t<tree_type_t>> trees_;
};

} // namespace model
} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
