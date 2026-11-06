#!/usr/bin/env python3
# *******************************************************************************
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************
"""Binding to the forward SDPA cost model in the oneDNN library.

Part of scripts/sdpa_tuner; imported by the other scripts (README.md there,
Dev-mode interface). The model lives once, in src/gpu/intel/sdpa/select.cpp;
a ONEDNN_DEV_MODE build exports it through the dnnl_impl_sdpa_fwd_* C
functions (select_api.cpp), which this module calls with ctypes. The problem
and device are passed as the hw: and prb: strings of the fwd_select verbose
line; the dataclasses below only parse and re-create those strings and the
tuner's CSV columns.

The library is the path given to Library() (the scripts' --lib), else
$DNNL_LIB, else build/src/dnnl.dll or libdnnl.so under the repo.

    lib = Library()
    coefs = lib.seed_coefs("xe_hpg")   # dict, names from lib.coef_names
    cost, derived = lib.estimate(hw_str, prb_str, "32,16,16,8,16,2,8,4", coefs)
    ranked = lib.enumerate(hw_str, prb_str, coefs)

Run directly, it ranks the configs of a sample A750 problem, a quick check
that the library loads:

    python scripts/sdpa_tuner/sdpa_model.py [LIB]
"""

import ctypes
import os
import sys
from dataclasses import asdict, dataclass

MASKS = {"none": 0, "causal_top_left": 1, "causal_bottom_right": 2, "buffer": 3}


def cfg_str(c):
    return ",".join(str(x) for x in c)


def parse_cfg(s):
    v = tuple(int(x) for x in s.strip().split(","))
    if len(v) != 8:
        raise ValueError("config needs 8 integers: %r" % s)
    return v


@dataclass
class HW:
    arch: str = "xe_hpg"
    eu_count: int = 0
    eus_per_subslice: int = 0
    tpe128: int = 0
    tpe256: int = 0
    grf_bytes: int = 0
    slm_per_wg: int = 0
    slm_per_subslice: int = 0
    max_wg_items_128: int = 0
    l3_bytes: int = 0
    subgroup_size: int = 16
    integrated: bool = False
    systolic: bool = True

    @staticmethod
    def from_verbose(s):
        """Parse the hw:... field of the fwd_select verbose line."""
        kv = dict(tok.split("=", 1) for tok in s.split(";") if "=" in tok)
        tpe = kv.get("tpe", "0/0").split("/")
        slm = kv.get("slm", "0/0").split("/")
        return HW(
            arch=kv.get("arch", "xe_hpg"),
            eu_count=int(kv.get("eu", 0)),
            eus_per_subslice=int(kv.get("ss", 0)),
            tpe128=int(tpe[0]),
            tpe256=int(tpe[1]),
            grf_bytes=int(kv.get("grf", 0)),
            slm_per_wg=int(slm[0]),
            slm_per_subslice=int(slm[1]),
            max_wg_items_128=int(kv.get("wg", 0)),
            l3_bytes=int(kv.get("l3", 0)),
            subgroup_size=int(kv.get("sg", 16)),
            integrated=kv.get("integ", "0") == "1",
            systolic=kv.get("sys", "1") == "1",
        )

    def to_verbose(self):
        return (
            "arch=%s;eu=%d;ss=%d;tpe=%d/%d;grf=%d;slm=%d/%d;wg=%d;l3=%d;"
            "sg=%d;integ=%d;sys=%d"
            % (
                self.arch,
                self.eu_count,
                self.eus_per_subslice,
                self.tpe128,
                self.tpe256,
                self.grf_bytes,
                self.slm_per_wg,
                self.slm_per_subslice,
                self.max_wg_items_128,
                self.l3_bytes,
                self.subgroup_size,
                int(self.integrated),
                int(self.systolic),
            )
        )


@dataclass
class Problem:
    d_qk: int = 0
    d_v: int = 0
    d_max_kq: int = 0
    d_max_v: int = 0
    keys: int = 0
    queries: int = 0
    batch_heads: int = 0
    kv_group_size: int = 1
    mask: int = 0
    mask_broadcast_q: bool = True
    q_bits: int = 16
    k_bits: int = 16
    v_bits: int = 16
    dst_bits: int = 16
    mask_bits: int = 0
    k_quantized: bool = False
    v_quantized: bool = False
    k_group_size: int = 0
    v_group_size: int = 0
    kq_f16_acc: bool = False
    vs_f16_acc: bool = False
    f32: bool = False
    fma: bool = False
    q_align: int = 64
    k_align: int = 64
    v_align: int = 64
    dst_align: int = 64
    transpose_k: bool = False
    training: bool = False
    dropout: bool = False

    @staticmethod
    def from_verbose(s):
        """Parse the prb:... field of the fwd_select verbose line."""
        kv = dict(tok.split("=", 1) for tok in s.split(";") if "=" in tok)

        def pair(name, default="0/0"):
            return [int(x) for x in kv.get(name, default).split("/")]

        d = pair("d")
        dmax = pair("dmax")
        bits = pair("bits", "16/16/16/16/0")
        qz = pair("qz")
        gs = pair("gs")
        acc = pair("acc")
        al = pair("al", "64/64/64/64")
        return Problem(
            d_qk=d[0],
            d_v=d[1],
            d_max_kq=dmax[0],
            d_max_v=dmax[1],
            keys=int(kv.get("k", 0)),
            queries=int(kv.get("q", 0)),
            batch_heads=int(kv.get("bh", 0)),
            kv_group_size=int(kv.get("g", 1)),
            mask=int(kv.get("mask", 0)),
            mask_broadcast_q=kv.get("mbq", "1") == "1",
            q_bits=bits[0],
            k_bits=bits[1],
            v_bits=bits[2],
            dst_bits=bits[3],
            mask_bits=bits[4],
            k_quantized=qz[0] == 1,
            v_quantized=qz[1] == 1,
            k_group_size=gs[0],
            v_group_size=gs[1],
            kq_f16_acc=acc[0] == 1,
            vs_f16_acc=acc[1] == 1,
            f32=kv.get("f32", "0") == "1",
            fma=kv.get("fma", "0") == "1",
            q_align=al[0],
            k_align=al[1],
            v_align=al[2],
            dst_align=al[3],
            transpose_k=kv.get("tk", "0") == "1",
            training=kv.get("tr", "0") == "1",
            dropout=kv.get("do", "0") == "1",
        )

    def to_verbose(self):
        """The prb:... string, as fwd_problem_str() prints it."""
        return (
            "d=%d/%d;dmax=%d/%d;k=%d;q=%d;bh=%d;g=%d;mask=%d;mbq=%d;"
            "bits=%d/%d/%d/%d/%d;qz=%d/%d;gs=%d/%d;acc=%d/%d;f32=%d;fma=%d;"
            "al=%d/%d/%d/%d;tk=%d;tr=%d;do=%d"
            % (
                self.d_qk,
                self.d_v,
                self.d_max_kq,
                self.d_max_v,
                self.keys,
                self.queries,
                self.batch_heads,
                self.kv_group_size,
                self.mask,
                int(self.mask_broadcast_q),
                self.q_bits,
                self.k_bits,
                self.v_bits,
                self.dst_bits,
                self.mask_bits,
                int(self.k_quantized),
                int(self.v_quantized),
                self.k_group_size,
                self.v_group_size,
                int(self.kq_f16_acc),
                int(self.vs_f16_acc),
                int(self.f32),
                int(self.fma),
                self.q_align,
                self.k_align,
                self.v_align,
                self.dst_align,
                int(self.transpose_k),
                int(self.training),
                int(self.dropout),
            )
        )


PROBLEM_FIELDS = list(asdict(Problem()).keys())
HW_FIELDS = list(asdict(HW()).keys())


def _from_row(kind, row):
    obj = kind()
    for f in asdict(obj):
        if f not in row or row[f] == "":
            continue
        cur = getattr(obj, f)
        v = row[f]
        if isinstance(cur, bool):
            setattr(obj, f, v in ("1", "True", "true"))
        elif isinstance(cur, str):
            setattr(obj, f, v)
        else:
            setattr(obj, f, int(v))
    return obj


def problem_from_row(row):
    return _from_row(Problem, row)


def hw_from_row(row):
    return _from_row(HW, row)


def find_library(explicit=None):
    """The library to load: --lib, $DNNL_LIB, or build/src under the repo."""
    if explicit:
        return explicit
    if os.environ.get("DNNL_LIB"):
        return os.environ["DNNL_LIB"]
    names = ("dnnl.dll", "libdnnl.so", "libdnnl.so.3", "libdnnl.dylib")
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        for name in names:
            cand = os.path.join(here, "build", "src", name)
            if os.path.exists(cand):
                return cand
        here = os.path.dirname(here)
    raise FileNotFoundError(
        "no oneDNN library found; pass --lib or set DNNL_LIB"
    )


class Library:
    """ctypes wrapper of the dnnl_impl_sdpa_fwd_* dev-mode entry points."""

    def __init__(self, path=None):
        self.path = find_library(path)
        if sys.platform == "win32":
            # The Intel runtime DLLs the library needs are copied next to
            # benchdnn, not next to dnnl.dll, and ctypes does not search PATH
            # for dependencies; load them first so dnnl.dll resolves them
            # from the already loaded modules
            lib_dir = os.path.dirname(os.path.abspath(self.path))
            dirs = [lib_dir, os.path.join(lib_dir, "..", "tests", "benchdnn")]
            for root in filter(
                None,
                [
                    os.environ.get("ONEAPI_ROOT", ""),
                    r"C:\Program Files (x86)\Intel\oneAPI",
                ],
            ):
                dirs.append(os.path.join(root, "bin"))
                if os.path.isdir(root):
                    dirs += [
                        os.path.join(root, v, "bin")
                        for v in sorted(os.listdir(root), reverse=True)
                    ]
            for name in ("libiomp5md.dll", "svml_dispmd.dll", "libmmd.dll"):
                for d in dirs:
                    cand = os.path.join(d, name)
                    if os.path.isfile(cand):
                        try:
                            ctypes.CDLL(cand, winmode=0)
                            break
                        except OSError:
                            continue
            self._lib = ctypes.CDLL(self.path, winmode=0)
        else:
            self._lib = ctypes.CDLL(self.path)
        try:
            count = self._lib.dnnl_impl_sdpa_fwd_coef_count
        except AttributeError:
            raise RuntimeError(
                "%s has no dnnl_impl_sdpa_fwd_* symbols; it must "
                "be a ONEDNN_DEV_MODE build"
                % self.path
            )
        count.restype = ctypes.c_int
        name = self._lib.dnnl_impl_sdpa_fwd_coef_name
        name.restype = ctypes.c_char_p
        name.argtypes = [ctypes.c_int]
        self.coef_names = [name(i).decode() for i in range(count())]
        self._n = len(self.coef_names)
        self._seeds = self._lib.dnnl_impl_sdpa_fwd_coef_seeds
        self._seeds.restype = ctypes.c_int
        self._seeds.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self._estimate = self._lib.dnnl_impl_sdpa_fwd_estimate
        self._estimate.restype = ctypes.c_int
        self._estimate.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._enumerate = self._lib.dnnl_impl_sdpa_fwd_enumerate
        self._enumerate.restype = ctypes.c_int
        self._enumerate.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self._cost = ctypes.c_double()
        self._derived = (ctypes.c_int * 6)()

    def seed_coefs(self, arch):
        buf = (ctypes.c_double * self._n)()
        if self._seeds(arch.encode(), buf, self._n) != 0:
            raise ValueError("unknown architecture %r" % arch)
        return dict(zip(self.coef_names, buf))

    def vector(self, coefs):
        """ctypes array from a dict or sequence; None stays None (seeds)."""
        if coefs is None:
            return None
        if isinstance(coefs, dict):
            coefs = [coefs[n] for n in self.coef_names]
        return (ctypes.c_double * self._n)(*coefs)

    def estimate(self, hw, prb, cfg, coefs=None):
        """(cost_us, derived) or None when the config is invalid.

        derived = (kv_tile, q_tile, sg_per_wg, slm_bytes, grfs,
        wg_per_subslice). hw/prb/cfg may be str or pre-encoded bytes; coefs a
        dict, a sequence or a vector() result.
        """
        vec = (
            coefs
            if coefs is None or isinstance(coefs, ctypes.Array)
            else self.vector(coefs)
        )
        rc = self._estimate(
            _b(hw),
            _b(prb),
            _b(cfg),
            vec,
            self._n,
            ctypes.byref(self._cost),
            self._derived,
        )
        if rc < 0:
            raise ValueError(
                "unparsable hw/prb/cfg string: %r %r %r" % (hw, prb, cfg)
            )
        if rc > 0:
            return None
        return self._cost.value, tuple(self._derived)

    def enumerate(self, hw, prb, coefs=None, limit=4096):
        """Valid configs, best estimate first, as 8-tuples."""
        vec = (
            coefs
            if coefs is None or isinstance(coefs, ctypes.Array)
            else self.vector(coefs)
        )
        buf = (ctypes.c_int * (8 * limit))()
        n = self._enumerate(_b(hw), _b(prb), vec, self._n, buf, limit)
        if n < 0:
            raise ValueError("unparsable hw/prb string: %r %r" % (hw, prb))
        n = min(n, limit)
        return [tuple(buf[8 * i : 8 * i + 8]) for i in range(n)]


def _b(s):
    return s if isinstance(s, bytes) else s.encode()


def coefs_to_cpp(m, names, indent="    "):
    """C++ assignments for seed_coefs() in select.cpp."""
    return "\n".join("%sm.%s = %.10g;" % (indent, k, m[k]) for k in names)


if __name__ == "__main__":
    lib = Library(sys.argv[1] if len(sys.argv) > 1 else None)
    hw = HW(
        "xe_hpg", 448, 16, 8, 4, 32, 65536, 131072, 1024, 16 << 20, 8
    ).to_verbose()
    prb = Problem(
        d_qk=64,
        d_v=64,
        d_max_kq=64,
        d_max_v=64,
        keys=512,
        queries=512,
        batch_heads=8,
        mask=MASKS["buffer"],
        mask_bits=16,
    ).to_verbose()
    coefs = lib.seed_coefs("xe_hpg")
    print("library:", lib.path)
    print(
        "%d coefficients: %s ..."
        % (len(lib.coef_names), ", ".join(lib.coef_names[:5]))
    )
    ranked = lib.enumerate(hw, prb, coefs)
    print("%d candidates; top 5:" % len(ranked))
    for c in ranked[:5]:
        res = lib.estimate(hw, prb, cfg_str(c), coefs)
        if res is None:
            continue
        cost, d = res
        print(
            "  %-24s kv=%-4d q=%-4d sg=%-3d slm=%-6d grf=%d wgss=%d"
            " cost_us=%.3f"
            % (cfg_str(c), d[0], d[1], d[2], d[3], d[4], d[5], cost)
        )
