################################################################################
# Copyright 2020 Intel Corporation
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
################################################################################

import functools
import logging
import math
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Set, cast

from . import ir


def maybe_make_any_tag(md: ir.MemoryDescriptor):
    return "any" if "a" in md.properties else md.tag


def attribute_flag(name: str):
    def wrapper(converter: "Converter"):
        attr = getattr(converter.entry.exts, name)
        flag_name = name.replace("_", "-")
        if attr is None:
            return ""
        return f"--attr-{flag_name}={attr!s}"

    return property(wrapper)


class ConverterMeta(type):
    driver: str


class Converter(metaclass=ConverterMeta):
    def __init__(self, entry: ir.Entry):
        self.entry = entry

    def _get_dir(self):
        dirs = {
            "forward_training": "FWD_D",
            "forward_inference": "FWD_I",
            "backward_data": "BWD_D",
            "backward_weights": "BWD_W",
            "backward": "BWD_DW",
        }

        if self.entry.prop_kind not in dirs:
            return ""

        return dirs[self.entry.prop_kind]

    def _get_alg(self):
        return self.entry.aux.get("alg")

    @property
    def engine(self):
        return f"--engine={self.entry.engine}"

    @property
    def dir(self) -> str:
        if self._get_dir():
            return f"--dir={self._get_dir()}"
        return ""

    @property
    def bias_mask(self) -> str:
        return ""

    @property
    def dts(self):
        for md in self.entry.mds:
            if md.data_type == "undef":
                continue
            return f"--dt={md.data_type}"
        return ""

    @property
    def tags(self):
        for md in self.entry.mds:
            if not md.tag:
                continue
            return f"--tag={md.tag}"  # XXX: Don't use maybe_make_any_tag
        return ""

    @property
    def encoding(self) -> str:
        return ""

    @property
    def grouped(self) -> str:
        return ""

    @property
    def flags(self) -> str:
        return ""

    def _get_nondefault_args(self, values, defaults):
        parts: List[str] = []
        pairs = list(zip(values, defaults))
        seen_nondefault = False
        for value, default in reversed(pairs):
            if value != default:
                seen_nondefault = True
            if seen_nondefault:
                parts.append(str(value))
        return list(reversed(parts))

    def _convert_dw_post_op(self, po: ir.DepthwisePostOp):
        return f"dw:{po.ksp}:{po.dst_dt}"

    def _convert_sum_post_op(self, po: ir.SumPostOp):
        values = po.scale, po.zp, po.dt
        args = self._get_nondefault_args(values, defaults=(1.0, 0, ""))
        return ":".join(["sum"] + args)

    def _convert_prelu_post_op(self, po: ir.PreLUPostOp):
        if po.mask != 0:
            return f"prelu:{po.mask}"
        return "prelu"

    def _convert_eltwise_post_op(self, po: ir.EltwisePostOp):
        values = po.alpha, po.beta, po.scale
        args = self._get_nondefault_args(values, defaults=(0.0, 0.0, 1.0))
        return ":".join([po.alg] + args)

    def _convert_select_post_op(self, po: ir.SelectPostOp):
        src1_arg = f"{po.dt}.{po.mask}"
        if po.tag != "any":
            src1_arg = src1_arg + f".{po.tag}"
        src2_arg = f"{po.src2_mask}"
        if po.src2_tag != "any":
            src2_arg = src2_arg + f".{po.tag}"
        return ":".join([po.alg, src1_arg, src2_arg])

    def _convert_binary_post_op(self, po: ir.BinaryPostOp):
        if po.strides != "":
            return f"{po.alg}:{po.dt}:{po.mask}:{po.tag}:{po.strides}"
        if po.tag != "any":
            return f"{po.alg}:{po.dt}:{po.mask}:{po.tag}"
        return f"{po.alg}:{po.dt}:{po.mask}"

    @property
    def post_ops(self):
        post_ops = self.entry.exts.post_ops
        if post_ops is None:
            return ""
        results = []
        for post_op in post_ops:
            if post_op.alg == "dw":
                dw_po = cast(ir.DepthwisePostOp, post_op)
                results.append(self._convert_dw_post_op(dw_po))
            elif post_op.alg == "sum":
                sum_po = cast(ir.SumPostOp, post_op)
                results.append(self._convert_sum_post_op(sum_po))
            elif post_op.alg == "prelu":
                prelu_po = cast(ir.PreLUPostOp, post_op)
                results.append(self._convert_prelu_post_op(prelu_po))
            elif post_op.alg == "binary_select":
                select_po = cast(ir.SelectPostOp, post_op)
                results.append(self._convert_select_post_op(select_po))
            elif post_op.alg.startswith("binary_"):
                binary_po = cast(ir.BinaryPostOp, post_op)
                results.append(self._convert_binary_post_op(binary_po))
            elif post_op.alg.startswith("eltwise_"):
                eltwise_po = cast(ir.EltwisePostOp, post_op)
                results.append(self._convert_eltwise_post_op(eltwise_po))
        return "--attr-post-ops=" + "+".join(results)

    def _get_quantization(
        self,
        params: Optional[Mapping[str, ir.QuantizationParam]],
        def_value: float,
        def_type: str,
    ):
        if params is None:
            return ""
        results = []
        for arg, param in params.items():
            mask = param.mask
            policy = ""
            # Set policy to "host_scalar" if is_host_scalar is True
            if param.is_host_scalar:
                policy = "host_scalar"
            if param.quantization_mode == "dynamic_mx":
                policy = "mx"
            if param.quantization_mode == "dynamic_fp":
                policy = "dynamic_fp"
            result = f"{arg}:{policy or mask}"
            if mask == 0 or policy == "host_scalar":
                result += f":{def_value}"
            dt = param.data_type
            groups = param.groups
            if dt != def_type or groups:
                result += f":{dt}"
            if groups:
                result += f":{groups}"
            results.append(result)
        return "+".join(results)

    @property
    def scales(self):
        params = self._get_quantization(self.entry.exts.scales, 0.5, "f32")
        return f"--attr-scales={params}" if params else ""

    @property
    def zero_points(self):
        params = self._get_quantization(self.entry.exts.zero_points, 1, "s32")
        return f"--attr-zero-points={params}" if params else ""

    @property
    def precomputed_reductions(self):
        params = self._get_quantization(
            self.entry.exts.precomputed_reductions, 1, "s32"
        )
        return f"--attr-precomputed-reductions={params}" if params else ""

    @property
    def rounding_mode(self):
        rounding_modes = self.entry.exts.rounding_mode
        if rounding_modes is None:
            return ""
        results = []
        for arg, mode in rounding_modes.items():
            results.append(f"{arg}:{mode!s}")
        return "--attr-rounding-mode=" + "+".join(results)

    scratchpad_mode = attribute_flag("scratchpad")
    fpmath_mode = attribute_flag("fpmath")
    acc_mode = attribute_flag("acc_mode")

    @property
    def dropout(self):
        dropout = self.entry.exts.dropout
        if dropout is None:
            return ""
        # Use default p=0.5 and seed=12345 since those values are user data and
        # can't be obtained properly.
        result = "0.5:12345"
        if not dropout.tag:
            return f"--attr-dropout={result}"
        result += f":{dropout.tag}"
        # Seed dt is always s64 in benchdnn and is not passed to driver
        if dropout.use_offset:
            result += ":987654321"
        else:
            result += ":0"
        if dropout.use_host_scalars:
            result += f":{dropout.use_host_scalars:d}"
        return f"--attr-dropout={result}"

    deterministic = attribute_flag("deterministic")

    @property
    def attrs(self):
        attrs = (
            self.post_ops,
            self.scales,
            self.zero_points,
            self.precomputed_reductions,
            self.scratchpad_mode,
            self.fpmath_mode,
            self.acc_mode,
            self.rounding_mode,
            self.dropout,
            self.deterministic,
        )
        return " ".join(attr for attr in attrs if attr)

    @property
    def aux(self):
        alg = self._get_alg()
        if alg is not None:
            return f"--alg={alg}"
        return ""

    @property
    def shapes(self):
        return self.entry.shapes


class AlgorithmMixin:
    entry: ir.Entry

    def _get_alg(self):
        alg = self.entry.aux.get("alg")
        if alg is None:
            return None
        return alg.split(self.entry.prim_kind, 1)[1][1:]


class MultiSourceMixin:
    entry: ir.Entry

    @property
    def dts(self):
        src_dts: List[str] = []
        other_dts: Dict[str, str] = {}
        for md in self.entry.mds:
            dt = md.data_type
            if md.arg == "src":
                src_dts.append(dt)
            elif dt != "undef":
                other_dts[md.arg[0]] = dt
        sdt_flags = "--sdt=" + ":".join(src_dts)
        other_dt_flags = " ".join(f"--{k}dt={v}" for k, v in other_dts.items())
        return f"{sdt_flags} {other_dt_flags}".strip()

    @property
    def tags(self):
        src_tags: List[str] = []
        other_tags: Dict[str, str] = {}
        for md in self.entry.mds:
            if md.arg == "src":
                src_tags.append(maybe_make_any_tag(md))
            elif md.tag:
                other_tags[md.arg[0]] = maybe_make_any_tag(md)
        stag_flags = "--stag=" + ":".join(src_tags)
        other_tag_flags = " ".join(
            f"--{k}tag={v}" for k, v in other_tags.items()
        )
        return f"{stag_flags} {other_tag_flags}".strip()


class CommonDataTypeMixin:
    entry: ir.Entry

    @property
    def dts(self):
        dts: Dict[str, str] = {}
        for md in self.entry.mds:
            c = md.arg[0]
            if c in dts:
                continue
            dts[c] = md.data_type
        return " ".join(f"--{k}dt={v}" for k, v in dts.items())


class TagTripletMixin:
    entry: ir.Entry

    @property
    def tags(self):
        md_map = {md.arg: md for md in self.entry.mds}
        has_fused_dw = "src_fused" in md_map
        # Fused dw defines dst tag by src_fused argument
        dst_name = "src_fused" if has_fused_dw else "dst"
        tags = []
        if "src" in md_map:
            md = md_map["src"]
            tag = maybe_make_any_tag(md)
            tags.append(f"--stag={tag}")
        if "wei" in md_map:
            md = md_map["wei"]
            tag = maybe_make_any_tag(md)
            # pass wtag any for cases with compensation
            if str(md.flags.value) != "f0":
                tag = "any"
            tags.append(f"--wtag={tag}")
        if dst_name in md_map:
            md = md_map[dst_name]
            tag = maybe_make_any_tag(md)
            tags.append(f"--dtag={tag}")
        return " ".join(tags)


class StridesMixin(TagTripletMixin):
    @property
    def tags(self):
        tags = []
        strides = []

        def add_strides_or_tag(arg, md: ir.MemoryDescriptor):
            if md.format_kind == "sparse":
                strides.append(md.strides)
                return
            tag = maybe_make_any_tag(md)
            if arg == "wei" and str(md.flags.value) != "f0":
                tag = "any"
            if tag != "any" and tag.lower() == tag and md.strides:
                strides.append(md.strides)
            else:
                tags.append(f"--{arg[0]}tag={tag}")
                strides.append("")

        md_map = {md.arg: md for md in self.entry.mds}
        args = "src", "wei", "dst"
        for arg in args:
            if arg not in md_map:
                continue
            md = md_map[arg]
            add_strides_or_tag(arg, md)
        stride_flag = "--strides=" + ":".join(strides)
        return " ".join(tags + [stride_flag])


class MultiDataTypeMixin:
    entry: ir.Entry

    @property
    def dts(self):
        dt_map = {md.arg: md.data_type for md in self.entry.mds}
        # Fused dw defines dst_dt by src_fused argument
        has_fused_dw = "src_fused" in dt_map
        dst_name = "src_fused" if has_fused_dw else "dst"
        dts = [
            dt_map.get("src", ""),
            dt_map.get("wei", ""),
            dt_map.get(dst_name, ""),
        ]
        return "--dt=" + ":".join(dt for dt in dts if dt)


class MultiDataTypeWithBiasMixin(MultiDataTypeMixin):
    @property
    def dts(self):
        dts = super().dts
        for md in self.entry.mds:
            if md.arg != "bia":
                continue
            return f"{dts} --bia-dt={md.data_type}".strip()
        return dts


class NormalizationMixin:
    entry: ir.Entry

    @property
    def aux(self):
        flags = self.entry.aux.get("flags")
        if flags is not None:
            return f"--flags={flags}"
        return ""


class BatchNormalizationConverter(NormalizationMixin, Converter):
    driver: str = "bnorm"


class BinaryConverter(AlgorithmMixin, MultiSourceMixin, Converter):
    driver: str = "binary"

    @property
    def shapes(self):
        return ":".join(self.entry.shapes.split(":", 2)[:2])

    @property
    def dts(self):
        src0, src1, *_, dst = self.entry.mds
        return f"--sdt={src0.data_type}:{src1.data_type} --ddt={dst.data_type}"

    @property
    def tags(self):
        src0, src1, *_, dst = self.entry.mds
        src0_tag = maybe_make_any_tag(src0)
        src1_tag = maybe_make_any_tag(src1)
        dst_tag = maybe_make_any_tag(dst)
        return f"--stag={src0_tag}:{src1_tag} --dtag={dst_tag}"


class BRGEMMConverter(MultiDataTypeMixin, Converter):
    driver: str = "brgemm"

    @property
    def aux(self):
        bs = self.entry.aux.get("bs", "")
        beta = self.entry.aux.get("beta", "")
        return f"--bs={bs} --beta={beta}"

    @property
    def tags(self):
        tags = []
        strides = []
        lds = []

        def add_strides_or_tag(arg, md):
            tag = maybe_make_any_tag(md)
            if arg == "wei":
                if str(md.flags.value) != "f0":
                    tag = "any"
                if tag != "any" and tag.lower() == tag and md.strides:
                    strides.append(md.strides)
                    lds.append("")
                else:
                    tags.append(f"--{arg[0]}tag={tag}")
                    strides.append("")
                    lds.append("")
            else:
                ld_val = md.strides.split("x")[0]
                lds.append(f"{ld_val}")
                strides.append("")

        md_map = {md.arg: md for md in self.entry.mds}
        args = "src", "wei", "dst"
        for arg in args:
            if arg not in md_map:
                continue
            md = md_map[arg]
            add_strides_or_tag(arg, md)
        stride_flag = "--strides=" + ":".join(strides)
        ld_flag = "--ld=" + ":".join(lds)
        return " ".join(tags + [stride_flag] + [ld_flag])


class ConcatConverter(CommonDataTypeMixin, MultiSourceMixin, Converter):
    driver: str = "concat"

    @property
    def aux(self):
        axis = self.entry.aux.get("axis")
        if axis is None:
            return ""
        return f"--axis={axis}"


class ConvolutionConverter(
    AlgorithmMixin,
    TagTripletMixin,
    MultiDataTypeWithBiasMixin,
    Converter,
):
    driver: str = "conv"

    @property
    def aux(self):
        alg = self._get_alg()
        if alg is not None:
            return f"--alg={alg}"
        return ""


class DeconvolutionConverter(ConvolutionConverter):
    driver: str = "deconv"


class EltwiseConverter(Converter):
    driver: str = "eltwise"

    @property
    def aux(self):
        alpha = self.entry.aux.get("alpha")
        beta = self.entry.aux.get("beta")
        flags = [f"--alg={self._get_alg()}"]
        if alpha is not None:
            flags.append(f"--alpha={alpha}")
        if beta is not None:
            flags.append(f"--beta={beta}")
        return " ".join(flags)


class GroupNormalizationConverter(
    MultiDataTypeMixin,
    BatchNormalizationConverter,
):
    driver: str = "gnorm"

    # --tag=SRC_TAG[:WEI_TAG][:DST_TAG]
    @property
    def tags(self):
        tag_map = {md.arg: maybe_make_any_tag(md) for md in self.entry.mds}
        args = "src", "wei", "dst"
        tags = [tag_map[arg] for arg in args if arg in tag_map]
        return "--tag=" + ":".join(tags)


class InnerProductConverter(
    TagTripletMixin, MultiDataTypeWithBiasMixin, Converter
):
    driver: str = "ip"


class LayerNormalizationConverter(GroupNormalizationConverter):
    driver: str = "lnorm"

    @property
    def dts(self):
        dts = super().dts
        shift_flag = None
        for md in self.entry.mds:
            if "scale" in md.arg:
                return f"{dts} --ss_dt={md.data_type}".strip()
            if "shift" in md.arg and shift_flag is None:
                shift_flag = f"--ss_dt={md.data_type}"
        if shift_flag is not None:
            return f"{dts} {shift_flag}".strip()
        return dts

    @property
    def tags(self):
        tags = super().tags
        for md in self.entry.mds:
            if md.arg == "stats":
                tags = f"{tags} --stat_tag={maybe_make_any_tag(md)}"
        return tags.strip()


class LRNConverter(AlgorithmMixin, Converter):
    driver: str = "lrn"

    @property
    def aux(self):
        alg = self._get_alg()
        algs = {"across_channels": "ACROSS", "within_channel": "WITHIN"}
        if alg not in algs:
            return ""
        return f"--alg={algs[alg]}"


class MatmulConverter(StridesMixin, MultiDataTypeWithBiasMixin, Converter):
    driver: str = "matmul"

    @property
    def bias_mask(self):
        for md in self.entry.mds:
            if md.arg != "bia":
                continue
            if "_" in md.flags.value:
                mask = md.flags.value.split("_")[1][4:]
                return f"--bia_mask={mask}"
        return ""

    @property
    def encoding(self):
        src, wei = self.entry.shapes.split(":", 2)[:2]
        *sb, m, sk = map(int, src.split("x"))
        *wb, wk, n = map(int, wei.split("x"))
        b = tuple(max(s, w) for s, w in zip(sb, wb))

        counts = {
            "src": math.prod(sb) * m * sk,
            "wei": math.prod(wb) * wk * n,
            "dst": math.prod(b) * m * n,
        }

        encodings: Dict[str, str] = {}
        for md in self.entry.mds:
            if md.arg not in counts:
                continue
            if md.format_kind != "sparse" or md.grouped:
                continue
            count = counts[md.arg]
            nnz = max(200, count // 100)
            sparsity = 1.00 - min(nnz / count, 1.00)
            encodings[md.arg] = f"{md.tag}+{sparsity:.2f}"
        if not encodings:
            return ""
        encoding = ":".join(encodings.get(arg, "") for arg in counts)
        return f"--encoding={encoding}"

    @property
    def grouped(self) -> str:
        # grouped params come from the src variable dim
        src_md = next((md for md in self.entry.mds if md.arg == "src"), None)
        desc = src_md.grouped if src_md is not None else None
        if desc is None:
            return ""
        var_idx, count = desc.variable_dim_idx, desc.group_count
        # verbose omits per-group sizes, so split the src dim evenly
        src_dims = self.entry.shapes.split(":")[0].split("x")
        total_dim = int(src_dims[var_idx])
        group_size, remainder = divmod(total_dim, count)
        sizes = [group_size] * count
        for i in range(remainder):
            sizes[i] += 1
        group_sizes = "+".join(map(str, sizes))
        return f"--grouped={var_idx}:{count}:{group_sizes}"

    @property
    def aux(self):
        rt_dim_masks = self.entry.aux.get("runtime_dims_masks", "")
        return f"--runtime_dims_masks={rt_dim_masks}"


class PoolingConverter(MultiDataTypeMixin, Converter):
    driver: str = "pool"

    @property
    def aux(self):
        return f"--alg={self._get_alg()}"


class PreLUConverter(Converter):
    driver: str = "prelu"

    @property
    def dts(self):
        data_dt, wei_dt = "", ""
        for md in self.entry.mds:
            if "data" in md.arg and not data_dt:
                data_dt = md.data_type
            if "wei" in md.arg and not wei_dt:
                wei_dt = md.data_type
            if data_dt and wei_dt:
                break
        return f"--sdt={data_dt}:{wei_dt}"

    @property
    def tags(self):
        data_tag, wei_tag = "", ""
        for md in self.entry.mds:
            if "data" in md.arg and not data_tag:
                data_tag = maybe_make_any_tag(md)
            if "wei" in md.arg and not wei_tag:
                wei_tag = maybe_make_any_tag(md)
            if data_tag and wei_tag:
                break
        return f"--stag={data_tag}:{wei_tag}"


class ReductionConverter(
    AlgorithmMixin,
    TagTripletMixin,
    CommonDataTypeMixin,
    Converter,
):
    driver: str = "reduction"

    @property
    def aux(self):
        p = self.entry.aux.get("p")
        eps = self.entry.aux.get("eps")
        args = [f"--alg={self._get_alg()}"]
        if p is not None:
            args.append(f"--p={p}")
        if eps is not None:
            args.append(f"--eps={eps}")
        return " ".join(args)


class ReorderConverter(StridesMixin, MultiDataTypeMixin, Converter):
    driver: str = "reorder"

    def _convert_flag(self, prefix, md: ir.MemoryDescriptor):
        flags = []
        fields = md.flags
        if fields.s8_comp_mask is not None:
            flags.append(f"s8s8_comp:{fields.s8_comp_mask}")
        if fields.zp_comp_mask is not None:
            flags.append(f"zp_comp:{fields.zp_comp_mask}")
        if flags:
            return f"--{prefix}flag=" + "+".join(flags)
        return ""

    @property
    def flags(self):
        flags = {}
        for md in self.entry.mds:
            if "src" in md.arg and "src" not in flags:
                flags["src"] = self._convert_flag("i", md)
            elif "dst" in md.arg and "dst" not in flags:
                flags["dst"] = self._convert_flag("o", md)

            if "src" in flags and "dst" in flags:
                break
        iflag = flags.get("src", "")
        oflag = flags.get("dst", "")
        return f"{iflag} {oflag}".strip()

    @property
    def aux(self):
        mask = self.entry.aux.get("runtime-dim-mask")
        if mask:
            return f"--runtime-dim-mask={mask}"
        return ""


class ResamplingConverter(AlgorithmMixin, CommonDataTypeMixin, Converter):
    driver: str = "resampling"


class RNNConverter(AlgorithmMixin, Converter):
    driver: str = "rnn"

    @property
    def flags(self):
        for md in self.entry.mds:
            if md.arg not in ("src_iter", "src_layer"):
                continue
            if md.strides == "":
                continue
            return "--trivial-strides=false"
        return "--trivial-strides=true"

    def _get_flag_from(self, flag_name, flag_values):
        flag = self.entry.aux.get(flag_name)
        if flag is None or flag not in flag_values:
            return ""
        return f"--{flag_name}={flag_values[flag]}"

    @property
    def aux(self):
        algs = {
            "vanilla_rnn": "VANILLA_RNN",
            "vanilla_lstm": "VANILLA_LSTM",
            "vanilla_gru": "VANILLA_GRU",
            "vanilla_augru": "VANILLA_AUGRU",
            "lbr_gru": "LBR_GRU",
            "lbr_augru": "LBR_AUGRU",
        }
        dirs = {
            "unidirectional_left2right": "left2right",
            "unidirectional_right2left": "right2left",
            "bidirectional_sum": "sum",
            "bidirectional_concat": "concat",
        }
        acts = {
            "eltwise_relu": "RELU",
            "eltwise_logistic": "LOGISTIC",
            "eltwise_tanh": "TANH",
        }
        all_flags = [
            self._get_flag_from("alg", algs),
            self._get_flag_from("direction", dirs),
            self._get_flag_from("activation", acts),
        ]
        flags = self.entry.aux.get("flags")
        if flags is not None:
            all_flags.append(f"--flags={flags}")
        return " ".join(flag for flag in all_flags if flag)

    @property
    def dir(self):
        dir = self._get_dir()
        return f"--prop={dir}"

    @property
    def dts(self):
        args = ["src_iter", "src_iter_c", "src_layer", "dst_iter", "dst_layer"]
        cfg_dts: str
        common_dt = True
        shared_dt = None
        bias_dt = None
        md_map: Dict[Optional[str], ir.MemoryDescriptor] = {}
        for md in self.entry.mds:
            md_map[md.arg] = md
            if md.arg == "bias":
                bias_dt = md.data_type
            elif md.arg in args:
                if shared_dt is None:
                    shared_dt = md.data_type
                elif md.data_type != shared_dt:
                    common_dt = False
        if common_dt and shared_dt in ["f32", "f16"]:
            cfg_dts = shared_dt
        elif common_dt and shared_dt == "bf16":
            cfg_dts = shared_dt
            # bias is part of cfg for bf16
            if bias_dt is not None and bias_dt != shared_dt:
                cfg_dts += bias_dt
        else:
            cfg_dts = ""
            for arg in args:
                if arg not in md_map:
                    continue
                md = md_map[arg]
                # src iter is skipped if it is f16
                if arg == "src_iter_c" and md.data_type == "f16":
                    continue
                cfg_dts += md.data_type
        return f"--cfg={cfg_dts}"

    @property
    def tags(self):
        # Tags for backward are driven by diff tensors, query them instead of
        # forward tensors. Latter will always have `any` format.
        has_diff_tensors = False
        for md in self.entry.mds:
            if "diff" in md.arg:
                has_diff_tensors = True
                break

        layer_names = ["src_layer", "wei_layer", "dst_layer"]
        if has_diff_tensors:
            layer_names = [f"diff_{name}" for name in layer_names]
        tags = []
        other_flags = []
        for md in self.entry.mds:
            arg = md.arg
            tag = maybe_make_any_tag(md)
            if arg in layer_names:
                tags.append(tag)
            elif md.tag == "undef":
                continue
            elif arg == "wei_proj":
                other_flags.append("--with-projection=true")
            elif arg == "wei_peephole":
                other_flags.append("--with-peephole=true")
        tag_flag = "--tag=" + ":".join(tags)
        return " ".join([tag_flag] + other_flags)


class ShuffleConverter(Converter):
    driver: str = "shuffle"

    @property
    def aux(self):
        axis = self.entry.aux.get("axis")
        group = self.entry.aux.get("group")
        args = []
        if axis is not None:
            args.append(f"--axis={axis}")
        if group is not None:
            args.append(f"--group={group}")
        return " ".join(args)


class SoftmaxConverter(TagTripletMixin, CommonDataTypeMixin, Converter):
    driver: str = "softmax"

    @property
    def aux(self):
        axis = self.entry.aux.get("axis")
        flags = f"--alg={self._get_alg()}"
        if axis is not None:
            flags += f" --axis={axis}"
        return flags


class SumConverter(MultiSourceMixin, Converter):
    driver: str = "sum"


class ZeroPadConverter(Converter):
    driver: str = "zeropad"

    @property
    def dts(self):
        return f"--dt={self.entry.mds[0].data_type}"

    @property
    def tags(self):
        return f"--tag={maybe_make_any_tag(self.entry.mds[0])}"


class SDPAConverter(Converter):
    driver: str = "sdpa"

    @property
    def aux(self):
        parts = []
        # Mask type
        msk = self.entry.aux.get("msk")
        if msk is not None:
            parts.append(f"--mask={msk}")

        # If a mask md is present but no auxiliary mask info, default to
        # --mask=buffer.
        for md in self.entry.mds:
            if md.arg == "msk":
                if msk is None:
                    parts.append("--mask=buffer")
                break

        # Scale type: scl:mul is the default (library mode), only emit
        # --scale= for non-default values like "div".
        scl = self.entry.aux.get("scl")
        if scl is not None:
            scale_mode = scl.split(":")[0]  # "mul" or "div"
            if scale_mode != "mul":
                parts.append(f"--scale={scale_mode}")

        return " ".join(parts)

    @property
    def dts(self):
        dt_map = {}
        mdt = None
        for md in self.entry.mds:
            if md.arg in ("query", "key", "val", "dst"):
                dt_map[md.arg] = md.data_type
            elif md.arg == "msk":
                mdt = md.data_type
        q_dt = dt_map.get("query", "f32")
        k_dt = dt_map.get("key", q_dt)
        v_dt = dt_map.get("val", q_dt)
        d_dt = dt_map.get("dst", q_dt)
        if q_dt == k_dt == v_dt == d_dt:
            dt = f"--dt={q_dt}"
        else:
            dt = f"--dt={q_dt}:{k_dt}:{v_dt}:{d_dt}"
        return f"{dt} --mdt={mdt}" if mdt else dt

    @property
    def tags(self):
        parts = []
        for md in self.entry.mds:
            if md.arg in ("query", "key", "val", "dst"):
                tag = maybe_make_any_tag(md)
                parts.append(f"--{md.arg[0]}tag={tag}")
        return " ".join(parts)


def get_converter(primitive: str) -> ConverterMeta:
    converters: Dict[str, ConverterMeta] = {
        "batch_normalization": BatchNormalizationConverter,
        "binary": BinaryConverter,
        "brgemm": BRGEMMConverter,
        "concat": ConcatConverter,
        "convolution": ConvolutionConverter,
        "deconvolution": DeconvolutionConverter,
        "eltwise": EltwiseConverter,
        "group_normalization": GroupNormalizationConverter,
        "inner_product": InnerProductConverter,
        "layer_normalization": LayerNormalizationConverter,
        "lrn": LRNConverter,
        "matmul": MatmulConverter,
        "pooling": PoolingConverter,
        "prelu": PreLUConverter,
        "reduction": ReductionConverter,
        "reorder": ReorderConverter,
        "resampling": ResamplingConverter,
        "rnn": RNNConverter,
        "sdpa": SDPAConverter,
        "shuffle": ShuffleConverter,
        "softmax": SoftmaxConverter,
        "sum": SumConverter,
        "zero_pad": ZeroPadConverter,
    }
    return converters[primitive]


class InputGenerator:
    """
    Generates an input for benchdnn from internal representation.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _warn_grouped(logger: Optional[logging.Logger]):
        if not logger:
            return
        logger.warning(
            "Verbose does not report per-group sizes; group sizes in"
            " --grouped are inferred by dividing the total dimension"
            " by the number of groups and may not match the original."
        )

    def _generate_case(self, entry: ir.Entry):
        Converter = get_converter(entry.prim_kind)
        converter = Converter(entry)
        grouped = converter.grouped
        if grouped:
            self._warn_grouped(self.logger)
        args = [
            "--reset",
            "--allow-enum-tags-only=0",
            converter.engine,
            converter.dir,
            converter.aux,
            converter.bias_mask,
            converter.dts,
            converter.encoding,
            converter.tags,
            converter.flags,
            converter.attrs,
            grouped,
            converter.shapes,
        ]
        return converter.driver, " ".join(arg for arg in args if arg)

    def generate(self, input, split_by_driver=False):
        missing: Set[str] = set()
        data: Dict[str, List[str]] = defaultdict(list)
        for value in input.values():
            try:
                driver, args = self._generate_case(value)
            except KeyError as e:
                if self.logger is not None and str(e) not in missing:
                    missing.add(str(e))
                    self.logger.warning(f"Missing converter: {e!s}")
                continue
            if not split_by_driver:
                driver, args = "all", f"--{driver} {args}"
            data[driver].append(args)
        return {k: "\n".join(v) for k, v in data.items()}
