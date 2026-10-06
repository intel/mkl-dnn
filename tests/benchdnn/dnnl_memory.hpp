/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef DNNL_MEMORY_HPP
#define DNNL_MEMORY_HPP

#include <unordered_map>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "utils/dims.hpp"
#include "utils/engine.hpp"
#include "utils/wrapper.hpp"

#define dnnl_mem_default_value 0xFF
#define dnnl_mem_default_perf_test_value 0x3F

static constexpr size_t dnnl_mem_large_page_size = 2u * 1024u * 1024u;
static constexpr size_t dnnl_mem_page_size = 4096u;

struct dnn_mem_t {
    struct handle_info_t {
        bool is_host_ptr;
        void *ptr;
        // Alignment used when allocating CPU buffers. Defaults to
        // `dnnl_mem_large_page_size`.
        size_t alignment;

        bool is_allocate() const { return ptr == DNNL_MEMORY_ALLOCATE; }

        static handle_info_t allocate(
                size_t alignment = dnnl_mem_large_page_size) {
            return {false, DNNL_MEMORY_ALLOCATE, alignment};
        }
    };

    dnn_mem_t() { map(); }
    dnn_mem_t(const_dnnl_memory_desc_t md, const engine_t &engine, bool prefill,
            const handle_info_t &handle_info = handle_info_t::allocate());

    dnn_mem_t(const_dnnl_memory_desc_t md, dnnl_data_type_t dt,
            const std::string &tag, const engine_t &engine, bool prefill);
    dnn_mem_t(const_dnnl_memory_desc_t md, dnnl_data_type_t dt,
            const dnnl_dims_t strides, const engine_t &engine, bool prefill);

    dnn_mem_t(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
            const std::string &tag, const engine_t &engine, bool prefill);
    dnn_mem_t(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
            const dnnl_dims_t strides, const engine_t &engine, bool prefill);

    dnn_mem_t(const dnn_mem_t &rhs, dnnl_data_type_t dt, const std::string &tag,
            const engine_t &engine);

    // Construct a host-scalar memory object, initialized with 0x3F pattern.
    dnn_mem_t(const_dnnl_memory_desc_t md);

    dnn_mem_t(const dnn_mem_t &rhs) = delete;
    dnn_mem_t &operator=(const dnn_mem_t &rhs) = delete;

    dnn_mem_t &operator=(dnn_mem_t &&rhs) {
        if (&rhs == this) return *this;
        cleanup();

        md_ = rhs.md_;
        m_ = rhs.m_;
        m_padded_ = rhs.m_padded_;
        data_ = std::move(rhs.data_);
        is_data_owner_ = rhs.is_data_owner_;
        active_ = rhs.active_;
        engine_ = std::move(rhs.engine_);
        is_mapped_ = (bool)rhs.is_mapped_;
        mapped_ptrs_ = std::move(rhs.mapped_ptrs_);

        rhs.active_ = false;
        return *this;
    }
    dnn_mem_t(dnn_mem_t &&rhs) : dnn_mem_t() { *this = std::move(rhs); }

    ~dnn_mem_t() { cleanup(); }

    int reorder(const dnn_mem_t &rhs, res_t *res,
            const_dnnl_primitive_attr_t attr,
            dnnl_data_type_t swap_dt = dnnl_data_type_undef);
    int reorder(const dnn_mem_t &rhs, res_t *res,
            dnnl_data_type_t swap_dt = dnnl_data_type_undef) {
        return reorder(rhs, res, nullptr, swap_dt);
    }

    size_t size(int index = 0) const;

    int64_t nelems(bool with_padded_dims = false) const;

    // Returns `true` if there're no "holes" between memory dimensions, and
    // `false`, otherwise.
    bool is_dense() const;

    // Queries from memory descriptor.
    int ndims() const;
    const dnnl_dims_t &dims() const;
    const dnnl_dims_t &padded_dims() const;
    dnnl_data_type_t dt(int buffer_index = 0) const;
    const dnnl_dims_t &padded_offsets() const;
    dnnl_dim_t offset0() const;
    dnnl_format_kind_t format_kind() const;
    const dnnl_dims_t &strides() const;
    int inner_nblks() const;
    const dnnl_dims_t &inner_blks() const;
    const dnnl_dims_t &inner_idxs() const;

    // Sparse memories require special handling for `no_ref_memory` modifier
    // because of indirect access. Thus, filling should apply to metadata and
    // it must be meaningful. It implies unconditional mapping.
    bool is_sparse_md() const;

    size_t sizeof_dt() const;

    template <typename T>
    explicit operator T *() const {
        assert(is_mapped_);
        // Always return 0th ptr.
        return static_cast<T *>(get_mapped_pointer<T>(0));
    }

    template <typename T>
    T *get_mapped_pointer(int index = 0) const {
        assert(is_mapped_);
        return static_cast<T *>(mapped_ptrs_[index]);
    }

    explicit operator bool() const { return active_; }

    // This interface is a shortcut version of the one below to speed up access
    // to the memory that is guaranteedly of f32 data type.
    // Keep the body in the header to help compiler to inline better.
    float get_f32_elem(int64_t idx) const {
        return static_cast<float *>(*this)[idx];
    }
    float get_elem(int64_t idx, int buffer_index = 0) const;

    // This interface is a shortcut version of set_elem below to speed up access
    // to the memory that is guaranteedly of f32 data type.
    // Keep the body in the header to help compiler to inline better.
    void set_f32_elem(int64_t idx, float value) const {
        static_cast<float *>(*this)[idx] = value;
    }
    // This interface allows to prevent roundings to float when
    // setting s64 values that are meant to not fit in f32, like seeds
    // or offsets.
    void set_s64_elem(int64_t idx, int64_t value) const {
        static_cast<int64_t *>(*this)[idx] = value;
    }
    void set_elem(int64_t idx, float value, int buffer_index = 0) const;

    int64_t get_idx(int64_t logical_idx, int dims_mask, const int ndims,
            const dims_t &groups = {}) const;
    int64_t get_idx(int64_t logical_idx, int dims_mask) const {
        return get_idx(logical_idx, dims_mask, ndims(), dims_t());
    }

    const engine_t &engine() const { return engine_; }

    bool is_mapped() const { return is_mapped_; }

    bool is_canary_protected() const { return is_canary_protected_; }

    void map() const;
    void unmap() const;
    void memset(int value, size_t size, int buffer_index) const;
#if (DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL)
    int gpu_fill_random(size_t size, int buffer_index) const;
#endif

    static dnn_mem_t create_from_host_ptr(const dnnl_memory_desc_t &md,
            const engine_t &engine, void *host_ptr);

    // Increases memory size to catch potential buffer overreads and
    // overwrites. The padded area is filled with a canary value.
    size_t pad_memory_size(size_t sz, bool *was_padded = nullptr) const;
    // Increases memory descriptor size to catch potential buffer overreads and
    // overwrites. The padded area is filled with a canary value.
    dnnl_memory_desc_t pad_memory_desc(
            const_dnnl_memory_desc_t md, bool *was_padded = nullptr) const;
    // Initializes zero memory descriptor
    static dnnl_status_t init_md(
            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> &md);
    // Initializes memory descriptor from sporadic tag or strides.
    static dnnl_status_t init_md(
            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> &md, int ndims,
            const dnnl_dims_t dims, dnnl_data_type_t data_type,
            const std::string &tag, const dims_t &strides_ = {});
    // Initializes memory descriptor for CSR encoding.
    static dnnl_status_t init_csr_md(
            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> &md, int ndims,
            const dnnl_dims_t dims, dnnl_data_type_t data_type, dnnl_dim_t nnz,
            dnnl_data_type_t indices_dt, dnnl_data_type_t pointers_dt);
    // Initializes memory descriptor for COO encoding.
    static dnnl_status_t init_coo_md(
            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> &md, int ndims,
            const dnnl_dims_t dims, dnnl_data_type_t data_type, dnnl_dim_t nnz,
            dnnl_data_type_t indices_dt);
    // Initializes memory descriptor for packed encoding.
    static dnnl_status_t init_sparse_packed_md(
            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> &md, int ndims,
            const dnnl_dims_t dims, dnnl_data_type_t data_type, dnnl_dim_t nnz);
    // Initializes memory descriptor for host scalar memory.
    static benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> init_host_scalar_md(
            dnnl_data_type_t data_type);
    // Initializes memory descriptor for grouped encoding.
    static dnnl_status_t init_grouped_md(
            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> &md, int ndims,
            const dnnl_dims_t dims, dnnl_data_type_t data_type,
            int variable_dim_idx, dnnl_dim_t group_count,
            dnnl_data_type_t offsets_dt = dnnl_s32);

    /* fields */
    dnnl_memory_desc_t md_ {};
    dnnl_memory_t m_ {};

    // "Base" memory with a canary-padded buffer for buffer overflow
    // protection.
    dnnl_memory_t m_padded_ {};
    bool is_canary_protected_ = false;

private:
    std::vector<void *> data_;
    bool is_data_owner_ = false;
    bool active_ = false;

    engine_t engine_;

    mutable bool is_mapped_ = false;
    mutable std::vector<void *> mapped_ptrs_;

    int init_memory(dnnl_memory_t *ret, const dnnl_memory_desc_t &md,
            dnnl_memory_t mem);
    int initialize_memory_create_sycl(const handle_info_t &handle_info);
    int initialize_memory_create_opencl(const handle_info_t &handle_info);
    int initialize_memory_create_ze(const handle_info_t &handle_info);
    int initialize_memory_create(const handle_info_t &handle_info);

    // `prefill` is a flag that controls whether the underlying memory buffer
    // will be accessed to set a special value across the buffer, such as NaN,
    // to catch issues with no/bad access.
    //
    // Some memory objects created will be filled right away with different
    // values, e.g., reference f32 memories. In that case, read/write access
    // with a special value is a waste of memory bandwidth so developer decides
    // when to enable/disable it.
    //
    // The flag is propagated to the toppest dnn_mem_t constructors. In case
    // when in doubt, always use `true` to stay on the safe side of things.
    int initialize(bool prefill,
            const handle_info_t &handle_info = handle_info_t::allocate());
    int initialize_by_host_scalar(const_dnnl_memory_desc_t md, void *value);

    void set_dt(dnnl_data_type_t dt) const;

    int cleanup();
};

using dnn_mem_map_t = std::unordered_map<int, dnn_mem_t>;

bool has_sparse_md(const dnn_mem_map_t &dnn_mem_map);

bool has_grouped_encoding(const_dnnl_memory_desc_t md);

dnnl_memory_desc_t clone_md(const_dnnl_memory_desc_t md);

// Returns the logical size of a memory descriptor, which is a product of nelems
// by data_type_size.
size_t get_logical_size(const_dnnl_memory_desc_t md);

// Checks that zero padding is preserved.
int check_zero_padding(const dnn_mem_t &mem, int arg, res_t *res = nullptr,
        int *error_count = nullptr);

// Checks that the buffer is not overrun if it was protected by a canary.
int check_buffer_overwrite(const dnn_mem_t &mem, int arg, res_t *res = nullptr);

// Returns physical offset by logical one. Logical offset is represented by an
// array pos. If is_pos_padded is true pos represents the position in already
// padded area.
dnnl_dim_t md_off_v(const dnn_mem_t &mem, const dnnl_dims_t pos,
        bool is_pos_padded = false);

#endif
