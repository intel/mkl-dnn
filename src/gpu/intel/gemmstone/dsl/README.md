gemmstone DSL
===========================================

Gemmstone DSL is a C++ library for generating optimized Intel GPU kernels. This library aims to fill a gap that exists between assembly and most other programming languages for the purpose of implementing highly optimized kernels. The general goal of this library is to provide:

* Assembly like control when implementing optimized kernels
* Implementation simplicity of high level programing languages
* Fast JIT kernel generation
* Tensor abstraction to simplify implementing GEMM adjacent operations

To accomplish these goals, gemmstone DSL kernel generation generally is composed of the following steps:
1. Building an IR kernel representation via the DSL API
2. Optimize the IR kernel via a sequence of optimization passes
    * This process is highly customizable
    * A few core transformations like expression simplification and common expression elimination are provided by default.
3. Binary/Runtime kernel generation

More details on these particular steps are contained in the following sections. A basic example of this process can be found in the [vector_scale](../examples/vector_scale.cpp) example, the core implementation of which is similar to the following:

```c++
    // Create kernel interface
    kernel::iface_t iface("vector_scale");
    expr_t buffer = iface.register_global("buffer", f32);
    expr_t size = iface.register_arg("size", u32);
    expr_t alpha = iface.register_arg("alpha", f32);

    // Begin building IR
    declare_kernel(iface, kernel::options_t(get_hardware(ocl_device, ocl_context)));

    auto offset = let("global_id", global_id(0, u32));
    auto is_inbounds = offset < size;
    auto b = def("b", load(buffer, offset, is_inbounds));
    b *= alpha;
    store(buffer, offset, b, is_inbounds);

    // End building IR and apply default optimization passes
    kernel_t kernel = end_kernel();

    // Generate runtime kernel object
    cl_kernel runtime_kernel = make_kernel(kernel, ocl_context, ocl_device);
```

In addition, gemmstone supports additional logging via the environment variable `GEMMSTONE_DEBUG`. Setting this variable to one of the following log levels will provide additional information when constructing an IR kernel:

```c++
enum class log_level_t {
    off = 0,
    warning = 100,
    suggestion = 120,
    info = 150,
    debug = 160,
    perf = 170,
    trace = 200,
};
```
# DSL API

At its core, the DSL API is an abstraction for building an IR kernel representation and, in particular, makes building IR kernels similar to a programming language. The core definition of this API is defined in [include/gemmstone/dsl](../include/gemmstone/dsl) files. To keep the DSL API as simple as possible, this API operates on an implicit (thread local) context which tracks information necessary for IR kernel generation (target hardware, known variables, etc.). As a consequence, the DSL API cannot be used to concurrently generate multiple IR kernels within a single thread and this context needs initialized/reset before and after use. This initialization implicitly happens upon a call to `declare_kernel`:

```c++
void declare_kernel(const kernel::iface_t &interface, const kernel::options_t &ctx);
```

After which subsequent DSL API calls are used to construct a sequence of IR statements. Reseting of this context then happens upon a call to `end_kernel`:

```c++
kernel_t end_kernel();
```

when an IR kernel is constructed and returned from the implicit context. This object can then be used to create a kernel in the appropriate runtime via the `make_*` APIs.

```c++
std::string make_asm(const kernel_t &kernel);
std::vector<uint8_t> make_binary(const kernel_t &kernel);
::sycl::kernel make_kernel(const kernel_t &kernel, ::sycl::context ctx, ::sycl::device dev);
LevelZeroKernelAndModule make_kernel(const kernel_t &kernel, ze_context_handle_t ctx, ze_device_handle_t dev);
cl_kernel make_kernel(const kernel_t &kernel, cl_context ctx, cl_device_id dev);
```

In terms of IR construction, the DSL API provides many of the core concepts we expect from a programming language, such as variable declaration, data loading and storing,
and control flow constructs.

```c++

// Mutable variable declaration
lval_t def(const std::string &name, const type_t &type, const expr_t &value = {});
// Immutable variable declaration
lval_t let(const std::string &name, const type_t &type, const expr_t &value);

expr_t load(const expr_t &buf, const expr_t &off, const expr_t &mask = {}, const send_hint_t &hint = {});
void store(const expr_t &buf, const expr_t &off, const expr_t &val, const expr_t mask = {}, const send_hint_t &hint = {});

// F and G are callable objects
template <typename F, typename G>
void _if(const expr_t &cond, const F &if_body, const G &else_body);
template <typename F>
void _for(const expr_t &var, const expr_t &bound, const expr_t &step, const F &body);
template <typename F>
void _while(const expr_t &cond, const F &body);
```

The above list is incomplete, see [gemmstone/dsl.hpp](../include/gemmstone/dsl/dsl.hpp) for a more complete list of the core interfaces. The DSL also contains some higher level extensions via the [tensor API](../include/gemmstone/dsl/tensor.hpp) and the [GEMM API](../include/gemmstone/dsl/gemm) to simplify implementing GEMM like operations. Some [examples](../examples) are also provided to demonstrate expected use of DSL API.

# Optimizing IR Kernels
IR kernel optimization passes are applied upon the call to `end_kernel`. By default, these passes are quite limited in scope, focusing on things like expression simplification and various IR normalization passes to simplify code generation. In some cases, custom transformations may be required. For example, gemmstone provides a mock API for kernel strategies which modifies the expected behavior of a GEMM kernel to help identify performance bottlenecks. In order to isolate the performance of this modification from optimizations like dead code elimination and expression simplification, this modification should be applied after all relevant optimization has been performed.

Unlike the above example, many potential IR passes can be replaced via direct generation. For example, a loop unroll pass could be created at the DSL level via a construct akin to:

```c++
_while(bound, [&]() {
    for(int i = 0; i < unroll; i++) {
        generate_unroll_expr(i);
    }
});
```

When such choices are available, it is often simpler and faster to rely on direct generation as:

 * IR transformations often require extra logic to restore higher-level details trivially accessible when generating via the DSL API
 * IR transformations may interact with each other in a hard-to-control/unexpected ways

As such, this API should be used only when there is some tangible benefit such as:

 * The IR transformation cannot reasonably be implemented via direct generation
 * The IR transformation allows significant reduction in direct generation complexity
 * The IR transformation enables significant reduction in kernel generation time via reusing an existing IR kernel object

Future Work: The DSL API to handle custom IR passes is unimplemented. In the future, we will likely need an interface in kernel::options_t to provide a custom list of transformations to apply before/after the default optimization set. Add an example here once this work has been completed.

# Binary/Runtime Kernel Generation
The final stage in runtime kernel construction is to lower the IR representation to the resulting binary. Currently, this is done via [nGEN](https://github.com/uxlfoundation/oneDNN), although other code generation backends may be added in the future. The general lowering process involves using the visitor pattern to perform an in-order traversal the IR kernel and perform whatever code generation or state tracking is required while visiting an `ir::object_t`. The following is an example for how `while_t` could be lowered via nGEN:

```c++
class ir_to_ngen_t final : public ir_visitor_t {
    ...
    void _visit(const while_t &obj) override {
        auto guard = alloc_scope_guard();
        ngen::Label loop_end_label;
        ngen::Label loop_begin_label;

        host_->mark(loop_begin_label);
        auto cond = evaluate(obj.cond);
        host_->jmpi(1 | ~cond.flag(), loop_end_label);
        visit(obj.body);
        host_->jmpi(1, loop_begin_label);
        host_->mark(loop_end_label);
    }
    ...
}
```
