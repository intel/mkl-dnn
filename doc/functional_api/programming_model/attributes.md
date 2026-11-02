Primitive Attributes {#dev_guide_attributes}
============================================

A quick recap of the primitive creation step, which consists of the following:
1. Creating a primitive descriptor based on the engine, operation parameters,
   and **attributes**. During creation of a primitive for backward propagation,
   the primitive descriptor from the forward propagation might be required as
   well (see [Training-Specific Aspects](@ref dev_guide_inference_and_training_aspects_training)).
2. Creating a primitive, solely based on a primitive descriptor.

Details on why all these steps are required can be found in
@ref dev_guide_basic_concepts. The fact that is important for us now is that
a primitive descriptor created at step 1 fully defines the operation that the
corresponding primitive will execute. Once the primitive descriptor is created,
it cannot be changed.

The parameters passed to create a primitive descriptor specify the problem. An
engine specifies where the primitive will be executed. Primitive parameters
specify the basics: the operation kind; the propagation kind; the source,
destination, and other tensors; the strides (if applicable); and so on.

**Attributes** specify some extra properties of the primitive. The attributes
were designed to be extensible, hence they are an opaque structure. Users must
create them before use and must set required specifics using the corresponding
setters. The attributes are copied during primitive descriptor creation, so
users can change or destroy attributes right after that.

If not modified, attributes can stay empty, which is equivalent to the default
attributes. For that purpose, in the C API users can pass `NULL` as an
attribute upon primitive descriptor creation. In the C++ API,
primitive descriptors' constructors have empty attributes as default
parameters, so, unless they are required, users can simply omit them.

## Attributes Usage

Below are the skeletons of using attributes with the C and C++ APIs. Error
handling is omitted to simplify reading.

~~~cpp
// ### C API ###

dnnl_primitive_attr_t attr; // opaque attributes
dnnl_primitive_attr_create(&attr);
dnnl_primitive_attr_set_SOMETHING(attr, params); // setting attributes params
dnnl_primitive_attr_set_SOMETHING_ELSE(attr, other_params);
dnnl_eltwise_backward_primitive_desc_create(&op_pd, engine, ..., hint_fwd_pd, attr);

// changing attr object here does not have any effect on op_pd

// once attr is no more used we can immediately destroy it
dnnl_primitive_attr_destroy(attr);

...

// ### C++ API ###

dnnl::primitive_attr attr;
attr.set_SOMETHING(params);
attr.set_SOMETHING_ELSE(params);

primitive::primitive_desc pd(..., attr);

// in C++ destroying of attr happens automatically

~~~

## Supported Attributes

As mentioned above, the attributes enable extending or changing the default
primitive behavior. Currently the following attributes are supported.
The detailed explanation is provided in the corresponding sections.

- [Scratchpad](@ref dev_guide_attributes_scratchpad) behavior: handling the
  intermediate temporary memory by the library or a user.
- [Floating-point math mode](@ref dev_guide_attributes_fpmath_mode) to
  allow implicit down-conversions of `f32` values during computation.
- [Accumulation mode](@ref dev_guide_attributes_accumulation_mode) to
  allow the usage of lower precision datatypes for accumulation.
- [Rounding mode](@ref dev_guide_attributes_rounding_mode) to control
  rounding mode upon specific argument downconversions.
- [Deterministic mode](@ref dev_guide_attributes_deterministic) to enforce
  run-to-run deterministic primitive execution.
- [Dropout](@ref dev_guide_attributes_dropout) to apply pseudo-random dropout
  to the output buffer.
- [Quantization](@ref dev_guide_attributes_quantization) settings used in int8
  inference.
- [Post-ops](@ref dev_guide_attributes_post_ops) to fuse a primitive with
  some operation applied to the primitive's result. Used mostly for inference.


## Attribute Related Error Handling
@anchor dev_guide_attributes_error_handling

Because the attributes are created separately from the corresponding primitive
descriptor, consistency checks are delayed. Users can successfully set
attributes in whatever configuration they want. However, when they try to
create a primitive descriptor with the attributes they set, it might happen
that there is no primitive implementation that supports such a configuration.
In this case, the library will return #dnnl_unimplemented in the case of the C
API or throw a corresponding @ref dnnl::error exception in the case of the C++
API. Unfortunately, the library does not currently provide any hints about what
exactly is going wrong in this case. The corresponding section of the
documentation simply documents the primitives' capabilities.

If the configuration is not supported only because the processor lacks
support for a data type used by the attributes, for example the data type of a
binary post-op argument, the library will return #dnnl_unsupported_data_type
instead. Refer to [Hardware Limitations](@ref data_types_hardware_limitations)
for details.

