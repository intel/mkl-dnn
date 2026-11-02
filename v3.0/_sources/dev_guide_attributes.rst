.. index:: pair: page; Primitive Attributes
.. _doxid-dev_guide_attributes:

Primitive Attributes
====================

A quick recap of the primitive creation step, which consists of the following:

#. Creating a primitive descriptor based on the engine, operation parameters, and attributes. During creation of a primitive for backward propagation, the primitive descriptor from the forward propagation might be required as well (see :ref:`Training-Specific Aspects <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_training>`).

#. Creating a primitive, solely based on a primitive descriptor.

Details on why all these steps are required can be found in :ref:`Basic Concepts <doxid-dev_guide_basic_concepts>`. The fact that is important for us now is that a primitive descriptor created at step 2 fully defines the operation that the corresponding primitive will execute. Once the primitive descriptor is created, it cannot be changed.

The parameters passed to create a primitive descriptor specify the problem. An engine specifies where the primitive will be executed. Primitive parameters specify the basics: the operation kind; the propagation kind; the source, destination, and other tensors; the strides (if applicable); and so on.

Attributes specify some extra properties of the primitive. The attributes were designed to be extensible, hence they are an opaque structure. Users must create them before use and must set required specifics using the corresponding setters. The attributes are copied during primitive descriptor creation, so users can change or destroy attributes right after that.

If not modified, attributes can stay empty, which is equivalent to the default attributes. For that purpose, in the C API users can pass ``NULL`` as an attribute upon primitive descriptor creation. In the C++ API, primitive descriptors' constructors have empty attributes as default parameters, so, unless they are required, users can simply omit them.

Attributes Usage
~~~~~~~~~~~~~~~~

Below are the skeletons of using attributes with the C and C++ APIs. Error handling is omitted to simplify reading.

.. ref-code-block:: cpp

	// ### C API ###
	
	:ref:`dnnl_primitive_attr_t <doxid-structdnnl__primitive__attr>` attr; // opaque attributes
	:ref:`dnnl_primitive_attr_create <doxid-group__dnnl__api__attributes_1gaf630fdc0d8d0fd8522ec93852a559081>`(&attr);
	dnnl_primitive_attr_set_SOMETHING(attr, params); // setting attributes params
	dnnl_primitive_attr_set_SOMETHING_ELSE(attr, other_params);
	:ref:`dnnl_eltwise_backward_primitive_desc_create <doxid-group__dnnl__api__eltwise_1gaba11ca62016a1c23d997db47bcd6c27d>`(&op_pd, engine, ..., hint_fwd_pd, attr);
	
	// changing attr object here does not have any effect on op_pd
	
	// once attr is no more used we can immediately destroy it
	:ref:`dnnl_primitive_attr_destroy <doxid-group__dnnl__api__attributes_1ga96a7539382945195627f2932bff8fadb>`(attr);
	
	...
	
	// ### C++ API ###
	
	dnnl::primitive_attr attr;
	attr.set_SOMETHING(params);
	attr.set_SOMETHING_ELSE(params);
	
	primitive::primitive_desc pd(..., attr);
	
	// in C++ destroying of attr happens automatically

Supported Attributes
~~~~~~~~~~~~~~~~~~~~

As mentioned above, the attributes enable extending or changing the default primitive behavior. Currently the following attributes are supported. The detailed explanation is provided in the corresponding sections.

* :ref:`Scratchpad <doxid-dev_guide_attributes_scratchpad>` behavior: handling the intermediate temporary memory by the library or a user;

* :ref:`Floating-point math mode <doxid-dev_guide_attributes_fpmath_mode>` to allow implicit down-conversions of f32 values during computation;

* :ref:`Quantization <doxid-dev_guide_attributes_quantization>` settings used in INT8 inference;

* :ref:`Post-ops <doxid-dev_guide_attributes_post_ops>` to fuse a primitive with some operation applied to the primitive's result. Used mostly for inference.

Attribute Related Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:target:`doxid-dev_guide_attributes_1dev_guide_attributes_error_handling` Because the attributes are created separately from the corresponding primitive descriptor, consistency checks are delayed. Users can successfully set attributes in whatever configuration they want. However, when they try to create a primitive descriptor with the attributes they set, it might happen that there is no primitive implementation that supports such a configuration. In this case, the library will return :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` in the case of the C API or throw a corresponding :ref:`dnnl::error <doxid-structdnnl_1_1error>` exception in the case of the C++ API. Unfortunately, the library does not currently provide any hints about what exactly is going wrong in this case. The corresponding section of the documentation simply documents the primitives' capabilities.


.. toctree::
   :hidden:

   dev_guide_attributes_fpmath_mode.rst
   dev_guide_attributes_quantization.rst
   dev_guide_attributes_post_ops.rst
   dev_guide_attributes_scratchpad.rst


