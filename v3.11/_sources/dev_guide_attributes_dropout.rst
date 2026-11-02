.. index:: pair: page; Dropout
.. _doxid-dev_guide_attributes_dropout:

Dropout
=======

Introduction
~~~~~~~~~~~~

In many DNN and GNN models, `Dropout <https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout>`__ is used to improve training results. In some cases this layer can take a significant amount of time. To enhance training performance, optimize dropout by fusing it with the primitive.

Implementation
~~~~~~~~~~~~~~

In oneDNN, dropout is a special operation akin to a binary post-op that gets applied to the output values of a primitive right before post-ops. It depends on a deterministic PRNG (current implementation uses a variation of Philox algorithm) and transforms the values as follows:

.. math::

	\mathrm{mask}[:] = (\mathrm{PRNG}(S, \mathrm{off}, :) > P) \\ \mathrm{dst}[:] = \mathrm{mask}[:] \cdot {{\mathrm{dst}[:]} \over {1 - P}}

where:

* :math:`\mathrm{mask}` values may be either 0 if the corresponding value in :math:`\mathrm{dst}` got zeroed (a.k.a. dropped out) or 1, otherwise.

* :math:`S, off` are the seed and the offset for the PRNG algorithm.

* :math:`P` is the probability for any given value to get dropped out, :math:`0 \leq P \leq 1`

API
~~~

* C: :ref:`dnnl_primitive_attr_get_dropout <doxid-group__dnnl__api__attributes_1gaa99e375cd5745218290260ff7b9e56e2>`, :ref:`dnnl_primitive_attr_set_dropout <doxid-group__dnnl__api__attributes_1ga29956085de89106951a10f290e4aab87>`

* C++: :ref:`dnnl::primitive_attr::get_dropout <doxid-structdnnl_1_1primitive__attr_1a2f037ab4ec2520d9a0b12777b30522d6>`, :ref:`dnnl::primitive_attr::set_dropout <doxid-structdnnl_1_1primitive__attr_1abe989b6c932434a755bade257d299755>`

The dropout primitive attribute has the following parameters:

* ``mask_desc`` : when set to a zero (or empty) memory descriptor, mask values are not written to the memory. Otherwise, it should have the same dimensions and the same layout as :math:`\mathrm{dst}`, as well as ``u8`` data type.

* ``seed_dt`` : data type of the seed argument :math:`S`, ``s64`` is recommended, ``s32`` is supported as a backward compatibility option.

* ``use_offset`` : boolean to express if an offset argument will be provided by the user at the execution time. When false, an offset of 0 is assumed.

* ``use_host_scalars`` : boolean specifying if probability, seed, and offset memory arguments will be passed as host_scalar memory objects when ``true``, or as device memory objects, otherwise.

When the dropout primitive attribute is set, the user must provide two additional memory arguments to the primitive execution:

* ``DNNL_ARG_ATTR_DROPOUT_PROBABILITY`` : this is a single-value ``f32`` input memory argument that holds :math:`P`.

* ``DNNL_ARG_ATTR_DROPOUT_SEED`` : this is a single-value input memory argument that holds :math:`S`. Its data type is specified by the ``seed_dt`` primitive attribute parameter and can be either ``s32`` or ``s64``.

Additionally, the following arguments conditionally need to be passed at the execution time as well:

* ``DNNL_ARG_ATTR_DROPOUT_MASK`` : if the ``mask_desc`` primitive attribute parameter is not a zero memory descriptor, the user must pass the :math:`\mathrm{mask}` through this output memory argument.

* ``DNNL_ARG_ATTR_DROPOUT_OFFSET`` : if the ``use_offset`` primitive attribute parameter is set, the user must pass the :math:`\mathrm{off}` through this input memory argument. This is a single-value ``s64`` memory argument.

