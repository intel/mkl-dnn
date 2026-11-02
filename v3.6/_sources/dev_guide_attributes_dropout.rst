.. index:: pair: page; Primitive Attributes: dropout
.. _doxid-dev_guide_attributes_dropout:

Primitive Attributes: dropout
=============================

Introduction
~~~~~~~~~~~~

In many DNN and GNN models, `Dropout <https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout>`__ is used to improve training results. In some cases this layer can take a significant amount of time. To enhance training performance, optimize dropout by fusing it with the primitive.

Implementation
~~~~~~~~~~~~~~

In oneDNN, dropout is a special operation akin to a binary post-op that gets applied to the output values of a primitive right before post-ops. It depends on a deterministic PRNG (current implementation uses a variation of Philox algorithm) and transforms the values as follows:

.. math::

	\mathrm{mask}[:] = (\mathrm{PRNG}(S, ...) > P) \\ \mathrm{dst}[:] = \mathrm{mask}[:] \cdot {{\mathrm{dst}[:]} \over {1 - P}}

where:

* :math:`\mathrm{mask}` is the output buffer (always of the same dimensions and usually of the same layout as :math:`\mathrm{dst}`, but potentially differing from it in type that can only be ``u8``) whose values may be either 0 if the corresponding value in :math:`\mathrm{dst}` got zeroed (a.k.a. dropped out) or 1 otherwise

* :math:`S` is the integer seed for the PRNG algorithm

* :math:`P` is the probability for any given value to get dropped out, :math:`0 \leq P \leq 1`

API
~~~

* C: :ref:`dnnl_primitive_attr_get_dropout <doxid-group__dnnl__api__attributes_1ga4491df7294d57081caf721d9512e9d82>`, :ref:`dnnl_primitive_attr_set_dropout <doxid-group__dnnl__api__attributes_1ga2f09a285f20c4f09a716dcff82fa19e9>`

* C++: :ref:`dnnl::primitive_attr::get_dropout <doxid-structdnnl_1_1primitive__attr_1a2f037ab4ec2520d9a0b12777b30522d6>`, :ref:`dnnl::primitive_attr::set_dropout <doxid-structdnnl_1_1primitive__attr_1abe989b6c932434a755bade257d299755>`

If the dropout operation gets specified in the primitive's attributes, the user must provide three additional buffers to it on execution:

* ``DNNL_ARG_ATTR_DROPOUT_MASK`` : through this ID the user has to pass the :math:`\mathrm{mask}` output buffer

* ``DNNL_ARG_ATTR_DROPOUT_PROBABILITY`` : this is a single-value ``f32`` input buffer that holds :math:`P`

* ``DNNL_ARG_ATTR_DROPOUT_SEED`` : this is a single-value ``s32`` input buffer that holds :math:`S`

