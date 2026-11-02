.. index:: pair: page; RNN
.. _doxid-dev_guide_rnn:

RNN
===

:ref:`API Reference <doxid-group__dnnl__api__rnn>`

General
~~~~~~~

The RNN primitive computes a stack of unrolled recurrent cells, as depicted in Figure 1. :math:`\bias`, :math:`\srciter` and :math:`\dstiter` are optional parameters (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`). If not provided, :math:`\bias` and :math:`\srciter` will default to 0.

.. image:: unrolled_stack_rnn.jpg
	:alt: Figure 1: Example of stacked recurrent cells unrolled over the time dimension and executed with the `left2right` direction. Dashed lines represent optional parameters.

The RNN primitive supports four modes for evaluation direction:

* ``left2right`` will process the input data timestamps by increasing order

* ``right2left`` will process the input data timestamps by decreasing order

* ``bidirectional_concat`` will process all the stacked layers from ``left2right`` and from ``right2left`` independently, and will concatenate the output in :math:`\dstlayer` over the channel dimension.

* ``bidirectional_sum`` will process all the stacked layers from ``left2right`` and from ``right2left`` independently, and will sum the two outputs to :math:`\dstlayer`.

Even though the RNN primitive supports passing a different number of channels for :math:`\srclayer`, :math:`\srciter`, :math:`\dstlayer`, and :math:`\dstiter`, we always require the following conditions in order for the dimension to be consistent:

* :math:`channels(\dstlayer) = channels(\dstiter)`,

* when :math:`T > 1`, :math:`channels(\srciter) = channels(\dstiter)`,

* when :math:`L > 1`, :math:`channels(\srclayer) = channels(\dstlayer)`,

* when using the ``bidirectional_concat`` direction, :math:`channels(\dstlayer) = 2 * channels(\dstiter)`.

The general formula for the execution of a stack of unrolled recurrent cells depends on the current iteration of the previous layer (:math:`h_{t,l-1}` and :math:`c_{t,l-1}`) and the previous iteration of the current layer (:math:`h_{t-1, l}`). Here is the exact equation for non-LSTM cells:

.. math::

	\begin{align} h_{t, l} = Cell(h_{t, l-1}, h_{t-1, l}) \end{align}

where :math:`t,l` are the indices of the timestamp and the layer of the cell being executed.

And here is the equation for LSTM cells:

.. math::

	\begin{equation*} (h_{t, l},c_{t,l}) = Cell(h_{t, l-1}, h_{t-1, l}, c_{t-1,l}) \end{equation*}

where :math:`t,l` are the indices of the timestamp and the layer of the cell being executed.

Cell Functions
~~~~~~~~~~~~~~

The RNN API provides four cell functions:

* `Vanilla RNN <#vanilla-rnn>`__, a single-gate recurrent cell,

* `LSTM <#lstm>`__, a four-gate long short-term memory cell,

* `GRU <#gru>`__, a three-gate gated recurrent unit cell,

* `Linear-before-reset GRU <#linear-before-reset-gru>`__, a three-gate recurrent unit cell with the linear layer before the reset gate,

* `AUGRU <#augru>`__, a three-gate gated recurrent unit cell with the attention update gate,

* `Linear-before-reset AUGRU <#linear-before-reset-augru>`__, a three-gate recurrent unit cell with the linear layer before the reset gate and the attention update gate.

Vanilla RNN
-----------

A single-gate recurrent cell initialized with :ref:`dnnl::vanilla_rnn_forward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1vanilla__rnn__forward_1_1primitive__desc_1ac5cd59057ae4b2aa3bf1b6dbbafaa49d>` or :ref:`dnnl::vanilla_rnn_backward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1vanilla__rnn__backward_1_1primitive__desc_1a8d7e9e966d54a04d4ac8e2d47df637e4>` as in the following example.

.. ref-code-block:: cpp

	auto vanilla_rnn_pd = :ref:`dnnl::vanilla_rnn_forward::primitive_desc <doxid-structdnnl_1_1vanilla__rnn__forward_1_1primitive__desc>`(
	    engine, aprop, activation, direction, src_layer_desc, src_iter_desc,
	    weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc,
	    dst_iter_desc);

The Vanilla RNN cell supports the ReLU, Tanh and Sigmoid activation functions. The following equations defines the mathematical operation performed by the Vanilla RNN cell for the forward pass:

.. math::

	\begin{align} a_t &= W \cdot h_{t,l-1} + U \cdot h_{t-1, l} + B \\ h_t &= activation(a_t) \end{align}

LSTM
----

LSTM (or Vanilla LSTM)
++++++++++++++++++++++

A four-gate long short-term memory recurrent cell initialized with :ref:`dnnl::lstm_forward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc_1a5148ad45607a698afb1093c5ede64a91>` or :ref:`dnnl::lstm_backward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1lstm__backward_1_1primitive__desc_1a46152ef35d4c4004ae1878bef29b2775>` as in the following example.

.. ref-code-block:: cpp

	auto lstm_pd = lstm_forward::primitive_desc(
	    engine, aprop, direction, src_layer_desc, src_iter_h_desc,
	    src_iter_c_desc, weights_layer_desc, weights_iter_desc, bias_desc,
	    dst_layer_desc, dst_iter_h_desc, dst_iter_c_desc);

Note that for all tensors with a dimension depending on the gate number, we implicitly require the order of these gates to be ``i``, ``f``, :math:`\tilde c`, and ``o``. The following equation gives the mathematical description of these gates and output for the forward pass:

.. math::

	\begin{align} i_t &= \sigma(W_i \cdot h_{t,l-1} + U_i \cdot h_{t-1, l} + B_i) \\ f_t &= \sigma(W_f \cdot h_{t,l-1} + U_f \cdot h_{t-1, l} + B_f) \\ \\ \tilde c_t &= \tanh(W_{\tilde c} \cdot h_{t,l-1} + U_{\tilde c} \cdot h_{t-1, l} + B_{\tilde c}) \\ c_t &= f_t * c_{t-1} + i_t * \tilde c_t \\ \\ o_t &= \sigma(W_o \cdot h_{t,l-1} + U_o \cdot h_{t-1, l} + B_o) \\ h_t &= \tanh(c_t) * o_t \end{align}

where :math:`W_*` are stored in :math:`\weightslayer`, :math:`U_*` are stored in :math:`\weightsiter` and :math:`B_*` are stored in :math:`\bias`.

.. note:: 

   In order for the dimensions to be consistent, we require :math:`channels(\srciterc) = channels(\dstiterc) = channels(\dstiter)`.
   
   


LSTM with Peephole
++++++++++++++++++

A four-gate long short-term memory recurrent cell with peephole initialized with :ref:`dnnl::lstm_forward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc_1a5148ad45607a698afb1093c5ede64a91>` or :ref:`dnnl::lstm_backward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1lstm__backward_1_1primitive__desc_1a46152ef35d4c4004ae1878bef29b2775>` as in the following example.

.. ref-code-block:: cpp

	auto lstm_pd = :ref:`dnnl::lstm_forward::primitive_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc>`(
	    engine, aprop, direction, src_layer_desc, src_iter_h_desc, src_iter_c_desc,
	    weights_layer_desc, weights_iter_desc, weights_peephole_desc,
	    bias_desc, dst_layer_desc, dst_iter_h_desc, dst_iter_c_desc);

Similarly to vanilla LSTM, we implicitly require the order of the gates to be ``i``, ``f``, :math:`\tilde c`, and ``o`` for all tensors with a dimension depending on the gates. For peephole weights, the gates order is ``i``, ``f``, ``o``. The following equation gives the mathematical description of these gates and output for the forward pass:

.. math::

	\begin{align} i_t &= \sigma(W_i \cdot h_{t,l-1} + U_i \cdot h_{t-1, l} + P_i \cdot c_{t-1} + B_i) \\ f_t &= \sigma(W_f \cdot h_{t,l-1} + U_f \cdot h_{t-1, l} + P_f \cdot c_{t-1} + B_f) \\ \\ \tilde c_t &= \tanh(W_{\tilde c} \cdot h_{t,l-1} + U_{\tilde c} \cdot h_{t-1, l} + B_{\tilde c}) \\ c_t &= f_t * c_{t-1} + i_t * \tilde c_t \\ \\ o_t &= \sigma(W_o \cdot h_{t,l-1} + U_o \cdot h_{t-1, l} + P_o \cdot c_t + B_o) \\ h_t &= \tanh(c_t) * o_t \end{align}

where :math:`P_*` are stored in ``weights_peephole``, and the other parameters are the same as in vanilla LSTM.

.. note:: 

   If the ``weights_peephole_desc`` passed to the primitive descriptor constructor is a zero memory desciptor, the primitive will behave the same as in LSTM primitive without peephole.
   
   


LSTM with Projection (or LSTMP)
+++++++++++++++++++++++++++++++

A four-gate long short-term memory recurrent cell with projection initialized with :ref:`dnnl::lstm_forward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc_1a5148ad45607a698afb1093c5ede64a91>` or :ref:`dnnl::lstm_backward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1lstm__backward_1_1primitive__desc_1a46152ef35d4c4004ae1878bef29b2775>` as in the following example.

.. ref-code-block:: cpp

	auto lstm_pd = :ref:`dnnl::lstm_forward::primitive_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc>`(
	    engine, aprop, direction, src_layer_desc, src_iter_h_desc, src_iter_c_desc,
	    weights_layer_desc, weights_iter_desc, weights_peephole_desc,
	    weights_projection_desc, bias_desc, dst_layer_desc, dst_iter_h_desc,
	    dst_iter_c_desc);

Similarly to vanilla LSTM, we implicitly require the order of the gates to be ``i``, ``f``, :math:`\tilde c`, and ``o`` for all tensors with a dimension depending on the gates. The following equation gives the mathematical description of these gates and output for the forward pass (for simplicity, LSTM without peephole is shown):

.. math::

	\begin{align} i_t &= \sigma(W_i \cdot h_{t,l-1} + U_i \cdot h_{t-1,l} + B_i) \\ f_t &= \sigma(W_f \cdot h_{t,l-1} + U_f \cdot h_{t-1,l} + B_f) \\ & \\ \tilde{c}_t &= \tanh(W_{\tilde{c}} \cdot h_{t,l-1} + U_{\tilde{c}} \cdot h_{t-1,l} + B_{\tilde{c}}) \\ c_t &= f_t * c_{t-1} + i_t * \tilde{c}_t \\ & \\ o_t &= \sigma(W_o \cdot h_{t,l-1} + U_o \cdot h_{t-1,l} + B_o) \\ h_t &= R \cdot (\tanh(c_t) * o_t) \end{align}

where :math:`R` is stored in ``weights_projection``, and the other parameters are the same as in vanilla LSTM.

.. note:: 

   If the ``weights_projection_desc`` passed to the primitive descriptor constructor is a zero memory desciptor, the primitive will behave the same as in LSTM primitive without projection.
   
   


GRU
---

A three-gate gated recurrent unit cell, initialized with :ref:`dnnl::gru_forward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a7dec21003d026bf7cec53362134be4a6>` or :ref:`dnnl::gru_backward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1gru__backward_1_1primitive__desc_1a9d9a3296626309417b3fb9f65e22f9ca>` as in the following example.

.. ref-code-block:: cpp

	auto gru_pd = :ref:`dnnl::gru_forward::primitive_desc <doxid-structdnnl_1_1gru__forward_1_1primitive__desc>`(
	    engine, aprop, direction, src_layer_desc, src_iter_desc,
	    weights_layer_desc, weights_iter_desc, bias_desc,
	    dst_layer_desc, dst_iter_desc);

Note that for all tensors with a dimension depending on the gate number, we implicitly require the order of these gates to be ``u``, ``r``, and ``o``. The following equation gives the mathematical definition of these gates.

.. math::

	\begin{align} u_t &= \sigma(W_u \cdot h_{t,l-1} + U_u \cdot h_{t-1, l} + B_u) \\ r_t &= \sigma(W_r \cdot h_{t,l-1} + U_r \cdot h_{t-1, l} + B_r) \\ o_t &= \tanh(W_o \cdot h_{t,l-1} + U_o \cdot (r_t * h_{t-1, l}) + B_o) \\ h_t &= u_t * h_{t-1, l} + (1 - u_t) * o_t \end{align}

where :math:`W_*` are in :math:`\weightslayer`, :math:`U_*` are in :math:`\weightsiter`, and :math:`B_*` are stored in :math:`\bias`.

.. note:: 

   If you need to replace u_t by (1-u_t) when computing h_t, you can achieve this by multiplying :math:`W_u`, :math:`U_u` and :math:`B_u` by :math:`-1`. This is possible as :math:`u_t = \sigma(W_u \cdot h_{t,l-1} + U_u \cdot h_{t-1, l} + B_u)`, and :math:`1 – \sigma(a) = \sigma(-a)`.
   
   


Linear-Before-Reset GRU
-----------------------

A three-gate gated recurrent unit cell with linear layer applied before the reset gate, initialized with :ref:`dnnl::lbr_gru_forward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1lbr__gru__forward_1_1primitive__desc_1ad4f6b8d67cf3232d20e880128bde2660>` or :ref:`dnnl::lbr_gru_backward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a72261d946a2fe263c884aaabaaca2164>` as in the following example.

.. ref-code-block:: cpp

	auto lbr_gru_pd = :ref:`dnnl::lbr_gru_forward::primitive_desc <doxid-structdnnl_1_1lbr__gru__forward_1_1primitive__desc>`(
	    engine, aprop, direction, src_layer_desc, src_iter_desc,
	    weights_layer_desc, weights_iter_desc, bias_desc,
	    dst_layer_desc, dst_iter_desc);

The following equation describes the mathematical behavior of the Linear-Before-Reset GRU cell.

.. math::

	\begin{align} u_t &= \sigma(W_u \cdot h_{t,l-1} + U_u \cdot h_{t-1, l} + B_u) \\ r_t &= \sigma(W_r \cdot h_{t,l-1} + U_r \cdot h_{t-1, l} + B_r) \\ o_t &= \tanh(W_o \cdot h_{t,l-1} + r_t *(U_o \cdot h_{t-1, l} + B_{u'}) + B_o) \\ h_t &= u_t * h_{t-1, l} + (1 - u_t) * o_t \end{align}

Note that for all tensors with a dimension depending on the gate number, except the bias, we implicitly require the order of these gates to be ``u``, ``r``, and ``o``. For the :math:`\bias` tensor, we implicitly require the order of the gates to be ``u``, ``r``, ``o``, and u `.

.. note:: 

   If you need to replace u_t by (1-u_t) when computing h_t, you can achieve this by multiplying :math:`W_u`, :math:`U_u` and :math:`B_u` by :math:`-1`. This is possible as :math:`u_t = \sigma(W_u \cdot h_{t,l-1} + U_u \cdot h_{t-1, l} + B_u)`, and :math:`1 – \sigma(a) = \sigma(-a)`.
   
   


AUGRU
-----

A three-gate gated recurrent unit cell, initialized with :ref:`dnnl::augru_forward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1augru__forward_1_1primitive__desc_1a46f05b511d0b13704f6bc3af4e0c5804>` or :ref:`dnnl::augru_backward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1augru__backward_1_1primitive__desc_1a7ca7158311dc864d1bd6c56d5915defc>` as in the following example.

.. ref-code-block:: cpp

	auto augru_pd = :ref:`dnnl::augru_forward::primitive_desc <doxid-structdnnl_1_1augru__forward_1_1primitive__desc>`(
	    engine, aprop, direction, src_layer_desc, src_iter_desc, attention_desc,
	    weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc,
	    dst_iter_desc);

Note that for all tensors with a dimension depending on the gate number, we implicitly require the order of these gates to be ``u``, ``r``, and ``o``. The following equation gives the mathematical definition of these gates.

.. math::

	\begin{align} u_t &= \sigma(W_u \cdot h_{t,l-1} + U_u \cdot h_{t-1, l} + B_u) \\ r_t &= \sigma(W_r \cdot h_{t,l-1} + U_r \cdot h_{t-1, l} + B_r) \\ o_t &= \tanh(W_o \cdot h_{t,l-1} + U_o \cdot (r_t * h_{t-1, l}) + B_o) \\ \tilde u_t &= (1 - a_t) * u_t \\ h_t &= \tilde u_t * h_{t-1, l} + (1 - \tilde u_t) * o_t \end{align}

where :math:`W_*` are in :math:`\weightslayer`, :math:`U_*` are in :math:`\weightsiter`, and :math:`B_*` are stored in :math:`\bias`.

Linear-Before-Reset AUGRU
-------------------------

A three-gate gated recurrent unit cell with linear layer applied before the reset gate, initialized with :ref:`dnnl::lbr_augru_forward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1lbr__augru__forward_1_1primitive__desc_1acbbe1666db758249e8637ef2a04d8ab4>` or :ref:`dnnl::lbr_augru_backward::primitive_desc::primitive_desc() <doxid-structdnnl_1_1lbr__augru__backward_1_1primitive__desc_1aeef40e28d5b5279ac195f92a5ee6b067>` as in the following example.

.. ref-code-block:: cpp

	auto lbr_augru_pd = :ref:`dnnl::lbr_augru_forward::primitive_desc <doxid-structdnnl_1_1lbr__augru__forward_1_1primitive__desc>`(
	    engine, aprop, direction, src_layer_desc, src_iter_desc, attention_desc,
	    weights_layer_desc, weights_iter_desc, bias_desc,
	    dst_layer_desc, dst_iter_desc);

The following equation describes the mathematical behavior of the Linear-Before-Reset AUGRU cell.

.. math::

	\begin{align} u_t &= \sigma(W_u \cdot h_{t,l-1} + U_u \cdot h_{t-1, l} + B_u) \\ r_t &= \sigma(W_r \cdot h_{t,l-1} + U_r \cdot h_{t-1, l} + B_r) \\ o_t &= \tanh(W_o \cdot h_{t,l-1} + r_t *(U_o \cdot h_{t-1, l} + B_{u'}) + B_o) \\ \tilde u_t &= (1 - a_t) * u_t \\ h_t &= \tilde u_t * h_{t-1, l} + (1 - \tilde u_t) * o_t \end{align}

Note that for all tensors with a dimension depending on the gate number, except the bias, we implicitly require the order of these gates to be ``u``, ``r``, and ``o``. For the :math:`\bias` tensor, we implicitly require the order of the gates to be ``u``, ``r``, ``o``, and u `.

Considerations for Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the RNN API for training, the forward pass should use the ``forward_training`` propagation kind, and a workspace should be passed to both the forward pass and the backward pass. Note that after executing the backward pass, the workspace is no more valid and should be populated once again by another forward pass.

The RNN primitive backward pass accumulates gradients to its weight outputs (namely :math:`\diffweightslayer`, :math:`\diffweightsiter`, :math:`\diffweightspeephole`, :math:`\diffweightsprojection`, :math:`\diffbias`). Hence, these tensors should be properly initialized to zero before their first use, and can be reused across calls to accumulate gradients if need be. This behavior can be altered by the RNN flag ``diff_weights_overwrite``. If this flag is set weight gradients will be initialized by zeros by the RNN primitive.

:target:`doxid-dev_guide_rnn_1dg_rnn_impl_limits`

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

===============================  ==================================  
Primitive input/output           Execution argument index            
===============================  ==================================  
:math:`\srclayer`                DNNL_ARG_SRC_LAYER                  
:math:`\srclayerattention`       DNNL_ARG_SRC_LAYER_ATTENTION        
:math:`\srciter`                 DNNL_ARG_SRC_ITER                   
:math:`\srciterc`                DNNL_ARG_SRC_ITER_C                 
:math:`\weightslayer`            DNNL_ARG_WEIGHTS_LAYER              
:math:`\weightsiter`             DNNL_ARG_WEIGHTS_ITER               
:math:`\weightspeephole`         DNNL_ARG_WEIGHTS_PEEPHOLE           
:math:`\weightsprojection`       DNNL_ARG_WEIGHTS_PROJECTION         
:math:`\bias`                    DNNL_ARG_BIAS                       
:math:`\dstlayer`                DNNL_ARG_DST_LAYER                  
:math:`\dstiter`                 DNNL_ARG_DST_ITER                   
:math:`\dstiterc`                DNNL_ARG_DST_ITER_C                 
:math:`\workspace`               DNNL_WORKSPACE                      
:math:`\diffsrclayer`            DNNL_ARG_DIFF_SRC_LAYER             
:math:`\diffsrclayerattention`   DNNL_ARG_DIFF_SRC_LAYER_ATTENTION   
:math:`\diffsrciter`             DNNL_ARG_DIFF_SRC_ITER              
:math:`\diffsrciterc`            DNNL_ARG_DIFF_SRC_ITER_C            
:math:`\diffweightslayer`        DNNL_ARG_DIFF_WEIGHTS_LAYER         
:math:`\diffweightsiter`         DNNL_ARG_DIFF_WEIGHTS_ITER          
:math:`\diffweightspeephole`     DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE      
:math:`\diffweightsprojection`   DNNL_ARG_DIFF_WEIGHTS_PROJECTION    
:math:`\diffbias`                DNNL_ARG_DIFF_BIAS                  
:math:`\diffdstlayer`            DNNL_ARG_DIFF_DST_LAYER             
:math:`\diffdstiter`             DNNL_ARG_DIFF_DST_ITER              
:math:`\diffdstiterc`            DNNL_ARG_DIFF_DST_ITER_C            
===============================  ==================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

Data Type Support
-----------------

The following table lists the combination of data types supported by the RNN primitive for each input and output memory object.

=======================  ============================  ===========  ===================  ========  =====  ============  
Propagation              Cell Function                 Input data   Recurrent data (1)   Weights   Bias   Output Data   
=======================  ============================  ===========  ===================  ========  =====  ============  
Forward / Backward       All                           f32          f32                  f32       f32    f32           
Forward / Backward (2)   All (3)                       bf16         bf16                 bf16      f32    bf16          
Forward                  All (3)                       f16          f16                  f16       f16    f16           
Forward inference        Vanilla LSTM, LSTMP and GRU   u8           u8                   s8        f32    u8, f32       
Forward inference        Vanilla LSTM, LSTMP           s8           s8                   s8        f32    s8, f32       
=======================  ============================  ===========  ===================  ========  =====  ============

(1) With LSTM and Peephole LSTM cells, the cell state datatype is f32, except for the f16 configuration.

(2) In backward propagation, all ``diff_*`` tensors are in f32.

(3) Projection LSTM is not supported.

.. warning:: 

   There might be hardware and/or implementation specific restrictions. Check :ref:`Implementation Limitations <doxid-dev_guide_rnn_1dg_rnn_impl_limits>` section below.
   
   


Data Representation
-------------------

In the oneDNN programming model, the RNN primitive is one of a few that support the placeholder memory format :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` (shortened to ``any`` from now on) and can define data and weight memory objects format based on the primitive parameters.

The following table summarizes the data layouts supported by the RNN primitive.

===================  ================================================================================================================================================================================================================================================  ==================================================================================================================================  ==================================================================================================================================  ========================================================================================================================  ==================================================================================================================================  
Propagation          Input/Output Data                                                                                                                                                                                                                                 Recurrent Data                                                                                                                      Layer and Iteration Weights                                                                                                         Peephole Weights and Bias                                                                                                 Projection LSTM Weights                                                                                                             
===================  ================================================================================================================================================================================================================================================  ==================================================================================================================================  ==================================================================================================================================  ========================================================================================================================  ==================================================================================================================================  
Forward / Backward   :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`                                                                                                                 :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`   :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`   :ref:`dnnl_ldgo <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2bc162b37fd0049dceab3b12300a26c7>`   :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`   
Forward              :ref:`dnnl_ntc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5d73ca7a68559ef44241be5a096e6bff>` , :ref:`dnnl_tnc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2a9735ec024c9362b717304edbfe2237>`   :ref:`dnnl_ldnc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4fd1cf9fdb67c554bcd8281695b65b3c>`             :ref:`dnnl_ldigo <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da96a227ab1a1be1825c1fa596c38847fc>`            :ref:`dnnl_ldgo <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2bc162b37fd0049dceab3b12300a26c7>`   :ref:`dnnl_ldio <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2d5a8e24d6b4904b4e8986d9b0fb4613>`             
Backward             :ref:`dnnl_ntc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5d73ca7a68559ef44241be5a096e6bff>` , :ref:`dnnl_tnc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2a9735ec024c9362b717304edbfe2237>`   :ref:`dnnl_ldnc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4fd1cf9fdb67c554bcd8281695b65b3c>`             :ref:`dnnl_ldgoi <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5fc9120d8f52d7d7fa853aa79bf654fe>`            :ref:`dnnl_ldgo <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2bc162b37fd0049dceab3b12300a26c7>`   :ref:`dnnl_ldoi <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da475da8ead8c761bac894e6c87042355d>`             
===================  ================================================================================================================================================================================================================================================  ==================================================================================================================================  ==================================================================================================================================  ========================================================================================================================  ==================================================================================================================================

While an RNN primitive can be created with memory formats specified explicitly, the performance is likely to be sub-optimal. When using ``any``, it is necessary to first create an RNN primitive descriptor and then query it for the actual data and weight memory objects formats.

.. note:: 

   The RNN primitive supports padded tensors and views. So even if two memory descriptors share the same data layout, they might still be different.
   
   


Post-Ops and Attributes
-----------------------

Currently post-ops and attributes are only used by the int8 variants of LSTM and GRU. See the markdown :ref:`RNN int8 inference example <doxid-cpu_rnn_inference_int8_cpp>` for more details on how to use and set these quantization parameters.

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. Bias must always be present (that is, the corresponding memory descriptor argument cannot be zero memory descriptor when the RNN primitive descriptor is initialized).



#. CPU
   
   * oneDNN supports s8 as input data only on systems with Advanced Matrix Extension(AMX) support.
   
   * Projection LSTM for bf16 data type is not supported.
   
   * f16 data type is not supported.



#. GPU
   
   * No support for AUGRU.
   
   * No support for Peephole LSTM and Projection LSTM.
   
   * Int8 support is provided for LSTM only.
   
   * Bias and cell state of bf16 data type is not supported.

Example
~~~~~~~

:ref:`LSTM RNN Primitive Example <doxid-lstm_example_cpp>`

This C++ API example demonstrates how to create and execute an :ref:`LSTM RNN <doxid-dev_guide_rnn>` primitive in forward training propagation mode.

Key optimizations included in this example:

* Creation of optimized memory format from the primitive descriptor.

