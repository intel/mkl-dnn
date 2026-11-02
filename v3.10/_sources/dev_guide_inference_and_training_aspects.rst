.. index:: pair: page; Inference and Training Aspects
.. _doxid-dev_guide_inference_and_training_aspects:

Inference and Training Aspects
==============================

:target:`doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_prop_kinds`

Propagation Kinds
~~~~~~~~~~~~~~~~~

oneDNN provides performance critical primitives to accelerate operations used both during training deep learning models and during the operations performed when the models are used for inference.

During inference, the input data is fed into the trained model which in turn produces a result (e.g. makes a prediction). This process is usually called forward propagation and corresponds to the :ref:`dnnl::prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>` propagation kind in oneDNN.

Training usually consists of the following steps.

#. Make prediction based on the current state of the model. As in the case of inference, this step is called forward propagation, but corresponds to the :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>` propagation kind. Note the difference in the names' suffixes: ``_training`` here versus ``_inference`` mentioned above. The differences are covered below in the corresponding :ref:`section <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_forward_training_vs_inference>` below.

#. Compute an error between predicted and the actual answer.

#. Perform the backward propagation of errors to compute the weights (learnable parameters) gradient. For a given operation (layer) the backward propagation in turn can be split into two steps:
   
   * Propagating error with respect to data, i.e. computing ``diff_src`` from ``diff_dst`` (see :ref:`Naming Conventions <doxid-dev_guide_conventions>`). This step corresponds to the :ref:`dnnl::prop_kind::backward_data <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faa12627cacf73ecb7ef088beedd650e96>` propagation kind;
   
   * Propagating error with respect to weights, i.e. computing ``diff_weights`` from ``diff_dst``. This step makes sense only for the operations that have learnable parameters and corresponds to the :ref:`dnnl::prop_kind::backward_weights <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa1a002980f340e61153a9f7de4f794cf6>` propagation kind.

#. Use computed gradients to modify the weights according to the chosen solver to improve the accuracy of the model.

Difference Between Forward Propagation on Training and Inference
----------------------------------------------------------------

:target:`doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_forward_training_vs_inference` Even though, mathematically, the forward propagation that happens during training and inference should be the same, in practice there are some differences mostly due to the performance considerations.

When executing inference, one may not care about values in the intermediate buffers during a model execution; hence one can reuse them as desired. However, if this is a forward propagation of a training it is beneficial to preserve input data, output data, or sometimes some intermediate data, that will later be used at the backward propagation to compute the gradients.

For example, let's take max pooling (:ref:`Pooling <doxid-dev_guide_pooling>` with algorithm kind :ref:`dnnl::algorithm::pooling_max <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8c73d4bb88a0497586a74256bb338e88>`) as an example. The forward pass consists of computing the maximum values in the sliding window over the source tensor. Hence the output is just another tensor that contain these maximum values. However, in order to compute source gradient on backward propagation one needs to know the position of these maximum values in the source tensor. Of course, it is possible to use the original source tensor to locate the maximums again, but this might be more expensive compared to preserving the positions of the maximum values in another tensor, that will be then used during the backward propagation. oneDNN uses the latter approach: for max pooling primitive when the propagation kind is set to :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>` the library produces one extra output called :ref:`Workspace <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_workspace>` which will be covered later in this document.

.. note:: 

   Key takeaways:
   
   * Always use :ref:`dnnl::prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>` when running inference; use :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>` for forward pass of training.
   
   * The number of a primitive's outputs might be greater by 1 if it was created with :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>` because of the extra :ref:`Workspace <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_workspace>` memory.
   
   


Different Backward Propagation Kinds
------------------------------------

:target:`doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_difference_backward_prop_kinds` As mentioned above, oneDNN separates error back-propagation with respect to data and error back-propagation with respect to weights. The former corresponds to :ref:`dnnl::prop_kind::backward_data <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faa12627cacf73ecb7ef088beedd650e96>`, while the latter corresponds to :ref:`dnnl::prop_kind::backward_weights <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa1a002980f340e61153a9f7de4f794cf6>` (for example: :ref:`Convolution <doxid-dev_guide_convolution>`).

Inference-Specific Aspects
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following list outlines the key specifics of running inference with oneDNN:

#. As described above, always use :ref:`dnnl::prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>` as a propagation kind.

#. To get maximum performance, consider performing operations in-place whenever possible (e.g. :ref:`Eltwise <doxid-dev_guide_eltwise>` and :ref:`Batch Normalization <doxid-dev_guide_batch_normalization>`). Check the primitives documentation pages to check which primitives support in-place operations.

#. Create primitives once, and reuse them across multiple model invocations. This is especially relevant for the frameworks integration.

#. Some primitives can be chained/fused with others using the :ref:`post-ops attributes <doxid-dev_guide_attributes_post_ops>`. This allows reducing memory bandwidth pressure and typically leads to better performance.

Most of these techniques are shown in the following examples:

* :ref:`CNN f32 inference example <doxid-cnn_inference_f32_cpp>`

* :ref:`CNN int8 inference example <doxid-cnn_inference_int8_cpp>`

:target:`doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_training`

Training-Specific Aspects
~~~~~~~~~~~~~~~~~~~~~~~~~

The following list outlines the key specifics of running training with oneDNN:

#. During the forward propagation, use :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>` as a propagation kind.

#. During backward propagation, perform backward by data and backward by weights using the corresponding propagation kinds.
   
   * Note that some primitives may combine these operations if that is beneficial from a performance perspective. For example, :ref:`RNN <doxid-dev_guide_rnn>` and :ref:`Batch Normalization <doxid-dev_guide_batch_normalization>` compute both ``diff_src`` and ``diff_weights`` at the same time. To highlight this behavior, the propagation kind is set to :ref:`dnnl::prop_kind::backward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa195fe59b6f103787a914aead0f3db502>`.

#. Create primitives once, and reuse them across multiple model invocations. This is especially relevant for the frameworks integration.

#. The :ref:`post-ops attributes <doxid-dev_guide_attributes_post_ops>` in general are not applicable for training, because the fused computations they result in do not produce the intermediate tensors which may be required during the backward propagation. For example, if you fuse :ref:`Convolution <doxid-dev_guide_convolution>` and :ref:`Eltwise <doxid-dev_guide_eltwise>` the direct output of the convolution would not be produced. However, it might be required during the backward propagation of the corresponding element-wise. (To compute ``diff_src``, one must pass ``diff_dst`` memory and the original ``src`` memory, which was exactly the intermediate one.)

#. To compute backward propagation, different primitives might require different tensors. The variety is caused by the mathematical formulas. For example, to compute backward propagation for :ref:`Eltwise <doxid-dev_guide_eltwise>` one needs to pass ``diff_dst`` and ``src``, but to compute backward propagation for :ref:`Softmax <doxid-dev_guide_softmax>`, one needs to pass ``diff_dst`` and ``dst``. Check the documentation for each primitive to see what is required for each particular primitive.

#. For the primitives that are created with :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` memory format tag, there are no guarantees that the memory format on forward and backward propagations will match. So the robust integration should always be ready to emit :ref:`Reorder <doxid-dev_guide_reorder>` when necessary. For example, it is not guaranteed that the ``src`` memory format of a convolution on forward propagation will always match the ``src`` memory format of the corresponding convolution on backward by weights propagation. Of course, the library tries to avoid unnecessary reorder, so in most cases the formats will be the same, but this would by no means always be true.

#. For the memory bandwidth bound primitives like :ref:`Eltwise <doxid-dev_guide_eltwise>`, :ref:`Pooling <doxid-dev_guide_pooling>`, and :ref:`Batch Normalization <doxid-dev_guide_batch_normalization>`, it is important to have ``diff_dst`` in the same memory format as the original ``dst``. The mismatch of the formats would lead to significant performance issues. To ensure the proper format, users should always use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` memory format for gradient tensors (``diff_dst``, ``diff_src``). If a primitive requires original data tensors (e.g. ``src`` in :ref:`Eltwise <doxid-dev_guide_eltwise>` or ``dst`` in :ref:`Softmax <doxid-dev_guide_softmax>`) user must pass fully defined memory descriptor for these tensors. In other words ``src`` and ``dst`` memory descriptors cannot be initialized with :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` for backward propagation. Based on the format of the original tensors, if any, and on forward primitive descriptor hint (see bullet 9 below) a primitive picks the proper format for the gradients. Occasionally, it might appear that the ``diff_dst`` that comes in is in other memory format than the primitive requires, hence robust integration code must be prepared to emit a reorder.
   
   * Alternatively, users may manually enforce ``diff_dst`` to have the same memory format as ``dst``, though this is not recommended.

#. Some primitives require an additional tensor to be passed between forward and backward propagation, which is called :ref:`Workspace <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_workspace>`.

#. When creating primitive descriptors on backward propagation, you might need to pass a primitive descriptor of the corresponding primitive from the forward propagation (in the API this primitive descriptor is typically called hint). This hint is required for the primitive to choose a proper implementation that would correspond to the one from the forward propagation. This is required only for the primitives that produce ``workspace``, because it might be different for different implementations.

#. When creating your working memory and memory descriptor, specify the type of memory you want to work with. This can be either 16-bit Brain Float (bf16) or 32-bit Floating Point (fp32). More details about using bf16 for training are detailed in the section :ref:`Bfloat16 Training <doxid-dev_guide_training_bf16>`.

Most of these techniques are shown in the following examples:

* :ref:`CNN f32 training example <doxid-cnn_training_f32_cpp>`

* :ref:`CNN bf16 training example <doxid-cnn_training_bf16_cpp>`

:target:`doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_workspace`

Workspace
~~~~~~~~~

oneDNN uses the notion of ``workspace`` for some very particular cases. Specifically, the ``workspace`` is a tensor that the primitive fills in during forward propagation and that will then be used by the corresponding backward propagation operation. The example with max pooling was already discussed above.

The workflow for using workspace is:

#. When creating a primitive for the forward propagation, query the primitive descriptor about the workspace requirement using ``.workspace_desc()``.
   
   * If the returned memory descriptor is essentially empty (i.e. is equal to ``:ref:`dnnl::memory::desc() <doxid-structdnnl_1_1memory_1_1desc>``` or for which :ref:`dnnl::memory::desc::get_size() <doxid-structdnnl_1_1memory_1_1desc_1abfa095ac138d4d2ef8efd3739e343f08>` returns 0), no extra action is required the workspace is not required for this primitive in this configuration.
   
   * Otherwise, create a workspace memory based on the memory descriptor obtained and pass it to the execution function with ``DNNL_ARG_WORKSPACE`` tag.

#. On backward propagation, attach that same workspace memory during the execution as well. The state of the workspace memory after backward computations are done is undefined.

.. note:: 

   Even if workspace is not required, it is perfectly valid to create a ``workspace`` memory of zero size and follow the logic where the workspace is indeed required. Such an approach may simplify the integration because the common pass is used.
   
   


.. ref-code-block:: cpp

	// FWD
	
	auto forward_primitive_desc = ...::primitive_desc(); // create a primitive desc
	auto :ref:`workspace_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3874c56bb4069607213666573dff2a96>` = forward_primitive_desc.workspace_desc(); // query workspace
	memory workspace(workspace_md, engine); // create a memory (even if empty)
	
	primitive_forward.execute(stream, {
	        ...,
	        {:ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, workspace} // this is output
	        });
	// The workspace contains required information for the backward propagation,
	// hence should not be used anywhere else.
	
	// ...
	
	// BWD
	primitive_backward.execute(stream, {
	        ...,
	        {:ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, workspace} // this input/output
	        });
	// The state of the workspace is undefined here

.. warning:: 

   Do not confuse workspace with the :ref:`Primitive Attributes: Scratchpad <doxid-dev_guide_attributes_scratchpad>`. The scratchpad is a temporary buffer that might be required by a primitive (no matter what propagation kind is) to perform an operation. It is used only during the primitive execution and should not be preserved across the calls.


.. toctree::
   :hidden:

   dev_guide_inference.rst
   dev_guide_inference_int8.rst
   dev_guide_training_bf16.rst


