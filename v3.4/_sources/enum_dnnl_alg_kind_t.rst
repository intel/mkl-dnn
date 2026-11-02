.. index:: pair: enum; dnnl_alg_kind_t
.. _doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23:

enum dnnl_alg_kind_t
====================

Overview
~~~~~~~~

Kinds of algorithms. :ref:`More...<details-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_alg_kind_t
	{
	    :target:`dnnl_alg_kind_undef<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aefd8cdc9427a7649537731dc8912b458>`,
	    :ref:`dnnl_convolution_direct<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8258635c519746dbf543ac13054acb5a>`               = 0x1,
	    :ref:`dnnl_convolution_winograd<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a4fb6efcd2a2e8766d50e70d37df1d971>`             = 0x2,
	    :ref:`dnnl_convolution_auto<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a62e85aff18d57ac4c3806234dcbafe2b>`                 = 0x3,
	    :ref:`dnnl_deconvolution_direct<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a575e3d69d108a8a1e62af755dda0ef5f>`             = 0xa,
	    :ref:`dnnl_deconvolution_winograd<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9b11a60748225144fdb960988e9b0cb9>`           = 0xb,
	    :ref:`dnnl_eltwise_relu<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>`                     = 0x20,
	    :ref:`dnnl_eltwise_tanh<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a81b20d8f0b54c7114024186a9fbb698e>`,
	    :ref:`dnnl_eltwise_elu<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a7afda2aa9bac4a229909522235f461b5>`,
	    :ref:`dnnl_eltwise_square<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a4da34cea03ccb7cc2701b2f2023bcc2e>`,
	    :ref:`dnnl_eltwise_abs<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a2ac04aed39c46f6d6356744d9d12df43>`,
	    :ref:`dnnl_eltwise_sqrt<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a2152d4664761b356bbceed3d9afe2189>`,
	    :ref:`dnnl_eltwise_linear<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aed5eec69000ddfe6ac96e161b0d723b4>`,
	    :ref:`dnnl_eltwise_soft_relu<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a82d95f7071af086d4b1652160d9a972f>`,
	    :ref:`dnnl_eltwise_hardsigmoid<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a19381a5adcfa6889394eab43c3fc4ee3>`,
	    :ref:`dnnl_eltwise_logistic<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ab560981bee9e7711017423e29ba46071>`,
	    :ref:`dnnl_eltwise_exp<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a4859f1326783a273500ef294bb7c7d5c>`,
	    :ref:`dnnl_eltwise_gelu_tanh<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a18c14d6904040bff94bce8a43c039c62>`,
	    :ref:`dnnl_eltwise_swish<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a63447dedf2e45ab535f1365502ff3240>`,
	    :ref:`dnnl_eltwise_log<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8ea10785816fd41353b49445852e0b74>`,
	    :ref:`dnnl_eltwise_clip<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a026ef822b5cc28653e0730f8c8c2cf32>`,
	    :ref:`dnnl_eltwise_clip_v2<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a911e6995e534a9f8e6af121bc2aba2d6>`,
	    :ref:`dnnl_eltwise_pow<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aa1d0f7a69b7dfbfbd817623552558054>`,
	    :ref:`dnnl_eltwise_gelu_erf<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a676e7d4e899ab2bbddc72f73a54c7779>`,
	    :ref:`dnnl_eltwise_round<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23adda28cb0389d39c0c43967352b116d9d>`,
	    :ref:`dnnl_eltwise_mish<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ae3b2cacb38f7aa0a115e631caa5d63d5>`,
	    :ref:`dnnl_eltwise_hardswish<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9ee6277dfff509e9fde3d5329b8eacd9>`,
	    :ref:`dnnl_eltwise_relu_use_dst_for_bwd<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aa2fffcdde8480cd08a0d6e4dee7dec53>`     = 0x100,
	    :ref:`dnnl_eltwise_tanh_use_dst_for_bwd<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a04e559b66a5d43a74a9f1b91da78151c>`,
	    :ref:`dnnl_eltwise_elu_use_dst_for_bwd<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a975aea11dce8571bf1d4b2552c652a27>`,
	    :ref:`dnnl_eltwise_sqrt_use_dst_for_bwd<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a45b82064ee41f69c5463895c41ec24d0>`,
	    :ref:`dnnl_eltwise_logistic_use_dst_for_bwd<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad224a5a4730407c8b97a10fb53d1fe0f>`,
	    :ref:`dnnl_eltwise_exp_use_dst_for_bwd<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ae7f15ca067ce527eb66a35767d253e81>`,
	    :ref:`dnnl_eltwise_clip_v2_use_dst_for_bwd<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a50d7ed64b4ab2a5c4a156291ac7cb98d>`,
	    :ref:`dnnl_pooling_max<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23acf3529ba1c4761c0da90eb6750def6c7>`                      = 0x1ff,
	    :ref:`dnnl_pooling_avg_include_padding<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ac13a4cc7c0dc1edfcbf1bac23391d5cb>`      = 0x2ff,
	    :ref:`dnnl_pooling_avg_exclude_padding<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a00156580493fd7c2f4cdbaaf9fcbde79>`      = 0x3ff,
	    :ref:`dnnl_lrn_across_channels<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a540b116253bf1290b9536929198d18fd>`              = 0xaff,
	    :ref:`dnnl_lrn_within_channel<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a922fdd348b6a3e6bbe589025691d7171>`               = 0xbff,
	    :ref:`dnnl_vanilla_rnn<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a65d20a62fd39cfe09b3deb2e35752449>`                      = 0x1fff,
	    :ref:`dnnl_vanilla_lstm<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a765cecdcf6f7c524833a241ecc9bf41d>`                     = 0x2fff,
	    :ref:`dnnl_vanilla_gru<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a7ba4a460b8bff80dcdf1240d7ad34208>`                      = 0x3fff,
	    :ref:`dnnl_lbr_gru<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a7cd2e2970fefcdeb255415d0363279e2>`                          = 0x4fff,
	    :ref:`dnnl_vanilla_augru<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aead66a932914267245d7539fb0aec943>`                    = 0x5fff,
	    :ref:`dnnl_lbr_augru<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aa592359a2800e4da61fef4133d1048b6>`                        = 0x6fff,
	    :ref:`dnnl_binary_add<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad4c6d69ac6f6b443449923d51325886d>`                       = 0x1fff0,
	    :ref:`dnnl_binary_mul<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ade272a5bcb8af2b2cb0bc691c78b4e36>`                       = 0x1fff1,
	    :ref:`dnnl_binary_max<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23af93b25a1cd108fbecfdbee9f1cfcdd88>`                       = 0x1fff2,
	    :ref:`dnnl_binary_min<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a21a9b503c9d06cea5f231fd170e623cc>`                       = 0x1fff3,
	    :ref:`dnnl_binary_div<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad63a6855c4f438cabd245b0bbff61cf4>`                       = 0x1fff4,
	    :ref:`dnnl_binary_sub<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a551dc23f954000fe81a97c9bd8ca4899>`                       = 0x1fff5,
	    :ref:`dnnl_binary_ge<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8303a5bb9566ad2cd1323653a81dc494>`                        = 0x1fff6,
	    :ref:`dnnl_binary_gt<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aae40b748b416aa218f420be2f6afbce4>`                        = 0x1fff7,
	    :ref:`dnnl_binary_le<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23acd36606bc4250410a573a15b2a984457>`                        = 0x1fff8,
	    :ref:`dnnl_binary_lt<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23abd093dc24480cf7a3e7a11c4d77dcafe>`                        = 0x1fff9,
	    :ref:`dnnl_binary_eq<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5b81e36f1c758682df8070d344d6f9b8>`                        = 0x1fffa,
	    :ref:`dnnl_binary_ne<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a3f48bade6a3e91fc7880fe823bd4d263>`                        = 0x1fffb,
	    :ref:`dnnl_resampling_nearest<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23af3f4351e11d0792cdfddff5e12e806be>`               = 0x2fff0,
	    :ref:`dnnl_resampling_linear<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a1db5bfa7000fa71a7b8bce1c3497ae1b>`                = 0x2fff1,
	    :ref:`dnnl_reduction_max<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aae4722e394206cf9774ae45db959854e>`,
	    :ref:`dnnl_reduction_min<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a3edeac87290d164cfd3e79adcb6ed91a>`,
	    :ref:`dnnl_reduction_sum<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ae74491a0b7bfe0720be69e3732894818>`,
	    :ref:`dnnl_reduction_mul<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9ff432e67749e211f5f0f64d5f707359>`,
	    :ref:`dnnl_reduction_mean<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ac88d2b9bc130483c177868888c705694>`,
	    :ref:`dnnl_reduction_norm_lp_max<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad6459b4162ab312f59fa48bf9dcf35c3>`,
	    :ref:`dnnl_reduction_norm_lp_sum<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a21c93597a1be438219bbbd832830f096>`,
	    :ref:`dnnl_reduction_norm_lp_power_p_max<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a3838df4d5d37de3237359043ccebfba1>`,
	    :ref:`dnnl_reduction_norm_lp_power_p_sum<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23adcb83e9f76b3beaeb831a59cd257d7dd>`,
	    :ref:`dnnl_softmax_accurate<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a0df1f8d88eb88b4d6c955e8473f54ade>`                 = 0x30000,
	    :ref:`dnnl_softmax_log<doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a92892f2d392fe424f3387b07dde9c680>`,
	};

.. _details-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Kinds of algorithms.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_convolution_direct
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8258635c519746dbf543ac13054acb5a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_convolution_direct

Direct convolution.

.. index:: pair: enumvalue; dnnl_convolution_winograd
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a4fb6efcd2a2e8766d50e70d37df1d971:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_convolution_winograd

Winograd convolution.

.. index:: pair: enumvalue; dnnl_convolution_auto
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a62e85aff18d57ac4c3806234dcbafe2b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_convolution_auto

Convolution algorithm(either direct or Winograd) is chosen just in time.

.. index:: pair: enumvalue; dnnl_deconvolution_direct
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a575e3d69d108a8a1e62af755dda0ef5f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_deconvolution_direct

Direct deconvolution.

.. index:: pair: enumvalue; dnnl_deconvolution_winograd
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9b11a60748225144fdb960988e9b0cb9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_deconvolution_winograd

Winograd deconvolution.

.. index:: pair: enumvalue; dnnl_eltwise_relu
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_relu

Eltwise: ReLU.

.. index:: pair: enumvalue; dnnl_eltwise_tanh
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a81b20d8f0b54c7114024186a9fbb698e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_tanh

Eltwise: hyperbolic tangent non-linearity (tanh)

.. index:: pair: enumvalue; dnnl_eltwise_elu
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a7afda2aa9bac4a229909522235f461b5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_elu

Eltwise: exponential linear unit (elu)

.. index:: pair: enumvalue; dnnl_eltwise_square
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a4da34cea03ccb7cc2701b2f2023bcc2e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_square

Eltwise: square.

.. index:: pair: enumvalue; dnnl_eltwise_abs
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a2ac04aed39c46f6d6356744d9d12df43:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_abs

Eltwise: abs.

.. index:: pair: enumvalue; dnnl_eltwise_sqrt
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a2152d4664761b356bbceed3d9afe2189:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_sqrt

Eltwise: square root.

.. index:: pair: enumvalue; dnnl_eltwise_linear
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aed5eec69000ddfe6ac96e161b0d723b4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_linear

Eltwise: linear.

.. index:: pair: enumvalue; dnnl_eltwise_soft_relu
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a82d95f7071af086d4b1652160d9a972f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_soft_relu

Eltwise: soft_relu.

.. index:: pair: enumvalue; dnnl_eltwise_hardsigmoid
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a19381a5adcfa6889394eab43c3fc4ee3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_hardsigmoid

Eltwise: hardsigmoid.

.. index:: pair: enumvalue; dnnl_eltwise_logistic
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ab560981bee9e7711017423e29ba46071:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_logistic

Eltwise: logistic.

.. index:: pair: enumvalue; dnnl_eltwise_exp
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a4859f1326783a273500ef294bb7c7d5c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_exp

Eltwise: exponent.

.. index:: pair: enumvalue; dnnl_eltwise_gelu_tanh
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a18c14d6904040bff94bce8a43c039c62:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_gelu_tanh

Eltwise: gelu.

.. note:: 

   Tanh approximation formula is used to approximate the cumulative distribution function of a Gaussian here

.. index:: pair: enumvalue; dnnl_eltwise_swish
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a63447dedf2e45ab535f1365502ff3240:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_swish

Eltwise: swish.

.. index:: pair: enumvalue; dnnl_eltwise_log
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8ea10785816fd41353b49445852e0b74:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_log

Eltwise: natural logarithm.

.. index:: pair: enumvalue; dnnl_eltwise_clip
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a026ef822b5cc28653e0730f8c8c2cf32:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_clip

Eltwise: clip.

.. index:: pair: enumvalue; dnnl_eltwise_clip_v2
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a911e6995e534a9f8e6af121bc2aba2d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_clip_v2

Eltwise: clip version 2.

.. index:: pair: enumvalue; dnnl_eltwise_pow
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aa1d0f7a69b7dfbfbd817623552558054:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_pow

Eltwise: pow.

.. index:: pair: enumvalue; dnnl_eltwise_gelu_erf
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a676e7d4e899ab2bbddc72f73a54c7779:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_gelu_erf

Eltwise: erf-based gelu.

.. index:: pair: enumvalue; dnnl_eltwise_round
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23adda28cb0389d39c0c43967352b116d9d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_round

Eltwise: round.

.. index:: pair: enumvalue; dnnl_eltwise_mish
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ae3b2cacb38f7aa0a115e631caa5d63d5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_mish

Eltwise: mish.

.. index:: pair: enumvalue; dnnl_eltwise_hardswish
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9ee6277dfff509e9fde3d5329b8eacd9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_hardswish

Eltwise: hardswish.

.. index:: pair: enumvalue; dnnl_eltwise_relu_use_dst_for_bwd
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aa2fffcdde8480cd08a0d6e4dee7dec53:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_relu_use_dst_for_bwd

Eltwise: ReLU (dst for backward)

.. index:: pair: enumvalue; dnnl_eltwise_tanh_use_dst_for_bwd
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a04e559b66a5d43a74a9f1b91da78151c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_tanh_use_dst_for_bwd

Eltwise: hyperbolic tangent non-linearity (tanh) (dst for backward)

.. index:: pair: enumvalue; dnnl_eltwise_elu_use_dst_for_bwd
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a975aea11dce8571bf1d4b2552c652a27:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_elu_use_dst_for_bwd

Eltwise: exponential linear unit (elu) (dst for backward)

.. index:: pair: enumvalue; dnnl_eltwise_sqrt_use_dst_for_bwd
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a45b82064ee41f69c5463895c41ec24d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_sqrt_use_dst_for_bwd

Eltwise: square root (dst for backward)

.. index:: pair: enumvalue; dnnl_eltwise_logistic_use_dst_for_bwd
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad224a5a4730407c8b97a10fb53d1fe0f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_logistic_use_dst_for_bwd

Eltwise: logistic (dst for backward)

.. index:: pair: enumvalue; dnnl_eltwise_exp_use_dst_for_bwd
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ae7f15ca067ce527eb66a35767d253e81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_exp_use_dst_for_bwd

Eltwise: exp (dst for backward)

.. index:: pair: enumvalue; dnnl_eltwise_clip_v2_use_dst_for_bwd
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a50d7ed64b4ab2a5c4a156291ac7cb98d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise_clip_v2_use_dst_for_bwd

Eltwise: clip version 2 (dst for backward)

.. index:: pair: enumvalue; dnnl_pooling_max
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23acf3529ba1c4761c0da90eb6750def6c7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_pooling_max

Max pooling.

.. index:: pair: enumvalue; dnnl_pooling_avg_include_padding
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ac13a4cc7c0dc1edfcbf1bac23391d5cb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_pooling_avg_include_padding

Average pooling include padding.

.. index:: pair: enumvalue; dnnl_pooling_avg_exclude_padding
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a00156580493fd7c2f4cdbaaf9fcbde79:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_pooling_avg_exclude_padding

Average pooling exclude padding.

.. index:: pair: enumvalue; dnnl_lrn_across_channels
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a540b116253bf1290b9536929198d18fd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_lrn_across_channels

Local response normalization (LRN) across multiple channels.

.. index:: pair: enumvalue; dnnl_lrn_within_channel
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a922fdd348b6a3e6bbe589025691d7171:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_lrn_within_channel

LRN within a single channel.

.. index:: pair: enumvalue; dnnl_vanilla_rnn
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a65d20a62fd39cfe09b3deb2e35752449:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_vanilla_rnn

RNN cell.

.. index:: pair: enumvalue; dnnl_vanilla_lstm
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a765cecdcf6f7c524833a241ecc9bf41d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_vanilla_lstm

LSTM cell.

.. index:: pair: enumvalue; dnnl_vanilla_gru
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a7ba4a460b8bff80dcdf1240d7ad34208:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_vanilla_gru

GRU cell.

.. index:: pair: enumvalue; dnnl_lbr_gru
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a7cd2e2970fefcdeb255415d0363279e2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_lbr_gru

GRU cell with linear before reset.

Modification of original GRU cell. Differs from :ref:`dnnl_vanilla_gru <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a7ba4a460b8bff80dcdf1240d7ad34208>` in how the new memory gate is calculated:

.. math::

	c_t = tanh(W_c*x_t + b_{c_x} + r_t*(U_c*h_{t-1}+b_{c_h}))

Primitive expects 4 biases on input: :math:`[b_{u}, b_{r}, b_{c_x}, b_{c_h}]`

.. index:: pair: enumvalue; dnnl_vanilla_augru
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aead66a932914267245d7539fb0aec943:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_vanilla_augru

AUGRU cell.

.. index:: pair: enumvalue; dnnl_lbr_augru
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aa592359a2800e4da61fef4133d1048b6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_lbr_augru

AUGRU cell with linear before reset.

.. index:: pair: enumvalue; dnnl_binary_add
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad4c6d69ac6f6b443449923d51325886d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_add

Binary add.

.. index:: pair: enumvalue; dnnl_binary_mul
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ade272a5bcb8af2b2cb0bc691c78b4e36:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_mul

Binary mul.

.. index:: pair: enumvalue; dnnl_binary_max
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23af93b25a1cd108fbecfdbee9f1cfcdd88:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_max

Binary max.

.. index:: pair: enumvalue; dnnl_binary_min
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a21a9b503c9d06cea5f231fd170e623cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_min

Binary min.

.. index:: pair: enumvalue; dnnl_binary_div
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad63a6855c4f438cabd245b0bbff61cf4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_div

Binary div.

.. index:: pair: enumvalue; dnnl_binary_sub
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a551dc23f954000fe81a97c9bd8ca4899:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_sub

Binary sub.

.. index:: pair: enumvalue; dnnl_binary_ge
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8303a5bb9566ad2cd1323653a81dc494:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_ge

Binary greater or equal.

.. index:: pair: enumvalue; dnnl_binary_gt
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aae40b748b416aa218f420be2f6afbce4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_gt

Binary greater than.

.. index:: pair: enumvalue; dnnl_binary_le
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23acd36606bc4250410a573a15b2a984457:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_le

Binary less or equal.

.. index:: pair: enumvalue; dnnl_binary_lt
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23abd093dc24480cf7a3e7a11c4d77dcafe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_lt

Binary less than.

.. index:: pair: enumvalue; dnnl_binary_eq
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5b81e36f1c758682df8070d344d6f9b8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_eq

Binary equal.

.. index:: pair: enumvalue; dnnl_binary_ne
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a3f48bade6a3e91fc7880fe823bd4d263:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary_ne

Binary not equal.

.. index:: pair: enumvalue; dnnl_resampling_nearest
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23af3f4351e11d0792cdfddff5e12e806be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_resampling_nearest

Nearest Neighbor Resampling Method.

.. index:: pair: enumvalue; dnnl_resampling_linear
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a1db5bfa7000fa71a7b8bce1c3497ae1b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_resampling_linear

Linear Resampling Method.

.. index:: pair: enumvalue; dnnl_reduction_max
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aae4722e394206cf9774ae45db959854e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reduction_max

Reduction using max.

.. index:: pair: enumvalue; dnnl_reduction_min
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a3edeac87290d164cfd3e79adcb6ed91a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reduction_min

Reduction using min.

.. index:: pair: enumvalue; dnnl_reduction_sum
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ae74491a0b7bfe0720be69e3732894818:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reduction_sum

Reduction using sum.

.. index:: pair: enumvalue; dnnl_reduction_mul
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9ff432e67749e211f5f0f64d5f707359:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reduction_mul

Reduction using mul.

.. index:: pair: enumvalue; dnnl_reduction_mean
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ac88d2b9bc130483c177868888c705694:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reduction_mean

Reduction using mean.

.. index:: pair: enumvalue; dnnl_reduction_norm_lp_max
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad6459b4162ab312f59fa48bf9dcf35c3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reduction_norm_lp_max

Reduction using lp norm.

.. index:: pair: enumvalue; dnnl_reduction_norm_lp_sum
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a21c93597a1be438219bbbd832830f096:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reduction_norm_lp_sum

Reduction using lp norm.

.. index:: pair: enumvalue; dnnl_reduction_norm_lp_power_p_max
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a3838df4d5d37de3237359043ccebfba1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reduction_norm_lp_power_p_max

Reduction using lp norm without final pth-root.

.. index:: pair: enumvalue; dnnl_reduction_norm_lp_power_p_sum
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23adcb83e9f76b3beaeb831a59cd257d7dd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reduction_norm_lp_power_p_sum

Reduction using lp norm without final pth-root.

.. index:: pair: enumvalue; dnnl_softmax_accurate
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a0df1f8d88eb88b4d6c955e8473f54ade:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_softmax_accurate

Softmax.

.. index:: pair: enumvalue; dnnl_softmax_log
.. _doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a92892f2d392fe424f3387b07dde9c680:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_softmax_log

Logsoftmax.

