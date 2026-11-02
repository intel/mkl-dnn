.. index:: pair: enum; algorithm
.. _doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640:

enum dnnl::algorithm
====================

Overview
~~~~~~~~

Kinds of algorithms. :ref:`More...<details-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum algorithm
	{
	    :ref:`undef<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af31ee5e3824f1f5e5d206bdf3029f22b>`                            = dnnl_alg_kind_undef,
	    :ref:`convolution_auto<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640acfdececd63a8bc0cfe1021ad614e2ded>`                 = dnnl_convolution_auto,
	    :ref:`convolution_direct<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`               = dnnl_convolution_direct,
	    :ref:`convolution_winograd<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a6a73ed41c67672f76f53a69e96c31ed4>`             = dnnl_convolution_winograd,
	    :ref:`deconvolution_direct<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a73f81608d2b7315f04c438fb8be5f99c>`             = dnnl_deconvolution_direct,
	    :ref:`deconvolution_winograd<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aeb5da82914413424bf941fa04eb065aa>`           = dnnl_deconvolution_winograd,
	    :ref:`eltwise_relu<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`                     = dnnl_eltwise_relu,
	    :ref:`eltwise_tanh<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a38dd7159307eab45742c78e72f06abb0>`                     = dnnl_eltwise_tanh,
	    :ref:`eltwise_elu<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a690376cad4a7f001e967a6a40d04da3a>`                      = dnnl_eltwise_elu,
	    :ref:`eltwise_square<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a448fc250a28e651c02966142f7f66cce>`                   = dnnl_eltwise_square,
	    :ref:`eltwise_abs<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a30046c43b9399a7aaa4281da7cff2d5c>`                      = dnnl_eltwise_abs,
	    :ref:`eltwise_sqrt<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a83b978b280304349860c7cdc3519865f>`                     = dnnl_eltwise_sqrt,
	    :ref:`eltwise_swish<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a97395201ce1744f18c19d243d24fbd13>`                    = dnnl_eltwise_swish,
	    :ref:`eltwise_linear<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a21aba6844d2de47b92ab1d110f561945>`                   = dnnl_eltwise_linear,
	    :ref:`eltwise_soft_relu<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a23175afeee905a4de38e5a24b46cb6e2>`                = dnnl_eltwise_soft_relu,
	    :ref:`eltwise_mish<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a018c284e3880c32050f62aee715c9dde>`                     = dnnl_eltwise_mish,
	    :ref:`eltwise_logistic<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a282836cce833348ee509b0ecd764d217>`                 = dnnl_eltwise_logistic,
	    :ref:`eltwise_exp<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a6506e1f0284a4bb8cb999b49c6142599>`                      = dnnl_eltwise_exp,
	    :ref:`eltwise_gelu_tanh<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640abab46bda36dbfea612e68c5345351194>`                = dnnl_eltwise_gelu_tanh,
	    :ref:`eltwise_gelu_erf<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8fd73f441d3ef70416c010f6aea56d5b>`                 = dnnl_eltwise_gelu_erf,
	    :ref:`eltwise_log<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af2a460e3ec273e904c6f6b39baa5e477>`                      = dnnl_eltwise_log,
	    :ref:`eltwise_clip<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a2aa1edfbe21c6770d63301058ec753ed>`                     = dnnl_eltwise_clip,
	    :ref:`eltwise_clip_v2<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a22b6d0ccfd823457c36e94d1a0fc4e50>`                  = dnnl_eltwise_clip_v2,
	    :ref:`eltwise_pow<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a30977b2a33c3a2bd6a5d71a352fa4bcc>`                      = dnnl_eltwise_pow,
	    :ref:`eltwise_round<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad3bc4dd78c67cb6040d6885b405b872b>`                    = dnnl_eltwise_round,
	    :ref:`eltwise_hardswish<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aa7519a9b7f19e4e27c8c092833d653c4>`                = dnnl_eltwise_hardswish,
	    :ref:`eltwise_hardsigmoid<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a3a2eed6c303aeb6d04cc3c28262470b1>`              = dnnl_eltwise_hardsigmoid,
	    :ref:`eltwise_relu_use_dst_for_bwd<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ac227b99192be41d4060b700d1e788fd6>`     = dnnl_eltwise_relu_use_dst_for_bwd,
	    :ref:`eltwise_tanh_use_dst_for_bwd<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a0be1df5ff86383197efdfeb01b10e046>`     = dnnl_eltwise_tanh_use_dst_for_bwd,
	    :ref:`eltwise_elu_use_dst_for_bwd<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a09b26fc530b10556b14bc8e9b2d29d35>`      = dnnl_eltwise_elu_use_dst_for_bwd,
	    :ref:`eltwise_sqrt_use_dst_for_bwd<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad5704aaf680703d473f983e3f19b8d4d>`     = dnnl_eltwise_sqrt_use_dst_for_bwd,
	    :ref:`eltwise_logistic_use_dst_for_bwd<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a40f1d6681d0ad7bf3e9313641e03ae14>` = dnnl_eltwise_logistic_use_dst_for_bwd,
	    :ref:`eltwise_exp_use_dst_for_bwd<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad10dade245ba1cbe70859fcbdad27637>`      = dnnl_eltwise_exp_use_dst_for_bwd,
	    :ref:`eltwise_clip_v2_use_dst_for_bwd<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a4253206d681eaea3183e9593547196c1>`  = dnnl_eltwise_clip_v2_use_dst_for_bwd,
	    :ref:`lrn_across_channels<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab9e2d858b551792385a4b5b86672b24b>`              = dnnl_lrn_across_channels,
	    :ref:`lrn_within_channel<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aa25c4ccc5c7c9bd3a977a2089189a494>`               = dnnl_lrn_within_channel,
	    :ref:`pooling_max<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8c73d4bb88a0497586a74256bb338e88>`                      = dnnl_pooling_max,
	    :ref:`pooling_avg_include_padding<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad62617dd43972a6bbff8d263a5c4462c>`      = dnnl_pooling_avg_include_padding,
	    :ref:`pooling_avg_exclude_padding<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a7ed10af2056450d8f6ac7277b48fc6a8>`      = dnnl_pooling_avg_exclude_padding,
	    :ref:`vanilla_rnn<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab2b3bfcaad515fd1bf0245e8ae6bd407>`                      = dnnl_vanilla_rnn,
	    :ref:`vanilla_lstm<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a716a71a7a6ec13785b552dc2412380ce>`                     = dnnl_vanilla_lstm,
	    :ref:`vanilla_gru<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a272fa682d36b810a24ec89202b7a31f7>`                      = dnnl_vanilla_gru,
	    :ref:`lbr_gru<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a792e966b14bd4e16625ee4df357a0723>`                          = dnnl_lbr_gru,
	    :ref:`vanilla_augru<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a452eeae9e5c7d0dcb6ff429d408bdc91>`                    = dnnl_vanilla_augru,
	    :ref:`lbr_augru<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af393fd3a60cc0a9c7b14f40476657882>`                        = dnnl_lbr_augru,
	    :ref:`binary_add<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab2c3faf084cf82b5603946995f637b35>`                       = dnnl_binary_add,
	    :ref:`binary_mul<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a0905fc5c22e79a8eed0988681eb6a0ae>`                       = dnnl_binary_mul,
	    :ref:`binary_max<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a4ae0aa1c0bf5bd1bf54564a2485618be>`                       = dnnl_binary_max,
	    :ref:`binary_min<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad096699e4ea7ada3d71428484e6119d8>`                       = dnnl_binary_min,
	    :ref:`binary_div<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a2835085341c109a886106f1b671aff71>`                       = dnnl_binary_div,
	    :ref:`binary_sub<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a979309f9436f7ebfa278b0ce682dd706>`                       = dnnl_binary_sub,
	    :ref:`binary_ge<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a2f7ccdf82eaf578e9781ae6a1d9faaa9>`                        = dnnl_binary_ge,
	    :ref:`binary_gt<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a67be16f4dcf48afa723ab72d1620a2b1>`                        = dnnl_binary_gt,
	    :ref:`binary_le<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aa64e69aed2ae6b5e5fbe82bc2b83a6d0>`                        = dnnl_binary_le,
	    :ref:`binary_lt<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af6386e60811ef5a0893c87fd9f270d60>`                        = dnnl_binary_lt,
	    :ref:`binary_eq<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a7b3df3d31607267fc2f72dda23784aad>`                        = dnnl_binary_eq,
	    :ref:`binary_ne<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a98a6e50562cc9e63fa81e71b8bbb265f>`                        = dnnl_binary_ne,
	    :ref:`resampling_nearest<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a091b262cfae5e610fcad1490bfb79431>`               = dnnl_resampling_nearest,
	    :ref:`resampling_linear<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a05731680a83c6dccd19a758b72a4ecbd>`                = dnnl_resampling_linear,
	    :ref:`reduction_max<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a42483cece49698bff4fc700cd2de655c>`                    = dnnl_reduction_max,
	    :ref:`reduction_min<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8cc607027ba7c9399ba610f22b88b3eb>`                    = dnnl_reduction_min,
	    :ref:`reduction_sum<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af710a5a162f3ac1c005876b5c96a6087>`                    = dnnl_reduction_sum,
	    :ref:`reduction_mul<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aab991281eb5a55118f233a6d906be210>`                    = dnnl_reduction_mul,
	    :ref:`reduction_mean<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ae2afc4d79ef0f2e8de6c4d7292ea1a8e>`                   = dnnl_reduction_mean,
	    :ref:`reduction_norm_lp_max<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a67f19545c23e777ffbae900ab1b8cde5>`            = dnnl_reduction_norm_lp_max,
	    :ref:`reduction_norm_lp_sum<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad73ba421a6b061afe16e49a6acc0733a>`            = dnnl_reduction_norm_lp_sum,
	    :ref:`reduction_norm_lp_power_p_max<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a423e9857dd013140a323235cd520bad5>`    = dnnl_reduction_norm_lp_power_p_max,
	    :ref:`reduction_norm_lp_power_p_sum<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a161d82b9c140979ea6fc5ce7ad1afcbd>`    = dnnl_reduction_norm_lp_power_p_sum,
	    :ref:`softmax_accurate<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a6aba07e95369523dbbb9cab366480dd4>`                 = dnnl_softmax_accurate,
	    :ref:`softmax_log<doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a0888321c22541f5d7094cf44bc261ec9>`                      = dnnl_softmax_log,
	};

.. _details-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Kinds of algorithms.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined algorithm.

.. index:: pair: enumvalue; convolution_auto
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640acfdececd63a8bc0cfe1021ad614e2ded:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	convolution_auto

Convolution algorithm that is chosen to be either direct or Winograd automatically.

.. index:: pair: enumvalue; convolution_direct
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	convolution_direct

Direct convolution.

.. index:: pair: enumvalue; convolution_winograd
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a6a73ed41c67672f76f53a69e96c31ed4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	convolution_winograd

Winograd convolution.

.. index:: pair: enumvalue; deconvolution_direct
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a73f81608d2b7315f04c438fb8be5f99c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	deconvolution_direct

Direct deconvolution.

.. index:: pair: enumvalue; deconvolution_winograd
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aeb5da82914413424bf941fa04eb065aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	deconvolution_winograd

Winograd deconvolution.

.. index:: pair: enumvalue; eltwise_relu
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_relu

Elementwise: rectified linear unit (ReLU)

.. index:: pair: enumvalue; eltwise_tanh
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a38dd7159307eab45742c78e72f06abb0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_tanh

Elementwise: hyperbolic tangent non-linearity (tanh)

.. index:: pair: enumvalue; eltwise_elu
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a690376cad4a7f001e967a6a40d04da3a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_elu

Elementwise: exponential linear unit (ELU)

.. index:: pair: enumvalue; eltwise_square
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a448fc250a28e651c02966142f7f66cce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_square

Elementwise: square.

.. index:: pair: enumvalue; eltwise_abs
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a30046c43b9399a7aaa4281da7cff2d5c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_abs

Elementwise: abs.

.. index:: pair: enumvalue; eltwise_sqrt
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a83b978b280304349860c7cdc3519865f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_sqrt

Elementwise: square root.

.. index:: pair: enumvalue; eltwise_swish
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a97395201ce1744f18c19d243d24fbd13:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_swish

Elementwise: swish (:math:`x \cdot sigmoid(a \cdot x)`)

.. index:: pair: enumvalue; eltwise_linear
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a21aba6844d2de47b92ab1d110f561945:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_linear

Elementwise: linear.

.. index:: pair: enumvalue; eltwise_soft_relu
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a23175afeee905a4de38e5a24b46cb6e2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_soft_relu

Elementwise: soft_relu.

.. index:: pair: enumvalue; eltwise_mish
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a018c284e3880c32050f62aee715c9dde:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_mish

Elementwise: mish.

.. index:: pair: enumvalue; eltwise_logistic
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a282836cce833348ee509b0ecd764d217:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_logistic

Elementwise: logistic.

.. index:: pair: enumvalue; eltwise_exp
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a6506e1f0284a4bb8cb999b49c6142599:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_exp

Elementwise: exponent.

.. index:: pair: enumvalue; eltwise_gelu_tanh
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640abab46bda36dbfea612e68c5345351194:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_gelu_tanh

Elementwise: tanh-based gelu.

.. index:: pair: enumvalue; eltwise_gelu_erf
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8fd73f441d3ef70416c010f6aea56d5b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_gelu_erf

Elementwise: erf-based gelu.

.. index:: pair: enumvalue; eltwise_log
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af2a460e3ec273e904c6f6b39baa5e477:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_log

Elementwise: natural logarithm.

.. index:: pair: enumvalue; eltwise_clip
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a2aa1edfbe21c6770d63301058ec753ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_clip

Elementwise: clip.

.. index:: pair: enumvalue; eltwise_clip_v2
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a22b6d0ccfd823457c36e94d1a0fc4e50:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_clip_v2

Eltwise: clip version 2.

.. index:: pair: enumvalue; eltwise_pow
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a30977b2a33c3a2bd6a5d71a352fa4bcc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_pow

Elementwise: pow.

.. index:: pair: enumvalue; eltwise_round
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad3bc4dd78c67cb6040d6885b405b872b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_round

Elementwise: round.

.. index:: pair: enumvalue; eltwise_hardswish
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aa7519a9b7f19e4e27c8c092833d653c4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_hardswish

Elementwise: hardswish.

.. index:: pair: enumvalue; eltwise_hardsigmoid
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a3a2eed6c303aeb6d04cc3c28262470b1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_hardsigmoid

Elementwise: hardsigmoid.

.. index:: pair: enumvalue; eltwise_relu_use_dst_for_bwd
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ac227b99192be41d4060b700d1e788fd6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_relu_use_dst_for_bwd

Elementwise: rectified linar unit (ReLU) (dst for backward)

.. index:: pair: enumvalue; eltwise_tanh_use_dst_for_bwd
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a0be1df5ff86383197efdfeb01b10e046:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_tanh_use_dst_for_bwd

Elementwise: hyperbolic tangent non-linearity (tanh) (dst for backward)

.. index:: pair: enumvalue; eltwise_elu_use_dst_for_bwd
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a09b26fc530b10556b14bc8e9b2d29d35:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_elu_use_dst_for_bwd

Elementwise: exponential linear unit (ELU) (dst for backward)

.. index:: pair: enumvalue; eltwise_sqrt_use_dst_for_bwd
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad5704aaf680703d473f983e3f19b8d4d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_sqrt_use_dst_for_bwd

Elementwise: square root (dst for backward)

.. index:: pair: enumvalue; eltwise_logistic_use_dst_for_bwd
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a40f1d6681d0ad7bf3e9313641e03ae14:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_logistic_use_dst_for_bwd

Elementwise: logistic (dst for backward)

.. index:: pair: enumvalue; eltwise_exp_use_dst_for_bwd
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad10dade245ba1cbe70859fcbdad27637:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_exp_use_dst_for_bwd

Elementwise: exponent (dst for backward)

.. index:: pair: enumvalue; eltwise_clip_v2_use_dst_for_bwd
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a4253206d681eaea3183e9593547196c1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise_clip_v2_use_dst_for_bwd

Elementwise: clip version 2 (dst for backward)

.. index:: pair: enumvalue; lrn_across_channels
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab9e2d858b551792385a4b5b86672b24b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	lrn_across_channels

Local response normalization (LRN) across multiple channels.

.. index:: pair: enumvalue; lrn_within_channel
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aa25c4ccc5c7c9bd3a977a2089189a494:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	lrn_within_channel

LRN within a single channel.

.. index:: pair: enumvalue; pooling_max
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8c73d4bb88a0497586a74256bb338e88:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	pooling_max

Max pooling.

.. index:: pair: enumvalue; pooling_avg_include_padding
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad62617dd43972a6bbff8d263a5c4462c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	pooling_avg_include_padding

Average pooling include padding.

.. index:: pair: enumvalue; pooling_avg_exclude_padding
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a7ed10af2056450d8f6ac7277b48fc6a8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	pooling_avg_exclude_padding

Average pooling exclude padding.

.. index:: pair: enumvalue; vanilla_rnn
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab2b3bfcaad515fd1bf0245e8ae6bd407:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	vanilla_rnn

RNN cell.

.. index:: pair: enumvalue; vanilla_lstm
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a716a71a7a6ec13785b552dc2412380ce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	vanilla_lstm

LSTM cell.

.. index:: pair: enumvalue; vanilla_gru
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a272fa682d36b810a24ec89202b7a31f7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	vanilla_gru

GRU cell.

.. index:: pair: enumvalue; lbr_gru
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a792e966b14bd4e16625ee4df357a0723:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	lbr_gru

GRU cell with linear before reset.

Differs from the vanilla GRU in how the new memory gate is calculated: :math:`c_t = tanh(W_c*x_t + b_{c_x} + r_t*(U_c*h_{t-1}+b_{c_h}))` LRB GRU expects 4 bias tensors on input: :math:`[b_{u}, b_{r}, b_{c_x}, b_{c_h}]`

.. index:: pair: enumvalue; vanilla_augru
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a452eeae9e5c7d0dcb6ff429d408bdc91:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	vanilla_augru

AUGRU cell.

.. index:: pair: enumvalue; lbr_augru
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af393fd3a60cc0a9c7b14f40476657882:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	lbr_augru

AUGRU cell with linear before reset.

.. index:: pair: enumvalue; binary_add
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab2c3faf084cf82b5603946995f637b35:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_add

Binary add.

.. index:: pair: enumvalue; binary_mul
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a0905fc5c22e79a8eed0988681eb6a0ae:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_mul

Binary mul.

.. index:: pair: enumvalue; binary_max
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a4ae0aa1c0bf5bd1bf54564a2485618be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_max

Binary max.

.. index:: pair: enumvalue; binary_min
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad096699e4ea7ada3d71428484e6119d8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_min

Binary min.

.. index:: pair: enumvalue; binary_div
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a2835085341c109a886106f1b671aff71:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_div

Binary div.

.. index:: pair: enumvalue; binary_sub
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a979309f9436f7ebfa278b0ce682dd706:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_sub

Binary sub.

.. index:: pair: enumvalue; binary_ge
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a2f7ccdf82eaf578e9781ae6a1d9faaa9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_ge

Binary greater than or equal.

.. index:: pair: enumvalue; binary_gt
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a67be16f4dcf48afa723ab72d1620a2b1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_gt

Binary greater than.

.. index:: pair: enumvalue; binary_le
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aa64e69aed2ae6b5e5fbe82bc2b83a6d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_le

Binary less than or equal.

.. index:: pair: enumvalue; binary_lt
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af6386e60811ef5a0893c87fd9f270d60:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_lt

Binary less than.

.. index:: pair: enumvalue; binary_eq
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a7b3df3d31607267fc2f72dda23784aad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_eq

Binary equal.

.. index:: pair: enumvalue; binary_ne
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a98a6e50562cc9e63fa81e71b8bbb265f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary_ne

Binary not equal.

.. index:: pair: enumvalue; resampling_nearest
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a091b262cfae5e610fcad1490bfb79431:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	resampling_nearest

Nearest Neighbor resampling method.

.. index:: pair: enumvalue; resampling_linear
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a05731680a83c6dccd19a758b72a4ecbd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	resampling_linear

Linear (Bilinear, Trilinear) resampling method.

.. index:: pair: enumvalue; reduction_max
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a42483cece49698bff4fc700cd2de655c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reduction_max

Reduction using max operation.

.. index:: pair: enumvalue; reduction_min
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8cc607027ba7c9399ba610f22b88b3eb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reduction_min

Reduction using min operation.

.. index:: pair: enumvalue; reduction_sum
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af710a5a162f3ac1c005876b5c96a6087:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reduction_sum

Reduction using sum operation.

.. index:: pair: enumvalue; reduction_mul
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aab991281eb5a55118f233a6d906be210:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reduction_mul

Reduction using mul operation.

.. index:: pair: enumvalue; reduction_mean
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ae2afc4d79ef0f2e8de6c4d7292ea1a8e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reduction_mean

Reduction using mean operation.

.. index:: pair: enumvalue; reduction_norm_lp_max
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a67f19545c23e777ffbae900ab1b8cde5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reduction_norm_lp_max

Reduction using norm_lp_max operation.

.. index:: pair: enumvalue; reduction_norm_lp_sum
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ad73ba421a6b061afe16e49a6acc0733a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reduction_norm_lp_sum

Reduction using norm_lp_sum operation.

.. index:: pair: enumvalue; reduction_norm_lp_power_p_max
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a423e9857dd013140a323235cd520bad5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reduction_norm_lp_power_p_max

Reduction using norm_lp_power_p_max operation.

.. index:: pair: enumvalue; reduction_norm_lp_power_p_sum
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a161d82b9c140979ea6fc5ce7ad1afcbd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reduction_norm_lp_power_p_sum

Reduction using norm_lp_power_p_sum operation.

.. index:: pair: enumvalue; softmax_accurate
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a6aba07e95369523dbbb9cab366480dd4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	softmax_accurate

Softmax, numerically stable.

.. index:: pair: enumvalue; softmax_log
.. _doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a0888321c22541f5d7094cf44bc261ec9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	softmax_log

LogSoftmax, numerically stable.

