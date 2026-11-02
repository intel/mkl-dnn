.. index:: pair: enum; format_tag
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f:

enum dnnl::memory::format_tag
=============================

Overview
~~~~~~~~

Memory format tag specification. :ref:`More...<details-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum format_tag
	{
	    :ref:`undef<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf31ee5e3824f1f5e5d206bdf3029f22b>`           = dnnl_format_tag_undef,
	    :ref:`any<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`             = dnnl_format_tag_any,
	    :ref:`a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0cc175b9c0f1b6a831c399e269772661>`               = dnnl_a,
	    :ref:`ab<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`              = dnnl_ab,
	    :ref:`ba<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa07159c47ee1b19ae4fb9c40d480856c4>`              = dnnl_ba,
	    :ref:`abc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa900150983cd24fb0d6963f7d28e17f72>`             = dnnl_abc,
	    :ref:`acb<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5ff832d9bca8241d653279756f3ccd11>`             = dnnl_acb,
	    :ref:`bac<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79ec16df80b57696a03bb364410061f3>`             = dnnl_bac,
	    :ref:`bca<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab64eab8ce39e013604e243089c687e4f>`             = dnnl_bca,
	    :ref:`cba<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3944b025c9ca7eec3154b44666ae04a0>`             = dnnl_cba,
	    :ref:`abcd<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae2fc714c4727ee9395f324cd2e7f331f>`            = dnnl_abcd,
	    :ref:`abdc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5e822a0b52c71ce2ff2ecd3aa8ca46fc>`            = dnnl_abdc,
	    :ref:`acbd<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa49780bb1cc9c6c4be2da2a34283f72a3>`            = dnnl_acbd,
	    :ref:`acdb<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa776649ab3d00664587b889457db87da7>`            = dnnl_acdb,
	    :ref:`adbc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaa6ec9bf967afd962bf57cda5c588104>`            = dnnl_adbc,
	    :ref:`bacd<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5a3b732e0600663606054f6d8dfd465e>`            = dnnl_bacd,
	    :ref:`bcda<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa078c4e73146ba90c3e4e9daa1facac28>`            = dnnl_bcda,
	    :ref:`cdba<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faff2458511a39d703bd273eabbb1395d6>`            = dnnl_cdba,
	    :ref:`dcab<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9737604d6ad5938095054dc4413bf8e9>`            = dnnl_dcab,
	    :ref:`abcde<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab56b4d92b40713acc5af89985d4b786>`           = dnnl_abcde,
	    :ref:`abdec<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7a05226d93263d6fc613f52400f7416d>`           = dnnl_abdec,
	    :ref:`acbde<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae0e56a95aff0ee48fc44e8a4faf69adb>`           = dnnl_acbde,
	    :ref:`acdeb<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf95d333a023ac787b43babc5b5c480ae>`           = dnnl_acdeb,
	    :ref:`bacde<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabb59e2a27c3cb4e24384952fc6847948>`           = dnnl_bacde,
	    :ref:`bcdea<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6debec47fc73823aa65cc49b94b7e1fb>`           = dnnl_bcdea,
	    :ref:`cdeba<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafafcbfdfcede01281676ab63df88a919>`           = dnnl_cdeba,
	    :ref:`decab<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab9a152e018d02395e6e94c5619da4a91>`           = dnnl_decab,
	    :ref:`abced<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faebb080afaac3a990ad3f1d0f21742fac>`           = dnnl_abced,
	    :ref:`abcdef<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae80b5017098950fc58aad83c8c14978e>`          = dnnl_abcdef,
	    :ref:`abdfce<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7c4600767ddcc2dba9d42133b390e593>`          = dnnl_abdfce,
	    :ref:`acbdef<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa66fd87374b1ec6be9fdbea534e17adeb>`          = dnnl_acbdef,
	    :ref:`abdefc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0a0eee0ab79fa687f0f63332c252d13a>`          = dnnl_abdefc,
	    :ref:`defcab<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac9cdbbb9640597fea44206cee7f8a151>`          = dnnl_defcab,
	    :ref:`abcdfe<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeb4c431d71c552a2bc16fab86156738e>`          = dnnl_abcdfe,
	    :ref:`abcdefg<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7ac66c0f148de9519b8bd264312c4d64>`         = dnnl_abcdefg,
	    :ref:`abcdegf<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa189fe48d1050763b571517a9e2aa113b>`         = dnnl_abcdegf,
	    :ref:`abcdefgh<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae8dc4081b13434b45189a720b77b6818>`        = dnnl_abcdefgh,
	    :ref:`abcdefhg<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9899b60d60de786c689156d87433c254>`        = dnnl_abcdefhg,
	    :ref:`abcdefghi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8aa99b1f439ff71293e95357bac6fd94>`       = dnnl_abcdefghi,
	    :ref:`abcdefgih<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa838730cad01311dbfba64353015e329b>`       = dnnl_abcdefgih,
	    :ref:`abcdefghij<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa925576942e94b2ef57a066101b48876>`      = dnnl_abcdefghij,
	    :ref:`abcdefghji<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa84356605428eb3b9ad6da81d3d0b5b26>`      = dnnl_abcdefghji,
	    :ref:`abcdefghijk<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa92b9cccc0b98c3a0b8d0df25a421c0e3>`     = dnnl_abcdefghijk,
	    :ref:`abcdefghikj<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeb9ee2026bb8d37a7e777264cf4bcde2>`     = dnnl_abcdefghikj,
	    :ref:`abcdefghijkl<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9fc9d606912030dca86582ed62595cf7>`    = dnnl_abcdefghijkl,
	    :ref:`abcdefghijlk<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9b19e0f5975e33696433bcc7360c61aa>`    = dnnl_abcdefghijlk,
	    :ref:`x<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9dd4e461268c8034f5c8564e155c67a6>`               = a,
	    :ref:`nc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1e7342845e24eb3b5b3554490da1c128>`              = ab,
	    :ref:`cn<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7efdfc94655a25dcea3ec85e9bb703fa>`              = ba,
	    :ref:`tn<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaafb96b2fa8806be307c4496867bad56>`              = ab,
	    :ref:`nt<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa25930e3036f13852cb0b29694bbab611>`              = ba,
	    :ref:`ncw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabc4eee572ad38f8a5c2534e0b2533100>`             = abc,
	    :ref:`nwc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faccc0543dc9272f9fffd720fe55f818e4>`             = acb,
	    :ref:`nchw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>`            = abcd,
	    :ref:`nhwc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa763cbf7ba1b7b8793dcdc6e2157b5c42>`            = acdb,
	    :ref:`chwn<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafd2263500e815d8ce46e79e6b178a10a>`            = bcda,
	    :ref:`ncdhw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae7a84f08ef10d4fe3096307c6d9770fd>`           = abcde,
	    :ref:`ndhwc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad2f2770ca9883423f680fd4a29481d64>`           = acdeb,
	    :ref:`oi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa2e63ee01401aaeca78be023dfbb8c59>`              = ab,
	    :ref:`io<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf98ed07a4d5f50f7de1410d905f1477f>`              = ba,
	    :ref:`oiw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3a8cb832bca6da72735b742014560a5b>`             = abc,
	    :ref:`owi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa55e9a1b1cc2aaf0f5b7417d8098ea4b3>`             = acb,
	    :ref:`wio<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5b12270b2f1c52cb9ae37905880a5a6f>`             = cba,
	    :ref:`iwo<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7afa5d2c0d2aa139eb213430ec70d91d>`             = bca,
	    :ref:`oihw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa14b72a467aeefa06a5cb802ec4a7743c>`            = abcd,
	    :ref:`hwio<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadd942f902c741de7193e1869b13ba55a>`            = cdba,
	    :ref:`ohwi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab889a0cb4d051674706433941a2e867f>`            = acdb,
	    :ref:`ihwo<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa56fe35ca9f339b4948384c2a2fa02de5>`            = bcda,
	    :ref:`iohw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac16710619e6550753801926b5388bef8>`            = bacd,
	    :ref:`oidhw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa98407962f915285419b23f5b0a223882>`           = abcde,
	    :ref:`dhwio<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeafcb3d831eef23cde0b46f64646327c>`           = cdeba,
	    :ref:`odhwi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabf55006478a9dae3506eb67923703a1d>`           = acdeb,
	    :ref:`iodhw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa940f633559dc193ac5a96584394774a3>`           = bacde,
	    :ref:`idhwo<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2b3d6389089fb08059ece8a9642b8991>`           = bcdea,
	    :ref:`goiw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1f1bc797706732c22eb9987e3cea3e54>`            = abcd,
	    :ref:`gowi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa113968d2d4d3a82b7880657ac86060ed>`            = abdc,
	    :ref:`wigo<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1609c2708a916705421742969a35351c>`            = dcab,
	    :ref:`gohwi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa371d33c7ac584e28d7243cd6f2e72bfb>`           = abdec,
	    :ref:`goihw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3dc0e53430a6368210356118196f821a>`           = abcde,
	    :ref:`hwigo<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafd710c828421b3c91725b0e5aa53ecc6>`           = decab,
	    :ref:`giohw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5f1d6f8a3a8842611d15affa574b864f>`           = acbde,
	    :ref:`goidhw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa678de4e5a5e34c9a71a632b8f334b8b>`          = abcdef,
	    :ref:`giodhw<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4f2399cfdc9861fee9d39cb1d703aadc>`          = acbdef,
	    :ref:`godhwi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faad93659c675a47264a01968e00dc1714>`          = abdefc,
	    :ref:`dhwigo<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab291a9717fdf353288d2d6082d578059>`          = defcab,
	    :ref:`tnc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac775cf954921a129a65eb929476de911>`             = abc,
	    :ref:`ntc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6758344f9f89d390953c6718f34e9be1>`             = bac,
	    :ref:`ldnc<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab49be97ff353a86d84d06d98f846b61d>`            = abcd,
	    :ref:`ldigo<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`           = abcde,
	    :ref:`ldgoi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4c850fa02c9e56a4c20dbfde737f1f8d>`           = abdec,
	    :ref:`ldio<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa201b31c3cfbeec4962a869bdc27da090>`            = abcd,
	    :ref:`ldoi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79ca41a567de72f7df2782e76dea722d>`            = abdc,
	    :ref:`ldgo<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab8690cd92ccee6a0ad55faccc0346aab>`            = abcd,
	    :target:`AB16b16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae696e4932512650482fec8856ac7b396>`        = dnnl_AB16b16a,
	    :target:`AB16b32a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa082aafb062610d5d784ce342bdd8cdd4>`        = dnnl_AB16b32a,
	    :target:`AB16b64a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabe6cb9a0a2db25671504613fd6ea52bd>`        = dnnl_AB16b64a,
	    :target:`AB8b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad6cee9fe17ab3df600e555345ad08b13>`       = dnnl_AB8b16a2b,
	    :target:`AB8b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac707d55115ef824284f55cb5d22614ed>`       = dnnl_AB8b32a2b,
	    :target:`AB8b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4f33f935648b9c05dc63ebeb8f30510d>`       = dnnl_AB8b64a2b,
	    :target:`AB4b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8e40cf67f4211c9f05a296ed143e358d>`       = dnnl_AB4b16a4b,
	    :target:`AB4b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa452e4cfc802e0ea8f2c8ab1362eb996e>`       = dnnl_AB4b32a4b,
	    :target:`AB4b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa74fe0faa12cac9cda3f5ad72c726d5fa>`       = dnnl_AB4b64a4b,
	    :target:`AB16b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa61b5976b5f5310d69f7b2ce08461020b>`      = dnnl_AB16b16a4b,
	    :target:`AB16b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa72cab25d10998f05285d19f0fce9212f>`      = dnnl_AB16b32a4b,
	    :target:`AB16b48a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4d18ec630c17fa245161d3c47c45dff2>`      = dnnl_AB16b48a4b,
	    :target:`AB16b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2cf2fc684de265c8a3989e8ce89ebc1f>`      = dnnl_AB16b64a4b,
	    :target:`AB16b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa74b72df747a37728dbc28a63d087d1d2>`      = dnnl_AB16b16a2b,
	    :target:`AB16b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa047418a7b5455b1350811b520945a43c>`      = dnnl_AB16b32a2b,
	    :target:`AB16b48a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2688e2c2442b5e02ac042b6a840268a3>`      = dnnl_AB16b48a2b,
	    :target:`AB16b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadbbe4016e92bb8ef7ede4e14ac0e2a75>`      = dnnl_AB16b64a2b,
	    :target:`Abc16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa77c145adee31be0b94281ba7ebca6fcb>`          = dnnl_Abc16a,
	    :target:`ABc16a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8adf2fda72cf03635d30469dde3640fe>`       = dnnl_ABc16a16b,
	    :target:`ABc4a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8e87a370997f9151038106471b48f174>`         = dnnl_ABc4a4b,
	    :target:`aBc16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7e6b87e989e2a78a5f69d2c1af35b462>`          = dnnl_aBc16b,
	    :target:`aBc32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaf7b35192a95be6e60d420f5cee7bc7c>`          = dnnl_aBc32b,
	    :target:`ABc16b16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4bb100f4bfde4cf12c696cd8873dc738>`       = dnnl_ABc16b16a,
	    :target:`ABc16b32a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa882a119ac837c774633555018837950a>`       = dnnl_ABc16b32a,
	    :target:`ABc16b64a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa39095713d5418335fd3874b7984fd35f>`       = dnnl_ABc16b64a,
	    :target:`Abc4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2a135231ba246bcff4e4a76a7783061f>`           = dnnl_Abc4a,
	    :target:`aBc4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa84114cd99efa89c7d0d437863332a67f>`           = dnnl_aBc4b,
	    :target:`ABc4b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8939404016732ff4ece68dfcaf40c75e>`      = dnnl_ABc4b16a4b,
	    :target:`ABc4b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa064a58b3ccb7142cd70183c8a04acffa>`      = dnnl_ABc4b32a4b,
	    :target:`ABc4b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9b49824025af253af452d303be3c8cec>`      = dnnl_ABc4b64a4b,
	    :target:`ABc2b8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4c76ce9b095243e79cb310e1b041057b>`       = dnnl_ABc2b8a4b,
	    :target:`ABc16a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5b88176484acc275bb92816a75cfd7e7>`     = dnnl_ABc16a16b2a,
	    :target:`ABc16b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa148d3babd8ed47f882db7e59da7d7201>`     = dnnl_ABc16b16a4b,
	    :target:`ABc16b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8809706a18ededea4d4e6c5d3e7be623>`     = dnnl_ABc16b32a4b,
	    :target:`ABc16b48a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa3b2c0cd081551e20a1ccf38e01e0238>`     = dnnl_ABc16b48a4b,
	    :target:`ABc16b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1956fc67a49b2ea7f9fb75786e4cdfb1>`     = dnnl_ABc16b64a4b,
	    :target:`ABc16b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa115fc61bc26d58d987220eab16f5670>`     = dnnl_ABc16b16a2b,
	    :target:`ABc16b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2513c3568c9a4ff8e0f0a94e744b00fe>`     = dnnl_ABc16b32a2b,
	    :target:`ABc16b48a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faefe9db85237618e91a470bd1526c69be>`     = dnnl_ABc16b48a2b,
	    :target:`ABc16b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8b8b0b0cba0752efc70202ce1657211c>`     = dnnl_ABc16b64a2b,
	    :target:`ABc4b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0d815f0b88178e2f4d06b82cc23b689b>`         = dnnl_ABc4b4a,
	    :target:`ABc8a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79b9870f007647a15905c0069b6ba996>`      = dnnl_ABc8a16b2a,
	    :target:`ABc8a8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2665331a935aeba5d5c76eee34fe528a>`         = dnnl_ABc8a8b,
	    :target:`ABc8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa98947cec3d32b09b2e9dc4011b5d0ea1>`         = dnnl_ABc8a4b,
	    :target:`aBc8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac79e8d3b22b4b6283a4d47fb29ada5ba>`           = dnnl_aBc8b,
	    :target:`ABc8b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf7b1e3bb02bcb88be3ff57d52066db29>`      = dnnl_ABc8b16a2b,
	    :target:`ABc8b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8670024ee0c473d00c5fc6e654a8859a>`      = dnnl_ABc8b32a2b,
	    :target:`ABc8b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab268fa2ac23eb270ace231082c4438fb>`      = dnnl_ABc8b64a2b,
	    :target:`ABc8b8a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadc1e646f5a4dbdb66a68703924f818d0>`         = dnnl_ABc8b8a,
	    :target:`Abcd8a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9b52b9089aaf72e0ca35cfd06c3b9c08>`          = dnnl_Abcd8a,
	    :target:`Abcd16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab1756ecea8a29754af14815b9cd2cdf3>`         = dnnl_Abcd16a,
	    :target:`Abcd32a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facc543c011dec0fc6af8e6306e167bb9b>`         = dnnl_Abcd32a,
	    :target:`ABcd16a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8de48edabb9809687b4942e1b19bf326>`      = dnnl_ABcd16a16b,
	    :target:`aBcd16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5d55d90de66e593624aee60adb9f8540>`         = dnnl_aBcd16b,
	    :target:`aBcd32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa61d6c958096b122e341e1072ac3353fd>`         = dnnl_aBcd32b,
	    :target:`ABcd16b16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa32e2a00a75ff1810113dab64fff1a68f>`      = dnnl_ABcd16b16a,
	    :target:`ABcd16b32a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6946869905dddae367af298d130ebe44>`      = dnnl_ABcd16b32a,
	    :target:`ABcd16b64a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa35689b74d14bb8bfe9c15143e3786b8b>`      = dnnl_ABcd16b64a,
	    :target:`aBCd16b16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facdf43d4ebfe0a9d95e8932b3e70517c7>`      = dnnl_aBCd16b16c,
	    :target:`aBCd16c16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1eb6b367c2ccab891bc1f002bcf445c0>`      = dnnl_aBCd16c16b,
	    :target:`Abcd4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa91028d97462edc87adafecbeac0fda2a>`          = dnnl_Abcd4a,
	    :target:`aBcd4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac2360cf243576d4dfdc31f8eabf95e5e>`          = dnnl_aBcd4b,
	    :target:`ABcd4b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab585b6df77eedca0785c1cda91f31e29>`     = dnnl_ABcd4b16a4b,
	    :target:`ABcd4b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa767abf39568b10407ff1801361a8ddb>`     = dnnl_ABcd4b32a4b,
	    :target:`ABcd4b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa66c030a0c7d1271d33cd93e4f18306c4>`     = dnnl_ABcd4b64a4b,
	    :target:`ABcd2b8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa61957e79765808bd586c355faf8ba6bc>`      = dnnl_ABcd2b8a4b,
	    :target:`ABcd4b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae33cdf340198dede7a5e2a5777601a05>`        = dnnl_ABcd4b4a,
	    :target:`ABcd4a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafcaa6b01f2f09f4724aa61b86b75a18e>`        = dnnl_ABcd4a4b,
	    :target:`aBCd4c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae349b11785f3ab115cfd826cb3273af4>`     = dnnl_aBCd4c16b4c,
	    :target:`aBCd2c8b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad4dfedc7f576eb8445184b019e4996c2>`      = dnnl_aBCd2c8b4c,
	    :target:`ABcd16a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa12b88ef27bb48152a2e9f56085ea207e>`    = dnnl_ABcd16a16b2a,
	    :target:`ABcd16b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8770ad8ba755b2b8b31fbd2b7001fdca>`    = dnnl_ABcd16b16a4b,
	    :target:`ABcd16b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2dc585624d7fbe7367273fdc66413d87>`    = dnnl_ABcd16b32a4b,
	    :target:`ABcd16b48a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9a5dd71450ad756dc256704a9326a856>`    = dnnl_ABcd16b48a4b,
	    :target:`ABcd16b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadbd891340f20f041b1c79660fd27da70>`    = dnnl_ABcd16b64a4b,
	    :target:`ABcd16b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3c8cf7645456e108799c8d0f6b9ca7c7>`    = dnnl_ABcd16b16a2b,
	    :target:`ABcd16b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa220f9999e85d51f9f30d2c92aa46976a>`    = dnnl_ABcd16b32a2b,
	    :target:`ABcd16b48a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa498385518386c0c3bffe2f7e92a330b4>`    = dnnl_ABcd16b48a2b,
	    :target:`ABcd16b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac9deb9edd7306b6f653df2d01fb075f7>`    = dnnl_ABcd16b64a2b,
	    :target:`aBCd16b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf8bf6e2352f19ca4727cf5e4d95da5d6>`    = dnnl_aBCd16b16c2b,
	    :target:`aBCd16c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa54737862ae66bf0bc1902cab98a703e>`    = dnnl_aBCd16c16b4c,
	    :target:`aBCd16c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa813eeb17878a29ffa447f9bf25454a5f>`    = dnnl_aBCd16c16b2c,
	    :target:`aBCd4c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa34041da6e29a487758640adc3b076c8>`        = dnnl_aBCd4c4b,
	    :target:`aBCd4b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf078b1cb553e3626b2c81039f29db6b8>`        = dnnl_aBCd4b4c,
	    :target:`ABcd8a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6597bf8274a8f7ac8ef39e9cd61b9ec1>`     = dnnl_ABcd8a16b2a,
	    :target:`ABcd8a8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadfa7f0ee91a58b03b5d65c293125858a>`        = dnnl_ABcd8a8b,
	    :target:`ABcd8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa68adbe75d944d8f28b6b0f506a701bc9>`        = dnnl_ABcd8a4b,
	    :target:`ABcd8a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0da9a96d375f3d24fd2009de428ae5dc>`        = dnnl_ABcd8a2b,
	    :ref:`aBcd8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa448a7fc9219294ce172b0edf9498b5c4>`          = dnnl_aBcd8b,
	    :target:`ABcd8b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0c035ab408bbad64c4599d82204eca70>`     = dnnl_ABcd8b16a2b,
	    :target:`ABcd8b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa510d3c14034871efae12eb75380f1314>`     = dnnl_ABcd8b32a2b,
	    :target:`ABcd8b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faefaca8377de0b3cad31c02644fee1ee3>`     = dnnl_ABcd8b64a2b,
	    :target:`aBCd8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae6c88aeae25973eaa8ea089c2a6a42f0>`     = dnnl_aBCd8b16c2b,
	    :ref:`ABcd8b8a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabcbce50e9c241458767871fa053e1ba0>`        = dnnl_ABcd8b8a,
	    :target:`aBCd8b8c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa22a04acd0981d3df20a6946f2d63d50e>`        = dnnl_aBCd8b8c,
	    :target:`aBCd8b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa058f6c5cf231a29e0d01dc686ae472b4>`        = dnnl_aBCd8b4c,
	    :target:`aBCd8c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2e0728bac73db59d8e0481e88f29fad1>`     = dnnl_aBCd8c16b2c,
	    :target:`aBCd8c8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2e63d893f1711440e5f8935c1d20c164>`        = dnnl_aBCd8c8b,
	    :target:`Abcde16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa12db43bbe0c6b63c3b4b04d0fadd802a>`        = dnnl_Abcde16a,
	    :target:`Abcde32a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6ddd5d85e7ecdd273334dcc2ecdb7899>`        = dnnl_Abcde32a,
	    :target:`ABcde16a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa120b9b67382c458a97a9d1cc3c4e0902>`     = dnnl_ABcde16a16b,
	    :target:`aBcde16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac45847e0e12ecd8cac5d4a98ccc6bef6>`        = dnnl_aBcde16b,
	    :target:`aBcde32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7e1802ef8a37d9f27ae9ef23414df82a>`        = dnnl_aBcde32b,
	    :target:`ABcde16b16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa78e33c68c36a4f4b97cc8958a8c4aa95>`     = dnnl_ABcde16b16a,
	    :target:`ABcde16b32a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa68d72fcbe52a6ff996099f7051f4588a>`     = dnnl_ABcde16b32a,
	    :target:`ABcde16b64a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa08009e5d71d13fe6ba3a9f218f63a44>`     = dnnl_ABcde16b64a,
	    :target:`aBCde16b16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4789d2b682a01bed34318be417e9a206>`     = dnnl_aBCde16b16c,
	    :target:`aBCde16c16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa275b87a4c6da71c18c65a691c6af2f3e>`     = dnnl_aBCde16c16b,
	    :target:`aBCde2c8b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac59accaba01874d96f9ac65b7d66540a>`     = dnnl_aBCde2c8b4c,
	    :target:`Abcde4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8ecca782aa49990ab26ff202f0251e66>`         = dnnl_Abcde4a,
	    :target:`aBcde4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeaa37c2eca13f8a741d741640efd7656>`         = dnnl_aBcde4b,
	    :target:`ABcde4b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa019f84e0b32929be123b952be11426db>`       = dnnl_ABcde4b4a,
	    :target:`ABcde4a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa54a09e0006cadcb8be5656c6f1a06549>`       = dnnl_ABcde4a4b,
	    :target:`aBCde4b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0ecdca6185803e918e4544323c41d3e9>`       = dnnl_aBCde4b4c,
	    :target:`aBCde4c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faff14e3a6dabb2e8874b7bf36f12e5c53>`    = dnnl_aBCde4c16b4c,
	    :target:`aBCde16b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad849f504066db3c6d09629a014e9c46d>`   = dnnl_aBCde16b16c2b,
	    :target:`aBCde16c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faebf11ce876e2960876f67abb01df3471>`   = dnnl_aBCde16c16b4c,
	    :target:`aBCde16c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faefad4e30fbe3ebe853fd2599ec6449cd>`   = dnnl_aBCde16c16b2c,
	    :target:`aBCdef16c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6f2b98917bc5bd4596f3e20ac685d926>`  = dnnl_aBCdef16c16b2c,
	    :target:`aBCde4c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab80d94e8f97f164b66d4b478b9410cde>`       = dnnl_aBCde4c4b,
	    :target:`Abcde8a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf31b83b4d7bf6091d47b0f57d0e10264>`         = dnnl_Abcde8a,
	    :target:`ABcde8a8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa954a31af1ed0b3b9a9e375d6f6d4257a>`       = dnnl_ABcde8a8b,
	    :target:`ABcde8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa66fa7bb6501a24eda28c6e22210b9cea>`       = dnnl_ABcde8a4b,
	    :target:`aBcde8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2928a5cd5c7e5e95998bcb4f24335220>`         = dnnl_aBcde8b,
	    :target:`ABcde8b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa00df0930569d6a4b9c62f3bd4c27f5a4>`    = dnnl_ABcde8b16a2b,
	    :target:`ABcde8b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa35b7b934d2be2d06da98855c7637b170>`    = dnnl_ABcde8b32a2b,
	    :target:`ABcde8b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7feb6e8da3edfd912eb7f3f9cbae944c>`    = dnnl_ABcde8b64a2b,
	    :target:`ABcde4b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9a04276c66b717c1ea67d44c25b4838a>`    = dnnl_ABcde4b16a4b,
	    :target:`ABcde4b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac9cf1a6e96b49a6a5abd007d1abcad98>`    = dnnl_ABcde4b32a4b,
	    :target:`ABcde4b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa577fddc6924d64ed373f904364557e01>`    = dnnl_ABcde4b64a4b,
	    :target:`ABcde16b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa86d68decb77a15f9e06a2b9c35929898>`   = dnnl_ABcde16b16a4b,
	    :target:`ABcde16b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa160a997f9f712b47a75f27d01c13606a>`   = dnnl_ABcde16b32a4b,
	    :target:`ABcde16b48a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4012eb697ea2e55073ada0db96b07151>`   = dnnl_ABcde16b48a4b,
	    :target:`ABcde16b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa2e083a389e018313124081250f8400a>`   = dnnl_ABcde16b64a4b,
	    :target:`ABcde16b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa09e446541f494a8433a9597418ccf1d2>`   = dnnl_ABcde16b16a2b,
	    :target:`ABcde16b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafa9aa285c069eb2b41e454e5e488f714>`   = dnnl_ABcde16b32a2b,
	    :target:`ABcde16b48a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa686706c9a993c0d61e639b6b5c65715e>`   = dnnl_ABcde16b48a2b,
	    :target:`ABcde16b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa422c1ebf7f7a389830924451a61c5b15>`   = dnnl_ABcde16b64a2b,
	    :target:`ABcde2b8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa96c0e90f2b6f1804aab9721ee04f8ec7>`     = dnnl_ABcde2b8a4b,
	    :target:`aBCde8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa92f337765b473338389ce17961927d8f>`    = dnnl_aBCde8b16c2b,
	    :target:`ABcde8b8a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faedf9318e4c601759f4474fc94464a459>`       = dnnl_ABcde8b8a,
	    :target:`aBCde8b8c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6390664d064e577e4c068fcfc6de9134>`       = dnnl_aBCde8b8c,
	    :target:`aBCde8b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa88d3b7478dc875887586f868e48e3e0b>`       = dnnl_aBCde8b4c,
	    :target:`ABcd4a8b8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa70621dffd1398acc2e78eeb8a3fcda49>`    = dnnl_ABcd4a8b8a4b,
	    :target:`ABcd2a8b8a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa750678c620c0119068dc66f751a9e43c>`    = dnnl_ABcd2a8b8a2b,
	    :target:`aBCde4b8c8b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa00025ae6907e34c690de6ddd60423523>`   = dnnl_aBCde4b8c8b4c,
	    :target:`aBCde2b8c8b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac6c2b18886f80e76c2efd5e81018e89a>`   = dnnl_aBCde2b8c8b2c,
	    :target:`aBCde8c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa39dcf6c6e6061494dc3d79128a7b7e16>`    = dnnl_aBCde8c16b2c,
	    :target:`aBCde8c8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa38796ed143bff89b696b8b19eeeda44b>`       = dnnl_aBCde8c8b,
	    :target:`aBcdef16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6a19babc839e184e5ca2b277f376bfaf>`       = dnnl_aBcdef16b,
	    :target:`aBCdef16b16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab0cf98561c58a11c0bde0a0a31aa3a59>`    = dnnl_aBCdef16b16c,
	    :target:`aBCdef16c16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa36766eadad940ab2804bcccb6a7838c5>`    = dnnl_aBCdef16c16b,
	    :target:`aBcdef4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa78db8ad202df89454a921b7c5432d151>`        = dnnl_aBcdef4b,
	    :target:`aBCdef2c8b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa291731a2fd28b29f717e853d223fbf06>`    = dnnl_aBCdef2c8b4c,
	    :target:`aBCdef4c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6c6f2aba250ce2441a9ff2976ab0e4ed>`      = dnnl_aBCdef4c4b,
	    :target:`aBCdef4b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facb829143c19eb16c7b91eb42e76a4e30>`      = dnnl_aBCdef4b4c,
	    :target:`aBCdef8b8c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa60e1c7bfad21352d904f80f5671d082b>`      = dnnl_aBCdef8b8c,
	    :target:`aBCdef8b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1941c20b002ea8eb4d2f5364ae87ea1c>`      = dnnl_aBCdef8b4c,
	    :target:`aBCdef8c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa10d6e2f36d793a8de7fd5ea97d2e158b>`   = dnnl_aBCdef8c16b2c,
	    :target:`aBCdef4c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa52d5a75e1bb4a6b6a704f478a6c27bb7>`   = dnnl_aBCdef4c16b4c,
	    :target:`aBCdef8c8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faea3f9e57566cad20acdac5254194cf2c>`      = dnnl_aBCdef8c8b,
	    :target:`aBdc16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafba0f9e6c5d27843e2f66bfa7e277612>`         = dnnl_aBdc16b,
	    :target:`aBdc4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1b1edb0117d39115ccdbb72fd13ab926>`          = dnnl_aBdc4b,
	    :target:`aBdc8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5e01a5405304144b1cc88bcd3921a999>`          = dnnl_aBdc8b,
	    :target:`aBdec16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa80c37d3114c0c477b95126c208145e24>`        = dnnl_aBdec16b,
	    :target:`aBdec4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faefbc4428be375dd4cfe493fe397d8c65>`         = dnnl_aBdec4b,
	    :target:`aBdec8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8babba3825af4f9ee534ba0292918d3b>`         = dnnl_aBdec8b,
	    :target:`aBdefc16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa73c193ce0e5f20b6649fbfa89cd85806>`       = dnnl_aBdefc16b,
	    :target:`aCBdef16c16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7d1aa12b0392d673efe4d8fd8fa5498c>`    = dnnl_aCBdef16c16b,
	    :target:`aCBdef16b16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa8e232c4f2de2b45db60e33e979194eb>`    = dnnl_aCBdef16b16c,
	    :target:`aBdefc4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa77f7431a6a7228b1d7941be09e133455>`        = dnnl_aBdefc4b,
	    :target:`aBdefc8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facd3979acb531a3932147cbb8fe8c480e>`        = dnnl_aBdefc8b,
	    :target:`Acb16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab0f77caa3dd94494f9a14ed5bf05b9d7>`          = dnnl_Acb16a,
	    :target:`Acb4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa723520a2226cac57233f4e24ea6259ca>`           = dnnl_Acb4a,
	    :target:`Acb8a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faef80848cfaf08b3edbf35ba52834dd9b>`           = dnnl_Acb8a,
	    :target:`aCBd16b16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2409de8f31bda21af9d61b2bfd569263>`      = dnnl_aCBd16b16c,
	    :target:`aCBd16c16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad18a1b4d6f9aece1b426cc95d1a34d90>`      = dnnl_aCBd16c16b,
	    :target:`aCBde16b16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa34f370b4d4c2f03494747d02f5d4970b>`     = dnnl_aCBde16b16c,
	    :target:`aCBde16c16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae1f4826029f30fa9a89284cb6109a1bd>`     = dnnl_aCBde16c16b,
	    :target:`Acdb16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5bc76df72f48c4212994f8ba890c583d>`         = dnnl_Acdb16a,
	    :target:`Acdb4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa119cdc50aa9f3f36d7d3125c778c620c>`          = dnnl_Acdb4a,
	    :target:`Acdb8a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa650aee93cdcdcaf91e2bf0657088b864>`          = dnnl_Acdb8a,
	    :target:`Acdeb16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0cb28993934aff498bcda6f6f3cee8d6>`        = dnnl_Acdeb16a,
	    :target:`Acdeb4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa334cd01f22c0e6f36095c76a66504543>`         = dnnl_Acdeb4a,
	    :target:`Acdeb8a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac7f114c480275779f0ae4484d83b6f41>`         = dnnl_Acdeb8a,
	    :target:`BAc16a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab1055b9db4db65014b9c0c00adddb1c7>`       = dnnl_BAc16a16b,
	    :target:`BAc16b16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa46c299696c756e994c0a8285c8a9a884>`       = dnnl_BAc16b16a,
	    :target:`BAcd16a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad98b0acb88c4867796a3cfd0c15127ed>`      = dnnl_BAcd16a16b,
	    :target:`BAcd16b16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7d188af63c65095e2b34ef401f752799>`      = dnnl_BAcd16b16a,
	    :target:`ABcd32a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab937580442cc8b50cf459bb534876e5e>`      = dnnl_ABcd32a32b,
	    :target:`BAcde16b16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa09e89b2758266ba2ea266c84369a5ab6>`     = dnnl_BAcde16b16a,
	    :target:`BAcde16a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa802d13cf6ad6e14fafdf471c97df5a5f>`     = dnnl_BAcde16a16b,
	    :target:`aBdec32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa868c53346cf855961472b7c7516e2257>`        = dnnl_aBdec32b,
	    :target:`Abcdef16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4f6cb73f8ac3d13ba31d78c9bfd33232>`       = dnnl_Abcdef16a,
	    :target:`Abcdef32a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa71d88a282605bcf34c14cd4b97483941>`       = dnnl_Abcdef32a,
	    :target:`Acdb32a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad8119b9e070a87198eaf3564122805e3>`         = dnnl_Acdb32a,
	    :target:`aBCd2b4c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf707a39e9bc81057a8f4d7f96beea078>`      = dnnl_aBCd2b4c2b,
	    :target:`aBCde2b4c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8eff2ffbf6564ff766d9aadb26eff6d3>`     = dnnl_aBCde2b4c2b,
	    :target:`aBCdef2b4c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9a3224480d168f92cbe03a1611c9a046>`    = dnnl_aBCdef2b4c2b,
	    :target:`aBCd2c4b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8f4456fbf604283332857e8a2164c00e>`      = dnnl_aBCd2c4b2c,
	    :target:`aBCde2c4b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1cd6ceac6f7b69ffbfe014bc7e372960>`     = dnnl_aBCde2c4b2c,
	    :target:`aBCdef2c4b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa40147d580e59a3a2617632fcbae8c649>`    = dnnl_aBCdef2c4b2c,
	    :target:`aBCd4b8c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facf757e0782891dcd8533d22aa963a2cb>`      = dnnl_aBCd4b8c2b,
	    :target:`aBCde4b8c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa0bfe2467e7546cb30d0f482a789a54c>`     = dnnl_aBCde4b8c2b,
	    :target:`aBCdef4b8c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeb0734151b49071dd726fdddf5983a47>`    = dnnl_aBCdef4b8c2b,
	    :target:`aBCd4c8b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2dc655c6b0e242b96908c6ec42a21279>`      = dnnl_aBCd4c8b2c,
	    :target:`aBCde4c8b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa41bb5b49467c596935a0925d524094e8>`     = dnnl_aBCde4c8b2c,
	    :target:`aBCdef4c8b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3b147225642d3a9f8afd4f34c0ba7630>`    = dnnl_aBCdef4c8b2c,
	    :target:`AB32a32b8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa235270d63611411252eb2ee8b154b8ae>`    = dnnl_AB32a32b8a4b,
	    :target:`AB32a32b8a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa648e2d7f561473aa47889906a9493971>`    = dnnl_AB32a32b8a2b,
	    :target:`AB8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf862a5103a447f4ee8a024d24c20843a>`          = dnnl_AB8a4b,
	    :target:`AB8a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab6b19312da9179f07b87273c45951588>`          = dnnl_AB8a2b,
	    :target:`abDc32d<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa337dbd6c1e6baa7747e091ca64178407>`         = dnnl_abDc32d,
	    :target:`abDC32d4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4bb5cc8413870035a846e8af99266595>`       = dnnl_abDC32d4c,
	    :target:`abCd32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8459ab5eba67f93c34735d50c5a811e2>`         = dnnl_abCd32c,
	    :target:`abdEc32e<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa84622339f69f0e426cb046313f0e0f39>`        = dnnl_abdEc32e,
	    :target:`abdEC32e2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1a143fd96d36290950ce735565aea5bf>`      = dnnl_abdEC32e2c,
	    :target:`abdEC32e4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa99340b0c84db6edc75647b9950a5d37b>`      = dnnl_abdEC32e4c,
	    :target:`abdCe32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6cecb154177a7cc52ca16bb7f4ef5a2a>`        = dnnl_abdCe32c,
	    :target:`abdCE32c2e<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa52ac0c523be15926bd99a2a628b6d025>`      = dnnl_abdCE32c2e,
	    :target:`aBCdef16c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa81daf4f05c60156fac5ddbab2abc04fd>`  = dnnl_aBCdef16c16b4c,
	    :target:`aBdC16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6792d8968b1f942d49fae5279c5aa364>`       = dnnl_aBdC16b4c,
	    :target:`aBdeC16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9266d8b9bfe63e064e4606f9ee4c943e>`      = dnnl_aBdeC16b4c,
	    :target:`AcB16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7b8c5962408786684d649774b5dcaa09>`        = dnnl_AcB16a4b,
	    :target:`AcdB16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1a56fb39c3b1f61a6700eb895981e2ce>`       = dnnl_AcdB16a2b,
	    :target:`aBdefC16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa49e0c7ba899e9c4bb7c96bce278c4da5>`     = dnnl_aBdefC16b4c,
	    :target:`AcdeB16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa236c1241a284a49f5085c030583dcc07>`      = dnnl_AcdeB16a4b,
	    :target:`Acb32a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa61c7f67c1a8e32f99b05e6e7ccd8d4b5>`          = dnnl_Acb32a,
	    :target:`AcB32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae690b5192dfddb6deb474362354ef56c>`        = dnnl_AcB32a2b,
	    :target:`AcB32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4b7cafdf15cba40582ad9405af61d8e8>`        = dnnl_AcB32a4b,
	    :target:`Acb48a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa00c22d0fcb6b55247c37eaf708524a82>`          = dnnl_Acb48a,
	    :target:`AcB48a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa64c7a216fc54e9d6dcf9ac025859d07>`        = dnnl_AcB48a2b,
	    :target:`AcB48a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa491d02094800a1235907c5305b5c5855>`        = dnnl_AcB48a4b,
	    :target:`Acb64a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa99ca5166045b451632eb978d3c49f3d4>`          = dnnl_Acb64a,
	    :target:`AcB64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7ebda2472e365d160f917d68b87aa67e>`        = dnnl_AcB64a2b,
	    :target:`AcB64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2f2dede029c95451b235ccee62448ef8>`        = dnnl_AcB64a4b,
	    :target:`cBa2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad469940b7c6f4cf7c28d0f0e481e1ca1>`           = dnnl_cBa2b,
	    :target:`cBa4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa214a7e0946b5169b5a4a0db93e7f5bd9>`           = dnnl_cBa4b,
	    :target:`aBdc32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac2369fee25d468d63a9e6b9afcbc3dc2>`         = dnnl_aBdc32b,
	    :target:`aBdC32b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa55ad2edad06458cca9275b409ba45047>`       = dnnl_aBdC32b2c,
	    :target:`aBdC32b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa20e00273a7177fe69efc4c9da5f5cc5a>`       = dnnl_aBdC32b4c,
	    :target:`aBdc48b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabb217c41e651f9655058a9fe13c2f918>`         = dnnl_aBdc48b,
	    :target:`aBdC48b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faacfab6d19fce4ac5323c583d9c8c8a61>`       = dnnl_aBdC48b2c,
	    :target:`aBdC48b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab41f5aa1d835ed5a1817e04035f1c5ea>`       = dnnl_aBdC48b4c,
	    :target:`aBdc64b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4cb506e53716d5c6ec13ba94d52b2b92>`         = dnnl_aBdc64b,
	    :target:`aBdC64b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4d1830230960d84dc767c3a3193ee6b1>`       = dnnl_aBdC64b2c,
	    :target:`aBdC64b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa56c6dd67e261f7df6363fd243d389638>`       = dnnl_aBdC64b4c,
	    :target:`adcb<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabace68067690f39a0e089b23acebdda3>`            = dnnl_adcb,
	    :target:`adCb2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac467e9ada00bd67727313184431a233d>`          = dnnl_adCb2c,
	    :target:`adCb4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa40df54c5d1dacf1b6c4181fea9e9bb34>`          = dnnl_adCb4c,
	    :target:`AcdB32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad277a83ebd8cc0be41e3c0ddd88530cf>`       = dnnl_AcdB32a2b,
	    :target:`AcdB32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf7cefdcc9859a8c1b9146be9ed85f922>`       = dnnl_AcdB32a4b,
	    :target:`Acdb48a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2019b6963482fa0a041768cc13975f24>`         = dnnl_Acdb48a,
	    :target:`AcdB48a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae0a8ac7eecbe796e664401beb6af3718>`       = dnnl_AcdB48a2b,
	    :target:`AcdB48a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa53cd4fbffbde6b0149072e90c50cf9c4>`       = dnnl_AcdB48a4b,
	    :target:`Acdb64a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa21a1dc5d732512452755ccb5cc2ef003>`         = dnnl_Acdb64a,
	    :target:`AcdB64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa24ac512c79267c7dafa098c20f761e39>`       = dnnl_AcdB64a2b,
	    :target:`AcdB64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7410799f0e801c9b2c2863556afac1c8>`       = dnnl_AcdB64a4b,
	    :target:`cdBa2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8954410e7f96bbbed5e368efd77ff87c>`          = dnnl_cdBa2b,
	    :target:`cdBa4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab34bbb0e339fdb21e7f679aefba58b8b>`          = dnnl_cdBa4b,
	    :target:`aBdeC32b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa939a407387db44a58b32b450cf9a9953>`      = dnnl_aBdeC32b2c,
	    :target:`aBdeC32b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7a6853aae5d2ac321285c9622fabdae1>`      = dnnl_aBdeC32b4c,
	    :target:`aBdec48b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5743fb0ddecac21441de52847d4eb119>`        = dnnl_aBdec48b,
	    :target:`aBdeC48b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa87382ffeabb275469fdd49f417daea2a>`      = dnnl_aBdeC48b2c,
	    :target:`aBdeC48b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa9261f68a07d8d6dda48b4e02605a666>`      = dnnl_aBdeC48b4c,
	    :target:`aBdec64b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae0caa3185aa487ff179da0ddfc1c6004>`        = dnnl_aBdec64b,
	    :target:`aBdeC64b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa36ca21726a435a2663953e05ce345482>`      = dnnl_aBdeC64b2c,
	    :target:`aBdeC64b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2ac19b8b4befa8052222416fecaa2c4d>`      = dnnl_aBdeC64b4c,
	    :target:`adecb<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadeeaafeee929b4f6bfce4d5828267516>`           = dnnl_adecb,
	    :target:`adeCb2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6c540e07c5a6499622b760134d18d35f>`         = dnnl_adeCb2c,
	    :target:`adeCb4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9d91394f1e793f72469d34fa28179c0e>`         = dnnl_adeCb4c,
	    :target:`Acdeb32a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae8d1e2538952bec6cd6c122e8ffc9b35>`        = dnnl_Acdeb32a,
	    :target:`AcdeB32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1746395db0cc57ade8b765cb6707d0a8>`      = dnnl_AcdeB32a2b,
	    :target:`AcdeB32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa9b41bade96c02f8a6f80dfd427be20e>`      = dnnl_AcdeB32a4b,
	    :target:`Acdeb48a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa90c6892259c2904cbb9e1b665f41f2cf>`        = dnnl_Acdeb48a,
	    :target:`AcdeB48a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae4ed1d0100ee70c79e130a1ccf6f318f>`      = dnnl_AcdeB48a2b,
	    :target:`AcdeB48a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5f958619bc5c971e7fa7a5361a9e5cf7>`      = dnnl_AcdeB48a4b,
	    :target:`Acdeb64a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac6dbc575e5a1bb737a7f1ac574521275>`        = dnnl_Acdeb64a,
	    :target:`AcdeB64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1df7e921cb80a9050c03a909a856c79a>`      = dnnl_AcdeB64a2b,
	    :target:`AcdeB64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa63c323b7505665d370fdf374a5c75b52>`      = dnnl_AcdeB64a4b,
	    :target:`cdeBa2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab80970b24894885a4653dd311354d8d5>`         = dnnl_cdeBa2b,
	    :target:`cdeBa4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa24951ab690a19d035b4d5e5c03efcca5>`         = dnnl_cdeBa4b,
	    :target:`aBdefc32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa38596d3e36ba5f28870ee0290f3ebde3>`       = dnnl_aBdefc32b,
	    :target:`aBdefC32b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5eb1b6d656def244cf79efd862d592b5>`     = dnnl_aBdefC32b2c,
	    :target:`aBdefC32b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facc70aa8b20a5b4aca24d4f73255bf9b0>`     = dnnl_aBdefC32b4c,
	    :target:`aBdefc48b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa90eb85abdcc09e08954163319f470730>`       = dnnl_aBdefc48b,
	    :target:`aBdefC48b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3197f9d7913268ab51a6c4af322df471>`     = dnnl_aBdefC48b2c,
	    :target:`aBdefC48b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa60e83a8845e572bb4d910a0128352b3a>`     = dnnl_aBdefC48b4c,
	    :target:`aBdefc64b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa639d53f21fa29bdc3e968b2cc328656e>`       = dnnl_aBdefc64b,
	    :target:`aBdefC64b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa89ee2ce9916e48b517c2456a9eaf7353>`     = dnnl_aBdefC64b2c,
	    :target:`aBdefC64b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0ad31a1b43ed35999247dfc6106c9df7>`     = dnnl_aBdefC64b4c,
	    :target:`adefcb<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad4c53ef898ae4e9d3ca4b2ac996e2c0e>`          = dnnl_adefcb,
	    :target:`adefCb2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2b683ca04cbad049dd7ed55ea5b9b895>`        = dnnl_adefCb2c,
	    :target:`adefCb4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa152ed0d7121dd21ccacda7c0c78e6d2f>`        = dnnl_adefCb4c,
	    :target:`ABc32a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79f50c532ac03e7f860c80e7bbb8cb02>`       = dnnl_ABc32a32b,
	    :target:`BAc8a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1a86bbab6f7963fa4f5573c96cb84740>`      = dnnl_BAc8a16b2a,
	    :target:`BAcd8a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7779cc1e884bbe11c341dc43a30d8c58>`     = dnnl_BAcd8a16b2a,
	    :target:`ABcde8a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2d8eb644709ecb6e80d3f3bf368bbbfd>`    = dnnl_ABcde8a16b2a,
	    :target:`aCBd8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab454f12090dc922df1a9a483463155c6>`     = dnnl_aCBd8b16c2b,
	    :target:`BAcde8a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa61d15e92362ba4d6fbb148f76cc46a7d>`    = dnnl_BAcde8a16b2a,
	    :target:`aCBde8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7fa8c9f686c19498165af5961b8a0f40>`    = dnnl_aCBde8b16c2b,
	    :target:`ABcde32a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa215ff6dcf189b30f0213be627b28688e>`     = dnnl_ABcde32a32b,
	    :target:`ABc4a8b8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab3ba545a6be7a4bf66cfe21ad5ed7bcf>`     = dnnl_ABc4a8b8a4b,
	    :target:`ABcde4a8b8a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0a22bece95339b762f9c32ae1776a90a>`   = dnnl_ABcde4a8b8a4b,
	    :target:`BAc4b8a8b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf6a388cf1ab1328bfcc120944932d482>`     = dnnl_BAc4b8a8b4a,
	    :target:`BAcd4b8a8b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab4c6ea291d88564a6346adf734d37899>`    = dnnl_BAcd4b8a8b4a,
	    :target:`BAcde4b8a8b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa046f440f0bc0a617a5bd388ad4462eec>`   = dnnl_BAcde4b8a8b4a,
	    :target:`aBCd4b8c8b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9d284e0d13b80123b349794ef1872110>`    = dnnl_aBCd4b8c8b4c,
	    :target:`aBCdef4b8c8b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa31507c9899612511c1abbf0f46a398fc>`  = dnnl_aBCdef4b8c8b4c,
	    :target:`aBCdef8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2bc45ebdb0bb94d63d8ada7d0ee3ef16>`   = dnnl_aBCdef8b16c2b,
	    :target:`aCBdef8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa64d213720b10ae0191116a9b1ee427cd>`   = dnnl_aCBdef8b16c2b,
	    :target:`aBdC16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa78a5505b26d08d0b4c8c51e1367b4454>`       = dnnl_aBdC16b2c,
	    :target:`aBdeC16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad2ec22aedf4a490248bfe5beef03eb9b>`      = dnnl_aBdeC16b2c,
	    :target:`aBdefC16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8149421ef23728049be98b1f2deb8cf4>`     = dnnl_aBdefC16b2c,
	    :target:`aBedc16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf5900d4c08a8706bca0ba2f31722b93d>`        = dnnl_aBedc16b,
	    :target:`AcB16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faed0ce859f017d2c1ece403b77d962ec6>`        = dnnl_AcB16a2b,
	    :target:`AcdB16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa41f54ee30b3afc2cae35de2c2123b5e4>`       = dnnl_AcdB16a4b,
	    :target:`AcdeB16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9f84d9ddba9ca777a13404aabe9bf65d>`      = dnnl_AcdeB16a2b,
	    :target:`Adcb16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa04ebd79eb4e37d8c1c6a3b60ace4ca55>`         = dnnl_Adcb16a,
	    :target:`aCBd4c8b8c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad8fa2b64d6bc6fe449a030af9a0d0708>`    = dnnl_aCBd4c8b8c4b,
	    :target:`aCBde4c8b8c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7b9dc7609ea2a43fd53cee628be250c8>`   = dnnl_aCBde4c8b8c4b,
	    :target:`aCBdef4c8b8c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3ed4cd161f8b6ad0f2b4bbd4a9636661>`  = dnnl_aCBdef4c8b8c4b,
	    :target:`ABc32a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4ae2f1ea9b7b6109186cd1490202b1d6>`       = dnnl_ABc32a16b,
	    :target:`ABcd16a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa13ff6367039f38e9f58b3f5a67a5061c>`      = dnnl_ABcd16a32b,
	    :target:`ABcd32a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac04990846ec1c8dfc588361a0117ce55>`      = dnnl_ABcd32a16b,
	    :target:`ABcde32a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8bd76bcf4e0d7db26629ae216f9d82c6>`     = dnnl_ABcde32a16b,
	    :target:`AB48a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa24adc9ebffd98ca13bdd3e253621fc50>`        = dnnl_AB48a16b,
	    :target:`AB48a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafa7a58335726b2d9f28e179a58f5159b>`        = dnnl_AB48a32b,
	    :target:`ABc40a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa826d7bc2b949c311682a0d2fceaa8982>`       = dnnl_ABc40a16b,
	    :target:`ABc40a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa53ab793b86a42a4f62e0b3147b7d30ee>`       = dnnl_ABc40a32b,
	    :target:`aBC48b16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1460971bf824af5bd563f6cc627e4098>`       = dnnl_aBC48b16c,
	    :target:`aBC48b32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa283aa259ca474f537c605d90d5362b29>`       = dnnl_aBC48b32c,
	    :target:`ABcd40a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3748fa3e5b9bc3119734dd9baba8d368>`      = dnnl_ABcd40a16b,
	    :target:`ABcd40a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab3557ea0565925179b348a17ad324f11>`      = dnnl_ABcd40a32b,
	    :target:`BA16a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa796b937255c3a372bcb9f6f39f0f9a82>`        = dnnl_BA16a16b,
	    :target:`BA16a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1bfbb80a60e2cf9dea27ce1d974de702>`        = dnnl_BA16a32b,
	    :target:`BA16a48b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1cc9ffce57195f8e815d4b411abc45f9>`        = dnnl_BA16a48b,
	    :target:`BA16a64b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa118de9a07d80c2260ba80e2f982de8d8>`        = dnnl_BA16a64b,
	    :target:`BA16a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa041f8ff98e4cfa03f29fc582e7674830>`      = dnnl_BA16a16b2a,
	    :target:`BA16a32b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae9d598ab2594ef0a69b009a7429d6d53>`      = dnnl_BA16a32b2a,
	    :target:`BA16a48b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facdd7b774b73e302a68bf7b5285f6d7eb>`      = dnnl_BA16a48b2a,
	    :target:`BA16a64b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabf04545a4df3f33ddbf281cecb7d9407>`      = dnnl_BA16a64b2a,
	    :target:`BA16a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0690c897878064abc208c2dcb906393b>`      = dnnl_BA16a16b4a,
	    :target:`BA16a32b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae1b636a4bbdaba68f6abd79bb1f669ba>`      = dnnl_BA16a32b4a,
	    :target:`BA16a48b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1c0bdb5f1cdc85538ed081b992d8ec6d>`      = dnnl_BA16a48b4a,
	    :target:`BA16a64b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa50904732710de5faff6d8668f1df47f9>`      = dnnl_BA16a64b4a,
	    :target:`decbA16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa837c021a2d20c12460a0b60c601bef0b>`        = dnnl_decbA16a,
	    :target:`decbA8a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7b8484afa2cd72f8875264e9c202b11f>`         = dnnl_decbA8a,
	    :target:`aCB16b16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac722cd7235d8bf40688c00158fff479c>`       = dnnl_aCB16b16c,
	    :target:`aCB16b32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa52e15af2235f0fee87df1d69f8f72705>`       = dnnl_aCB16b32c,
	    :target:`aCB16b48c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae85bdf3eab2e61b012bb231b2ff851af>`       = dnnl_aCB16b48c,
	    :target:`aCB16b64c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa598524c37f256fe9ed94e36c4960ab96>`       = dnnl_aCB16b64c,
	    :target:`aCB16b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf21a4dcc5d19241de775e606f2a9965b>`     = dnnl_aCB16b16c2b,
	    :target:`aCB16b32c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae9bf150790591115f5b33ff590bb5940>`     = dnnl_aCB16b32c2b,
	    :target:`aCB16b48c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa359d66b1b9ba9dcb4e55f61d46c7965a>`     = dnnl_aCB16b48c2b,
	    :target:`aCB16b64c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa00298715481748f675e1f5d59c56d044>`     = dnnl_aCB16b64c2b,
	    :target:`aCB16b16c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae5b0efebec34cbec7706d8b8f7d7e58e>`     = dnnl_aCB16b16c4b,
	    :target:`aCB16b32c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa64f2d0f99c0e54f3cd4faf3c5adaf041>`     = dnnl_aCB16b32c4b,
	    :target:`aCB16b48c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad2c33f27498c13226c409d5c7b8c3612>`     = dnnl_aCB16b48c4b,
	    :target:`aCB16b64c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faccd15a39ee216a4d770713d8db6e55bc>`     = dnnl_aCB16b64c4b,
	    :target:`format_tag_last<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3d78b4610da52a1b6bff0c40b5a02404>` = dnnl_format_tag_last,
	    :target:`nCdhw16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6861aaac0752cd0c618f68453416ea5e>`        = dnnl_nCdhw16c,
	    :target:`nCdhw4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa94adcd68022f332e02475001d4fc9201>`         = dnnl_nCdhw4c,
	    :target:`nCdhw8c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafced4a15e4acb04396e30b54955fa1a2>`         = dnnl_nCdhw8c,
	    :target:`nChw16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf0f101166691298d3aad79d7f347f2ef>`         = dnnl_nChw16c,
	    :target:`nChw4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad649ed0a6bf979419b24d6c7c58d97d7>`          = dnnl_nChw4c,
	    :target:`nChw8c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1c644d98a1e2e21c610ef0eadb24d094>`          = dnnl_nChw8c,
	    :target:`nCw16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa41e0c6dde8335c790dc7be6f28c5b845>`          = dnnl_nCw16c,
	    :target:`nCw4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1d5b916eac88a854f7596bb7778b5d84>`           = dnnl_nCw4c,
	    :target:`nCw8c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa53de045f004e830b06a37f3e59fbf29>`           = dnnl_nCw8c,
	    :target:`NCw16n16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab1b9bf96de481c827f3403ba3e9c4d81>`       = dnnl_NCw16n16c,
	    :target:`NChw16n16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac37f342bac76c6566daec13bd5106594>`      = dnnl_NChw16n16c,
	    :target:`NCdhw16n16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa62a823240787c4b65f45570772156c18>`     = dnnl_NCdhw16n16c,
	    :target:`NCdhw32n32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa39d485639568c18a89ff66c093b63a47>`     = dnnl_NCdhw32n32c,
	    :target:`NChw32n32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad37ac31732879264c2bce09f7caea13b>`      = dnnl_NChw32n32c,
	    :target:`IOhw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac42ef6e97425f3e97913783169c2261e>`      = dnnl_IOhw16i16o,
	    :target:`OI16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa20f4926c6842db05b8ee80f559d5cad>`        = dnnl_OI16i16o,
	    :target:`OI16i32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3767db3c50ee72c8976d8bbb61b17dd6>`        = dnnl_OI16i32o,
	    :target:`OI16i64o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafee7cdbca4f5a3f7035023dce010587f>`        = dnnl_OI16i64o,
	    :target:`OI8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4aeb984eee437eef8482d96e67cf2be1>`       = dnnl_OI8i16o2i,
	    :target:`OI8i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeea74c26057470bb593c95b06cca29bc>`       = dnnl_OI8i32o2i,
	    :target:`OI8i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa36e18f085d97af75c3750a505c37b7c9>`       = dnnl_OI8i64o2i,
	    :target:`OI4i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa08aa4df5946c40b552128721235749c0>`       = dnnl_OI4i16o4i,
	    :target:`OI4i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5d53bcff3d0af974ea8310f15ba3cc82>`       = dnnl_OI4i32o4i,
	    :target:`OI4i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3f880ef73adddab01814022a58dbaef0>`       = dnnl_OI4i64o4i,
	    :target:`Ohwi32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa90759a72cc812c5345cf587c90cc9268>`         = dnnl_Ohwi32o,
	    :target:`IOdhw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa621bb2ac71efada565d275934a4e6c15>`     = dnnl_IOdhw16i16o,
	    :target:`gIOhw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa36c0eda0e14719c161c8181ad2d4b942>`     = dnnl_gIOhw16i16o,
	    :target:`gOhwi32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faee023523d45caa828c6e5bbe8f3d188a>`        = dnnl_gOhwi32o,
	    :target:`Goidhw16g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa84f69746b0fc107edf92275009c2d4b0>`       = dnnl_Goidhw16g,
	    :target:`IOw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaf551a573134b0970e281d247677749e>`       = dnnl_IOw16o16i,
	    :target:`OIw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaa209bd1ce63baa9d9fdc73c5f557b8c>`       = dnnl_OIw16i16o,
	    :target:`OIw16i32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faad31f51070974f4aa90f3596ea351846>`       = dnnl_OIw16i32o,
	    :target:`OIw16i64o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadff5196eb3e24bc356b2b2830d6133ea>`       = dnnl_OIw16i64o,
	    :target:`IOw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafcadbccc60230a29d1e86a0414bb72f7>`       = dnnl_IOw16i16o,
	    :target:`gIOw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa24645ce5c92a6aaa5fe12cc33f5eb23d>`      = dnnl_gIOw16i16o,
	    :target:`OIw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa76ecfa744a9efaa40822abc529e73a42>`       = dnnl_OIw16o16i,
	    :target:`Oiw16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf4d5683ba89a1401146d0c119c2236f1>`          = dnnl_Oiw16o,
	    :target:`OIw4i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa018de1f2c5427248a698153211901ff3>`      = dnnl_OIw4i16o4i,
	    :target:`OIw4i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7ea4d51e08dcc89fb2a7254aebe7244c>`      = dnnl_OIw4i32o4i,
	    :target:`OIw4i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa03bbe39d92e11d3ab6184c93c40433e2>`      = dnnl_OIw4i64o4i,
	    :target:`OIw2i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae9b53e6e72c82a9600b6f805add71b57>`       = dnnl_OIw2i8o4i,
	    :target:`OIw4i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5db05d7bcc70e711661db2d3b1d4415f>`         = dnnl_OIw4i4o,
	    :target:`OIw4o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab967e60f35d689a6838209e16ef7481d>`         = dnnl_OIw4o4i,
	    :target:`Oiw4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf4a06d521a3b1a21cc2b6c51c5e3d757>`           = dnnl_Oiw4o,
	    :target:`OIw8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa0bfb3e88be7ef569a33f9206f36c5cb>`      = dnnl_OIw8i16o2i,
	    :target:`OIw8i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa71cf41159db972d3e995fbc85390b0e3>`      = dnnl_OIw8i32o2i,
	    :target:`OIw8i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafbb983b22c56d79d39c07b08f0f49f49>`      = dnnl_OIw8i64o2i,
	    :target:`OIw8i8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa11734e3a192668eae54d0b55a3a5f900>`         = dnnl_OIw8i8o,
	    :target:`OIw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae100564e5659e696df92c19d87514076>`      = dnnl_OIw8o16i2o,
	    :target:`OIw8o8i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9bf4142ef7541fe2440cd339d1fd78a7>`         = dnnl_OIw8o8i,
	    :target:`OIw8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa94bae6f8097adc154129b2fffe515971>`         = dnnl_OIw8o4i,
	    :target:`OIw16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2e03aac84ecfed291a9c1fbce08c2597>`     = dnnl_OIw16i16o4i,
	    :target:`OIw16i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa30307d7a5170938777204cd99756ad6d>`     = dnnl_OIw16i32o4i,
	    :target:`OIw16i48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa45bc2af26cc9b09100f141ef184ab393>`     = dnnl_OIw16i48o4i,
	    :target:`OIw16i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2e2073e340ab15f43bc4b8ebb4e4b7de>`     = dnnl_OIw16i64o4i,
	    :target:`OIw16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaafb77e74b085bb1724eecdb25d66ef7>`     = dnnl_OIw16i16o2i,
	    :target:`OIw16i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabae977d670c572d1a37a455309bb2031>`     = dnnl_OIw16i32o2i,
	    :target:`OIw16i48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae118038939e5f80bc2e3bb7645465429>`     = dnnl_OIw16i48o2i,
	    :target:`OIw16i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa101d2cc6c72193586ca514f1f454f4a4>`     = dnnl_OIw16i64o2i,
	    :target:`OIw16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa55832b1e2f4403bf68ccaf8763764b62>`     = dnnl_OIw16o16i2o,
	    :target:`Owi16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa09af0573ac44a98805eab5c635c173c2>`          = dnnl_Owi16o,
	    :target:`OwI16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa45362250f163d3297c53c7c38c99fa45>`        = dnnl_OwI16o2i,
	    :target:`Iwo16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabe9729cd0b78f6fc0d3727862b0f036a>`          = dnnl_Iwo16i,
	    :target:`IwO16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf4573e5a811ad6662d6867e2288302fc>`        = dnnl_IwO16i2o,
	    :target:`IwO16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9f82d70b5d6aae687b9ed794a6917ade>`        = dnnl_IwO16i4o,
	    :target:`Owi4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa12204086f4262b5c2aa029f43d30eec6>`           = dnnl_Owi4o,
	    :target:`Owi8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa09c453d0c74291741493768ca45cf765>`           = dnnl_Owi8o,
	    :target:`IOhw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2be6713b0c2d362ff6e8df2a611fa876>`      = dnnl_IOhw16o16i,
	    :target:`Ohwi16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa25e357d03f1dfdc48bd49f573f5ec871>`         = dnnl_Ohwi16o,
	    :target:`OhwI16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae9daae9f14357c81ee8c3c1f444e5cb4>`       = dnnl_OhwI16o2i,
	    :target:`Ihwo16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79a4b7a419280a78880fff43e3e5e3cf>`         = dnnl_Ihwo16i,
	    :target:`IhwO16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5fa560f5f1085739da23bc0abcd70e4f>`       = dnnl_IhwO16i2o,
	    :target:`IhwO16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9ae607e487014cbb6eded6a812781778>`       = dnnl_IhwO16i4o,
	    :target:`Ohwi4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafb9c8980396d6ea229dac5f83b191b86>`          = dnnl_Ohwi4o,
	    :target:`Ohwi8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadab0df66a9483754bd555a390cb4ef0b>`          = dnnl_Ohwi8o,
	    :target:`OIhw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3376ab492bc782eecfac5dfe1671d847>`      = dnnl_OIhw16i16o,
	    :target:`OIhw16i32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf37495198740238d38ca87fcefed6fe5>`      = dnnl_OIhw16i32o,
	    :target:`OIhw16i64o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa65f0f07f0afac2503de570b1fdae82ea>`      = dnnl_OIhw16i64o,
	    :target:`OIhw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae31366e22f0970dba5a4ca4d36e9a2d8>`      = dnnl_OIhw16o16i,
	    :target:`Oihw16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa435ad696b07dd91fdb8637385284df1b>`         = dnnl_Oihw16o,
	    :target:`OIhw4i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa881d72f28db8825143b916c79b9e78b5>`     = dnnl_OIhw4i16o4i,
	    :target:`OIhw4i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf925c18329b1f127380c0c0bda25dd80>`     = dnnl_OIhw4i32o4i,
	    :target:`OIhw4i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8453fb39189b12636585f4e600d803c6>`     = dnnl_OIhw4i64o4i,
	    :target:`OIhw4i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa91b0da7423d9eca50e9abb61a0cd3fa2>`        = dnnl_OIhw4i4o,
	    :target:`OIhw4o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6e08a5845009f77deaf6701aafaf53cc>`        = dnnl_OIhw4o4i,
	    :target:`Oihw4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac4942f3bd7aacaf3dd5aae4b79ef7093>`          = dnnl_Oihw4o,
	    :target:`OIhw8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa971c53f0333bb7bfe14240a2a7bb072>`     = dnnl_OIhw8i16o2i,
	    :target:`OIhw8i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa959c91fac1a790ca2e2c99988bfe4c7c>`     = dnnl_OIhw8i32o2i,
	    :target:`OIhw8i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa13c39fbd748dcede9adba437f8153e18>`     = dnnl_OIhw8i64o2i,
	    :target:`OIhw8i8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa94359730df7ca59e037041533a9ac909>`        = dnnl_OIhw8i8o,
	    :target:`OIhw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7bea00fcbf4c7b35c69ef47c16172149>`     = dnnl_OIhw8o16i2o,
	    :target:`OIhw8o8i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faba8ddd99732f6f7cee745a1a594a7efa>`        = dnnl_OIhw8o8i,
	    :target:`OIhw8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab024c7313e8ccc2b19c72ab7c72301b>`        = dnnl_OIhw8o4i,
	    :target:`OIhw2i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac6c1c664f1bfe641318406a9b7b2f414>`      = dnnl_OIhw2i8o4i,
	    :target:`IOdhw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79f40a00b5aeaf3b7fc0d1af838bda9f>`     = dnnl_IOdhw16o16i,
	    :target:`Odhwi16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadfe8f0bea94fb653d49b7e97635b34a2>`        = dnnl_Odhwi16o,
	    :target:`OdhwI16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaf7904d3d79d89c1c82db6959a01b446>`      = dnnl_OdhwI16o2i,
	    :target:`Idhwo16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa97466a82db1853d35b4cefedba9b1029>`        = dnnl_Idhwo16i,
	    :target:`IdhwO16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa26b7e5468c8468ac8c3cdd69135aed5e>`      = dnnl_IdhwO16i2o,
	    :target:`IdhwO16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa207c2f56caf4c3aa9bd58a9539cbf041>`      = dnnl_IdhwO16i4o,
	    :target:`Odhwi4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8982e59611d1498d8f0b66ea5e34b00d>`         = dnnl_Odhwi4o,
	    :target:`Odhwi8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa34c681ee06dd9199cdfba6ce995e92f0>`         = dnnl_Odhwi8o,
	    :target:`OIdhw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7364c31504080477789c9f0c02557409>`     = dnnl_OIdhw16i16o,
	    :target:`OIdhw16i32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e9536b869e23018d88bd515103aa0fe>`     = dnnl_OIdhw16i32o,
	    :target:`OIdhw16i64o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac630ab9e3f0359c50e3a17fe3dbbece1>`     = dnnl_OIdhw16i64o,
	    :target:`OIdhw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabd4b3b295e376c89af8dfd2eb44e1b4f>`     = dnnl_OIdhw16o16i,
	    :target:`OIdhw16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa839c7220133e1be11d19d77b3429e6ea>`   = dnnl_OIdhw16o16i2o,
	    :target:`Oidhw16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa44b6c29022e2781ec5cbee750d888cea>`        = dnnl_Oidhw16o,
	    :target:`OIdhw4i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0dcda1b9e35dafea4ad166e594e28dda>`       = dnnl_OIdhw4i4o,
	    :target:`OIdhw4o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa5620d9722a779df82eaaf2d80c6d418>`       = dnnl_OIdhw4o4i,
	    :target:`Oidhw4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad2531b0046f779044f5130c362aed248>`         = dnnl_Oidhw4o,
	    :target:`OIdhw8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faac7db4ea6b9389481bc6797bc39016ee>`    = dnnl_OIdhw8i16o2i,
	    :target:`OIdhw8i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0f3de80c33805df1ef23459587417420>`    = dnnl_OIdhw8i32o2i,
	    :target:`OIdhw8i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facc02487e23639e2df4b43efdc95bfd10>`    = dnnl_OIdhw8i64o2i,
	    :target:`OIdhw4i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae529f09596c340395ed8dbb0096e987a>`    = dnnl_OIdhw4i16o4i,
	    :target:`OIdhw16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae64eef219b16629effc1540a4be01eef>`   = dnnl_OIdhw16i16o4i,
	    :target:`OIdhw16i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4b3cbaf66d6b2a675338625f21defc49>`   = dnnl_OIdhw16i32o4i,
	    :target:`OIdhw16i48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6dc7cbec873e4ac163841436f1a0062d>`   = dnnl_OIdhw16i48o4i,
	    :target:`OIdhw16i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf606a911c9efeab0cf37bd98d2a34ce5>`   = dnnl_OIdhw16i64o4i,
	    :target:`OIdhw16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa47fc54c58e425c9cb02f4dac3b303b6a>`   = dnnl_OIdhw16i16o2i,
	    :target:`OIdhw16i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faedb5f17773e719594743376011151c2b>`   = dnnl_OIdhw16i32o2i,
	    :target:`OIdhw16i48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa724ebb4f0bc07cf3735bfb640e07449>`   = dnnl_OIdhw16i48o2i,
	    :target:`OIdhw16i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab66e948aad2e205157f39a050f085ce2>`   = dnnl_OIdhw16i64o2i,
	    :target:`OIdhw4i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa928acc956c3370b8acae8085c7cfa06>`    = dnnl_OIdhw4i32o4i,
	    :target:`OIdhw4i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3ca72a666e05dfc705f684c9c38ba24a>`    = dnnl_OIdhw4i64o4i,
	    :target:`OIdhw2i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8fd16cb655cf6137169d7f0843f14faa>`     = dnnl_OIdhw2i8o4i,
	    :target:`OIdhw8i8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa90ecbec00f847951c1b2114afd877285>`       = dnnl_OIdhw8i8o,
	    :target:`OIdhw8o8i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa96f841238973a6d8d28299377a99981f>`       = dnnl_OIdhw8o8i,
	    :target:`OIdhw8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3da48b570ada0a815198f8b80d366868>`       = dnnl_OIdhw8o4i,
	    :target:`gIOw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafa027d58629c684b73bcc1714a98e2c3>`      = dnnl_gIOw16o16i,
	    :target:`gOIw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa69ff0c51b99e2b3a2242c5573db75680>`      = dnnl_gOIw16i16o,
	    :target:`gOIw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafa987d0f6bf33d361270f4fd0e4b6b5c>`      = dnnl_gOIw16o16i,
	    :target:`gOiw16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa372a20447ce0335fe5c4f0126fb1d041>`         = dnnl_gOiw16o,
	    :target:`gOIw4i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa68f856a42aaec4e873e1086429cde66f>`     = dnnl_gOIw4i16o4i,
	    :target:`gOIw2i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa05516d69a71416e700d0eda0feb16524>`      = dnnl_gOIw2i8o4i,
	    :target:`gOIw4i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad056246fae702dc91c93b0acb9c5b976>`        = dnnl_gOIw4i4o,
	    :target:`gOIw4o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa696b071737d3766203b79539d3973cad>`        = dnnl_gOIw4o4i,
	    :target:`gOiw4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9d932a4601f34d94dd9cc0933f4ab154>`          = dnnl_gOiw4o,
	    :target:`gOIw8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faceb58ad5277238cb287876d1e6072736>`     = dnnl_gOIw8i16o2i,
	    :target:`gOIw8i8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9063aabfe5d944e94135c4cb67ba2584>`        = dnnl_gOIw8i8o,
	    :target:`gOIw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac1793fd656f546c03dea8bc7baa1f305>`     = dnnl_gOIw8o16i2o,
	    :target:`gOIw8o8i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafa861d36e362fc22ef0bb6d513dd9a89>`        = dnnl_gOIw8o8i,
	    :target:`gOIw8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadc01922e085179440158604b5a20e724>`        = dnnl_gOIw8o4i,
	    :target:`gOIw16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3664a511532c9182bb8ed64ceef6b993>`    = dnnl_gOIw16i16o4i,
	    :target:`gOIw16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad4e8eb5939cad18c6bc190aa461b6ef6>`    = dnnl_gOIw16i16o2i,
	    :target:`gOIw16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab9763f89de724c2b695756571459785a>`    = dnnl_gOIw16o16i2o,
	    :target:`gOwi16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4adc998e59225b906421a37a53881137>`         = dnnl_gOwi16o,
	    :target:`gOwI16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0c974071f8ea7be19b95a8326f8be906>`       = dnnl_gOwI16o2i,
	    :target:`gIwo16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faba79d5a894c1c87a15399cffd6e0b746>`         = dnnl_gIwo16i,
	    :target:`gIwO16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad0ff4be38dfccfe42e78bc342c1d469e>`       = dnnl_gIwO16i2o,
	    :target:`gIwO16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadb39ee7d3199fbe45234aa1666bd0512>`       = dnnl_gIwO16i4o,
	    :target:`gOwi4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3ba5d84f4cb5f66841c2dabc7fe4fc6c>`          = dnnl_gOwi4o,
	    :target:`gOwi8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faad638741237c3f4c2af1964459ef0997>`          = dnnl_gOwi8o,
	    :target:`Goiw8g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2f0595f2c9ae8dbb639505e73eed7b5e>`          = dnnl_Goiw8g,
	    :target:`Goiw16g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8ce71d2d8663566ac8a3dd50e283b394>`         = dnnl_Goiw16g,
	    :target:`gIOhw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab4329019d9beb50e7c0d6b4149b7a842>`     = dnnl_gIOhw16o16i,
	    :target:`gOhwi16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4d7fce02dd37cb1593cb1ad7b828ee9b>`        = dnnl_gOhwi16o,
	    :target:`gOhwI16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7d4c0a56b0843125e1ac44b5b1ee4b12>`      = dnnl_gOhwI16o2i,
	    :target:`gIhwo16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafa4de44103e13b28a9e863aabd945240>`        = dnnl_gIhwo16i,
	    :target:`gIhwO16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6169243ea609e826b0f900045879969b>`      = dnnl_gIhwO16i2o,
	    :target:`gIhwO16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7342c4cb60d2b9e7fddf02d9b30304ad>`      = dnnl_gIhwO16i4o,
	    :target:`gOhwi4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5b0706b3d8de37265ef1ca7da0c95db0>`         = dnnl_gOhwi4o,
	    :target:`gOhwi8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa054e5fad6ff701447c1f04372f214e0b>`         = dnnl_gOhwi8o,
	    :target:`Goihw16g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa159b2a2e896f7484ddcf9c8e4038eca0>`        = dnnl_Goihw16g,
	    :target:`gOIhw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1ca4dda6de1722abc8ef8bdfb642a4b0>`     = dnnl_gOIhw16i16o,
	    :target:`gOIhw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7d79e7d651180b0e8817bcf913e8663e>`     = dnnl_gOIhw16o16i,
	    :target:`gOihw16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faed5c4dae27a0a62c9f924075927aa154>`        = dnnl_gOihw16o,
	    :target:`gOIhw4i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafeec7a3e52ae9e28b19c65f53edc7808>`    = dnnl_gOIhw4i16o4i,
	    :target:`gOIhw2i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faebd09bf00bcf64d1c79ae144e7ebe5e2>`     = dnnl_gOIhw2i8o4i,
	    :target:`gOIhw4i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa57a6611ccc3768ae5c7bdab2fbef26ac>`       = dnnl_gOIhw4i4o,
	    :target:`gOIhw4o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa57bdbc7a7cf1be84b5ba129959280ce4>`       = dnnl_gOIhw4o4i,
	    :target:`gOihw4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4b845a934bc59efddf9ef1045cc0b7d9>`         = dnnl_gOihw4o,
	    :target:`Goihw8g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaa36c1e32c51ddec41593ae43f9e080a>`         = dnnl_Goihw8g,
	    :target:`gOIhw8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa531e0d8539a3732cf3fb2cd62f0ad887>`    = dnnl_gOIhw8i16o2i,
	    :target:`gOIhw8i8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1d727cbaa05cb6f4e793e447bd7aee0d>`       = dnnl_gOIhw8i8o,
	    :target:`gOIhw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7e2e3ba3efc10e24283753abafb2afdc>`    = dnnl_gOIhw8o16i2o,
	    :target:`OIw4o8i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaba5aa6bc9babf518224361d68731126>`     = dnnl_OIw4o8i8o4i,
	    :target:`OIdhw4o8i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7664b9d3e206fcc6001fa5d83d279e8c>`   = dnnl_OIdhw4o8i8o4i,
	    :target:`OIhw4o8i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facd9391bd093a92195718626d6b3b9829>`    = dnnl_OIhw4o8i8o4i,
	    :target:`OIhw2o8i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facfdd91de4104243541be095b2f2e0794>`    = dnnl_OIhw2o8i8o2i,
	    :target:`gOIw4o8i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa846190f2375e258e71f76fa74daed9c6>`    = dnnl_gOIw4o8i8o4i,
	    :target:`gOIdhw4o8i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa808f7d58a2d7202155e4901c9d398cd3>`  = dnnl_gOIdhw4o8i8o4i,
	    :target:`gOIhw4o8i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5acdc941617a90dd95110fb1d71830f4>`   = dnnl_gOIhw4o8i8o4i,
	    :target:`gOIhw2o8i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa905367adecd07519b5e12a8da02d4e04>`   = dnnl_gOIhw2o8i8o2i,
	    :target:`OIhw16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa80cc1614ef1ffce929046674c8ad83ca>`    = dnnl_OIhw16i16o4i,
	    :target:`OIhw16i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8c4e68ae2eb8c655338d94cd7deb952f>`    = dnnl_OIhw16i32o4i,
	    :target:`OIhw16i48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facedee44ae4a2ad7738c3dfc57e11606f>`    = dnnl_OIhw16i48o4i,
	    :target:`OIhw16i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac7cc4094921e1bf91f3570ca5d5531f4>`    = dnnl_OIhw16i64o4i,
	    :target:`OIhw16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7d67a3c0f3e6c533a77a894409017b55>`    = dnnl_OIhw16i16o2i,
	    :target:`OIhw16i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3c2fb01721999cefb3848b9fe8a1e5eb>`    = dnnl_OIhw16i32o2i,
	    :target:`OIhw16i48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0fdb34ba231cc0dd5d3a696440a06073>`    = dnnl_OIhw16i48o2i,
	    :target:`OIhw16i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa511a9438fd27613cff5f98e553fc727a>`    = dnnl_OIhw16i64o2i,
	    :target:`OIhw16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac16315623990f1d8ad2834d222aa6f2c>`    = dnnl_OIhw16o16i2o,
	    :target:`gOIhw16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa46da8d87f79318580c735c241449d9a5>`   = dnnl_gOIhw16i16o4i,
	    :target:`gOIhw16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2d3b0878dfb60fd641359da8f0e138e0>`   = dnnl_gOIhw16i16o2i,
	    :target:`gOIhw16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5637ff54e9868af2a907bef900b56bba>`   = dnnl_gOIhw16o16i2o,
	    :target:`gOIhw8o8i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9ba3ca93854586fcd7f3c158efd869bf>`       = dnnl_gOIhw8o8i,
	    :target:`gOIhw8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7bbe7df155560d5d88c7a7feb2801ce1>`       = dnnl_gOIhw8o4i,
	    :target:`gIOdhw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa747b1b2818d0de608f236e8815ffe192>`    = dnnl_gIOdhw16i16o,
	    :target:`gIOdhw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa33beeac1eb5bc81f5c2259347e8ebd6b>`    = dnnl_gIOdhw16o16i,
	    :target:`gOdhwi16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa81e4899579f993df8c2f7d4c7fc22847>`       = dnnl_gOdhwi16o,
	    :target:`gOdhwI16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6b6699fd9768a197cd07ef9ffaf56795>`     = dnnl_gOdhwI16o2i,
	    :target:`gIdhwo16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8cfdcb6408ae618e1196df1bf4fea1ef>`       = dnnl_gIdhwo16i,
	    :target:`gIdhwO16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac9ff661d3157ddbbd490a0a1888399e6>`     = dnnl_gIdhwO16i2o,
	    :target:`gIdhwO16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf9c1d923b032dab9c5b6ea7f76189bb5>`     = dnnl_gIdhwO16i4o,
	    :target:`gOdhwi4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5eb1abc93e2027336989730c391a5498>`        = dnnl_gOdhwi4o,
	    :target:`gOdhwi8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafdc86aad95764f0aae1ad15c60e3da80>`        = dnnl_gOdhwi8o,
	    :target:`gOIdhw16i16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa08acb28c1b80aa888ec6bb555937299f>`    = dnnl_gOIdhw16i16o,
	    :target:`gOIdhw16o16i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa38c03747d6b033b007ebfbe0fa89a1ff>`    = dnnl_gOIdhw16o16i,
	    :target:`gOIdhw16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa67a83c2e89e7dcbb41473f13d0e46846>`  = dnnl_gOIdhw16o16i2o,
	    :target:`gOidhw16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab3ad516ad551b8436335045a1bfcb057>`       = dnnl_gOidhw16o,
	    :target:`gOIdhw4i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3ff966b4edaf4c6a88815e88db0d3e80>`      = dnnl_gOIdhw4i4o,
	    :target:`gOIdhw4o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa352f19cfe30ed59f5877379c3cfb67b5>`      = dnnl_gOIdhw4o4i,
	    :target:`gOidhw4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa96950c8210b11b4cbf8e4d8f3aefac55>`        = dnnl_gOidhw4o,
	    :target:`gOIdhw8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad464a542f1137020b390aa6e0348b882>`   = dnnl_gOIdhw8i16o2i,
	    :target:`gOIdhw4i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5131f96b95382e84e1d26e1e0352ea44>`   = dnnl_gOIdhw4i16o4i,
	    :target:`gOIdhw16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad872cd1610e3ba13e255a16cf648aabd>`  = dnnl_gOIdhw16i16o4i,
	    :target:`gOIdhw16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4fa0dd9f1f5d47738114e8ebff370cc0>`  = dnnl_gOIdhw16i16o2i,
	    :target:`gOIdhw2i8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf2915dd18193aa6847e79b393f077096>`    = dnnl_gOIdhw2i8o4i,
	    :target:`gOIdhw8i8o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa07a5d807871093893d396b7e7e8eaddb>`      = dnnl_gOIdhw8i8o,
	    :target:`gOIdhw8o8i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae59e06dff785ca742a1d78147e7cc267>`      = dnnl_gOIdhw8o8i,
	    :target:`gOIdhw8o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa91cbcc7d523c9626b053a9af4bf2d17c>`      = dnnl_gOIdhw8o4i,
	    :target:`gOIw2i4o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac853483610c6e0be8e4a67b71bcefa06>`      = dnnl_gOIw2i4o2i,
	    :target:`gOIhw2i4o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab02e15a301a65c885e73875150421557>`     = dnnl_gOIhw2i4o2i,
	    :target:`gOIdhw2i4o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa919c6fe8d12deed5419dc785272a41e6>`    = dnnl_gOIdhw2i4o2i,
	    :target:`gOIw2o4i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab8d523262b77ad2c550860fa03ed318>`      = dnnl_gOIw2o4i2o,
	    :target:`gOIhw2o4i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa26fc0a2cce0ced17c59a50ba3ac1a21a>`     = dnnl_gOIhw2o4i2o,
	    :target:`gOIdhw2o4i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa18114293f6dbf838ea365244f3a84827>`    = dnnl_gOIdhw2o4i2o,
	    :target:`gOIw4i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6c59aca085940321bf972a1b950bdaa5>`      = dnnl_gOIw4i8o2i,
	    :target:`gOIhw4i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faad7251dbabb603a70d2b0a625bdb0cf7>`     = dnnl_gOIhw4i8o2i,
	    :target:`gOIdhw4i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0b611afe51d618c7e3b5b5580e13b451>`    = dnnl_gOIdhw4i8o2i,
	    :target:`gOIw4o8i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac8078e57989e2eb291f06869a5e8d343>`      = dnnl_gOIw4o8i2o,
	    :target:`gOIhw4o8i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5da1ba7266eff1873c0fa61001a972b1>`     = dnnl_gOIhw4o8i2o,
	    :target:`gOIdhw4o8i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facec5ae64c9907e7f0ad288c81ca90898>`    = dnnl_gOIdhw4o8i2o,
	    :target:`ldOi32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa34c727930109706841e55323e4595229>`         = abDc32d,
	    :target:`ldOI32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa877f0a4f6ec46dd64981822c30d92dc1>`       = abDC32d4c,
	    :target:`ldgOi32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4899d02491495b6f7faab51f0a7a9be3>`        = abdEc32e,
	    :target:`ldgOI32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf5429a17c0564d21eaaf245f1145d130>`      = abdEC32e2c,
	    :target:`ldgOI32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac036069426eaf684fefd7a2ae93b7286>`      = abdEC32e4c,
	    :target:`OwI16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab693117f7a0017cc7c3e59c167780662>`        = dnnl_OwI16o4i,
	    :target:`OhwI16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faed107119643ccf241765544d74c95ed1>`       = dnnl_OhwI16o4i,
	    :target:`gOwI16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5da33b1819cf4caa1c8b407adcce5be8>`       = dnnl_gOwI16o4i,
	    :target:`gOhwI16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab17446bbd941adef98e9b13a970da1c2>`      = dnnl_gOhwI16o4i,
	    :target:`OdhwI16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7df9be72fcc9c8d1c373829e84ab8a5d>`      = dnnl_OdhwI16o4i,
	    :target:`gOdhwI16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6955ffb856847f3c578225984edaa357>`     = dnnl_gOdhwI16o4i,
	    :target:`Owi32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab2f775eba2a8bf5f6bded75b1401342e>`          = dnnl_Owi32o,
	    :target:`OwI32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa735880d468c8a9db54ac6835d845435>`        = dnnl_OwI32o2i,
	    :target:`OwI32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0de1ed50ee9c5fb5d3819007fcb9686e>`        = dnnl_OwI32o4i,
	    :target:`Owi48o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa59da66e905e6f5b50b1daedea14e810b>`          = dnnl_Owi48o,
	    :target:`OwI48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa718d4f26951678ef98b7f207446fd767>`        = dnnl_OwI48o2i,
	    :target:`OwI48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6d7a9f4375c40c629308d8c018084f28>`        = dnnl_OwI48o4i,
	    :target:`Owi64o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa47e5eb03b0a7c4d86c87f98f210421b9>`          = dnnl_Owi64o,
	    :target:`OwI64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3e2776d3a81f27ff293ab8055e7af60b>`        = dnnl_OwI64o2i,
	    :target:`OwI64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3469fbbd834c332526af842faf1ece31>`        = dnnl_OwI64o4i,
	    :target:`Iwo32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9e0978fd91b4dc3b50d37454150c81ba>`          = dnnl_Iwo32i,
	    :target:`IwO32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf43b7853366f4e29ca9dab76819fec80>`        = dnnl_IwO32i2o,
	    :target:`IwO32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac45c52406932f6bd96f3e00ad470e434>`        = dnnl_IwO32i4o,
	    :target:`Iwo48i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9938f62e62e10b0efef6b873dde45d5b>`          = dnnl_Iwo48i,
	    :target:`IwO48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad8549d87c399b14dc1bf15f206d826c8>`        = dnnl_IwO48i2o,
	    :target:`IwO48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8414786e59b6abaf1df38593a045ffa6>`        = dnnl_IwO48i4o,
	    :target:`Iwo64i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faccc741ced929f4e3310af1ab6a06ede0>`          = dnnl_Iwo64i,
	    :target:`IwO64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa223d38ce5ae47b3d2f507822921030e>`        = dnnl_IwO64i2o,
	    :target:`IwO64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadcf17947c7fdaaa39054ed108b81bd54>`        = dnnl_IwO64i4o,
	    :target:`wIo2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadf19ed9a411bf2c34333e5cb36fbce24>`           = dnnl_wIo2i,
	    :target:`wIo4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad94592a22ce032391784ca8608c1a1bf>`           = dnnl_wIo4i,
	    :target:`gOwi32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf7f1a6f6599dc4586055ce55c611c3bc>`         = dnnl_gOwi32o,
	    :target:`gOwI32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad794cd419a5e9fb5a8ab31dd1e2f9250>`       = dnnl_gOwI32o2i,
	    :target:`gOwI32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7e924b35cc179f9a53cc5fe409526c3c>`       = dnnl_gOwI32o4i,
	    :target:`gOwi48o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa78cfc6b55b5193a4d3f6e6225c9f441c>`         = dnnl_gOwi48o,
	    :target:`gOwI48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae58d1066bac091de2aac6cc9d5ec6d8c>`       = dnnl_gOwI48o2i,
	    :target:`gOwI48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5b15e58bb6a6138c4858d9e9beddc266>`       = dnnl_gOwI48o4i,
	    :target:`gOwi64o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabe2302583c2c36475dfd197117eac3e4>`         = dnnl_gOwi64o,
	    :target:`gOwI64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4163c4fb1d0c81a1e4173efc738ed3f9>`       = dnnl_gOwI64o2i,
	    :target:`gOwI64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad51fdb51dace88da14ca0aeb9f877052>`       = dnnl_gOwI64o4i,
	    :target:`gIwo32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6f1b002cdaebe577f3a773718ba4edef>`         = dnnl_gIwo32i,
	    :target:`gIwO32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa299080ee9668fa10a1356ae93a7f58f6>`       = dnnl_gIwO32i2o,
	    :target:`gIwO32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadbd4928222e435be4924c7246bfd3d9a>`       = dnnl_gIwO32i4o,
	    :target:`gIwo48i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa685a2e9140177c0dd7a5738a38ae4394>`         = dnnl_gIwo48i,
	    :target:`gIwO48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa834caccd16be628c85bf9d4ff8038759>`       = dnnl_gIwO48i2o,
	    :target:`gIwO48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa912da03946e27f5d7c1a15c3735bc20a>`       = dnnl_gIwO48i4o,
	    :target:`gIwo64i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabd7313a722496b8f3a48e14cce835fec>`         = dnnl_gIwo64i,
	    :target:`gIwO64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa800d8c65e94783e73d1b1a05b9bb321d>`       = dnnl_gIwO64i2o,
	    :target:`gIwO64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa194ff55c479f5832c8d885fe449f8234>`       = dnnl_gIwO64i4o,
	    :target:`gwio<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facf04f85df4192095f70cc646bdb72cb7>`            = dnnl_gwio,
	    :target:`gwIo2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8b4f24521a166a1326e5c7197df858dc>`          = dnnl_gwIo2i,
	    :target:`gwIo4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4455693875f1d3d1d9ef984cdf6586ee>`          = dnnl_gwIo4i,
	    :target:`OhwI32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4df6168ba8d020bd00e89d9d6d045c5b>`         = dnnl_OhwI32o,
	    :target:`OhwI32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6b777f6abcc001e38fe1b9648ff14bd6>`       = dnnl_OhwI32o2i,
	    :target:`OhwI32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4845d06886f3283c653baed0f5994b57>`       = dnnl_OhwI32o4i,
	    :target:`Ohwi48o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3a6fc82af262ec29424021173080fd85>`         = dnnl_Ohwi48o,
	    :target:`OhwI48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2f79a70f4481b3d0a5fab26c07f47a35>`       = dnnl_OhwI48o2i,
	    :target:`OhwI48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae833be2caab4add2532bd99865cc8bf5>`       = dnnl_OhwI48o4i,
	    :target:`Ohwi64o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad645fe21c614a6dc9edc855f55515f0d>`         = dnnl_Ohwi64o,
	    :target:`OhwI64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab134a42f4ff7403591629ab286a3a695>`       = dnnl_OhwI64o2i,
	    :target:`OhwI64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2cf6fb20a117032e37f6213161c1d769>`       = dnnl_OhwI64o4i,
	    :target:`Ihwo32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa751a084c7ca40cfe34084a60463934d7>`         = dnnl_Ihwo32i,
	    :target:`IhwO32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0d53c301cdd3f6336f91b920d5a16248>`       = dnnl_IhwO32i2o,
	    :target:`IhwO32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa16062381d554952a4801519a5a08a474>`       = dnnl_IhwO32i4o,
	    :target:`Ihwo48i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa86833c5fa803dfb003c0eac47cdd1a1c>`         = dnnl_Ihwo48i,
	    :target:`IhwO48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faef73d5cd8e257d541190437d1ed8f0e6>`       = dnnl_IhwO48i2o,
	    :target:`IhwO48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa55503b69845e147e865c992f1fb53b1d>`       = dnnl_IhwO48i4o,
	    :target:`Ihwo64i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2c5dab069ed3b66a3e80b4f47ac99771>`         = dnnl_Ihwo64i,
	    :target:`IhwO64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4f4fb729df92670962a5d39967221b5b>`       = dnnl_IhwO64i2o,
	    :target:`IhwO64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa615896fbe169a7a725f5b7f0c8b265ef>`       = dnnl_IhwO64i4o,
	    :target:`hwIo2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa20928584cf824e01004567eb12789d0d>`          = dnnl_hwIo2i,
	    :target:`hwIo4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8990041df1afc6644265e1cd4b6274bd>`          = dnnl_hwIo4i,
	    :target:`gOhwI32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0a381ced908949e52d52dad3155a69e4>`        = dnnl_gOhwI32o,
	    :target:`gOhwI32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabc6f9851d09a34f65cc17330a1d93866>`      = dnnl_gOhwI32o2i,
	    :target:`gOhwI32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa11d88af60d6522b5e490581b5fa445c8>`      = dnnl_gOhwI32o4i,
	    :target:`gOhwi48o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa929f13063de1f02a7f9718b748cc11f>`        = dnnl_gOhwi48o,
	    :target:`gOhwI48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1772dc1dfb5e559f7582735145dbb841>`      = dnnl_gOhwI48o2i,
	    :target:`gOhwI48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad80e93448823f0203f617046bb12fb87>`      = dnnl_gOhwI48o4i,
	    :target:`gOhwi64o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faed8c7731d2b2a21b635b35b3c1602918>`        = dnnl_gOhwi64o,
	    :target:`gOhwI64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac3c8dd82c5bf996e304c6bf400bea0f9>`      = dnnl_gOhwI64o2i,
	    :target:`gOhwI64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa0d6aa8969a0e9aab528e0c8420a9642>`      = dnnl_gOhwI64o4i,
	    :target:`gIhwo32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faed6cf6af45d592600698011ee2b30c80>`        = dnnl_gIhwo32i,
	    :target:`gIhwO32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab9bb754b695453454926914ca046b914>`      = dnnl_gIhwO32i2o,
	    :target:`gIhwO32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa2c3ce1daafec421c96d6dd1d3569b53>`      = dnnl_gIhwO32i4o,
	    :target:`gIhwo48i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad226dd6d7f1ec2f944281cb1e699fc0a>`        = dnnl_gIhwo48i,
	    :target:`gIhwO48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7f164f7349b43bcceb8e7c07f8c2cd30>`      = dnnl_gIhwO48i2o,
	    :target:`gIhwO48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8436d87973f5d7c08ed8f9800b0003d9>`      = dnnl_gIhwO48i4o,
	    :target:`gIhwo64i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae0d2352a5f10b679139b40cc5bde5710>`        = dnnl_gIhwo64i,
	    :target:`gIhwO64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa109399e7f8fe53d926e09553f6357300>`      = dnnl_gIhwO64i2o,
	    :target:`gIhwO64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa240a73e52225b2ac05eb68e23eb492f>`      = dnnl_gIhwO64i4o,
	    :target:`ghwio<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7f192e7092d8a12f7b8cf5069228047b>`           = dnnl_ghwio,
	    :target:`ghwIo2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facfa2d4b9b57833173aa06b767342018c>`         = dnnl_ghwIo2i,
	    :target:`ghwIo4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad1fdec6d2e1430474b1cd5232decc1ef>`         = dnnl_ghwIo4i,
	    :target:`Odhwi32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa87471317b98cd7b6f67cb7b06fe00c0f>`        = dnnl_Odhwi32o,
	    :target:`OdhwI32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa264537270d1237d9bc575d8f0b15df50>`      = dnnl_OdhwI32o2i,
	    :target:`OdhwI32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa984902cd3efe8da8a029281b66c98f87>`      = dnnl_OdhwI32o4i,
	    :target:`Odhwi48o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4779d52f97848c014d33090138b073bb>`        = dnnl_Odhwi48o,
	    :target:`OdhwI48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf30be921d71c1caa6ce6e9dd1cfd5cc2>`      = dnnl_OdhwI48o2i,
	    :target:`OdhwI48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac5718d01d1f2c332575dc70953044924>`      = dnnl_OdhwI48o4i,
	    :target:`Odhwi64o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa38549ab1d0debdfa1702987319f46252>`        = dnnl_Odhwi64o,
	    :target:`OdhwI64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2c90128a412014a999fb027cf83ab825>`      = dnnl_OdhwI64o2i,
	    :target:`OdhwI64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae16e1dba255d0a4b7397c5fb7aabab19>`      = dnnl_OdhwI64o4i,
	    :target:`Idhwo32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa99444285be924a578cd3fb5a99108985>`        = dnnl_Idhwo32i,
	    :target:`IdhwO32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa46ccedb1ac24dd19ae5a9e8ae0aa7949>`      = dnnl_IdhwO32i2o,
	    :target:`IdhwO32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad45b739463a3cf43cc81528a8e627188>`      = dnnl_IdhwO32i4o,
	    :target:`Idhwo48i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad28189b300558ae4206159aa49fab19f>`        = dnnl_Idhwo48i,
	    :target:`IdhwO48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa32c1f906f8dfd6cb1d6330a8399ca5f5>`      = dnnl_IdhwO48i2o,
	    :target:`IdhwO48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeca647b2db13f92b5160c9719f36f3fa>`      = dnnl_IdhwO48i4o,
	    :target:`Idhwo64i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafb68225eed3f2c2b3d97dc9eb418bcd7>`        = dnnl_Idhwo64i,
	    :target:`IdhwO64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4897b960bf35c26ec035b3c25b5d578a>`      = dnnl_IdhwO64i2o,
	    :target:`IdhwO64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facbe29436511aaa458ea860762a8696f3>`      = dnnl_IdhwO64i4o,
	    :target:`dhwIo2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7e804c434045ab42160b57fc4b0fe0bb>`         = dnnl_dhwIo2i,
	    :target:`dhwIo4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab90ec068bb9b908eae63b1e2a9aa6e12>`         = dnnl_dhwIo4i,
	    :target:`gOdhwi32o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa86a6297c0dbef962b041de09e3ff8b96>`       = dnnl_gOdhwi32o,
	    :target:`gOdhwI32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3590ce62033da4268520848c7b6469e8>`     = dnnl_gOdhwI32o2i,
	    :target:`gOdhwI32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab76e71c21668262443fd1d4598d922d>`     = dnnl_gOdhwI32o4i,
	    :target:`gOdhwi48o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8682e145caf738629bb8b518bdb8bfbc>`       = dnnl_gOdhwi48o,
	    :target:`gOdhwI48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa62262c37ac2b440961bd2bd2908c2b28>`     = dnnl_gOdhwI48o2i,
	    :target:`gOdhwI48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac33d670374f674431bd2f163cc1c0657>`     = dnnl_gOdhwI48o4i,
	    :target:`gOdhwi64o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa8c1305c5d6275d503b46f1c901ae07b>`       = dnnl_gOdhwi64o,
	    :target:`gOdhwI64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facc57e33baff9e730f4e18ab54077e2bb>`     = dnnl_gOdhwI64o2i,
	    :target:`gOdhwI64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3350c4c1bd2be3383250ec9389036a08>`     = dnnl_gOdhwI64o4i,
	    :target:`gIdhwo32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0b142bc884d37a6a378ad902fa684de1>`       = dnnl_gIdhwo32i,
	    :target:`gIdhwO32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8f5929606b2d074345c05364a66399be>`     = dnnl_gIdhwO32i2o,
	    :target:`gIdhwO32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafccb22f04ccf57460a89e8a3b658b8dd>`     = dnnl_gIdhwO32i4o,
	    :target:`gIdhwo48i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4d4a08daa7993a262130d712399d7019>`       = dnnl_gIdhwo48i,
	    :target:`gIdhwO48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6f0063415b7e6984dfd5e31b3adeb3c4>`     = dnnl_gIdhwO48i2o,
	    :target:`gIdhwO48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae1cb4f93407a63a58038b8fa2e55df93>`     = dnnl_gIdhwO48i4o,
	    :target:`gIdhwo64i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae5bc04e8fc180a459706816d3905b886>`       = dnnl_gIdhwo64i,
	    :target:`gIdhwO64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa72a5756bd1388c9e01c9ad1c7da81473>`     = dnnl_gIdhwO64i2o,
	    :target:`gIdhwO64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2470a07d30935e8f2d87c2e9a9df77af>`     = dnnl_gIdhwO64i4o,
	    :target:`gdhwio<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac216bbb5c2b760d45eb3e5d1ecc4e9f9>`          = dnnl_gdhwio,
	    :target:`gdhwIo2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1ad45a78edc2776c4763399c3109eb43>`        = dnnl_gdhwIo2i,
	    :target:`gdhwIo4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7b009c40c5142b8f31a254455880881e>`        = dnnl_gdhwIo4i,
	    :target:`ldIo32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faced5d1c1d76b6320969fa7a6a452a036>`         = dnnl_ldIo32i,
	    :target:`ldgIo32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf42b9e37327307c1974bc5e70ce48272>`        = dnnl_ldgIo32i,
	    :target:`ldgIO32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9e13d3a09fff641fb4fbbb47a27683fb>`      = dnnl_ldgIO32i2o,
	    :target:`nCdhw32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabc6154d165066e532faf2d754c4a1f96>`        = dnnl_nCdhw32c,
	    :target:`nChw32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa66ed2e952e009f9171d981ba922054ec>`         = dnnl_nChw32c,
	    :target:`nCw32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad9f9b04264a2c2aaf6b79488fd4c1611>`          = dnnl_nCw32c,
	    :target:`NCw32n16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab2de1d1d95a49d88d78edd929c8735d0>`       = dnnl_NCw32n16c,
	    :target:`NChw32n16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad0948a2c411caa16c284d3b2ac50cfd1>`      = dnnl_NChw32n16c,
	    :target:`NCdhw32n16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa205ec83d57dda7090186941ae07c93af>`     = dnnl_NCdhw32n16c,
	    :target:`NCw32n32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7ff68a20daa403588ab5ca199ddaed57>`       = dnnl_NCw32n32c,
	    :target:`OI16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab766c9d8d29df0bee300327c2ab61f0>`      = dnnl_OI16i16o4i,
	    :target:`IOw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac0fdb2a3b7963bafec6c09b59f35bb3b>`      = dnnl_IOw8o16i2o,
	    :target:`IOhw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa27e5b716b445fb992b46c91449a13f8c>`     = dnnl_IOhw8o16i2o,
	    :target:`Owhi16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4d9b89bbab4ff3e92e98415caa68d1db>`         = dnnl_Owhi16o,
	    :target:`OIdhw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaed00d46cf380a8db345fc2feda639fd>`    = dnnl_OIdhw8o16i2o,
	    :target:`IOdhw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa2e26c7e426d56d367ae380cc1a1a5f3>`    = dnnl_IOdhw8o16i2o,
	    :target:`Goiw4g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fade42c570ba03f3bdbf59a87ba477f21a>`          = dnnl_Goiw4g,
	    :target:`gIOw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa4aeca62ad44cd6f38b8973e3593451e>`     = dnnl_gIOw8o16i2o,
	    :target:`Goiw32g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa78a6918eb2b0d2ae4700f9c1998f76f8>`         = dnnl_Goiw32g,
	    :target:`Goihw4g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf91320ad6f16b0c0f4298b5722766c2a>`         = dnnl_Goihw4g,
	    :target:`gIOhw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faceb871f7a9d25cee508dd6538cbd5e38>`    = dnnl_gIOhw8o16i2o,
	    :target:`Goihw32g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf39b4136b620f954f2d40e0d54a04222>`        = dnnl_Goihw32g,
	    :target:`gOwhi16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0fe96337e2f391d51c78d0179e20f9d2>`        = dnnl_gOwhi16o,
	    :target:`IOw4i8o8i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8f0f7a9f88092a740b57b0ba02cf91c5>`     = dnnl_IOw4i8o8i4o,
	    :target:`IOhw4i8o8i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa534f74f8c8c78448f0846f4435d8f5af>`    = dnnl_IOhw4i8o8i4o,
	    :target:`IOdhw4i8o8i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad6b537cda8017d23f439367f984a4729>`   = dnnl_IOdhw4i8o8i4o,
	    :target:`gIOw4i8o8i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faee3d953e0545d5a88eb7022802a4b828>`    = dnnl_gIOw4i8o8i4o,
	    :target:`gIOhw4i8o8i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab0b08f87ba5fb60881bfd93f583483d5>`   = dnnl_gIOhw4i8o8i4o,
	    :target:`gIOdhw4i8o8i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa80f37dc24e22da12d843b42022c4ac42>`  = dnnl_gIOdhw4i8o8i4o,
	    :target:`gOIdhw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab5a81083fa44fbbd70a008e532d2a7c9>`   = dnnl_gOIdhw8o16i2o,
	    :target:`gIOdhw8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae6ea09dad0e6d4168a682bfa5d35234e>`   = dnnl_gIOdhw8o16i2o,
	    :target:`Goidhw32g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1f8c3ff502f22b4a7c2a87b9b8e89c4b>`       = dnnl_Goidhw32g,
	    :target:`OI16i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9823fa384be57110a7d15c14434d0f84>`      = dnnl_OI16i32o4i,
	    :target:`OI16i48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac76e42a3468656cd81166f89cbb5bc3d>`      = dnnl_OI16i48o4i,
	    :target:`OI16i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf88832ab5eb35b236e0b95ef103e0dfc>`      = dnnl_OI16i64o4i,
	    :target:`OI16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa7ac8c00d10ff9d051d6005373bd0054>`      = dnnl_OI16i16o2i,
	    :target:`OI16i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae567b18711002ad4808cff094da2b4db>`      = dnnl_OI16i32o2i,
	    :target:`OI16i48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faee0e8f2a14f877352541a459580342e3>`      = dnnl_OI16i48o2i,
	    :target:`OI16i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac727d743d51e048279ef545333dcb4eb>`      = dnnl_OI16i64o2i,
	    :target:`aBdeC16c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6b45ffa382c4aa8d4dffcbdc3d14c35b>`   = dnnl_aBdeC16c16b4c,
	    :target:`AcB16b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa169fa43e948bd7af4131c4dc9a533958>`     = dnnl_AcB16b16a2b,
	    :target:`aBdC16c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0c5c2e94b73f048d56d8cd281ab8a305>`    = dnnl_aBdC16c16b2c,
	    :target:`AcB16b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad635bc2672d9f0129846bfb954210fd2>`     = dnnl_AcB16b16a4b,
	    :target:`aBdC16c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa67888664a1194408be179cfbab6f81b6>`    = dnnl_aBdC16c16b4c,
	    :target:`AcdB16b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9c066ef26714e29f45b3d65ddeba224c>`    = dnnl_AcdB16b16a2b,
	    :target:`aBdefC16c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0013fcbe7689286569c77c7b3c38319e>`  = dnnl_aBdefC16c16b4c,
	    :target:`AcdeB16b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa64e979a7ef57f746c5dd47fde480a19a>`   = dnnl_AcdeB16b16a4b,
	    :target:`AcB16b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae16b1a55a722a154ec6f269f8b0e5e0c>`     = dnnl_AcB16b32a2b,
	    :target:`AcB16b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0e70d6616a8aee12de495655a0aebe50>`     = dnnl_AcB16b32a4b,
	    :target:`AcB16b48a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4d7e29688379d2f315fa99cc8e755593>`     = dnnl_AcB16b48a2b,
	    :target:`AcB16b48a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa74b014b7aa78e9874628c70d59d6c920>`     = dnnl_AcB16b48a4b,
	    :target:`AcB16b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabf2abb2c1f8c0cee3f8c85f4652efeab>`     = dnnl_AcB16b64a2b,
	    :target:`AcB16b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa66cf60e6ef0d313b804987257f567184>`     = dnnl_AcB16b64a4b,
	    :target:`aBdC16c32b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa03445a03332677d12474f53015e95684>`    = dnnl_aBdC16c32b2c,
	    :target:`aBdC16c32b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad8be2786b99587ec035f7391b7db62b6>`    = dnnl_aBdC16c32b4c,
	    :target:`aBdC16c48b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad61ee4f223b1510b9cbbcf6cbdea1601>`    = dnnl_aBdC16c48b2c,
	    :target:`aBdC16c48b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac2e2cf9ccd3c9f8bfa71fe2cc5041486>`    = dnnl_aBdC16c48b4c,
	    :target:`aBdC16c64b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab98bce60ce76b358dc9348c0e2e3902d>`    = dnnl_aBdC16c64b2c,
	    :target:`aBdC16c64b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa42af5628042353e52062ddbe3073f1f2>`    = dnnl_aBdC16c64b4c,
	    :target:`AcdB16b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac630a6ff5fd9bb98e16522d45bc78869>`    = dnnl_AcdB16b32a2b,
	    :target:`AcdB16b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa70628a0a08264e89ef439aa293b733cb>`    = dnnl_AcdB16b32a4b,
	    :target:`AcdB16b48a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3face72a174bd2c7d12588e2b61dd639675>`    = dnnl_AcdB16b48a2b,
	    :target:`AcdB16b48a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa125f0b045795a3644644de0c6d38f0ed>`    = dnnl_AcdB16b48a4b,
	    :target:`AcdB16b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5761e8fd8edc5b741e24c997d25574f3>`    = dnnl_AcdB16b64a2b,
	    :target:`AcdB16b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac771b1575e8e13cadd1026af945e312e>`    = dnnl_AcdB16b64a4b,
	    :target:`aBdeC16c32b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2826f417c637be6199910e9ab22e3677>`   = dnnl_aBdeC16c32b2c,
	    :target:`aBdeC16c32b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4bb4526c4e0f7f299c524f5887a70cee>`   = dnnl_aBdeC16c32b4c,
	    :target:`aBdeC16c48b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf161e7c1e8cbb46ebce64cc48ecb57c1>`   = dnnl_aBdeC16c48b2c,
	    :target:`aBdeC16c48b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac0b11e4632859d10faf9203980075642>`   = dnnl_aBdeC16c48b4c,
	    :target:`aBdeC16c64b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad81cc1b5f433a868236c33d8db5b12fb>`   = dnnl_aBdeC16c64b2c,
	    :target:`aBdeC16c64b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa886edf1215155df1263764f0f44071b8>`   = dnnl_aBdeC16c64b4c,
	    :target:`AcdeB16b32a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2382521309d8747129e1a1ed49b10eb0>`   = dnnl_AcdeB16b32a2b,
	    :target:`AcdeB16b32a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf7299a7e2b2feba2026887224aedce40>`   = dnnl_AcdeB16b32a4b,
	    :target:`AcdeB16b48a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf2c9a15082e3fa9f566358aed992107d>`   = dnnl_AcdeB16b48a2b,
	    :target:`AcdeB16b48a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5c33d3a442a28081d1d374361a2129ac>`   = dnnl_AcdeB16b48a4b,
	    :target:`AcdeB16b64a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7ad1f3613cee696e386e30df4ee2b1f7>`   = dnnl_AcdeB16b64a2b,
	    :target:`AcdeB16b64a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa58cf573fce31b72050993cf6bfef7bc0>`   = dnnl_AcdeB16b64a4b,
	    :target:`aBdefC16c32b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa91d66a8a7e703f517a7c0fd2e87ea521>`  = dnnl_aBdefC16c32b2c,
	    :target:`aBdefC16c32b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa91f1e1adfd8244974fd3d359cd4787ff>`  = dnnl_aBdefC16c32b4c,
	    :target:`aBdefC16c48b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa39a8b1064f49ad0f7119a30f04f8dd07>`  = dnnl_aBdefC16c48b2c,
	    :target:`aBdefC16c48b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5f1f5fcb7687423bab636882d208c6c6>`  = dnnl_aBdefC16c48b4c,
	    :target:`aBdefC16c64b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac5c0af5c171cb03ebaeaf2ec3115c6e9>`  = dnnl_aBdefC16c64b2c,
	    :target:`aBdefC16c64b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad0c8c7ec9dfe62856d136df5414afc53>`  = dnnl_aBdefC16c64b4c,
	    :target:`OwI16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa096bee33daf8eb40ad2d2098db0d6b0c>`     = dnnl_OwI16i16o2i,
	    :target:`gOwI16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa33d81afd48be19fa7bef58e9663e0b92>`    = dnnl_gOwI16i16o2i,
	    :target:`OhwI16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3face075e8b50637f9b7bac1bd7ce22dd58>`    = dnnl_OhwI16i16o2i,
	    :target:`gOhwI16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae9de8543bf1b7034b21834e0474f700b>`   = dnnl_gOhwI16i16o2i,
	    :target:`OdhwI16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa847529de1641daabbb7cf021844f3474>`   = dnnl_OdhwI16i16o2i,
	    :target:`gOdhwI16i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa034e5cc77fbf6b98d3dbc55c8faee447>`  = dnnl_gOdhwI16i16o2i,
	    :target:`OwI16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa815a34e32814fd2dd892f9a89058af0a>`     = dnnl_OwI16i16o4i,
	    :target:`gOwI16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa860251e65b03ba6d40bd52915a16c661>`    = dnnl_gOwI16i16o4i,
	    :target:`OhwI16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa997c5fdb8b67c5b7e2af5900ed0e6a3c>`    = dnnl_OhwI16i16o4i,
	    :target:`gOhwI16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0d5789956cb55cb296f839225eb2d11e>`   = dnnl_gOhwI16i16o4i,
	    :target:`OdhwI16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab7cbfac98c962d99256379b9dc81f244>`   = dnnl_OdhwI16i16o4i,
	    :target:`gOdhwI16i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5db7b886d01b55de9360a7a57eb17bed>`  = dnnl_gOdhwI16i16o4i,
	    :target:`OwI16i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2210a24fd810eeb45ac3a70d12198ae0>`     = dnnl_OwI16i32o2i,
	    :target:`OwI16i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab36068359d0dba8c4626c5a6997e99ef>`     = dnnl_OwI16i32o4i,
	    :target:`OwI16i48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6d4ff1ab9552b04345d01fb927de16b3>`     = dnnl_OwI16i48o2i,
	    :target:`OwI16i48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faacdc5ace8523ff655a9365b5a3703e6e>`     = dnnl_OwI16i48o4i,
	    :target:`OwI16i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa23ccef8df3acbbe92218e661b6effd8d>`     = dnnl_OwI16i64o2i,
	    :target:`OwI16i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2a3245f4147774fa391d911803788bc0>`     = dnnl_OwI16i64o4i,
	    :target:`gOwI16i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabbaacdee5fdf42eb0df525c3965fbf9d>`    = dnnl_gOwI16i32o2i,
	    :target:`gOwI16i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa32b0066b9d63d93b755a6a0476c51b93>`    = dnnl_gOwI16i32o4i,
	    :target:`gOwI16i48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadc3170fb9659fe2676b5a11df949978f>`    = dnnl_gOwI16i48o2i,
	    :target:`gOwI16i48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa66f5f4af54af563ba4741d21b4670720>`    = dnnl_gOwI16i48o4i,
	    :target:`gOwI16i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7aafd9951f3aa8f8c71d007f082dca60>`    = dnnl_gOwI16i64o2i,
	    :target:`gOwI16i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3d57289a9dbb88b3aba761027d45afa4>`    = dnnl_gOwI16i64o4i,
	    :target:`OhwI16i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa912bde3641493e551885b95f784bf1bc>`    = dnnl_OhwI16i32o2i,
	    :target:`OhwI16i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6fe49d4595dc7662ba088d375facdaf7>`    = dnnl_OhwI16i32o4i,
	    :target:`OhwI16i48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac474adac454a8901dc10fd6351315b04>`    = dnnl_OhwI16i48o2i,
	    :target:`OhwI16i48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9d38f0135b8eb408456393114f86af1e>`    = dnnl_OhwI16i48o4i,
	    :target:`OhwI16i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faafefa145841860eda7646d9ef85e265c>`    = dnnl_OhwI16i64o2i,
	    :target:`OhwI16i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa848c27018938a8db75c361f5dc12b394>`    = dnnl_OhwI16i64o4i,
	    :target:`gOhwI16i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5eccb139a167485f87d874adf4080411>`   = dnnl_gOhwI16i32o2i,
	    :target:`gOhwI16i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa675b1c8ef937220161ba7538280f1cde>`   = dnnl_gOhwI16i32o4i,
	    :target:`gOhwI16i48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5acf6332609e0837ce2a0a5ddc660c8c>`   = dnnl_gOhwI16i48o2i,
	    :target:`gOhwI16i48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab2d4f5cf277bc8b86c0f32fb47f8f003>`   = dnnl_gOhwI16i48o4i,
	    :target:`gOhwI16i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9b8ccd1ca4fa7a0b92a6fe4dc882c063>`   = dnnl_gOhwI16i64o2i,
	    :target:`gOhwI16i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeb8012a6765017c984bfe157eccdc388>`   = dnnl_gOhwI16i64o4i,
	    :target:`OdhwI16i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facd2bd7a39e4f0d3c2718cc2dffe3eb8f>`   = dnnl_OdhwI16i32o2i,
	    :target:`OdhwI16i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3bfae2b1b42a4cfa6e7fcdf8fac1266a>`   = dnnl_OdhwI16i32o4i,
	    :target:`OdhwI16i48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2ecfcecd5926971964acdeb743606792>`   = dnnl_OdhwI16i48o2i,
	    :target:`OdhwI16i48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa289fd172969e65e319b078b73ce501b4>`   = dnnl_OdhwI16i48o4i,
	    :target:`OdhwI16i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac332635fa17a2e67842e9b8c8c05b93c>`   = dnnl_OdhwI16i64o2i,
	    :target:`OdhwI16i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa64c158931dfca496339f859d6816ce1a>`   = dnnl_OdhwI16i64o4i,
	    :target:`IdhwO16o32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa72b757a60feaf36bf8e58e943bf2ca18>`   = dnnl_IdhwO16o32i2o,
	    :target:`IdhwO16o32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa305bb40de17afbdaff6ceeaf7d394408>`   = dnnl_IdhwO16o32i4o,
	    :target:`IdhwO16o48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf56c6c7ece787432bb0cb3871ff53c0c>`   = dnnl_IdhwO16o48i2o,
	    :target:`IdhwO16o48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fade31fb841ac5ccfa5c01c1f13ecc83c1>`   = dnnl_IdhwO16o48i4o,
	    :target:`IdhwO16o64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa16d070028efe50c13eb0910e04613764>`   = dnnl_IdhwO16o64i2o,
	    :target:`IdhwO16o64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4fe952234eb15283e51d38093a696e25>`   = dnnl_IdhwO16o64i4o,
	    :target:`gOdhwI16i32o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabeab32886a6fa3f36fb6a671e740eee6>`  = dnnl_gOdhwI16i32o2i,
	    :target:`gOdhwI16i32o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa929f0901b307a4a8453941f9aca31931>`  = dnnl_gOdhwI16i32o4i,
	    :target:`gOdhwI16i48o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5b5325b67c2c6f6fbc6a82c42510ed9d>`  = dnnl_gOdhwI16i48o2i,
	    :target:`gOdhwI16i48o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa97ec7dbdad02d3296d06fe89f23b9f03>`  = dnnl_gOdhwI16i48o4i,
	    :target:`gOdhwI16i64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9ce59786532d79569c17dc7b18f03548>`  = dnnl_gOdhwI16i64o2i,
	    :target:`gOdhwI16i64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faec8c2f785c03cd52525b06a55376b7d4>`  = dnnl_gOdhwI16i64o4i,
	    :target:`gIdhwO16o32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8aef35f2d9f39b131af164ffeb44124d>`  = dnnl_gIdhwO16o32i2o,
	    :target:`gIdhwO16o32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4adf4c925a07b7d3b6e2885da722c8c3>`  = dnnl_gIdhwO16o32i4o,
	    :target:`gIdhwO16o48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faebeb8e47a96c3d5d4dcdca01a1e7df64>`  = dnnl_gIdhwO16o48i2o,
	    :target:`gIdhwO16o48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadbcd839123e9f41cc410827c42a79d41>`  = dnnl_gIdhwO16o48i4o,
	    :target:`gIdhwO16o64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad3687433de9bd4d69451d8b5821dcd41>`  = dnnl_gIdhwO16o64i2o,
	    :target:`gIdhwO16o64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa32ea5cbb0b02e88d514baed08859467d>`  = dnnl_gIdhwO16o64i4o,
	    :target:`IwO16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabba6d7f900b27e364ab2b406866a57ca>`     = dnnl_IwO16o16i2o,
	    :target:`IwO16o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faceb3eb7249033bf4d4ea86b2df119184>`     = dnnl_IwO16o16i4o,
	    :target:`IhwO16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1f2d65c1819e810cc34e12c744ede525>`    = dnnl_IhwO16o16i2o,
	    :target:`IhwO16o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facf2d027bb678ff362964dbbc89356a47>`    = dnnl_IhwO16o16i4o,
	    :target:`IdhwO16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0b043162d7355621555158ab599cc3f3>`   = dnnl_IdhwO16o16i2o,
	    :target:`IdhwO16o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5cb83b785a657286d03aa732d08a1244>`   = dnnl_IdhwO16o16i4o,
	    :target:`gIwO16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabb0df3d861f34ec0292962a915bd2e98>`    = dnnl_gIwO16o16i2o,
	    :target:`gIwO16o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa15c022fd8591b4bcc597cb5a7d81b145>`    = dnnl_gIwO16o16i4o,
	    :target:`gIhwO16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab8fc846138ce58e37ce72a7ec42ee528>`   = dnnl_gIhwO16o16i2o,
	    :target:`gIhwO16o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa963cfbc060791ebe2fd17b5a598529dd>`   = dnnl_gIhwO16o16i4o,
	    :target:`gIdhwO16o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa91e196c20193baed13856fe9d456e465>`  = dnnl_gIdhwO16o16i2o,
	    :target:`gIdhwO16o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa29a2e215217ae28814ecbd084eebad7d>`  = dnnl_gIdhwO16o16i4o,
	    :target:`IwO16o32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa978fcd376042e15196ef7318f1ad4989>`     = dnnl_IwO16o32i2o,
	    :target:`IwO16o32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa24d2dc675a5dc5a1369b74df475c8d2>`     = dnnl_IwO16o32i4o,
	    :target:`IwO16o48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab109f61b76d3f8b9c9ec8b017143b078>`     = dnnl_IwO16o48i2o,
	    :target:`IwO16o48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab0d0413653f8a783371a440ea531dc36>`     = dnnl_IwO16o48i4o,
	    :target:`IwO16o64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3ad44654664f27b32f9012afbe6fc212>`     = dnnl_IwO16o64i2o,
	    :target:`IwO16o64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad73481a3506f4f57fc33bbbfea862f3f>`     = dnnl_IwO16o64i4o,
	    :target:`gIwO16o32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9030d3a6f478d62bc44803d2565ce462>`    = dnnl_gIwO16o32i2o,
	    :target:`gIwO16o32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa22c319650e85cda4efb6b388ae04fc78>`    = dnnl_gIwO16o32i4o,
	    :target:`gIwO16o48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad2a48c1f0aba166287bccae2ab2cfa38>`    = dnnl_gIwO16o48i2o,
	    :target:`gIwO16o48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9cda9c1ea876421d96623b2a1703b64d>`    = dnnl_gIwO16o48i4o,
	    :target:`gIwO16o64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab9b9390be300d238b046901bf80651f0>`    = dnnl_gIwO16o64i2o,
	    :target:`gIwO16o64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9ca4f7088357e080266b12ffea9a16bd>`    = dnnl_gIwO16o64i4o,
	    :target:`IhwO16o32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac816b13041cf55a4cc981a9e2742a8d5>`    = dnnl_IhwO16o32i2o,
	    :target:`IhwO16o32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa106e488918961a13f5ac374b8dc3c7e9>`    = dnnl_IhwO16o32i4o,
	    :target:`IhwO16o48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa52c1dca7f1dfbade0731a4319af7d209>`    = dnnl_IhwO16o48i2o,
	    :target:`IhwO16o48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac8d8e592dedb69ae995f9bf34980f577>`    = dnnl_IhwO16o48i4o,
	    :target:`IhwO16o64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa179695a346734f69aaee7b789ce2ecff>`    = dnnl_IhwO16o64i2o,
	    :target:`IhwO16o64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7f996cd0c1f416bd1927fc0aa5c6357a>`    = dnnl_IhwO16o64i4o,
	    :target:`gIhwO16o32i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa21563f7bee508df73ef4c8c3361fd77c>`   = dnnl_gIhwO16o32i2o,
	    :target:`gIhwO16o32i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa25577e8a76f268589d0069996b496a7e>`   = dnnl_gIhwO16o32i4o,
	    :target:`gIhwO16o48i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa207c706edd98f22426f49a14ba15ae28>`   = dnnl_gIhwO16o48i2o,
	    :target:`gIhwO16o48i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5af8c321936632c5bbf8b26e801c61f5>`   = dnnl_gIhwO16o48i4o,
	    :target:`gIhwO16o64i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa3fb4b91d7ea0583d18fcc3c7e0dc3ee>`   = dnnl_gIhwO16o64i2o,
	    :target:`gIhwO16o64i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa59260fc9d17cc7e514622de8ef616d4c>`   = dnnl_gIhwO16o64i4o,
	    :target:`aBdeC16c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa569d674d74f99760082511f6b938544c>`   = dnnl_aBdeC16c16b2c,
	    :target:`aBdefC16c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa70308da20fb9db558d8bc99b7659a334>`  = dnnl_aBdefC16c16b2c,
	    :target:`AcdB16b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab38b71e04798e8769dc968d14c55576>`    = dnnl_AcdB16b16a4b,
	    :target:`AcdeB16b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf88c1b6e4cd959eb752adb5f9db0eaf5>`   = dnnl_AcdeB16b16a2b,
	    :target:`hwioG16g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf939f28776344029c3d66abc8d5d2880>`        = dnnl_hwioG16g,
	    :target:`hwioG8g<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa16d43054424159d879649eeb41f2e7be>`         = dnnl_hwioG8g,
	    :target:`ABc4a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2d4044517f5f2f177b684e302417f4c6>`         = dnnl_ABc4a2b,
	    :target:`ABc8a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1aba50c6638cd7da495080dcb02f2a96>`         = dnnl_ABc8a2b,
	    :target:`ABcd4a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa938cc5976207909dfbdc3aff3bf8814>`        = dnnl_ABcd4a2b,
	    :target:`ABcde4a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2e2c4b9d39bb71ed80570f37c33477c1>`       = dnnl_ABcde4a2b,
	    :target:`ABcde8a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa070a8b4a7bf05f104d2dec5e2bbfb8d2>`       = dnnl_ABcde8a2b,
	    :target:`ABcd4a8b8a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa8fcf48e505a5f7966e84cb6bfdcaa6c>`    = dnnl_ABcd4a8b8a2b,
	    :target:`NCdhw40n32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5b23b3b79d87ea03171bd567f01bd703>`     = dnnl_NCdhw40n32c,
	    :target:`NChw40n32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0ab27cdec94c5f7ad788d3ee066db5e2>`      = dnnl_NChw40n32c,
	    :target:`NCw40n32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabb08a7a8c7e9de2fbbfa9777e71fcff8>`       = dnnl_NCw40n32c,
	    :target:`OIdhw4o8i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4fac289c75913bbd732619c2c7e8a702>`   = dnnl_OIdhw4o8i8o2i,
	    :target:`OIhw4o8i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab853d15c1a8e2479c30d63946df1864a>`    = dnnl_OIhw4o8i8o2i,
	    :target:`OIw4o8i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2e3609f173b3af117fc1c3698b39d248>`     = dnnl_OIw4o8i8o2i,
	    :target:`gOIdhw4o8i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa0c711171ad7b1912c1eaa9b9859abca>`  = dnnl_gOIdhw4o8i8o2i,
	    :target:`gOIhw4o8i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa84ef1ff7083b64fce96c51a2e902d53b>`   = dnnl_gOIhw4o8i8o2i,
	    :target:`gOIw4o8i8o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa51a0aa0813d6459a5f77c120aabea360>`    = dnnl_gOIw4o8i8o2i,
	    :target:`IOdhw4i8o8i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab0b15064465466ee64f5e7d7a0d6c484>`   = dnnl_IOdhw4i8o8i2o,
	    :target:`IOhw4i8o8i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa77fba6b3c6869cb26ee4b53c22370245>`    = dnnl_IOhw4i8o8i2o,
	    :target:`IOw4i8o8i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae0a61f3255571b499269e9c0256002e5>`     = dnnl_IOw4i8o8i2o,
	    :target:`gIOdhw4i8o8i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf820b2e82034b78f09986ce94470820c>`  = dnnl_gIOdhw4i8o8i2o,
	    :target:`gIOhw4i8o8i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7c1c2d17207de609688b816962650073>`   = dnnl_gIOhw4i8o8i2o,
	    :target:`gIOw4i8o8i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1033cafe58e2df7b11efcd1c69448194>`    = dnnl_gIOw4i8o8i2o,
	    :target:`aBCd8b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7c12ab1904a00e7180fee6cbbf2a92d9>`        = dnnl_aBCd8b2c,
	    :target:`ABcde40a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facc8f7a28e8216026d39f430e08fcee8a>`     = dnnl_ABcde40a16b,
	    :target:`ABcde40a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa502e08873cfec67aac0729ac8ad72f63>`     = dnnl_ABcde40a32b,
	    :target:`aBCde8b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa05ec13ee4fe8afd26d9e0111ed1215e2>`       = dnnl_aBCde8b2c,
	    :target:`ABcde4a8b8a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faea6cf3092ef76fcbcbde8a9edc1e8c11>`   = dnnl_ABcde4a8b8a2b,
	    :target:`ABc4a8b8a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facf079910cd549ab73f50cfa62aa49fb7>`     = dnnl_ABc4a8b8a2b,
	    :target:`aBCdef4b8c8b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa718418ef67c040c63b57a0fb793ca022>`  = dnnl_aBCdef4b8c8b2c,
	    :target:`aBCde4b8c8b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1162b94d53164a376d59cc2502273d33>`   = dnnl_aBCde4b8c8b2c,
	    :target:`aBCd4b8c8b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae8f62476d9b9a9445d2ce6d32940a5f5>`    = dnnl_aBCd4b8c8b2c,
	    :target:`BAcde4b8a8b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa97359f27d90967a83f7b8783560c4765>`   = dnnl_BAcde4b8a8b2a,
	    :target:`BAcd4b8a8b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0c5c25fac2615561314962dcfffdc8bf>`    = dnnl_BAcd4b8a8b2a,
	    :target:`BAc4b8a8b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabd60262a112c5b873afece456bd6b176>`     = dnnl_BAc4b8a8b2a,
	    :target:`aCBdef4c8b8c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1c02c36eec85658cfc058975e0a7c89f>`  = dnnl_aCBdef4c8b8c2b,
	    :target:`aCBde4c8b8c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6d344185e09675d8e85cb6649ce136fe>`   = dnnl_aCBde4c8b8c2b,
	    :target:`aCBd4c8b8c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9109669b153ce0392a4da397f25d3836>`    = dnnl_aCBd4c8b8c2b,
	    :target:`aBCdef8b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae0edbd382692c2cca2607f00778c7b02>`      = dnnl_aBCdef8b2c,
	    :target:`AB32a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4b2b27218016a35257186150ec09a469>`        = dnnl_AB32a16b,
	    :target:`AB32a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9d538c71c5b158ca60e69d714b4ea01c>`        = dnnl_AB32a32b,
	    :target:`BA4b8a8b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0be0d6e3a9634de6a92b7be9bc5f7bc5>`      = dnnl_BA4b8a8b2a,
	    :target:`BA4b8a8b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3c33659b6dcf008f6d2aae9fd732ada6>`      = dnnl_BA4b8a8b4a,
	    :target:`aBC32b16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad4d6f606a2af7edbc5cc24fb9870274b>`       = dnnl_aBC32b16c,
	    :target:`aBC32b32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa17e9f1b6a7fa4f5d3ddef5eff69f2cbd>`       = dnnl_aBC32b32c,
	    :target:`aCB4c8b8c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa46847a3bf32404cd0fafd74d07e1a247>`     = dnnl_aCB4c8b8c2b,
	    :target:`aCB4c8b8c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf6852e47c367215c6a5c70e4eb30f120>`     = dnnl_aCB4c8b8c4b,
	    :target:`ABc2b8a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7e2b50db55d562a18b3a811d3fb9159e>`    = dnnl_ABc2b8a16b4a,
	    :target:`ABcd2b8a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa83a5b9cbf44355e7a06498a10a6bac2f>`   = dnnl_ABcd2b8a16b4a,
	    :target:`ABcde2b8a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad849762e4301cff95648daee73611165>`  = dnnl_ABcde2b8a16b4a,
	    :target:`ABc2a8b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6f1433bb2c26bcf3dee668c5912fd2b8>`    = dnnl_ABc2a8b16a4b,
	    :target:`ABc2a8b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6adda2dfd0aae815a1f0db7139cc5175>`    = dnnl_ABc2a8b16a2b,
	    :target:`ABc2b32a8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa464cf37f185dd2128d0007b7506f6411>`      = dnnl_ABc2b32a8b,
	    :target:`ABcd2a8b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa72d2665a766a0bceb17d1ea918cf7ded>`   = dnnl_ABcd2a8b16a4b,
	    :target:`ABcd2a8b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9d6f583f8642b92cf14fb9c6be0474ef>`   = dnnl_ABcd2a8b16a2b,
	    :target:`aCBd2c8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facbcfbd20e636720211bebc96cc7f6ca3>`   = dnnl_aCBd2c8b16c2b,
	    :target:`ABcd2b32a8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadc77cf470d98c141bb1c7321dd99eada>`     = dnnl_ABcd2b32a8b,
	    :target:`aBCd2c8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faffae808cb518d334376ec9d9fcdbb5a1>`   = dnnl_aBCd2c8b16c2b,
	    :target:`ABcde2a8b16a4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac7c6a9566328b485c55d0c58d25a8fbd>`  = dnnl_ABcde2a8b16a4b,
	    :target:`ABcde2a8b16a2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa95315349ee810f09727e52b487db3a10>`  = dnnl_ABcde2a8b16a2b,
	    :target:`aCBde2c8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1cc22f10c916443a9205cd45c1fccf5e>`  = dnnl_aCBde2c8b16c2b,
	    :target:`ABcde2b32a8b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad56b1c3c29085c7d89afb54e02a08e93>`    = dnnl_ABcde2b32a8b,
	    :target:`aBC2b8c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2063dc64d1236dab3a5b04308d73c3ee>`    = dnnl_aBC2b8c16b2c,
	    :target:`aBCd2b8c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa905511b378f45173e5eae563716b49a7>`   = dnnl_aBCd2b8c16b2c,
	    :target:`aBCde2b8c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafe87c6ed0a4743ab94a5c537dbf2e161>`  = dnnl_aBCde2b8c16b2c,
	    :target:`aBCdef2b8c16b2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa164309574bf1b3bd3d365fe2830e3501>` = dnnl_aBCdef2b8c16b2c,
	    :target:`BAcde2b8a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab7843219ffd2c6b16b6bd797b753995>`  = dnnl_BAcde2b8a16b4a,
	    :target:`BAcd2b8a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa68d53f9a23dca4e9807a5c67b4b9f695>`   = dnnl_BAcd2b8a16b4a,
	    :target:`BAc2b8a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad00fb667fbbd7cc42ab02626947942d3>`    = dnnl_BAc2b8a16b4a,
	    :target:`BAcde2b8a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3b85219dd3d31122a8215ed54f426bf4>`  = dnnl_BAcde2b8a16b2a,
	    :target:`BAcd2b8a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa529c85e72d82ca6c1a99c377fc891046>`   = dnnl_BAcd2b8a16b2a,
	    :target:`BAc2b8a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf6d968d23c4815ce5e1af850024ea2d6>`    = dnnl_BAc2b8a16b2a,
	    :target:`aBCde2c8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8b7732ff6175f777ebfd8bd65dc487e2>`  = dnnl_aBCde2c8b16c2b,
	    :target:`aBCdef2c8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5314b7c0e8c7b3b8174e7b7f568f50e9>` = dnnl_aBCdef2c8b16c2b,
	    :target:`aCBdef2c8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0318ee9f30e9ea04a51c051b78c53cc5>` = dnnl_aCBdef2c8b16c2b,
	    :target:`aBCd2b8c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa814e8697abc55ec700bbcd15f81ae716>`   = dnnl_aBCd2b8c16b4c,
	    :target:`aBCde2b8c16b4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafd838ffd0a3d4dd76db290857be91379>`  = dnnl_aBCde2b8c16b4c,
	    :target:`NCdhw40n16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8c3d0a164ff81bbe406b15185a696809>`     = dnnl_NCdhw40n16c,
	    :target:`NCw40n16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0dfa5ffc07a2cce5cbd069f58c7835a1>`       = dnnl_NCw40n16c,
	    :target:`NChw40n16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab8c1d6fc48fb80d0d3030e9cbcd7519e>`      = dnnl_NChw40n16c,
	    :target:`NCw2c32n8c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facfccfc0ffa0bb6b57ec5d1e621190004>`      = dnnl_NCw2c32n8c,
	    :target:`NChw2c32n8c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0ae980b47900bcc1fdda7e4478329b62>`     = dnnl_NChw2c32n8c,
	    :target:`NCdhw2c32n8c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadd014af2af3b40ffab5e1286ce29ebb4>`    = dnnl_NCdhw2c32n8c,
	    :target:`OIw2i8o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa38dffefb0addf848914595ff7b85491e>`    = dnnl_OIw2i8o16i4o,
	    :target:`OIhw2i8o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa094e1b40c54c4c7bb075b94e1db2dd47>`   = dnnl_OIhw2i8o16i4o,
	    :target:`OIdhw2i8o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facc9656ae5d46805d2a1f0f44115f5313>`  = dnnl_OIdhw2i8o16i4o,
	    :target:`OIw2o8i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa188319803c864836797ae61401068ddd>`    = dnnl_OIw2o8i16o4i,
	    :target:`OIw2o8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6b79ad34aec94906ec51c5027a318aa8>`    = dnnl_OIw2o8i16o2i,
	    :target:`IOw2i8o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1432c94a0e43069006d158fd4adc7bd6>`    = dnnl_IOw2i8o16i4o,
	    :target:`IOw2i8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faacfe12fb3cde459d3ed8ada29dc9b757>`    = dnnl_IOw2i8o16i2o,
	    :target:`OIhw2o8i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faede7d488f015a093081643f92f90823c>`   = dnnl_OIhw2o8i16o4i,
	    :target:`OIhw2o8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5e265d9d982ae4eb53d2a428d818f628>`   = dnnl_OIhw2o8i16o2i,
	    :target:`IOhw2i8o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa21fe2cad75626584db9b44a3ef7a8a06>`   = dnnl_IOhw2i8o16i4o,
	    :target:`IOhw2i8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa182ab2a3da8f6609b7ea924fc9f23c45>`   = dnnl_IOhw2i8o16i2o,
	    :target:`OIdhw2o8i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa96b884647d6fbdfe4990822cf3f98d1d>`  = dnnl_OIdhw2o8i16o4i,
	    :target:`OIdhw2o8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa736a23db0c119710f531f38f7a8f885a>`  = dnnl_OIdhw2o8i16o2i,
	    :target:`IOdhw2i8o16i4o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa15b8081a5339e2e3d20606c9c2e6a690>`  = dnnl_IOdhw2i8o16i4o,
	    :target:`IOdhw2i8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa64f1b16a8a70dc9ed6ca62b0783bdf5>`  = dnnl_IOdhw2i8o16i2o,
	    :target:`gOIw2o8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9975f09ad86386ea8096e7589f61c86e>`   = dnnl_gOIw2o8i16o2i,
	    :target:`gIOw2i8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7042d07988157d93cebece13049cb318>`   = dnnl_gIOw2i8o16i2o,
	    :target:`gIOhw2i8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad730cf58ae4e6a1879eedbb4360ea7af>`  = dnnl_gIOhw2i8o16i2o,
	    :target:`gIOdhw2i8o16i2o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa87ffa179f900c5805290b78a9f09984c>` = dnnl_gIOdhw2i8o16i2o,
	    :target:`gOIhw2o8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa67f2ab261174895aec2e6373be8595fa>`  = dnnl_gOIhw2o8i16o2i,
	    :target:`gOIdhw2o8i16o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa23d059a3c98513296427fd71abe8c327>` = dnnl_gOIdhw2o8i16o2i,
	    :target:`gOIw2o8i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5281527fed9dd4b5366b07da18007b95>`   = dnnl_gOIw2o8i16o4i,
	    :target:`gOIhw2o8i16o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab953a43c4ccf7e88f14c549b949b299a>`  = dnnl_gOIhw2o8i16o4i,
	    :target:`BA4b8a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa7416618645af07d01ea3f8e906f0881>`     = dnnl_BA4b8a16b2a,
	    :target:`BA4b8a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa85bf17ce462c6e7910668d98b7f87dd5>`     = dnnl_BA4b8a16b4a,
	    :target:`aCB4c8b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab6e98d598b52001f0c1cc2183f21e1ef>`    = dnnl_aCB4c8b16c2b,
	    :target:`aCB4c8b16c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8a9554ca7a2a871b4547eff935f4ea25>`    = dnnl_aCB4c8b16c4b,
	    :target:`aCB16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab4b99ad7314adfb9277a34fa45c1758a>`        = dnnl_aCB16c2b,
	    :target:`aCB16c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad95a43211a94198ec4890092c0b398f3>`        = dnnl_aCB16c4b,
	    :target:`BA16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faceb7c4e428e97f8ef032a29734a3bafd>`         = dnnl_BA16b2a,
	    :target:`BA16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad0927a4d6fe996d56996d15c3af02c67>`         = dnnl_BA16b4a,
	    :target:`aBC16b16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa4a27e1fbe144dcbd1a2d856841232e8>`       = dnnl_aBC16b16c,
	    :target:`aBC16b32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa90b2365f3d3b00022033e9d77fc6155f>`       = dnnl_aBC16b32c,
	    :target:`AB16a16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa134ec05f051ed1c8f429ab6b975e2196>`        = dnnl_AB16a16b,
	    :target:`AB16a32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3328ba495f42b182627f15db6b4808f2>`        = dnnl_AB16a32b,
	    :target:`ABcde16a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0c74273626008796ba0e1c8c0e89dc0c>`   = dnnl_ABcde16a16b2a,
	    :target:`aBCdef16b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8d0eac83472b0e7168776dc6a967430c>`  = dnnl_aBCdef16b16c2b,
	    :target:`Acedb16a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa90f3a663a798ce004c7c3b7033f36c51>`        = dnnl_Acedb16a,
	    :target:`aBdfec16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa07ade1cc4197067f8f486432769f7838>`       = dnnl_aBdfec16b,
	    :target:`Odwhi16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8c80afdcaab25767d4d4b3a8780d5876>`        = dnnl_Odwhi16o,
	    :target:`gOdwhi16o<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7a80533723dea3ce61a5980de2e67ef5>`       = dnnl_gOdwhi16o,
	    :target:`abdEC64e2c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabf2d42e421e302c3f85104cccfeec28d>`      = dnnl_abdEC64e2c,
	    :target:`abdEC64e4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faac7f56894c51a1469c61b85799546646>`      = dnnl_abdEC64e4c,
	    :target:`ldgOI64o2i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa84f8778245dfaf97bd181c1b5a17dff1>`      = abdEC64e2c,
	    :target:`ldgOI64o4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3e03c1b660c46a5292f0490ddf9d025d>`      = abdEC64e4c,
	    :target:`abCd4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa16c13979055de9b462508607165e3ee5>`          = dnnl_abCd4c,
	    :target:`abCde4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facaae54370e69c8076abe6a6d0e9a23fa>`         = dnnl_abCde4c,
	    :target:`abCdef4c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faacb13ddc4f84e66ff81b5277e41e042c>`        = dnnl_abCdef4c,
	    :target:`abCde32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9fa6e2192c362382164e2efd78b8f502>`        = dnnl_abCde32c,
	    :target:`abCdef32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa12c67c54dd56b735c7d4dfd6754db27e>`       = dnnl_abCdef32c,
	    :target:`aCdefB16b32c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa424e0ec5a111296e59c743f277f4940>`  = dnnl_aCdefB16b32c2b,
	    :target:`aCdefB16b32c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa20ede8cc256f145293d8ace24350bde4>`  = dnnl_aCdefB16b32c4b,
	    :target:`aCdefB16b48c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0abb087fbc2720b91c42727fa675f559>`  = dnnl_aCdefB16b48c2b,
	    :target:`aCdefB16b48c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2dcc983358f5b6ee9ead61663c2f5283>`  = dnnl_aCdefB16b48c4b,
	    :target:`aCdefB16b64c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafcb46ab461cb7069ed78e4d7f6c27ff3>`  = dnnl_aCdefB16b64c2b,
	    :target:`aCdefB16b64c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae87580de7e8616f5dcb71834564939cd>`  = dnnl_aCdefB16b64c4b,
	    :target:`BcdeA16a32b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5b687ad7b22fca08c28d3078a8c678cc>`   = dnnl_BcdeA16a32b2a,
	    :target:`BcdeA16a32b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa6f0de3944fabde637534a5b364b6d20>`   = dnnl_BcdeA16a32b4a,
	    :target:`BcdeA16a48b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae27d929df197628e8b2ff70f73daf698>`   = dnnl_BcdeA16a48b2a,
	    :target:`BcdeA16a48b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facc386dedb43ce3e1365df1c9d71254b8>`   = dnnl_BcdeA16a48b4a,
	    :target:`BcdeA16a64b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa67d3120a08531834f3bbd0ead792f050>`   = dnnl_BcdeA16a64b2a,
	    :target:`BcdeA16a64b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa554748ad01c497ec970e16559b4030fd>`   = dnnl_BcdeA16a64b4a,
	    :target:`aCdefb32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5967079a62a192c52ea209d7a9c244b2>`       = dnnl_aCdefb32c,
	    :target:`aCdefB32c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac3dcf626385e06ed258365910dd6f9da>`     = dnnl_aCdefB32c2b,
	    :target:`aCdefB32c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa83ebc4823d1438625c4f192a3e9ab8c7>`     = dnnl_aCdefB32c4b,
	    :target:`aCdefb48c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0686359f58a2452f29cd4d2b0e61f751>`       = dnnl_aCdefb48c,
	    :target:`aCdefB48c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2ffa5e9121b95235e438268865536188>`     = dnnl_aCdefB48c2b,
	    :target:`aCdefB48c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5a74a9e2994f2ab34ebec776d61886f0>`     = dnnl_aCdefB48c4b,
	    :target:`aCdefb64c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac764357caa06d361ea264d53c1c91e99>`       = dnnl_aCdefb64c,
	    :target:`aCdefB64c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2821566c35b0fc651c8396379e1b3f56>`     = dnnl_aCdefB64c2b,
	    :target:`aCdefB64c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa86afa1fd67399494aa15535a8b38a28c>`     = dnnl_aCdefB64c4b,
	    :target:`Bcdea32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa821d6406b504cd86a67801eb47332e0e>`        = dnnl_Bcdea32b,
	    :target:`BcdeA32b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8a7885845c4baf3452f2df8d21735175>`      = dnnl_BcdeA32b2a,
	    :target:`BcdeA32b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa45f27bd6bd0ec873472b5ca4c8bb4991>`      = dnnl_BcdeA32b4a,
	    :target:`Bcdea48b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac2dc56ea61d635be7dd8c287e83d460e>`        = dnnl_Bcdea48b,
	    :target:`BcdeA48b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6e78c54c3de42420cb51f2726c23bb3a>`      = dnnl_BcdeA48b2a,
	    :target:`BcdeA48b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac5cd4dc0e10f96bf254b06c9d95116e0>`      = dnnl_BcdeA48b4a,
	    :target:`Bcdea64b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1d699284adcb03c894e40342e9269325>`        = dnnl_Bcdea64b,
	    :target:`BcdeA64b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa03ccb7c8a8920d8f33d002118edf45d7>`      = dnnl_BcdeA64b2a,
	    :target:`BcdeA64b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0638e2d6d2a657bc6a4d16dcd3b3c70b>`      = dnnl_BcdeA64b4a,
	    :target:`Bca32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5ae0f1c77748b411755288aa1a0d29fb>`          = dnnl_Bca32b,
	    :target:`BcA32b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad2aa5cc3dce52cd3eab304a7d523ac86>`        = dnnl_BcA32b2a,
	    :target:`BcA32b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facda9bff1897a7a353efff44fa161ea9a>`        = dnnl_BcA32b4a,
	    :target:`Bca48b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4d820d42efca02b4236587c1a7fd93ac>`          = dnnl_Bca48b,
	    :target:`BcA48b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad66f3479475cf18eda9480d2e18c527a>`        = dnnl_BcA48b2a,
	    :target:`BcA48b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5da07d1d5ca871641518d776eea603d2>`        = dnnl_BcA48b4a,
	    :target:`Bca64b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa99c3d8182b185f25f4c6ac00f9a6ddaa>`          = dnnl_Bca64b,
	    :target:`BcA64b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa735335a7f90c4e8548c4f57d9110cc8b>`        = dnnl_BcA64b2a,
	    :target:`BcA64b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9e2a442942a687a1ca2a769244690730>`        = dnnl_BcA64b4a,
	    :target:`aCdb32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac6ae3c8179ca8854aac741ac3f07f4ed>`         = dnnl_aCdb32c,
	    :target:`aCdB32c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9832fd22e4db9679b31eaabc8f83beaa>`       = dnnl_aCdB32c2b,
	    :target:`aCdB32c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad3b367cd70605239f38612c582b8e7ff>`       = dnnl_aCdB32c4b,
	    :target:`aCdb48c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9953bf2e9edca3d0a55a41a6f957ff5f>`         = dnnl_aCdb48c,
	    :target:`aCdB48c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa73ca66ff93ad4e0d42ab88759400588d>`       = dnnl_aCdB48c2b,
	    :target:`aCdB48c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa60c0e4942e3a88d355454de1d5ae3ef1>`       = dnnl_aCdB48c4b,
	    :target:`aCdb64c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4f65fb3e7011059d5f50717f33b8d699>`         = dnnl_aCdb64c,
	    :target:`aCdB64c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa88618eda7e44457ae3aebd3659bd8546>`       = dnnl_aCdB64c2b,
	    :target:`aCdB64c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa87035771f768460d17594d5c015365db>`       = dnnl_aCdB64c4b,
	    :target:`BcA16a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa227289ac0b532d3b35f6ee62d204238b>`     = dnnl_BcA16a16b2a,
	    :target:`BcA16a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa735cf04ecd8aa8d6e4e25b46c4a03fcf>`     = dnnl_BcA16a16b4a,
	    :target:`BcdA16a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadaaa187a416bfe9ab36879440765252c>`    = dnnl_BcdA16a16b2a,
	    :target:`BcdA16a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf8d9bba0cff45e837aaf9863585a098a>`    = dnnl_BcdA16a16b4a,
	    :target:`BcdeA16a16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac5c389e04b7ab7489391d67e815dfe06>`   = dnnl_BcdeA16a16b2a,
	    :target:`BcdeA16a16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf9c2208ca753dbe85dbdccd1475a1cec>`   = dnnl_BcdeA16a16b4a,
	    :target:`aCdB16b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafa5676494029e908f2c0aaa1fc9549ee>`    = dnnl_aCdB16b16c2b,
	    :target:`aCdB16b16c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1e456ac75749f9a14eb38627b1c8b431>`    = dnnl_aCdB16b16c4b,
	    :target:`aCdeB16b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa86524329be453e99657bca7573630de6>`   = dnnl_aCdeB16b16c2b,
	    :target:`aCdeB16b16c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6f23c9c33397fe212f0dc75fb841e2ab>`   = dnnl_aCdeB16b16c4b,
	    :target:`aCdefB16b16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa40f28f856af876aa98223a8a3d47004b>`  = dnnl_aCdefB16b16c2b,
	    :target:`aCdefB16b16c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa25ec691c1559a5ccb3f64ea0d349ff3b>`  = dnnl_aCdefB16b16c4b,
	    :target:`BcA16a32b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0d3124fe1349c69f775edd01b818f829>`     = dnnl_BcA16a32b2a,
	    :target:`BcA16a32b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa04d522fa1e4bf05e599b638e6fdbdb5d>`     = dnnl_BcA16a32b4a,
	    :target:`BcA16a48b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf6a95f02c84e807ed56fbc3852b8cdf7>`     = dnnl_BcA16a48b2a,
	    :target:`BcA16a48b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9dcbd982125a0f968e0dffc493905947>`     = dnnl_BcA16a48b4a,
	    :target:`BcA16a64b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac3a01b2e5c7d53108ff11394b68133d3>`     = dnnl_BcA16a64b2a,
	    :target:`BcA16a64b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa454cc50ceaca48054989ff0b5ed142df>`     = dnnl_BcA16a64b4a,
	    :target:`aCdB16b32c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad01dab9e186e59b241a26b0819a888e8>`    = dnnl_aCdB16b32c2b,
	    :target:`aCdB16b32c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e2591911d79d13114647e0a5c366c64>`    = dnnl_aCdB16b32c4b,
	    :target:`aCdB16b48c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac84abe622b73fb7cdae4d5cbf20289b7>`    = dnnl_aCdB16b48c2b,
	    :target:`aCdB16b48c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabe71cd21518e5b8921aff1ca4a1749e4>`    = dnnl_aCdB16b48c4b,
	    :target:`aCdB16b64c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabe41a4f66d373c3f5449ff0f890d7858>`    = dnnl_aCdB16b64c2b,
	    :target:`aCdB16b64c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa83b15ffd9b71c06117577395f8bf777f>`    = dnnl_aCdB16b64c4b,
	    :target:`BcdA16a32b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf8df89d0c5e08443c3a4b0ebf62d4652>`    = dnnl_BcdA16a32b2a,
	    :target:`BcdA16a32b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab1d15922e361d0d245fdc4dbef55560c>`    = dnnl_BcdA16a32b4a,
	    :target:`BcdA16a48b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa06681bdb8f5a10121bcb205400721f7f>`    = dnnl_BcdA16a48b2a,
	    :target:`BcdA16a48b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa936041abd510ea410639a4d454b4c391>`    = dnnl_BcdA16a48b4a,
	    :target:`BcdA16a64b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae7dbc0e835825f9c0b162e837726a7f5>`    = dnnl_BcdA16a64b2a,
	    :target:`BcdA16a64b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa400553fff569bf06cf4851ad3103b8d5>`    = dnnl_BcdA16a64b4a,
	    :target:`aCdeB16b32c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa841c2cb53ed2e3e744883dc1a79d207c>`   = dnnl_aCdeB16b32c2b,
	    :target:`aCdeB16b32c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab4b8abec245265223e07f0a5b16fb4ca>`   = dnnl_aCdeB16b32c4b,
	    :target:`aCdeB16b48c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2d6e3ae54b57d1e10f67c0a5a063fb20>`   = dnnl_aCdeB16b48c2b,
	    :target:`aCdeB16b48c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa478115d5982fb3951d4cf1c9c47dcbd2>`   = dnnl_aCdeB16b48c4b,
	    :target:`aCdeB16b64c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4360e8f9a77dd31aab852cd65c8b89b4>`   = dnnl_aCdeB16b64c2b,
	    :target:`aCdeB16b64c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facc53ee9019a29334ea41c70287db3541>`   = dnnl_aCdeB16b64c4b,
	    :target:`Bca16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa279de3c8042d2b6c85d0e836b526d3c>`          = dnnl_Bca16b,
	    :target:`BcA16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab18f03284b55c5cc2bc4577d3a69805c>`        = dnnl_BcA16b2a,
	    :target:`BcA16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa311c77ef65fa856df8740b741547c675>`        = dnnl_BcA16b4a,
	    :target:`Bcda16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa02474caa2ac1009cb2d02ccfcace710c>`         = dnnl_Bcda16b,
	    :target:`BcdA16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa441206c6e84b0e5d8c1a1c3ca57b4ad4>`       = dnnl_BcdA16b2a,
	    :target:`BcdA16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa45caff544e88fd5a4f482ef717a0279d>`       = dnnl_BcdA16b4a,
	    :target:`Bcdea16b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa005afa3b893eca010e63a4e4d5e2ad60>`        = dnnl_Bcdea16b,
	    :target:`BcdeA16b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafe335fb316e17533a79b050c7d90167d>`      = dnnl_BcdeA16b2a,
	    :target:`BcdeA16b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6eb6327d195ba4097190ce68c36485a8>`      = dnnl_BcdeA16b4a,
	    :target:`aCdb16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faed336d385d0d93f19ada51267a4193cd>`         = dnnl_aCdb16c,
	    :target:`aCdB16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa44ce66cdde6846eef95f081642562a2>`       = dnnl_aCdB16c2b,
	    :target:`aCdB16c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf8d2671f0d465e2694a208fe7aa9cee8>`       = dnnl_aCdB16c4b,
	    :target:`aCdeb16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa06baf524a79696bdf4e5366ee5d21149>`        = dnnl_aCdeb16c,
	    :target:`aCdeB16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadb48f70cbb625364f7d9d2c49454d5c6>`      = dnnl_aCdeB16c2b,
	    :target:`aCdeB16c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa55ee84ab2aa0fec523e4a47884c6848e>`      = dnnl_aCdeB16c4b,
	    :target:`aCdefb16c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabbe82c521c22134cb943720c1c6dc532>`       = dnnl_aCdefb16c,
	    :target:`aCdefB16c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0db7c286c0304768aa85fcd0516010b4>`     = dnnl_aCdefB16c2b,
	    :target:`aCdefB16c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae6739de7947b03f3131881c09fb7679c>`     = dnnl_aCdefB16c4b,
	    :target:`Bcda32b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fada563c14b28be79cef186b9e8b827a7c>`         = dnnl_Bcda32b,
	    :target:`BcdA32b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3ae210f0bd238b2eda1f87a6e4721a31>`       = dnnl_BcdA32b2a,
	    :target:`BcdA32b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac77aae2296bab203b961a44bf71230e1>`       = dnnl_BcdA32b4a,
	    :target:`Bcda48b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2c77e03ca4b5fbcf0a316f74f3dbbbd9>`         = dnnl_Bcda48b,
	    :target:`BcdA48b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4c990f5a970ec124df2297fe33345bc6>`       = dnnl_BcdA48b2a,
	    :target:`BcdA48b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab906d4eba062890bf9d8cac6d3176cc8>`       = dnnl_BcdA48b4a,
	    :target:`Bcda64b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faadc3109326e2d60e147be43bed9cd0f4>`         = dnnl_Bcda64b,
	    :target:`BcdA64b2a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0fc5cf622f3c53bdfefbd680f9078962>`       = dnnl_BcdA64b2a,
	    :target:`BcdA64b4a<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabc01ef115c4b16e7096ac86effd8b063>`       = dnnl_BcdA64b4a,
	    :target:`aCdeb32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa77daa43bfd164ddec8bffae7c661d618>`        = dnnl_aCdeb32c,
	    :target:`aCdeB32c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa32aaca015b34c5916477933449a08def>`      = dnnl_aCdeB32c2b,
	    :target:`aCdeB32c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5dfaeed17eb62cc877afa2722db14240>`      = dnnl_aCdeB32c4b,
	    :target:`aCdeb48c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac41d03019d0f4ccaa2848b7c2698146d>`        = dnnl_aCdeb48c,
	    :target:`aCdeB48c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6d6ad200c12d03dc2b4cca37d0a97639>`      = dnnl_aCdeB48c2b,
	    :target:`aCdeB48c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad17e955e18dbab0611033650ab59d774>`      = dnnl_aCdeB48c4b,
	    :target:`aCdeb64c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0d4964caedf59d46fa1fa6e983061a99>`        = dnnl_aCdeb64c,
	    :target:`aCdeB64c2b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa54644bc5ce6eede7e3320a4227877299>`      = dnnl_aCdeB64c2b,
	    :target:`aCdeB64c4b<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3facd6c9a638a1620afde16a86596640124>`      = dnnl_aCdeB64c4b,
	    :target:`NChw16n32c<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2239f3c3707416d130ba2a98abeb1536>`      = dnnl_NChw16n32c,
	    :target:`goIw4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4da9f8d2d12932fbd7c660585ca81f5e>`          = dnnl_goIw4i,
	    :target:`goIw32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa53a685f4fe8144c9fbb7900be9c235f>`         = dnnl_goIw32i,
	    :target:`goIhw4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf0be2e0d5b17c2b9ee21a187bff53d86>`         = dnnl_goIhw4i,
	    :target:`goIhw32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa26bbdb2dc309775de7702df8454c674d>`        = dnnl_goIhw32i,
	    :target:`goIdhw4i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6f56834e68307c5b1f70f4b7c9536f7f>`        = dnnl_goIdhw4i,
	    :target:`goIdhw32i<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa361f1f8f3ef0981e15faf68bf280db9>`       = dnnl_goIdhw32i,
	    :target:`cab<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa16ecfd64586ec6c1ab212762c2c38a90>`             = dnnl_cab,
	    :target:`cdab<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faba8920719baed64ead4b2ee37770c56a>`            = dnnl_cdab,
	    :target:`cdeab<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafe9f9399613d44d39e47b04cf7dd10fd>`           = dnnl_cdeab,
	    :target:`woi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3d8cf6bf0f86c00fb2031bdef989bf91>`             = dnnl_woi,
	    :target:`hwoi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa44dff5ccf70539482e1ae7b47f9c7686>`            = dnnl_hwoi,
	    :target:`dhwoi<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa20a74ec68a0d00f1fd423ed04b8588a6>`           = dnnl_dhwoi,
	};

.. _details-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Memory format tag specification.

Memory format tags can be further divided into two categories:

* Domain-agnostic names, i.e. names that do not depend on the tensor usage in the specific primitive. These names use letters from ``a`` to ``f`` to denote logical dimensions and form the order in which the dimensions are laid in memory. For example, :ref:`dnnl::memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>` is used to denote a 2D tensor where the second logical dimension (denoted as ``b``) is the innermost, i.e. has stride = 1, and the first logical dimension (``a``) is laid out in memory with stride equal to the size of the second dimension. On the other hand, :ref:`dnnl::memory::format_tag::ba <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa07159c47ee1b19ae4fb9c40d480856c4>` is the transposed version of the same tensor: the outermost dimension (``a``) becomes the innermost one.

* Domain-specific names, i.e. names that make sense only in the context of a certain domain, such as CNN. These names are aliases to the corresponding domain-agnostic tags and used mostly for convenience. For example, :ref:`dnnl::memory::format_tag::nc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1e7342845e24eb3b5b3554490da1c128>` is used to denote 2D CNN activations tensor memory format, where the channels dimension is the innermost one and the batch dimension is the outermost one. Moreover, :ref:`dnnl::memory::format_tag::nc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1e7342845e24eb3b5b3554490da1c128>` is an alias for :ref:`dnnl::memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`, because for CNN primitives the logical dimensions of activations tensors come in order: batch, channels, spatial. In other words, batch corresponds to the first logical dimension (``a``), and channels correspond to the second one (``b``).

The following domain-specific notation applies to memory format tags:

* ``'n'`` denotes the mini-batch dimension

* ``'c'`` denotes a channels dimension

* When there are multiple channel dimensions (for example, in convolution weights tensor), ``'i'`` and ``'o'`` denote dimensions of input and output channels

* ``'g'`` denotes a groups dimension for convolution weights

* ``'d'``, ``'h'``, and ``'w'`` denote spatial depth, height, and width respectively

See :ref:`dnnl_format_tag_t <doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` for a detailed description.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined memory format tag.

.. index:: pair: enumvalue; any
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	any

Placeholder memory format tag.

Used to instruct the primitive to select a format automatically.

.. index:: pair: enumvalue; a
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0cc175b9c0f1b6a831c399e269772661:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	a

plain 1D tensor

.. index:: pair: enumvalue; ab
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ab

plain 2D tensor

.. index:: pair: enumvalue; ba
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa07159c47ee1b19ae4fb9c40d480856c4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ba

permuted 2D tensor

.. index:: pair: enumvalue; abc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa900150983cd24fb0d6963f7d28e17f72:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abc

plain 3D tensor

.. index:: pair: enumvalue; acb
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5ff832d9bca8241d653279756f3ccd11:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	acb

permuted 3D tensor

.. index:: pair: enumvalue; bac
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79ec16df80b57696a03bb364410061f3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bac

permuted 3D tensor

.. index:: pair: enumvalue; bca
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab64eab8ce39e013604e243089c687e4f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bca

permuted 3D tensor

.. index:: pair: enumvalue; cba
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3944b025c9ca7eec3154b44666ae04a0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cba

permuted 3D tensor

.. index:: pair: enumvalue; abcd
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae2fc714c4727ee9395f324cd2e7f331f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcd

plain 4D tensor

.. index:: pair: enumvalue; abdc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5e822a0b52c71ce2ff2ecd3aa8ca46fc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abdc

permuted 4D tensor

.. index:: pair: enumvalue; acbd
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa49780bb1cc9c6c4be2da2a34283f72a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	acbd

permuted 4D tensor

.. index:: pair: enumvalue; acdb
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa776649ab3d00664587b889457db87da7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	acdb

permuted 4D tensor

.. index:: pair: enumvalue; adbc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaa6ec9bf967afd962bf57cda5c588104:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	adbc

permuted 4D tensor

.. index:: pair: enumvalue; bacd
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5a3b732e0600663606054f6d8dfd465e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bacd

permuted 4D tensor

.. index:: pair: enumvalue; bcda
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa078c4e73146ba90c3e4e9daa1facac28:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bcda

permuted 4D tensor

.. index:: pair: enumvalue; cdba
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faff2458511a39d703bd273eabbb1395d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cdba

permuted 4D tensor

.. index:: pair: enumvalue; dcab
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9737604d6ad5938095054dc4413bf8e9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dcab

permuted 4D tensor

.. index:: pair: enumvalue; abcde
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab56b4d92b40713acc5af89985d4b786:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcde

plain 5D tensor

.. index:: pair: enumvalue; abdec
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7a05226d93263d6fc613f52400f7416d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abdec

permuted 5D tensor

.. index:: pair: enumvalue; acbde
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae0e56a95aff0ee48fc44e8a4faf69adb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	acbde

permuted 5D tensor

.. index:: pair: enumvalue; acdeb
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf95d333a023ac787b43babc5b5c480ae:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	acdeb

permuted 5D tensor

.. index:: pair: enumvalue; bacde
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabb59e2a27c3cb4e24384952fc6847948:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bacde

permuted 5D tensor

.. index:: pair: enumvalue; bcdea
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6debec47fc73823aa65cc49b94b7e1fb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bcdea

permuted 5D tensor

.. index:: pair: enumvalue; cdeba
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafafcbfdfcede01281676ab63df88a919:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cdeba

permuted 5D tensor

.. index:: pair: enumvalue; decab
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab9a152e018d02395e6e94c5619da4a91:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	decab

permuted 5D tensor

.. index:: pair: enumvalue; abced
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faebb080afaac3a990ad3f1d0f21742fac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abced

permuted 5D tensor

.. index:: pair: enumvalue; abcdef
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae80b5017098950fc58aad83c8c14978e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdef

plain 6D tensor

.. index:: pair: enumvalue; abdfce
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7c4600767ddcc2dba9d42133b390e593:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abdfce

permuted 6D tensor

.. index:: pair: enumvalue; acbdef
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa66fd87374b1ec6be9fdbea534e17adeb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	acbdef

permuted 6D tensor

.. index:: pair: enumvalue; abdefc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0a0eee0ab79fa687f0f63332c252d13a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abdefc

permuted 6D tensor

.. index:: pair: enumvalue; defcab
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac9cdbbb9640597fea44206cee7f8a151:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	defcab

permuted 6D tensor

.. index:: pair: enumvalue; abcdfe
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeb4c431d71c552a2bc16fab86156738e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdfe

permuted 6D tensor

.. index:: pair: enumvalue; abcdefg
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7ac66c0f148de9519b8bd264312c4d64:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefg

plain 7D tensor

.. index:: pair: enumvalue; abcdegf
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa189fe48d1050763b571517a9e2aa113b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdegf

permuted 7D tensor

.. index:: pair: enumvalue; abcdefgh
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae8dc4081b13434b45189a720b77b6818:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefgh

plain 8D tensor

.. index:: pair: enumvalue; abcdefhg
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9899b60d60de786c689156d87433c254:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefhg

permuted 8D tensor

.. index:: pair: enumvalue; abcdefghi
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa8aa99b1f439ff71293e95357bac6fd94:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefghi

plain 9D tensor

.. index:: pair: enumvalue; abcdefgih
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa838730cad01311dbfba64353015e329b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefgih

permuted 9D tensor

.. index:: pair: enumvalue; abcdefghij
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa925576942e94b2ef57a066101b48876:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefghij

plain 10D tensor

.. index:: pair: enumvalue; abcdefghji
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa84356605428eb3b9ad6da81d3d0b5b26:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefghji

permuted 10D tensor

.. index:: pair: enumvalue; abcdefghijk
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa92b9cccc0b98c3a0b8d0df25a421c0e3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefghijk

plain 11D tensor

.. index:: pair: enumvalue; abcdefghikj
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeb9ee2026bb8d37a7e777264cf4bcde2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefghikj

permuted 11D tensor

.. index:: pair: enumvalue; abcdefghijkl
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9fc9d606912030dca86582ed62595cf7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefghijkl

plain 12D tensor

.. index:: pair: enumvalue; abcdefghijlk
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9b19e0f5975e33696433bcc7360c61aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	abcdefghijlk

permuted 12D tensor

.. index:: pair: enumvalue; x
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9dd4e461268c8034f5c8564e155c67a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	x

1D tensor; an alias for :ref:`dnnl::memory::format_tag::a <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0cc175b9c0f1b6a831c399e269772661>`

.. index:: pair: enumvalue; nc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1e7342845e24eb3b5b3554490da1c128:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	nc

2D CNN activations tensor; an alias for :ref:`dnnl::memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`

.. index:: pair: enumvalue; cn
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7efdfc94655a25dcea3ec85e9bb703fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cn

2D CNN activations tensor; an alias for :ref:`dnnl::memory::format_tag::ba <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa07159c47ee1b19ae4fb9c40d480856c4>`

.. index:: pair: enumvalue; tn
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faaafb96b2fa8806be307c4496867bad56:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	tn

2D RNN statistics tensor; an alias for :ref:`dnnl::memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`

.. index:: pair: enumvalue; nt
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa25930e3036f13852cb0b29694bbab611:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	nt

2D RNN statistics tensor; an alias for :ref:`dnnl::memory::format_tag::ba <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa07159c47ee1b19ae4fb9c40d480856c4>`

.. index:: pair: enumvalue; ncw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabc4eee572ad38f8a5c2534e0b2533100:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ncw

3D CNN activations tensor; an alias for :ref:`dnnl::memory::format_tag::abc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa900150983cd24fb0d6963f7d28e17f72>`

.. index:: pair: enumvalue; nwc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faccc0543dc9272f9fffd720fe55f818e4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	nwc

3D CNN activations tensor; an alias for :ref:`dnnl::memory::format_tag::acb <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5ff832d9bca8241d653279756f3ccd11>`

.. index:: pair: enumvalue; nchw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	nchw

4D CNN activations tensor; an alias for :ref:`dnnl::memory::format_tag::abcd <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae2fc714c4727ee9395f324cd2e7f331f>`

.. index:: pair: enumvalue; nhwc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa763cbf7ba1b7b8793dcdc6e2157b5c42:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	nhwc

4D CNN activations tensor; an alias for :ref:`dnnl::memory::format_tag::acdb <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa776649ab3d00664587b889457db87da7>`

.. index:: pair: enumvalue; chwn
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafd2263500e815d8ce46e79e6b178a10a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	chwn

4D CNN activations tensor; an alias for :ref:`dnnl::memory::format_tag::bcda <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa078c4e73146ba90c3e4e9daa1facac28>`

.. index:: pair: enumvalue; ncdhw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae7a84f08ef10d4fe3096307c6d9770fd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ncdhw

5D CNN activations tensor; an alias for :ref:`dnnl::memory::format_tag::abcde <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab56b4d92b40713acc5af89985d4b786>`

.. index:: pair: enumvalue; ndhwc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fad2f2770ca9883423f680fd4a29481d64:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ndhwc

5D CNN activations tensor; an alias for :ref:`dnnl::memory::format_tag::acdeb <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf95d333a023ac787b43babc5b5c480ae>`

.. index:: pair: enumvalue; oi
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa2e63ee01401aaeca78be023dfbb8c59:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	oi

2D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`

.. index:: pair: enumvalue; io
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf98ed07a4d5f50f7de1410d905f1477f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	io

2D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::ba <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa07159c47ee1b19ae4fb9c40d480856c4>`

.. index:: pair: enumvalue; oiw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3a8cb832bca6da72735b742014560a5b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	oiw

3D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::abc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa900150983cd24fb0d6963f7d28e17f72>`

.. index:: pair: enumvalue; owi
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa55e9a1b1cc2aaf0f5b7417d8098ea4b3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	owi

3D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::acb <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5ff832d9bca8241d653279756f3ccd11>`

.. index:: pair: enumvalue; wio
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5b12270b2f1c52cb9ae37905880a5a6f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	wio

3D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::cba <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3944b025c9ca7eec3154b44666ae04a0>`

.. index:: pair: enumvalue; iwo
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7afa5d2c0d2aa139eb213430ec70d91d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	iwo

3D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::bca <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab64eab8ce39e013604e243089c687e4f>`

.. index:: pair: enumvalue; oihw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa14b72a467aeefa06a5cb802ec4a7743c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	oihw

4D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::abcd <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae2fc714c4727ee9395f324cd2e7f331f>`

.. index:: pair: enumvalue; hwio
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fadd942f902c741de7193e1869b13ba55a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	hwio

4D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::cdba <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faff2458511a39d703bd273eabbb1395d6>`

.. index:: pair: enumvalue; ohwi
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab889a0cb4d051674706433941a2e867f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ohwi

4D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::acdb <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa776649ab3d00664587b889457db87da7>`

.. index:: pair: enumvalue; ihwo
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa56fe35ca9f339b4948384c2a2fa02de5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ihwo

4D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::bcda <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa078c4e73146ba90c3e4e9daa1facac28>`

.. index:: pair: enumvalue; iohw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac16710619e6550753801926b5388bef8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	iohw

4D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::bacd <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5a3b732e0600663606054f6d8dfd465e>`

.. index:: pair: enumvalue; oidhw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa98407962f915285419b23f5b0a223882:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	oidhw

5D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::abcde <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab56b4d92b40713acc5af89985d4b786>`

.. index:: pair: enumvalue; dhwio
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faeafcb3d831eef23cde0b46f64646327c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dhwio

5D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::cdeba <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafafcbfdfcede01281676ab63df88a919>`

.. index:: pair: enumvalue; odhwi
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabf55006478a9dae3506eb67923703a1d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	odhwi

5D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::acdeb <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faf95d333a023ac787b43babc5b5c480ae>`

.. index:: pair: enumvalue; iodhw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa940f633559dc193ac5a96584394774a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	iodhw

5D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::bacde <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabb59e2a27c3cb4e24384952fc6847948>`

.. index:: pair: enumvalue; idhwo
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa2b3d6389089fb08059ece8a9642b8991:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	idhwo

5D CNN weights tensor; an alias for :ref:`dnnl::memory::format_tag::bcdea <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6debec47fc73823aa65cc49b94b7e1fb>`

.. index:: pair: enumvalue; goiw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1f1bc797706732c22eb9987e3cea3e54:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	goiw

4D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::abcd <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae2fc714c4727ee9395f324cd2e7f331f>`

.. index:: pair: enumvalue; gowi
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa113968d2d4d3a82b7880657ac86060ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	gowi

4D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::abdc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5e822a0b52c71ce2ff2ecd3aa8ca46fc>`

.. index:: pair: enumvalue; wigo
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1609c2708a916705421742969a35351c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	wigo

4D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::dcab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9737604d6ad5938095054dc4413bf8e9>`

.. index:: pair: enumvalue; gohwi
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa371d33c7ac584e28d7243cd6f2e72bfb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	gohwi

5D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::abdec <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7a05226d93263d6fc613f52400f7416d>`

.. index:: pair: enumvalue; goihw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa3dc0e53430a6368210356118196f821a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	goihw

5D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::abcde <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab56b4d92b40713acc5af89985d4b786>`

.. index:: pair: enumvalue; hwigo
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafd710c828421b3c91725b0e5aa53ecc6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	hwigo

5D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::decab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab9a152e018d02395e6e94c5619da4a91>`

.. index:: pair: enumvalue; giohw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5f1d6f8a3a8842611d15affa574b864f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	giohw

5D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::acbde <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae0e56a95aff0ee48fc44e8a4faf69adb>`

.. index:: pair: enumvalue; goidhw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faa678de4e5a5e34c9a71a632b8f334b8b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	goidhw

6D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::abcdef <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae80b5017098950fc58aad83c8c14978e>`

.. index:: pair: enumvalue; giodhw
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4f2399cfdc9861fee9d39cb1d703aadc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	giodhw

6D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::abcdef <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae80b5017098950fc58aad83c8c14978e>`

.. index:: pair: enumvalue; godhwi
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faad93659c675a47264a01968e00dc1714:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	godhwi

6D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::abdefc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0a0eee0ab79fa687f0f63332c252d13a>`

.. index:: pair: enumvalue; dhwigo
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab291a9717fdf353288d2d6082d578059:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dhwigo

6D CNN weights tensor with groups; an alias for :ref:`dnnl::memory::format_tag::defcab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac9cdbbb9640597fea44206cee7f8a151>`

.. index:: pair: enumvalue; tnc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac775cf954921a129a65eb929476de911:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	tnc

3D RNN data tensor in the format (seq_length, batch, input channels); an alias for :ref:`dnnl::memory::format_tag::abc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa900150983cd24fb0d6963f7d28e17f72>`.

.. index:: pair: enumvalue; ntc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa6758344f9f89d390953c6718f34e9be1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ntc

3D RNN data tensor in the format (batch, seq_length, input channels); an alias for :ref:`dnnl::memory::format_tag::bac <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79ec16df80b57696a03bb364410061f3>`.

.. index:: pair: enumvalue; ldnc
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab49be97ff353a86d84d06d98f846b61d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ldnc

4D RNN states tensor in the format (num_layers, num_directions, batch, state channels); an alias for :ref:`dnnl::memory::format_tag::abcd <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae2fc714c4727ee9395f324cd2e7f331f>`.

.. index:: pair: enumvalue; ldigo
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ldigo

5D RNN weights tensor in the format (num_layers, num_directions, input_channels, num_gates, output_channels); an alias for :ref:`dnnl::memory::format_tag::abcde <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faab56b4d92b40713acc5af89985d4b786>`.

* For LSTM cells, the gates order is input, forget, candidate and output gate.

* For GRU cells, the gates order is update, reset and output gate.

.. index:: pair: enumvalue; ldgoi
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4c850fa02c9e56a4c20dbfde737f1f8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ldgoi

5D RNN weights tensor in the format (num_layers, num_directions, num_gates, output_channels, input_channels); an alias for :ref:`dnnl::memory::format_tag::abdec <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa7a05226d93263d6fc613f52400f7416d>`.

* For LSTM cells, the gates order is input, forget, candidate and output gate.

* For GRU cells, the gates order is update, reset and output gate.

.. index:: pair: enumvalue; ldio
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa201b31c3cfbeec4962a869bdc27da090:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ldio

4D LSTM projection tensor in the format (num_layers, num_directions, num_channels_in_hidden_state, num_channels_in_recurrent_projection); an alias for :ref:`dnnl::memory::format_tag::abcd <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae2fc714c4727ee9395f324cd2e7f331f>`.

.. index:: pair: enumvalue; ldoi
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79ca41a567de72f7df2782e76dea722d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ldoi

4D LSTM projection tensor in the format (num_layers, num_directions, num_channels_in_recurrent_projection, num_channels_in_hidden_state); an alias for :ref:`dnnl::memory::format_tag::abdc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5e822a0b52c71ce2ff2ecd3aa8ca46fc>`.

.. index:: pair: enumvalue; ldgo
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab8690cd92ccee6a0ad55faccc0346aab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ldgo

4D RNN bias tensor in the format (num_layers, num_directions, num_gates, output_channels); an alias for :ref:`dnnl::memory::format_tag::abcd <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae2fc714c4727ee9395f324cd2e7f331f>`.

* For LSTM cells, the gates order is input, forget, candidate and output gate.

* For GRU cells, the gates order is update, reset and output gate.

.. index:: pair: enumvalue; aBcd8b
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa448a7fc9219294ce172b0edf9498b5c4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	aBcd8b

4D tensor blocked by 2nd dimension with block size 8

.. index:: pair: enumvalue; ABcd8b8a
.. _doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabcbce50e9c241458767871fa053e1ba0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ABcd8b8a

4D tensor blocked by 1st and 2nd dimension with block size 8

