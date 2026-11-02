.. index:: pair: enum; dnnl_format_tag_t
.. _doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d:

enum dnnl_format_tag_t
======================

Overview
~~~~~~~~

Memory format tag specification. :ref:`More...<details-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_format_tag_t
	{
	    :ref:`dnnl_format_tag_undef<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6f473b40f7e7470ba886ce3220ec69a7>` = 0,
	    :ref:`dnnl_format_tag_any<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`,
	    :ref:`dnnl_a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a72c401669bf1737439d6c4af17d0be>`,
	    :ref:`dnnl_ab<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53>`,
	    :ref:`dnnl_abc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadff5ea69392d7e4da23179dc0ba7cbc2>`,
	    :ref:`dnnl_abcd<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>`,
	    :ref:`dnnl_abcde<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da30d5d3c9de2931f06d265af81787ada3>`,
	    :ref:`dnnl_abcdef<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da74b2db34e6253e56f441764d16e6ec42>`,
	    :ref:`dnnl_abcdefg<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa6db6b10c9279959f5de845d454c19f0>`,
	    :ref:`dnnl_abcdefgh<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da12cecc4b1febc6f7bbc4ddced6482155>`,
	    :ref:`dnnl_abcdefghi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0a0796a36ef631a3f1bdd9210e59684e>`,
	    :ref:`dnnl_abcdefghij<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafaaf9a62cda825a376f1e35936dbf5b1>`,
	    :ref:`dnnl_abcdefghijk<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da571cdcd3d6fd3ae211377f7c2068846a>`,
	    :ref:`dnnl_abcdefghijkl<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da24084d001c72ff2e0543fe09d543678f>`,
	    :ref:`dnnl_ba<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6a6dbc0b30468d92e32a9cb3f6615c43>`,
	    :ref:`dnnl_acb<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8537ed269eb5d0586456db114039c00>`,
	    :ref:`dnnl_bac<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae31cc3d3a0bfbd4936f7b503cf6dc9de>`,
	    :ref:`dnnl_bca<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da37fcc057fe063d378bb5f1d5111e0e60>`,
	    :ref:`dnnl_cab<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4a42ef47bea9ef571f64944b62fddea3>`,
	    :ref:`dnnl_cba<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5a8de15eee2e5bcc2515eb7b01965789>`,
	    :ref:`dnnl_abdc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad7bcb765b861ed46b2b752e912989217>`,
	    :ref:`dnnl_acbd<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6a81c235d5a08451ac61edb1681bd925>`,
	    :ref:`dnnl_acdb<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fcce5dd7260b5b0740e3b37b1e9ad41>`,
	    :ref:`dnnl_adbc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf3800b4f18027abd248a61ae7c5fc8e0>`,
	    :ref:`dnnl_adcb<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac3b46f2409584aab3c10f8189a9fb47f>`,
	    :ref:`dnnl_bacd<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab1a39583c90f536df19e47d8262ef6f6>`,
	    :ref:`dnnl_bcda<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac42f365ea58d5b73c141c0788641f74a>`,
	    :ref:`dnnl_cdab<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e99e668dcd118f6f576c7be57d5c43d>`,
	    :ref:`dnnl_cdba<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2368e1a4ce9d9954fa10c242569eacb4>`,
	    :ref:`dnnl_dcab<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae6c549f1a265f8d61bffb24c24925a1d>`,
	    :ref:`dnnl_abced<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac2006f669dbddbc93e5fd8cf2abbe005>`,
	    :ref:`dnnl_abdec<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab417c26841486da2a0da84730d0874d>`,
	    :ref:`dnnl_acbde<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac5f398cc58f5625915719ec541bb152e>`,
	    :ref:`dnnl_acdeb<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfe86402763786b9b4d73062cfd2f05>`,
	    :ref:`dnnl_adecb<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf87ca2ebe0be8da766b84f3a6bdf72ff>`,
	    :ref:`dnnl_bacde<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da57c548e503e61b8c0f9a70a0a9acce1c>`,
	    :ref:`dnnl_bcdea<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad219c123f120d32a5f26d584b334af41>`,
	    :ref:`dnnl_cdeab<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf4870a90b840aed0c8162f0fa5822967>`,
	    :ref:`dnnl_cdeba<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae7dd2b81eb2b502efbf65c888665b358>`,
	    :ref:`dnnl_decab<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da31fa5f2eb4a236664583afc291310ec6>`,
	    :ref:`dnnl_abcdfe<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2e7f7dc1fa5cc578ee70b018ae78d63d>`,
	    :ref:`dnnl_abdefc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad23172b9338edbd306f84f24fd7b00be>`,
	    :ref:`dnnl_abdfce<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa9483a5b4d17c8aebd14a9894ff67fe5>`,
	    :ref:`dnnl_acbdef<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8463916a68cb816050fbaf71817093ae>`,
	    :ref:`dnnl_adefcb<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da934ab2ceb3d0f6b358f3724b407811a0>`,
	    :ref:`dnnl_defcab<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da73502e9308b365e74da295eaa83703ad>`,
	    :ref:`dnnl_abcdegf<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da28fa9c38c8aba3f7366a40c5f28f5894>`,
	    :ref:`dnnl_abcdefhg<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da013495943a14ba43871723e2a1f29de8>`,
	    :ref:`dnnl_abcdefgih<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad1ce4f5554fe6c5338ca3de649477545>`,
	    :ref:`dnnl_abcdefghji<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8dc3817d010338ab9150378f8fded79>`,
	    :ref:`dnnl_abcdefghikj<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da190b0571914d2675e1bc51d154e0de48>`,
	    :ref:`dnnl_abcdefghijlk<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadeb66e32bfc557ea52fbeeb532287b45>`,
	    :target:`dnnl_Abc16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1f3f02b350c925f3d9c897d73bace402>`,
	    :target:`dnnl_ABc16a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8e6eb934f2389e342c76e97a68ab0224>`,
	    :target:`dnnl_ABc32a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da980bc9736517045eaf801fbfb9875733>`,
	    :target:`dnnl_ABc4a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae2f36245a8520ef772172ceca76fb991>`,
	    :ref:`dnnl_aBc16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3e2da6258e387e0eb149b22f57209056>`,
	    :target:`dnnl_ABc16b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae179e674c582f816b4354bea5be4d28b>`,
	    :target:`dnnl_Abc4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da15c4f79e87162a887ab13eb0b0a7e4fc>`,
	    :ref:`dnnl_aBc32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8c6671d24cda4d6ea0fa7d8613f686c>`,
	    :ref:`dnnl_aBc4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4df9991897c2031b5fbdf97a4b3d2a02>`,
	    :target:`dnnl_ABc4b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cf6b7ee629a1b3bc03904d95e8255f4>`,
	    :target:`dnnl_ABc2b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac76353027eb441b08f92530da0c0fd48>`,
	    :target:`dnnl_ABc16b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa9cb05b0c68df7ca860dfa78ff61744c>`,
	    :target:`dnnl_ABc16b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac2d6936a95341123206597da7844f856>`,
	    :target:`dnnl_ABc4b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0297a6563d66c10695e7141d6d45b28c>`,
	    :target:`dnnl_ABc8a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2b98e94f21d758012a4f42b8a67cc448>`,
	    :target:`dnnl_ABc8a8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da63923570626e9ffe67216e8ec758f7be>`,
	    :target:`dnnl_ABc8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf11f206fb73103511ee3938716b82bcb>`,
	    :ref:`dnnl_aBc8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da96fcac28a789530765def2cfae2b61f6>`,
	    :target:`dnnl_ABc8b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da818c91596f569b7d3c3d8b21de045dc3>`,
	    :target:`dnnl_BAc8a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae8b5332e47484638a0bf9e553780ff1c>`,
	    :target:`dnnl_ABc8b8a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8bf604879f819db21d895dec9453b146>`,
	    :target:`dnnl_Abcd16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da00a609b0923525efd734be895d32fe65>`,
	    :target:`dnnl_Abcd8a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac84aa1648e12ca70a06f39bf94647b23>`,
	    :target:`dnnl_ABcd16a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9c0ae577e44f83d190d1a347f96e0357>`,
	    :target:`dnnl_Abcd32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7354dbec2128b136c2e894b947c59399>`,
	    :target:`dnnl_ABcd32a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83ae1d8b994297601096fdcfd5e59668>`,
	    :ref:`dnnl_aBcd16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da590d748ac3761a51b407555a9a0d5efe>`,
	    :target:`dnnl_ABcd16b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2caad9e6b5ef8da3a86b78f0a372eda7>`,
	    :target:`dnnl_aBCd16b16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacd4df356da84852303c7905052dd9142>`,
	    :target:`dnnl_aBCd16c16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0a84665eb583fd7574b5bb774e459b2f>`,
	    :target:`dnnl_Abcd4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6b482db77afbb285b288c679c9be1ae3>`,
	    :ref:`dnnl_aBcd32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadaa7aeb5746c398a1c8ee1b42362bc58>`,
	    :ref:`dnnl_aBcd4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2ec2a0d2900e5ddb7f88783e2d8a5e66>`,
	    :target:`dnnl_ABcd4b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae950dd7ff699d5f51b175bdac1b84890>`,
	    :target:`dnnl_ABcd16b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da536323299510bd78310e0a8f9b88042a>`,
	    :target:`dnnl_ABcd16b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafe2dc8f9b8ef99e7336f9bb315bd28f3>`,
	    :target:`dnnl_ABcd4b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab95ba206fe14422c836d989fcaa013a8>`,
	    :target:`dnnl_ABcd4a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daeb77a3e431e7d90ca0eb3cb3abbebb94>`,
	    :target:`dnnl_aBCd2c4b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadcf839a7edc3764d62c779ee12dd1872>`,
	    :target:`dnnl_aBCd4b8c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da67864e97ff75ae1d4e53eee2785e82f1>`,
	    :target:`dnnl_aBCd4c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad056012cef6bfa44927de3cf7d85328b>`,
	    :target:`dnnl_aBCd2c8b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5c383c88fc041dac096ebff85bc3b37f>`,
	    :target:`dnnl_aBCd16c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad2c5c7d5fd912c0eb078ce4b82649c28>`,
	    :target:`dnnl_aBCd16c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab55f7b773e4f38a31af3306372d23ee3>`,
	    :target:`dnnl_aBCd4c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1a69ffdfcc86697235751a426cf74781>`,
	    :target:`dnnl_aBCd4b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1dbb1956688d4883ecf79dc0a7b6e28b>`,
	    :target:`dnnl_ABcd8a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da359c3de34197a41b6b08a985f0deb577>`,
	    :target:`dnnl_ABcd2b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6910582d0278bc234d44d23de57cf8b1>`,
	    :target:`dnnl_ABcd8a8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab48d2feeaed5ee18e7c702438470f308>`,
	    :target:`dnnl_ABcd8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab8e8cdd8d83c7c806105eb58ced1cd36>`,
	    :ref:`dnnl_aBcd8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9523ccda38e86a8f7f58f8f780ef6eac>`,
	    :target:`dnnl_aBCd4c8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da97e68fe68baf3d45caac83b6bc5b1426>`,
	    :target:`dnnl_ABcd8b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7ed04c708d85c8616deb0b6f79ef7f4b>`,
	    :target:`dnnl_aBCd8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da85273cb852d097faf8d8c0da6c41feec>`,
	    :target:`dnnl_BAcd8a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0be4bf53f881fc912b029986269e2d9f>`,
	    :ref:`dnnl_ABcd8b8a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2062c0db6b460fd8ac254684bbc15c36>`,
	    :target:`dnnl_aBCd8b8c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaf3b40408d73e8ac8642225c9f25f2f1>`,
	    :target:`dnnl_aBCd8b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1f77af45396bb24d07760511364a5613>`,
	    :target:`dnnl_aBCd8c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6c1fad78bb53863daced660c0f89e2a2>`,
	    :target:`dnnl_ABcde8a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa1d7b92ba320315089e52cb44b409a24>`,
	    :target:`dnnl_aCBd8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf74406af448716e5cb7e579746a423e0>`,
	    :target:`dnnl_aBCd8c8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da77890505517693773deafbd83cace868>`,
	    :target:`dnnl_Abcde16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5b8167c8d939ef924e98256b4eb98aec>`,
	    :target:`dnnl_Abcde32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1f28a133f5dae595b94e4924057c18fb>`,
	    :target:`dnnl_ABcde16a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daced565c64805b395099381a37bcbb7aa>`,
	    :target:`dnnl_BAcde8a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2cdddb60db18b8555f85640f4b4a9cef>`,
	    :ref:`dnnl_aBCd2b4c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac8768b971e97816930a09b6199f5bc05>`,
	    :ref:`dnnl_ABcde4b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad8e56dc7af8c48b5d4e5d11650a19eab>`,
	    :ref:`dnnl_ABcde2b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6dc96fd5cf79d2d4395cd129830f17a3>`,
	    :ref:`dnnl_aBcde16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad4357b1ef924c1331e9e7e25facc3827>`,
	    :target:`dnnl_ABcde16b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da18eb2e398335fcb1ac2408513c7817da>`,
	    :target:`dnnl_aBCde16b16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf5c2345a4a4cc585514a993a9cafdb99>`,
	    :target:`dnnl_aBCde16c16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafae7512e4dec5d0a0dcbab100462fb7b>`,
	    :target:`dnnl_aBCde2c8b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da364f711b4979cbb347af415f92467437>`,
	    :target:`dnnl_Abcde4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4a3039a650604f32aa5075a83c5711e7>`,
	    :ref:`dnnl_aBcde32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabdf6daaaaa7e79356fb5b4ec7ae90e5a>`,
	    :ref:`dnnl_aBcde4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3b841f4fc9008e7a27f4dcc4b130d819>`,
	    :target:`dnnl_ABcde4b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da60287b711beae3dda2b8e99751310ced>`,
	    :target:`dnnl_ABcde4a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafb76f365d48a201239fbbf55528b9718>`,
	    :target:`dnnl_aBCde4b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3fb2bdd6dfa508a35349a448cd5eb6ad>`,
	    :target:`dnnl_aBCde2c4b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9efd1a953a1fdccd45c4ae257d07639c>`,
	    :target:`dnnl_aBCde4b8c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3d270c3aa4c400d9ba421ef7ffe79024>`,
	    :target:`dnnl_aBCde4c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0265637b037f150c9a6f44e097217df3>`,
	    :target:`dnnl_aBCde16c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab0a6f6af2d023c7cf4cb34c0aa851bdd>`,
	    :target:`dnnl_aBCde16c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4f9b587b9ebe5fa931bff5419d06d27d>`,
	    :target:`dnnl_aBCde4c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da785b527d318e2e354cc14dfeae4c195b>`,
	    :target:`dnnl_Abcde8a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da730b91732737502118324686d409cfb8>`,
	    :target:`dnnl_ABcde8a8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8eaa68e9b1ceafec83c08940b3b398d4>`,
	    :target:`dnnl_ABcde8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab312430da2278cc10e86de914b25d2c>`,
	    :target:`dnnl_BAcde16b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa1dd482da424c94a3b71b757f1b0a8a7>`,
	    :ref:`dnnl_aBcde8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa778e377f1b0afdee9a6bcbc63ae76bd>`,
	    :target:`dnnl_ABcde8b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da312dfffd6c27d39cb5de872a5b386245>`,
	    :target:`dnnl_aBCde8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da67cf192e55322acc60dd0a5e0218e023>`,
	    :target:`dnnl_aBCde4c8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6c1c731d5b253a1cc55c73f3fed87e7e>`,
	    :target:`dnnl_aCBde8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacda7bd7200123c4078bd0021e425ac4f>`,
	    :target:`dnnl_ABcde8b8a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6d74954d3d0e522ea212f62954572628>`,
	    :target:`dnnl_ABcde32a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8759f10eeae5b4c0c2a3f9ac139390a6>`,
	    :target:`dnnl_aBCde8b8c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac80c40dbc7ffe75fe1811e6f03416df6>`,
	    :target:`dnnl_aBCde8b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3074852c442c4a921510a63fd70a4376>`,
	    :target:`dnnl_ABc4a8b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4156a4bc4d12d3b92cd0d248b8f345b0>`,
	    :target:`dnnl_ABcd4a8b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacf6b9bf9f1fbc09357f6ec3e523c2e4e>`,
	    :target:`dnnl_ABcde4a8b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa3fa29cebdc49d51d9d96990db129f77>`,
	    :target:`dnnl_BAc4b8a8b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6fb4b638d77703af8633c2b02f384a19>`,
	    :target:`dnnl_BAcd4b8a8b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da47c101af1fc9784096f0ea8e82c8f75b>`,
	    :target:`dnnl_BAcde4b8a8b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadab31a42d1d9d26c5dcc1b1f573c7231>`,
	    :target:`dnnl_ABcd2a8b8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da508fb83e49f4b61a0c1f6559c1ad493b>`,
	    :target:`dnnl_aBCd4b8c8b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da417d9d0863cb8a9ffa41da1b36ad258c>`,
	    :target:`dnnl_aBCde4b8c8b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3a93b3ea3e2b878e0e652ee5eeeec3e1>`,
	    :target:`dnnl_aBCde2b8c8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac893c361ab66a3c6d4b330de62dd6da2>`,
	    :target:`dnnl_aBCde8c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4aa39c74ad124f7548de870bf22a3c28>`,
	    :target:`dnnl_aBCde8c8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da65bbcb9c1b1e428a8987a9bc603cf641>`,
	    :ref:`dnnl_aBCde2b4c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da451386bf211128eaa11a8863a9d97e2b>`,
	    :ref:`dnnl_aBcdef16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da431c182a4f2665030f29973efb556c94>`,
	    :target:`dnnl_aBCdef16b16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad49ced68989f7d54514bb6cd3138f8e6>`,
	    :target:`dnnl_aBCdef16c16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0a39498d5ace47415cf7af24ee709a8c>`,
	    :target:`dnnl_aBCdef4c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8054d00c19f5db7641b6ed7c360a9f90>`,
	    :ref:`dnnl_aBCdef2c8b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf9d8eabf65635e7a932e5f5b3339cdd8>`,
	    :target:`dnnl_aBCdef4c8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaae6ee2c7dbc1f2ea5fbff789a67155f>`,
	    :ref:`dnnl_aBCdef2b4c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da06fe4d629df62b68f499416b791da96b>`,
	    :ref:`dnnl_aBcdef4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2735d5d034a5abe998df55be8e47e5e8>`,
	    :target:`dnnl_aBCdef4c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3daff9d46680246acc4447330b7b74c9>`,
	    :target:`dnnl_aBCdef4b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da552f9164ee2cb21a5dcba1ed05628f78>`,
	    :target:`dnnl_aBCdef2c4b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabde37b9832e2b86715f72b46b8548294>`,
	    :target:`dnnl_aBCdef4b8c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da295a932e5fe4d99dfe72c5965a95303d>`,
	    :target:`dnnl_aBCdef8b8c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6d683f0464f7a010c430f304a1a0fae9>`,
	    :target:`dnnl_aBCdef8b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa1ca028ca42fbe9f38ff09cef0b3d994>`,
	    :target:`dnnl_aBCdef8c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf7a9058aa0164b32492b96e54bb5fbe2>`,
	    :target:`dnnl_aBCdef4b8c8b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da16e4169d41ed1d6d219eb0aa405fdeda>`,
	    :target:`dnnl_aBCdef8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da53f6ad15b1d93d28fc1c880d9dd61868>`,
	    :target:`dnnl_aCBdef8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5527188b8276ff375356dc1d247193f3>`,
	    :target:`dnnl_aBCdef8c8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9bedf873f137d40d08480fb2dcd1927a>`,
	    :target:`dnnl_aBdc16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadfddca9f759672b0735d3b8ea2e01034>`,
	    :target:`dnnl_aBdC16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da04cc05297225408031f4a0aa81864ff4>`,
	    :target:`dnnl_aBdC16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da88692a1fc85c72c803afb60157130b41>`,
	    :target:`dnnl_aBdc4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7b880b3fd856a32a5d559753b7da1d4d>`,
	    :target:`dnnl_aBdc8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da914d32dd6ddd195a4ada50fa8656821d>`,
	    :target:`dnnl_aBdec16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaed4d913aa27e12be0e7dca0b44f06f2>`,
	    :target:`dnnl_aBdeC16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da98ec760512ba076f6769d10f18036b13>`,
	    :target:`dnnl_aBdeC16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da38f1f9bb85d15e6a220deaee954decbc>`,
	    :target:`dnnl_aBdec32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da15b25713843ce5b79fe2acd4ecd46b2a>`,
	    :target:`dnnl_aBdec4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da46fca1587c9124a3986144666bc4d4dd>`,
	    :target:`dnnl_aBdec8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da80606d4c8ccaa616d5ee867ed341f526>`,
	    :target:`dnnl_aBdefc16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da225e71ab2cf35ec5779b666a10578645>`,
	    :target:`dnnl_aBdefC16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da85529ca87964a09623a737ad780077c4>`,
	    :target:`dnnl_aCBdef16c16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf7dc09eef597f617b621fed80510fa2a>`,
	    :target:`dnnl_aBdefc4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad33def63f74d7f5a9d5e4e4de1fd3549>`,
	    :target:`dnnl_aBdefc8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad24aa48c7b9ffb2fcca746d88d533b8e>`,
	    :target:`dnnl_Abcdef16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da92c4924d2b608a3bc86b72739e73d034>`,
	    :target:`dnnl_Abcdef32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7d19b6c6c1b9098eeafbbd2e6944bb09>`,
	    :target:`dnnl_aBedc16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da16245da9b0eedeec4e6876e268dd31bf>`,
	    :target:`dnnl_Acb16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bfab660a122d0cc962f6fe7b82f2e60>`,
	    :target:`dnnl_AcB16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf0b093620df280095b29f946cf21fb5b>`,
	    :target:`dnnl_AcB16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a66b3884eb919e1969c71adb3747024>`,
	    :target:`dnnl_Acb4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9908156c9bdb912d84eda9b1bad77683>`,
	    :target:`dnnl_Acb8a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae64ddaa317dfc38ba6115f26766bf437>`,
	    :target:`dnnl_aCBd16b16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafff3cc23e27926344aeab61ff6b80a9c>`,
	    :target:`dnnl_aCBd16c16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadfec7111bb7ac163af7f451e82b03eb5>`,
	    :target:`dnnl_aCBde16b16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafe88e948cdde58570ee1fe35396a176f>`,
	    :target:`dnnl_aCBde16c16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da99562760cb2b96130f3b317f5c0eb8a5>`,
	    :target:`dnnl_Acdb16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae899a9dcb69c10f091fe15f749f1b09d>`,
	    :target:`dnnl_AcdB16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7d468009e3110df78c59bb4c2ea8d250>`,
	    :target:`dnnl_AcdB16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da005f7d9395f6cb15a3d527a482131c03>`,
	    :target:`dnnl_Acdb32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae2e070bc38ab90196d11620330be4bdb>`,
	    :target:`dnnl_Acdb4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1deb63bb8ad8eef831c709ce8a3d3f32>`,
	    :target:`dnnl_Acdb8a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa81dc815b0ec8bd729f5b856139279a7>`,
	    :target:`dnnl_Acdeb16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f21562a38357ce04e99c9ad5d741e47>`,
	    :target:`dnnl_AcdeB16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6332a6ad7c83b8243bab3b77801c754c>`,
	    :target:`dnnl_Acdeb4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacb1169fe7b79d20161182caa0afc0c15>`,
	    :target:`dnnl_Acdeb8a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae1e147d05e1696dfb9e403cbad777200>`,
	    :target:`dnnl_Adcb16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da393dec9f3935ba04df0b58b2e8440964>`,
	    :target:`dnnl_BAc16a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabf4ef76fd5820b6836c1e09bace6361f>`,
	    :target:`dnnl_BAc16b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae903179bbf7c370469b71f9ef1fd12d2>`,
	    :target:`dnnl_BAcd16a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab4c46cc5104349632549a6afdecf5007>`,
	    :target:`dnnl_BAcd16b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf678aa7c0cccf761c2db4cf95d0791c9>`,
	    :target:`dnnl_aCBd4c8b8c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da94e4182b2bdc0e10b37d0a4045ddc2bb>`,
	    :target:`dnnl_aCBde4c8b8c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab845e5e99381eb7309136fc0cd73478f>`,
	    :target:`dnnl_aCBdef4c8b8c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf69769c7462007abf6756e4e9df1fe88>`,
	    :target:`dnnl_BAcde16a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da54d324628354f454420b12d38da108ac>`,
	    :target:`dnnl_aCBdef16b16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9d14fe44b8bce3ea498f40f353c05a84>`,
	    :target:`dnnl_ABc16b32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da14d52b34d82ed3efbb29ef0c0a0c8bcf>`,
	    :target:`dnnl_ABc16b64a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da71f46cc2eeef5c92e19ef1d57915b3f9>`,
	    :target:`dnnl_ABc4b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da51d47c8e707ba68e55d03257289e7dda>`,
	    :target:`dnnl_ABc4b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da47f0f82a0099088bee3911a75f8e72e7>`,
	    :target:`dnnl_ABc8b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf226ccb508a7b8b7e48106b2269c31dc>`,
	    :target:`dnnl_ABc8b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da728ddc6acabeb845722c4198a4f3d2a2>`,
	    :target:`dnnl_AB16b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da777478e39ff0e681535f13086dc4552b>`,
	    :target:`dnnl_AB16b32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa4df2f3c68986741425c2e6864d5675a>`,
	    :target:`dnnl_AB16b64a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafc30947164f8c7db71b8b967209b6073>`,
	    :target:`dnnl_AB8b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaab0a9dbe35dc66e5fe20f1ab3c74e52>`,
	    :target:`dnnl_AB8b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3b050901652debc432946f04f685d1a2>`,
	    :target:`dnnl_AB8b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad61d639f5f7030e4eb06a15f26588cd4>`,
	    :target:`dnnl_AB4b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da798047f6dbcd73fd36811ca367c10206>`,
	    :target:`dnnl_AB4b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1dc027b3324599058a466bc0c079e516>`,
	    :target:`dnnl_AB4b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad2db4e008f6c2507daf67734cdc9a2ca>`,
	    :target:`dnnl_AB16b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da678f1bfe1118160398bc003fe8f27484>`,
	    :target:`dnnl_ABcd16b32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4627f586159eca38b53175e571dac3f2>`,
	    :target:`dnnl_ABcd16b64a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da19ce677a4abbab9074f7de681a9cddd0>`,
	    :target:`dnnl_ABcd4b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae95b3a94eeee7204458186cb3ef1aa0b>`,
	    :target:`dnnl_ABcd4b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae8270dac1d48effb250fc0794a055cfa>`,
	    :target:`dnnl_ABcd8b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da411273e59ae944f57b8c626988ecfca0>`,
	    :target:`dnnl_ABcd8b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab1dad5683d2f432a02361b3eb602daf2>`,
	    :target:`dnnl_ABcde4b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf076ac2beefdeedc4b227aead8adc2f3>`,
	    :target:`dnnl_ABcde4b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da776cc0d4bdeaa97bfa4f364120e21283>`,
	    :target:`dnnl_ABcde16b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daabbb82fefa4aa12a87473f699a3f0d5e>`,
	    :target:`dnnl_ABcde16b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da16367cb08d0490efa8f053d7ee9052e1>`,
	    :target:`dnnl_ABcde16b32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4a56233ff6db7f33e5658da7bf0d3682>`,
	    :target:`dnnl_ABcde16b64a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5db93942708a7a99f827d6a3db33b3ea>`,
	    :target:`dnnl_ABcde8b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7ab8f2f922aa858c69c8cf70905d1f31>`,
	    :target:`dnnl_ABcde8b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da82bc040a011c501940d88de65f7dfa40>`,
	    :target:`dnnl_aBCdef16c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadc0f0749228c29cb8b0ff7b43f09ba37>`,
	    :target:`dnnl_aBCdef16c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa1b2af8a48d865cfc6b1e5c36035b12b>`,
	    :target:`dnnl_AB32a32b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab3150a8c2f4c2d8c8c139e3df78fc604>`,
	    :target:`dnnl_AB8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8b21cc55e83a128627622e3106042eb1>`,
	    :target:`dnnl_AB32a32b8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafc4180d1ce52dc318ef8efd3ae0f6169>`,
	    :target:`dnnl_AB8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac370f6bfc6f87fdf8a8f579460beb990>`,
	    :target:`dnnl_abDc32d<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf34703dacf80591bef4790639c4b2ca1>`,
	    :target:`dnnl_abDC32d4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da79d575e07854c45ce00eebe812c893d2>`,
	    :target:`dnnl_abdEc32e<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da026b527806539ff73d59bf301648f04d>`,
	    :target:`dnnl_abdEC32e2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da094a7e50aa2ad48acf9c2037e71fcaef>`,
	    :target:`dnnl_abdEC32e4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2bdf556c7e92989e29198fbb1db31f64>`,
	    :target:`dnnl_aBdefC16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad070e8d7487267e213e83759b6630e7f>`,
	    :target:`dnnl_AcdeB16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2838aeb6abb12b3e5f826b4a6bf79ee7>`,
	    :target:`dnnl_ABcd16a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf5e477131eee10aab9df318f8aa9e024>`,
	    :target:`dnnl_ABc16a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da85582c48895f40fc5bb647986baf09c5>`,
	    :target:`dnnl_aBCd16b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da16a59e715a9a1b9d06e9da38d91e5858>`,
	    :target:`dnnl_aBCde16b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daadf0337c91cbc9ddf7b7365f97d09831>`,
	    :target:`dnnl_Acb32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da54f5a82389efbaf71ce11daf237d47f3>`,
	    :target:`dnnl_AcB32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da270857a538d029a34e3b0d0db0d39332>`,
	    :target:`dnnl_AcB32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da33f5b0f14f0d6a41da515e400b4d1f80>`,
	    :target:`dnnl_Acb48a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4fe444278b839e3249cb9826bc23da84>`,
	    :target:`dnnl_AcB48a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4dd1b82de067b7a225d7acd4c90b7388>`,
	    :target:`dnnl_AcB48a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da72bd26b0f184c0147679079ec6d824e7>`,
	    :target:`dnnl_Acb64a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad1a54362279ffeae3ed77fc55875557e>`,
	    :target:`dnnl_AcB64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6f452c2d1a8af5ffabe7558e7b78837e>`,
	    :target:`dnnl_AcB64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da449572a15a2299a37d5d53976f112f49>`,
	    :target:`dnnl_cBa2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da00dc92fd7676de9afb33305c63a9aaca>`,
	    :target:`dnnl_cBa4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daec6a2b7a7dfc753d4daaf9072738d8b6>`,
	    :target:`dnnl_aBdc32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4da6738d4ca02b33685fdad35fea887b>`,
	    :target:`dnnl_aBdC32b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6ced4f3a6ac074ac8da77b40fe41f4dd>`,
	    :target:`dnnl_aBdC32b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da42e495f5d6d4f818df28bd247662280b>`,
	    :target:`dnnl_aBdc48b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da325bd72d9754637a661db80ffb42be04>`,
	    :target:`dnnl_aBdC48b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacd702d15d07e09362da80dd48e0fbbe8>`,
	    :target:`dnnl_aBdC48b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da846f2ab356c80ddff371d750761e6ac9>`,
	    :target:`dnnl_aBdc64b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da26c6196ebeb38615602630e85984605f>`,
	    :target:`dnnl_aBdC64b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2cf743e9a63f092c14043a1e5120416e>`,
	    :target:`dnnl_aBdC64b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafd57c680129f4239ee6876fe4216e71a>`,
	    :target:`dnnl_adCb2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1e43cfe80fd00d7249d33b03cbb3fdb6>`,
	    :target:`dnnl_adCb4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7f6682e8fc5545abaa0e906cf8ee5946>`,
	    :target:`dnnl_AcdB32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da06c5f623bcf12f80d9f1e50363b01495>`,
	    :target:`dnnl_AcdB32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da82393f019994262294b5324857d975cb>`,
	    :target:`dnnl_Acdb48a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf26a0937ecd2e5763ae5f9a8b1003ec9>`,
	    :target:`dnnl_AcdB48a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1090b5bed1a5838289dc4bc7a1b8404e>`,
	    :target:`dnnl_AcdB48a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7dbb2776cee79814a3274f28808c2b9b>`,
	    :target:`dnnl_Acdb64a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da360a005de840c4b2c96422d06477a847>`,
	    :target:`dnnl_AcdB64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2ab6b8969e8e8127303fe27a48e54743>`,
	    :target:`dnnl_AcdB64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac1b91e0a7496c4d5424288f7eff6f43b>`,
	    :target:`dnnl_cdBa2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da60a306d74635a7e343f40516a81050fb>`,
	    :target:`dnnl_cdBa4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9de66479aa3d956bf45d6d560e084a4b>`,
	    :target:`dnnl_aBdeC32b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1c3cba225c9924005725d3d67dac9a81>`,
	    :target:`dnnl_aBdeC32b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadd5ad13ab25ec9478a4cdfcc8969d788>`,
	    :target:`dnnl_aBdec48b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daacc43a6e9d19f4f80103cd21da005796>`,
	    :target:`dnnl_aBdeC48b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa089423dfbb7515430e4aa497a5e6987>`,
	    :target:`dnnl_aBdeC48b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5e5624db54a9e0bd32811ff4173bc46e>`,
	    :target:`dnnl_aBdec64b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da896fe47ad8adc0588268d79b93e347bd>`,
	    :target:`dnnl_aBdeC64b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadca236d3486c5dc151887e861f375624>`,
	    :target:`dnnl_aBdeC64b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac0a69ac6a8ee2fe02101b57f95b9e92f>`,
	    :target:`dnnl_adeCb2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0f1e4a225c98cde20da34285e3713911>`,
	    :target:`dnnl_adeCb4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae2a0d94cf4fa6fc9aa456e2279543ba0>`,
	    :target:`dnnl_Acdeb32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da584813f52c2362cdb2ac0cd012a0a887>`,
	    :target:`dnnl_AcdeB32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4321ed8b4b8a0e34b3a66290ba0c3c15>`,
	    :target:`dnnl_AcdeB32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da53112dc698149fa80bc86d149ca4fe5b>`,
	    :target:`dnnl_Acdeb48a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3e33cbcefb25c6ae45eadb7258f1ef78>`,
	    :target:`dnnl_AcdeB48a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae86d17f83bb212d6bc7e3127f8e735ca>`,
	    :target:`dnnl_AcdeB48a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab3cf16fd9c0a929418f03ef28d61cba4>`,
	    :target:`dnnl_Acdeb64a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daee189979deda242b566639593f9322db>`,
	    :target:`dnnl_AcdeB64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab08d6cbe632ecf7d8ec22787937c57ef>`,
	    :target:`dnnl_AcdeB64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa535c4e442dcda742ba8f39073909116>`,
	    :target:`dnnl_cdeBa2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac422aa3089d50a50b6087c6c31a4e493>`,
	    :target:`dnnl_cdeBa4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5473f95f72c7a15569c102693cf00a2b>`,
	    :target:`dnnl_aBdefc32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabb0deda24d5cf54675150ccee3c42797>`,
	    :target:`dnnl_aBdefC32b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da76e703bd9552941a9cc3708664066a37>`,
	    :target:`dnnl_aBdefC32b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da896c69f97ca7fb8f18d7f11d75d1ebef>`,
	    :target:`dnnl_aBdefc48b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da71919d29857464f5ebf85403b0f2cc23>`,
	    :target:`dnnl_aBdefC48b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da89a1bdf78a03350cc367ffb04defc4b5>`,
	    :target:`dnnl_aBdefC48b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab463a82cf7a016312b3927aedd0b48c8>`,
	    :target:`dnnl_aBdefc64b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa76fe00970cc49c750b4bdbfd1a43f23>`,
	    :target:`dnnl_aBdefC64b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da007f3d19bfd04cb51f88277afe769515>`,
	    :target:`dnnl_aBdefC64b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab497070b34a3f8515e0532c765502915>`,
	    :target:`dnnl_adefCb2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafa77d677234298d2b1e562569cae08ee>`,
	    :target:`dnnl_adefCb4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac00478cd95049a818b58bc2ecccabd1c>`,
	    :target:`dnnl_AB16b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da39126d9017044006a77482fb471c1f44>`,
	    :target:`dnnl_AB16b48a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf5317be1ff3ecc7684a68cdf16a781fc>`,
	    :target:`dnnl_AB16b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da27ccd12d73a7a4bac8d719e5850911c0>`,
	    :target:`dnnl_AB16b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf7f5b14e1998399b0893841f3617c16a>`,
	    :target:`dnnl_AB16b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da050b89a14efb05a175b3d9281d0ec76b>`,
	    :target:`dnnl_AB16b48a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae0e92935b6fc027e7adbeaf294a012f8>`,
	    :target:`dnnl_AB16b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad603920390de323288e53fbc73f912d4>`,
	    :target:`dnnl_ABc16b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3e782e9a4f657e30b987994a694f1f88>`,
	    :target:`dnnl_ABc16b48a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1befa1e0b209e43916873b634f3e23ee>`,
	    :target:`dnnl_ABc16b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5b0f087138dec3ee440ec906c2cfdc27>`,
	    :target:`dnnl_ABc16b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e768e4eb391b249b83578cfa296f2d4>`,
	    :target:`dnnl_ABc16b48a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da94d77d9acf08ef60a95724f74e857223>`,
	    :target:`dnnl_ABc16b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf49384b3b29b550d5f301c79856a1f3d>`,
	    :target:`dnnl_ABcd16b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab4c9ef18006b9a75e6f87fb1081069df>`,
	    :target:`dnnl_ABcd16b48a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daff11d69776e4706941e3362fe08ab389>`,
	    :target:`dnnl_ABcd16b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da11a77be93f1ac5bced5d0954c704f81a>`,
	    :target:`dnnl_ABcd16b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da673bed08a5a8158bd68dc71afcfed207>`,
	    :target:`dnnl_ABcd16b48a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9a1344dee09219c27fc0b355aeb80e6e>`,
	    :target:`dnnl_ABcd16b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da61ba5bd7f788f2fb6e0b46b8cf9ffb96>`,
	    :target:`dnnl_ABcde16b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daae184fbe1d32ab8b6549679d06ddcaa6>`,
	    :target:`dnnl_ABcde16b48a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da61eb51a96cbe485f5a0ccf8effcf13ff>`,
	    :target:`dnnl_ABcde16b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da653061f6eab4ef44c7fd84809ebd4970>`,
	    :target:`dnnl_ABcde16b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da84299ad12f5d5891be6a9cd3eed94e62>`,
	    :target:`dnnl_ABcde16b48a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac60b73be4ae964b0a28fe05d123244b4>`,
	    :target:`dnnl_ABcde16b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae8f008b3fe17dcecfcb4e5030d69d0d6>`,
	    :target:`dnnl_ABc32a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daafae9bc70fe9a23b88b466d52259bb06>`,
	    :target:`dnnl_ABcd32a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da38c3f33de484607f5754e1b4c478d8fd>`,
	    :target:`dnnl_ABcde32a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da19810b55c43b9208cd49d95eed6f64ec>`,
	    :target:`dnnl_AB48a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabd2d517bae9c636ac244f34068bfe680>`,
	    :target:`dnnl_AB48a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadada622ca5273cdaeaee4a49af818d53>`,
	    :target:`dnnl_ABc40a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa04b735c9ec7a8f9aaf4cfd02669292b>`,
	    :target:`dnnl_ABc40a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da51a8b6cf4d92610da25e4f54b267258e>`,
	    :target:`dnnl_aBC48b16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3bc3e6c319272d4a333df87d74a87f1f>`,
	    :target:`dnnl_aBC48b32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da291f1427d227135839a822180913c250>`,
	    :target:`dnnl_ABcd40a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafb296ed4962ddd45757d0853865d84f1>`,
	    :target:`dnnl_ABcd40a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da47f94491cccadf075f1937be499df9d7>`,
	    :target:`dnnl_abCd32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae470b4d6f8f88c7a3fb59689f1224e69>`,
	    :target:`dnnl_abdCe32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da196d052cc26948c3bdc0fe5fd69e5044>`,
	    :target:`dnnl_abdCE32c2e<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da85a3d0f985cd7b87cdc784035ea756c6>`,
	    :target:`dnnl_BA16a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7186dce0aad94eda97370946606a2b7a>`,
	    :target:`dnnl_BA16a32b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac6df5da97fbbee97eae7b8aeb17b5c4a>`,
	    :target:`dnnl_BA16a48b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8f19b05c084528572af30938f5e11aab>`,
	    :target:`dnnl_BA16a64b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa1cd258837e95fa074897fe4d6097506>`,
	    :target:`dnnl_BA16a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daff09551aa3fe6b38111694d20f7fabba>`,
	    :target:`dnnl_BA16a32b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da91d7454ba36d6caa5a0a0d6f1e789448>`,
	    :target:`dnnl_BA16a48b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf9a10c246a7d64037d28287c114a3848>`,
	    :target:`dnnl_BA16a64b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab68d3a89aa79267c32ffc130c4ed14a6>`,
	    :target:`dnnl_ABcd8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac846bf612620636024385ef645efdfa7>`,
	    :target:`dnnl_aBdeC16c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadb7e623134bd49cb75b1b0d5971217dd>`,
	    :target:`dnnl_aBdeC16c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da94025542a7c05a31b1216bb81a28d587>`,
	    :target:`dnnl_aBdefC16c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadf2c5f956118bf7f34eb1333984c6725>`,
	    :target:`dnnl_AcB16b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da65175c50d01faa10fee2433c478187c9>`,
	    :target:`dnnl_AcB16b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daefa450b40aca10cf64df469bf8c9d57a>`,
	    :target:`dnnl_AcdB16b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacc88fd7066ae40bb1870fc6113c7ba8f>`,
	    :target:`dnnl_AcdB16b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da33054786477d07e47d1abe07ba662ce2>`,
	    :target:`dnnl_AcdeB16b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da38d2c46d19d36146e9da0a19ad9ac4ef>`,
	    :target:`dnnl_aBdefC16c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da663532c9925ad78501f2622aeb0235b6>`,
	    :target:`dnnl_AcdeB16b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da508b2d828e9aa028166830a75fe6fd9a>`,
	    :target:`dnnl_AcB16b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da207a9d617245fbd43a1b7c50eee15b30>`,
	    :target:`dnnl_AcB16b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac0407d7c957ae3a5b9673fd3bd11f2d7>`,
	    :target:`dnnl_AcB16b48a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da78fc849f88a4a17007b7c586c704d0dc>`,
	    :target:`dnnl_AcB16b48a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da23f8566c5dda35e6bd6eab92e4226949>`,
	    :target:`dnnl_AcB16b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5c3b8f1f860bbe499d8d45f5a1e35132>`,
	    :target:`dnnl_AcB16b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daad0bd412fccbc108d2f2cc9f58591bc4>`,
	    :target:`dnnl_aBdC16c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa3f80959547ceaba57bade16bad72495>`,
	    :target:`dnnl_aBdC16c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2239bccda6e1c3bee6821ded1ed0ce86>`,
	    :target:`dnnl_aBdC16c32b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7119f5ed24f6be96ab77d6cbe81e7c9d>`,
	    :target:`dnnl_aBdC16c32b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8d6ed59402901ef5ba156367004a21e7>`,
	    :target:`dnnl_aBdC16c48b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaca072551731384c998472d23591fe7e>`,
	    :target:`dnnl_aBdC16c48b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf62f0d203daa06241b097d7636e7fd7f>`,
	    :target:`dnnl_aBdC16c64b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2d7b94e0252667909d7bbd1e9a6db764>`,
	    :target:`dnnl_aBdC16c64b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da76a54436f5045273610ee55ba48fbc15>`,
	    :target:`dnnl_AcdB16b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8a45deb82f3728e35c50bf7b38f8e342>`,
	    :target:`dnnl_AcdB16b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad26df68c646b134d63ae02546fdc8f40>`,
	    :target:`dnnl_AcdB16b48a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da911a0989f9b8adca544716f29818ff03>`,
	    :target:`dnnl_AcdB16b48a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2ae68a2ff418a7a7f38dcebf0981610a>`,
	    :target:`dnnl_AcdB16b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da62f8eb8ed88f2360ace5d15d8d44b9d3>`,
	    :target:`dnnl_AcdB16b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab9aab51e2e014c677a0f2e16384ee1e3>`,
	    :target:`dnnl_aBdeC16c32b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2105deb1b1ea5932abd8a4e67a0034d5>`,
	    :target:`dnnl_aBdeC16c32b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da37727ebdff6537dacf5ffc1959ce14e6>`,
	    :target:`dnnl_aBdeC16c48b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacffeb175ad0b212dcaa7dfb069bb8523>`,
	    :target:`dnnl_aBdeC16c48b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daef364f0fa99b4b7163564a795d246563>`,
	    :target:`dnnl_aBdeC16c64b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da005b5e48105abc123185a32a265f6d06>`,
	    :target:`dnnl_aBdeC16c64b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daea26e8a8c79f5ef2af7a051b152bb943>`,
	    :target:`dnnl_AcdeB16b32a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac68b47e3a4e4af4192c61de01c7640d5>`,
	    :target:`dnnl_AcdeB16b32a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da00529ded0a5c6ccbfb7c2e644e33c668>`,
	    :target:`dnnl_AcdeB16b48a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5bef09da061197cf22058324019cb895>`,
	    :target:`dnnl_AcdeB16b48a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacd01ffd149d6fec0b6668f4fe5b341c0>`,
	    :target:`dnnl_AcdeB16b64a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da93de80253f4bd56fd4f1151bc494678e>`,
	    :target:`dnnl_AcdeB16b64a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3f70dc239f435f74d119ba0c6c5a53c8>`,
	    :target:`dnnl_aBdefC16c32b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da43246cf09968075f35d1432d30a342cc>`,
	    :target:`dnnl_aBdefC16c32b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa42b79e1965bc56473945dad85135ba9>`,
	    :target:`dnnl_aBdefC16c48b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad09d40cf94c87aab3462aeb5b60cb518>`,
	    :target:`dnnl_aBdefC16c48b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da31dce58987b1e20a3d9eb3b00d300ea6>`,
	    :target:`dnnl_aBdefC16c64b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daafd0b10751fa28c48ff8c7f3f1a5d5ed>`,
	    :target:`dnnl_aBdefC16c64b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabd673047769c0b8e426d63a16712e7e8>`,
	    :target:`dnnl_decbA16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae56cc7143a5a396c2fe764cd7f97eaac>`,
	    :target:`dnnl_ABc4a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2b75dd110bdb6d972b3fa9c100570d3f>`,
	    :target:`dnnl_ABc8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da69066f29608b34f68355fc7e85ef8d9f>`,
	    :target:`dnnl_aBCd8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabff9c7165e9d9103686814faac1f8be6>`,
	    :target:`dnnl_ABcde4a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab5c93ee1448991d5f35686ed52bacd15>`,
	    :target:`dnnl_ABcde8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da785d0790c3df65c4ce6ac9184442b224>`,
	    :target:`dnnl_ABcde40a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae6f8c9bd120d224cc0b1a4129323cb6f>`,
	    :target:`dnnl_ABcde40a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da87eb5aed04983b33f852f0ce2ac4f650>`,
	    :target:`dnnl_aBCde8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaa668c7829deb0ee82453b4fa3378676>`,
	    :target:`dnnl_ABcde4a8b8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dade4f2dd701daed381753467feb46784e>`,
	    :target:`dnnl_ABcd4a8b8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daeea48cac5d186a52c8a52816111ff7ae>`,
	    :target:`dnnl_ABc4a8b8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0b88707da9bdae750993dd0954b080cf>`,
	    :target:`dnnl_aBCdef4b8c8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0262371206570fbec8ba75d1100dff21>`,
	    :target:`dnnl_aBCde4b8c8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da81f6d9eb3629d8e6e41cd75e617e1edc>`,
	    :target:`dnnl_aBCd4b8c8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae90241549325b5030b237478aea90b1f>`,
	    :target:`dnnl_BAcde4b8a8b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa43722c6c9b67479da967cf7ff8e7257>`,
	    :target:`dnnl_BAcd4b8a8b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daabd8a2009b12e0e7c8a816caad06db7a>`,
	    :target:`dnnl_BAc4b8a8b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf3a19ba3c180a11ef015d89d03728528>`,
	    :target:`dnnl_aCBdef4c8b8c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da65eb7531f0a5f0def57ee39b6c709dff>`,
	    :target:`dnnl_aCBde4c8b8c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da69025daaa1bc0b383a1682e1fe7ed16e>`,
	    :target:`dnnl_aCBd4c8b8c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9d17c09ad1355af9a702c861043eda8c>`,
	    :target:`dnnl_aBCdef8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab5118c2918ea43752e8dc8d5f19d4f1a>`,
	    :target:`dnnl_AB32a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7df9f59f1d4b6976a3c42ae08b04fb3a>`,
	    :target:`dnnl_AB32a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da07d4d8ff5695885e6ad5145e9087dca4>`,
	    :target:`dnnl_BA4b8a8b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf09b4c0090e0d0c5bf70172dd616f895>`,
	    :target:`dnnl_BA4b8a8b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da274a537f03ac0afcb6af6607266ae8cb>`,
	    :target:`dnnl_aBC32b16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da62a32375c88b69d4bdf5e7c4d977f6ce>`,
	    :target:`dnnl_aBC32b32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da01b7f0d01a3f3e6bb5713e3605d5494a>`,
	    :target:`dnnl_aCB4c8b8c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafbea1a9071cf362ebdc965171a876650>`,
	    :target:`dnnl_aCB4c8b8c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6f86f4a1e6b585e67c4197718eaa534b>`,
	    :target:`dnnl_ABcd4a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9a2b2047913fa52b908e2efabf2e4ce1>`,
	    :target:`dnnl_ABc2b8a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6f486efd1b73e60146c4e9894066eef0>`,
	    :target:`dnnl_ABcd2b8a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da284c1843638a3b5ccb5db37a1d43fc4c>`,
	    :target:`dnnl_ABcde2b8a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab5802d3b9a09868ddd7a02d6f956e034>`,
	    :target:`dnnl_ABc2a8b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1b0fc478ff4c2e4d97ae73c156766946>`,
	    :target:`dnnl_ABc2a8b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da01be829e4a5098f1b98ac90f2653ea50>`,
	    :target:`dnnl_ABc2b32a8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8d27660b8863779fce89f768da92c5b>`,
	    :target:`dnnl_ABcd2a8b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4327678202ec178f308fe89b7cd06b43>`,
	    :target:`dnnl_ABcd2a8b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5ee9316b2604e91b26bf1533a25d19c6>`,
	    :target:`dnnl_aCBd2c8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3c09a46b030c75ef8b083a04c410b3fb>`,
	    :target:`dnnl_ABcd2b32a8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafba6ae5fde0cf10398fb510366379587>`,
	    :target:`dnnl_aBCd2c8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabddb22bcf6383933bccb6369a9c6d989>`,
	    :target:`dnnl_ABcde2a8b16a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da39adf8089e27e80402758dffeba769d8>`,
	    :target:`dnnl_ABcde2a8b16a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0e2dc53a834681243fa35e68c10388e0>`,
	    :target:`dnnl_aCBde2c8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da84c4f45eda2ea33a8c48998378343a17>`,
	    :target:`dnnl_ABcde2b32a8b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daffe51f58d991dde0030f3a46c3b1274d>`,
	    :target:`dnnl_aBC2b8c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacaaf63c420701768e0a1b313b39de35a>`,
	    :target:`dnnl_aBCd2b8c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7b10c4b4dd70dd0b55d2a6ea3559d91d>`,
	    :target:`dnnl_aBCde2b8c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac7153bf872f55715ab4633f356aad7de>`,
	    :target:`dnnl_aBCdef2b8c16b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daee68ce0737e54a0a1276874a6dfef2ac>`,
	    :target:`dnnl_BAcde2b8a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab3f2d3b502ea96decc2296e1481e4c4d>`,
	    :target:`dnnl_BAcd2b8a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da44a2dee4d88db013cee45904372b8de5>`,
	    :target:`dnnl_BAc2b8a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da71fdd516150ea65bfc1e6875d41096e9>`,
	    :target:`dnnl_BAcde2b8a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab1ec4c01cbcb6c1dd0673387663b7d7>`,
	    :target:`dnnl_BAcd2b8a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3b2d9b59f00a1a864f784b33a4c6e46d>`,
	    :target:`dnnl_BAc2b8a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafe15f87b3d67dad93f6b4f7b6128aa09>`,
	    :target:`dnnl_aBCde2c8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a80865920fc17aed0d3c498c0c7824>`,
	    :target:`dnnl_aBCdef2c8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3434e6aad5ccf9aeb5c4caa8fa500042>`,
	    :target:`dnnl_aCBdef2c8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab09c313ae6553ba42fa42e442ebb06a0>`,
	    :target:`dnnl_aBCd2b8c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2fab4a76a2034de9a232ea95b8b33325>`,
	    :target:`dnnl_aBCde2b8c16b4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2fa9c32ba1d0d806de1a1f737dac77c9>`,
	    :target:`dnnl_BA4b8a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafe50d69f895b5b98945fdcfb2abb9e5b>`,
	    :target:`dnnl_BA4b8a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9e1067c19233eeb5cd1e4b5dcbff4c65>`,
	    :target:`dnnl_aCB4c8b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da31bd0b229a77a42ae7dc2a5702db78b4>`,
	    :target:`dnnl_aCB4c8b16c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da846923443f9f711eff81bdcc215db803>`,
	    :target:`dnnl_BA16a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da43efb33c6a2a433077fa589640e5a378>`,
	    :target:`dnnl_BA16a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaf20363d4b35b6ebdbc596639ba063cd>`,
	    :target:`dnnl_BA16a48b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab7cc850a98a02ee457d22c827a3604e8>`,
	    :target:`dnnl_BA16a64b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa850d6ef7e3f69c388683f7ac3b0adb1>`,
	    :target:`dnnl_aCB16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad9fe6ee6b56c1299958b111e5c21cc7e>`,
	    :target:`dnnl_aCB16c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8e5cddf640011ca2afb8a63ba943e61f>`,
	    :target:`dnnl_BA16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da725f6a1f3591ba1a67aa65b90b2ed27a>`,
	    :target:`dnnl_BA16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2ff953a657353c051320c32e995a32b2>`,
	    :target:`dnnl_aBC16b16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad76bccf1677d17cc00aad7dc765d142e>`,
	    :target:`dnnl_aBC16b32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabc128ab263d8c06bd5aebc7e95897a01>`,
	    :target:`dnnl_AB16a16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacb0439cd8653e7e08934ba15b540c3b4>`,
	    :target:`dnnl_AB16a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf886aecef1e5868f5509060ce36ce388>`,
	    :target:`dnnl_ABcde16a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4bfae5b885e40efdd79fc172ffcadfc4>`,
	    :target:`dnnl_aBCdef16b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0d2a3c56b49b3795d4dd18047e3695a5>`,
	    :target:`dnnl_Acedb16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da780ff957f64fcbc80d58157ea21ce03b>`,
	    :target:`dnnl_aBdfec16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8cf1d09004eb9e2005264c4eac2082d0>`,
	    :target:`dnnl_abdEC64e2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daeeb247d5e8956b11c19e6d366beaf7fa>`,
	    :target:`dnnl_abdEC64e4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadded38b52a465747cb27adc03efb349f>`,
	    :target:`dnnl_aCB16b16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa3b7f0abcc5c7492f65709318b20435a>`,
	    :target:`dnnl_aCB16b32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3cf7b642ee264da10298eb0bde7d9f7d>`,
	    :target:`dnnl_aCB16b48c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da159924ee2cc4f59570a4d878c16a2e31>`,
	    :target:`dnnl_aCB16b64c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da330d995ae671b65233e6d3cae1771ace>`,
	    :target:`dnnl_aCB16b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac9ea2c0dc9608f446efb6630a0e09ea7>`,
	    :target:`dnnl_aCB16b32c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da02394bf42da81a7565560da0444ef4cc>`,
	    :target:`dnnl_aCB16b48c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4da1386f7f02698f48729ac91b4530cf>`,
	    :target:`dnnl_aCB16b64c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da18995de3dde64dec3cc66cadfcdaf2c2>`,
	    :target:`dnnl_aCB16b16c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da887e2a129cdb46bdff06386784ca8205>`,
	    :target:`dnnl_aCB16b32c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab6d2c19e74a6208f6aa22a3a0bca61f>`,
	    :target:`dnnl_aCB16b48c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da795b2c6f2b8513f47608194d86f77a8b>`,
	    :target:`dnnl_aCB16b64c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da56eefc7c69875ea8b8a03081e7eee877>`,
	    :target:`dnnl_abCd4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6701993ba668a5c5a8be285b16296fce>`,
	    :target:`dnnl_abCde4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf7e9cd51d1cedd6c5a1f3b4f95a68988>`,
	    :target:`dnnl_abCdef4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf2151d7f7a2fcfc196732ffadd6a24da>`,
	    :target:`dnnl_abCde32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da173c82a993fef6fc0cb569b4c1b80a7b>`,
	    :target:`dnnl_abCdef32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa679fe7bc096a43e4435d57d82361bb3>`,
	    :target:`dnnl_ABcd16a32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf4026fb5f04b7d9d4f91153c2147de31>`,
	    :target:`dnnl_decbA8a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daea708e8bae5de45815676b362ed169b0>`,
	    :target:`dnnl_aCdefB16b32c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac270889dae879650c603758a2ddfa347>`,
	    :target:`dnnl_aCdefB16b32c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5cf858bce68aad01cbbf302ff6b5c1bb>`,
	    :target:`dnnl_aCdefB16b48c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da076f8b8185bc534d60ab9085f7ef481a>`,
	    :target:`dnnl_aCdefB16b48c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da90320455ec7e8ce6433168627a44bc15>`,
	    :target:`dnnl_aCdefB16b64c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da588762a64043a0b00b330b04613fa660>`,
	    :target:`dnnl_aCdefB16b64c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadd80c1a5acf4e3793c5c3b27662c2396>`,
	    :target:`dnnl_BcdeA16a32b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0b1150ffd05f8a46b1403d711433956f>`,
	    :target:`dnnl_BcdeA16a32b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da762d85efa830cb54aa42763b01c47d20>`,
	    :target:`dnnl_BcdeA16a48b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac5fc00d3d5299828c3c4a200eaa2ea53>`,
	    :target:`dnnl_BcdeA16a48b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf5041087fb0c64b68614239bbf976227>`,
	    :target:`dnnl_BcdeA16a64b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da00a6233d1a0419da029c3c2b17dc8701>`,
	    :target:`dnnl_BcdeA16a64b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da87760ad7d5c8cf5a859095558c849516>`,
	    :target:`dnnl_aCdefb32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da89eb552ee1f395b61d12702f13ec8bb0>`,
	    :target:`dnnl_aCdefB32c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabbb98c54829bae9d26b6c97edbfac0f3>`,
	    :target:`dnnl_aCdefB32c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7e688fd4b3a72553c0b6a5576e307f58>`,
	    :target:`dnnl_aCdefb48c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6853afba64cc7890595af766977d11da>`,
	    :target:`dnnl_aCdefB48c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab854f6d40da7dd060c80e8ee50fb94a2>`,
	    :target:`dnnl_aCdefB48c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2a2442f09fbb944f9e0fef6923048dfd>`,
	    :target:`dnnl_aCdefb64c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fd91164e917772c2ad17c382eb90ede>`,
	    :target:`dnnl_aCdefB64c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7af7c37adccb0acfc738fccba67139e3>`,
	    :target:`dnnl_aCdefB64c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4152815e3afefb0dbe9cc675ccff191b>`,
	    :target:`dnnl_Bcdea32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5700a5544a5db4bd6b96bb439ade978d>`,
	    :target:`dnnl_BcdeA32b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf5e4f0229d0388b03e09edf8d4353fb8>`,
	    :target:`dnnl_BcdeA32b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dada266c12e75a6b865fb4605ed50fa780>`,
	    :target:`dnnl_Bcdea48b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac3b9e0e0c4d19634f0f97a9138d4ae7e>`,
	    :target:`dnnl_BcdeA48b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad9c99ae7982b141724d8f92321f8fc1f>`,
	    :target:`dnnl_BcdeA48b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da61f38008805bef392c3d4f3ecefb88b7>`,
	    :target:`dnnl_Bcdea64b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7cef84c229e162d29905bac5774e4a79>`,
	    :target:`dnnl_BcdeA64b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac9b9f26b4ea7342521b8893aeded07db>`,
	    :target:`dnnl_BcdeA64b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5d8fc99dd87d8fe117398f03e030a84b>`,
	    :target:`dnnl_Bca32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa7ac3b5af631a32e2d122137dc45d707>`,
	    :target:`dnnl_BcA32b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da98f6e7c1277d646ae418c35d430dc28f>`,
	    :target:`dnnl_BcA32b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4617829735647568e410c4c6f0e98fc8>`,
	    :target:`dnnl_Bca48b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a1f51dd2c8e33dab2af96e889382de8>`,
	    :target:`dnnl_BcA48b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacc40561d508ffad591bc7ba4c5552f4f>`,
	    :target:`dnnl_BcA48b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da77193baecd8cc0117f30a0381569fe4b>`,
	    :target:`dnnl_Bca64b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daffcff09e11e7590f53e7755144ba557c>`,
	    :target:`dnnl_BcA64b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad10a2b9d60ea67690451a04b65b26788>`,
	    :target:`dnnl_BcA64b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da615f059c074e9704136024def50089a3>`,
	    :target:`dnnl_aCdb32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7e2e4683b026bde11cec4fa482491809>`,
	    :target:`dnnl_aCdB32c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafa8b4b9f94d8eb303252548fc5099a16>`,
	    :target:`dnnl_aCdB32c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7c809be88bae87692bb217040135d0f2>`,
	    :target:`dnnl_aCdb48c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5da635ff9849cf301ff339ef61f57382>`,
	    :target:`dnnl_aCdB48c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da921a9fb07c54f118b605a7b3967c09e1>`,
	    :target:`dnnl_aCdB48c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac29f1a725c79e8352ebd4ed27b584e50>`,
	    :target:`dnnl_aCdb64c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6b4a0e46e57b46defe4f98fda6bb84e9>`,
	    :target:`dnnl_aCdB64c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4f1a9fe7721a212627e024cb795de71b>`,
	    :target:`dnnl_aCdB64c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da53e54b90ce6eee31b8f1a1cb7f0bb665>`,
	    :target:`dnnl_BcA16a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da615961e857477501b6854df2d4e3c31e>`,
	    :target:`dnnl_BcA16a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da062c93b99148f9d97c8616f1564470cf>`,
	    :target:`dnnl_BcdA16a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da494cc1d0a5853f7d6e562e2d35030d7a>`,
	    :target:`dnnl_BcdA16a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da532ab8485226cd0cb752142ee566f34b>`,
	    :target:`dnnl_BcdeA16a16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daeddc6ded708bdf00b623307a2713e8b5>`,
	    :target:`dnnl_BcdeA16a16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaa5b5f68fbc4f83bde057922d25a28df>`,
	    :target:`dnnl_aCdB16b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae5ab42b316ef95bf1ab6552d1f422fd6>`,
	    :target:`dnnl_aCdB16b16c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2aba41e7e13e511712b271fe32612e33>`,
	    :target:`dnnl_aCdeB16b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da33e5260102541d43e3f3921aa8cbf35e>`,
	    :target:`dnnl_aCdeB16b16c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daff161cfbf11cd74e8d448eedb51e434c>`,
	    :target:`dnnl_aCdefB16b16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3fa82822a7d5d03244715dc4b1afcdd5>`,
	    :target:`dnnl_aCdefB16b16c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da01d03440b0a47f5371dab0717a569b73>`,
	    :target:`dnnl_BcA16a32b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4567c12982540ad1f9ea4d5aec886237>`,
	    :target:`dnnl_BcA16a32b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da636d7b3cf75267a37cdcc4fb4de912a0>`,
	    :target:`dnnl_BcA16a48b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7edb6dcb6fea8c07f7540dc1078fb642>`,
	    :target:`dnnl_BcA16a48b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daceab00f19d9032fddbfb40bd85e8c7de>`,
	    :target:`dnnl_BcA16a64b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daddfd29c4b54901b0d3ced40ce942b7d1>`,
	    :target:`dnnl_BcA16a64b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da686644c1574c9ed2b009b25dbc5d14b8>`,
	    :target:`dnnl_aCdB16b32c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab42e5c54d346f5af037a28944e2c8866>`,
	    :target:`dnnl_aCdB16b32c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2da67629303c71d3d140b343585f6adb>`,
	    :target:`dnnl_aCdB16b48c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0974ddfe38f9c49a6b3e7fe8f6446e05>`,
	    :target:`dnnl_aCdB16b48c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da283d3fb07824ef75e158fd6532192f42>`,
	    :target:`dnnl_aCdB16b64c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7fb4e55cb353b9216e895ad01eab3dfa>`,
	    :target:`dnnl_aCdB16b64c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da718c531b11281b5f27a5b1b519487fc3>`,
	    :target:`dnnl_BcdA16a32b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33580e0da9b63b60201250ba4c3dca4>`,
	    :target:`dnnl_BcdA16a32b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5c9aa6554fad4c31efa1571fddd9aa89>`,
	    :target:`dnnl_BcdA16a48b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da94c22cc356f7aae8337c1d476a7f04ee>`,
	    :target:`dnnl_BcdA16a48b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da87ccfa02c06a0f05397b01783006808a>`,
	    :target:`dnnl_BcdA16a64b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da00f40c279785af83ae3e618e32ed5d3f>`,
	    :target:`dnnl_BcdA16a64b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa94aaf10b3d8e74e6475e870e5e6f6b7>`,
	    :target:`dnnl_aCdeB16b32c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da53d1dc2428167dbd85d360717d934b5c>`,
	    :target:`dnnl_aCdeB16b32c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da88c9a1159a39f13f90f39006a62502c2>`,
	    :target:`dnnl_aCdeB16b48c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadb1e8e75d04eeaa24c4cc789f38d73cc>`,
	    :target:`dnnl_aCdeB16b48c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da64a9220107eb7e61b5ab39ded6e83fb5>`,
	    :target:`dnnl_aCdeB16b64c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9d9d77660ea6359ad69dac8061d56655>`,
	    :target:`dnnl_aCdeB16b64c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daae54415c2eef39ae944a4f522c727623>`,
	    :target:`dnnl_Bca16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8e5dc6ba750795066f4977ec996283a2>`,
	    :target:`dnnl_BcA16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da82e990f5302596424298f36501071cd3>`,
	    :target:`dnnl_BcA16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daae9116466db091828c6346170eca730d>`,
	    :target:`dnnl_Bcda16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa28c3b3ba4363fb8a8cd94384be2730d>`,
	    :target:`dnnl_BcdA16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daad41a8f09fdc44c4c8361989f4fddbcf>`,
	    :target:`dnnl_BcdA16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac04f6599f0552f6cad544a080e5270d5>`,
	    :target:`dnnl_Bcdea16b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafc2295dec677a90c6f2aa03e349d58f2>`,
	    :target:`dnnl_BcdeA16b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6a6132f6e020bc8eaffaf217682ed223>`,
	    :target:`dnnl_BcdeA16b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da710bbb4357987234e1b8fd069cbd3deb>`,
	    :target:`dnnl_aCdb16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da61bef909020eeea08eb2fdeb56b2c617>`,
	    :target:`dnnl_aCdB16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da20242253e4ee90a462c881704d574331>`,
	    :target:`dnnl_aCdB16c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5cff139072cc733924dbc5f3153b2c0f>`,
	    :target:`dnnl_aCdeb16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bb7caa7744439256cebf3f722c2a067>`,
	    :target:`dnnl_aCdeB16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da352136fe0709ab285c4c7326eefeebc7>`,
	    :target:`dnnl_aCdeB16c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3ee9853cb1f9add476456c6c67a0cfa8>`,
	    :target:`dnnl_aCdefb16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da275ca0b3253c4939e16c401b2b8dcc39>`,
	    :target:`dnnl_aCdefB16c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfca62833c015a1bc7240b3c229ad9d>`,
	    :target:`dnnl_aCdefB16c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae9a471b5f1a438b14aadb8c8db2e845c>`,
	    :target:`dnnl_Bcda32b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2d9c0c5c843b08ed57a355700c0854fa>`,
	    :target:`dnnl_BcdA32b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf605c64b826400b1789d47bf1a6c25fd>`,
	    :target:`dnnl_BcdA32b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4678d89b52053bcae380e03d6cef1952>`,
	    :target:`dnnl_Bcda48b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5798064e120d67a0278d0d81d4fd3ae1>`,
	    :target:`dnnl_BcdA48b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daee287435ea23b3ad3e27ac524b5ca34e>`,
	    :target:`dnnl_BcdA48b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da64c458c4c5be418e1ccb4bdb48a5cca0>`,
	    :target:`dnnl_Bcda64b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daec2a5fa8478c3ea665908afc1d5e6b8e>`,
	    :target:`dnnl_BcdA64b2a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8f5eab130f56276444b682b81121d09b>`,
	    :target:`dnnl_BcdA64b4a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9e33a96593d03b89a2a9154ee9f3d094>`,
	    :target:`dnnl_aCdeb32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3dceef36095ce66d10cea486d98a28ed>`,
	    :target:`dnnl_aCdeB32c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5693bb224f3e9d0ee32ef9cd2e8ec1f5>`,
	    :target:`dnnl_aCdeB32c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa8d2790d9a35978133e4b8e9c5352b77>`,
	    :target:`dnnl_aCdeb48c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daae600a35187a70a796ffe8717ffc635c>`,
	    :target:`dnnl_aCdeB48c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da54bf8da6d631cf43f97434b1187bae6d>`,
	    :target:`dnnl_aCdeB48c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab58da21513213f12e085a1c918cdb7cb>`,
	    :target:`dnnl_aCdeb64c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2902ebdd09c846b8105564b49f0e7243>`,
	    :target:`dnnl_aCdeB64c2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da09b515378231b012428d320918e46661>`,
	    :target:`dnnl_aCdeB64c4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa499d60cef3e8320434ba1d09362ceaa>`,
	    :target:`dnnl_Acb24a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafcd1cae6bad970e42eb3c0c33dd7ad80>`,
	    :target:`dnnl_Acdb24a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6aefc4e43d9b0a48af2d0f1b89d98af2>`,
	    :target:`dnnl_Acdeb24a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6dbba382920deeaace7ea3763ff72186>`,
	    :target:`dnnl_aBdc24b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3676767c4bfac7551264033095fae2cc>`,
	    :target:`dnnl_aBdec24b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5458c21bd99427f43bf40d781199941d>`,
	    :target:`dnnl_aBdefc24b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7d5fcd4c1b55016949e3bc1e6e3ebc86>`,
	    :target:`dnnl_abDc16d<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4e2dcc3deef9884d1885317232be8f16>`,
	    :target:`dnnl_abdEc16e<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab6986f74745bcd8a7aff83e56e7ee3f2>`,
	    :target:`dnnl_abdCe16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daca3fb21aeca67bb7fb1783c180ba6e4e>`,
	    :target:`dnnl_AcB24a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0219e884fc6ec20d3b617fcfa2ad78e7>`,
	    :target:`dnnl_AcdB24a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1e2fd261ecf1350c17fb556d4065a269>`,
	    :target:`dnnl_AcdeB24a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac65500f9e21313d5170a03a4aa3490f4>`,
	    :target:`dnnl_aBdC24b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da33d9f7efe3d33fe987ecc50c67d8838b>`,
	    :target:`dnnl_aBdeC24b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da14f90fc699c33fc62264319015dd7f39>`,
	    :target:`dnnl_aBdefC24b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da32a8eee41e67592d721c6ce1b8198e07>`,
	    :target:`dnnl_AcB8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da175b88236104a9a10f7c83bb55dc7a71>`,
	    :target:`dnnl_AcdB8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3188dba3c46ccb3be6ffb73598f8d5f5>`,
	    :target:`dnnl_AcdeB8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daabfa94181d91d0f1cc2ab7c691953304>`,
	    :target:`dnnl_aBdC8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3a2e5496bec2aaee679481bca226e6c9>`,
	    :target:`dnnl_aBdeC8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4fe2844ef88178439ecda4883138cd05>`,
	    :target:`dnnl_aBdefC8b2c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad7dcb0078ee84e8eb0bdc46205d03303>`,
	    :target:`dnnl_AB8b32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9e6df63d3919142c23bca4b324eca644>`,
	    :target:`dnnl_ABc8b32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadf1d674f70cf1844a402ee6e17cf3241>`,
	    :target:`dnnl_ABcd8b32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da557a151a572245efaff25dee155c7c86>`,
	    :target:`dnnl_ABcde8b32a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac650983683644e389cdfd551d77d6497>`,
	    :target:`dnnl_AB8b24a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafbe44431f6641af590ff07598c542e15>`,
	    :target:`dnnl_ABc8b24a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da94f6820183cec077363e2e014aef4265>`,
	    :target:`dnnl_ABcd8b24a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da82cfccda7527486f60d94529528595e9>`,
	    :target:`dnnl_ABcde8b24a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab6e26a00a4c997cc7982c12ac4c330f8>`,
	    :target:`dnnl_AB8b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da56594ebf1eff07b1024c5d755bf3df0c>`,
	    :target:`dnnl_ABc8b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8e37938f3b5feb2f8aab8881878068e0>`,
	    :target:`dnnl_ABcd8b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da54a48bb18ae127bdad08b962f7b65f1a>`,
	    :target:`dnnl_ABcde8b16a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1f9663f0dcb6e761aff0413a2e336860>`,
	    :target:`dnnl_AB8b8a<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da563c8f664e4fba1a16c498373f2775b2>`,
	    :target:`dnnl_AB4b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da33cb122890b217366af8cf30a9082134>`,
	    :target:`dnnl_AB4b24a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae4e4c8150d12db66636dd2cf03269be1>`,
	    :target:`dnnl_ABc4b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2fc75b6e298e03c83f9cbf08fbd27d79>`,
	    :target:`dnnl_ABc4b24a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae7afa4d3c06ded338ea8ced981f53137>`,
	    :target:`dnnl_ABcd4b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4526f2ac464d4ac41b1c95e123837673>`,
	    :target:`dnnl_ABcd4b24a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da70a74032ced73f8802af2e15c994a960>`,
	    :target:`dnnl_ABcde4b8a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3465b408a72d33e16769dbf90cb2077c>`,
	    :target:`dnnl_ABcde4b24a4b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da72b00e029b2f7be3284d37f96308d676>`,
	    :target:`dnnl_AB8b24a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8650ad4b2f6d9426c7a48bd6b8105fa>`,
	    :target:`dnnl_ABc8b24a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac84c6ee17a9abf5878a10d59f567d130>`,
	    :target:`dnnl_ABcd8b24a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7e9ea3c7e387e9591888984709885778>`,
	    :target:`dnnl_ABcde8b24a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1ac1f9e36ab7cf5b24eed7d06c9e3a68>`,
	    :target:`dnnl_AB8b8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacee2c612da29a89ef40c8147c1e38098>`,
	    :target:`dnnl_ABc8b8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da77a3bcbd2e325464d4a6e863cc10d216>`,
	    :target:`dnnl_ABcd8b8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da46485a16a9852f0edbd94c843d5811cb>`,
	    :target:`dnnl_ABcde8b8a2b<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac62a5df22bd5a8eea2f202ad0ee5e91e>`,
	    :ref:`dnnl_format_tag_last<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadece229e4705088e29d6cbfb94c64756>`,
	    :ref:`dnnl_x<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9ccb37bb1a788f0245efbffbaf81e145>`                = dnnl_a,
	    :ref:`dnnl_nc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>`               = dnnl_ab,
	    :ref:`dnnl_cn<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da11b661fe7ab82f99b34bcaf460a8dc66>`               = dnnl_ba,
	    :ref:`dnnl_tn<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac1ee747f881fc8cae805ae4051e671cf>`               = dnnl_ab,
	    :ref:`dnnl_nt<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7ede27a46172c1ab2775593ba4b118dc>`               = dnnl_ba,
	    :ref:`dnnl_ncw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab55cb1d54480dd7f796bf66eea3ad32f>`              = dnnl_abc,
	    :ref:`dnnl_nwc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f756dbdc1e949646c95f83e0f51bc43>`              = dnnl_acb,
	    :ref:`dnnl_nchw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>`             = dnnl_abcd,
	    :ref:`dnnl_nhwc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>`             = dnnl_acdb,
	    :ref:`dnnl_chwn<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab65a38658838cec19e718ba048cd459>`             = dnnl_bcda,
	    :ref:`dnnl_ncdhw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464>`            = dnnl_abcde,
	    :ref:`dnnl_ndhwc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>`            = dnnl_acdeb,
	    :ref:`dnnl_oi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daee91ff6fadfe8c5494be1595fd253c4e>`               = dnnl_ab,
	    :ref:`dnnl_io<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da46de139a50746a1a22d8d33c60d9081b>`               = dnnl_ba,
	    :ref:`dnnl_oiw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae6bd836b8515857585336a0921e35663>`              = dnnl_abc,
	    :ref:`dnnl_owi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da250d14a2373c43ea403098b4758152ab>`              = dnnl_acb,
	    :ref:`dnnl_wio<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da93eecc25f8ab1b07604b632401aa28e5>`              = dnnl_cba,
	    :ref:`dnnl_woi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da16fac91932b20b2f38776916a59765fb>`              = dnnl_cab,
	    :ref:`dnnl_iwo<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7811cefee328570707df45965b63844e>`              = dnnl_bca,
	    :ref:`dnnl_oihw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da11176ff202375dcd0d06e2fba5f8a8e0>`             = dnnl_abcd,
	    :ref:`dnnl_hwio<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4f4c7bd98c6d53fb3b69e1c8df0a80f6>`             = dnnl_cdba,
	    :ref:`dnnl_hwoi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa7f07aa691f07f8b4d04422537b82202>`             = dnnl_cdab,
	    :ref:`dnnl_ohwi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7543e08a69ca4f7770124d6424314d4b>`             = dnnl_acdb,
	    :ref:`dnnl_ihwo<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf1345b7d3d7aef806d9716287104dc96>`             = dnnl_bcda,
	    :ref:`dnnl_iohw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da17b2a778caa0f78f3387b09611ab9ceb>`             = dnnl_bacd,
	    :ref:`dnnl_oidhw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da18e605a5f6afe9995961fc21fddf314e>`            = dnnl_abcde,
	    :ref:`dnnl_iodhw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9681b8a3bb79ce73a21b5bb9c682bfde>`            = dnnl_bacde,
	    :ref:`dnnl_dhwio<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae4885779f955beeddc25443a3f8c2a63>`            = dnnl_cdeba,
	    :ref:`dnnl_dhwoi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daba857ec496e1489ed2692f6441e7ece5>`            = dnnl_cdeab,
	    :ref:`dnnl_odhwi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadf48b08738673a6746907b934d82248d>`            = dnnl_acdeb,
	    :ref:`dnnl_idhwo<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6fdbc8d31779a5548bc39c23d41466b2>`            = dnnl_bcdea,
	    :ref:`dnnl_goiw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a95a9ab4d0be153fe304d94a4f96138>`             = dnnl_abcd,
	    :ref:`dnnl_gowi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4ff8c44741175d152711815fb15a9da5>`             = dnnl_abdc,
	    :ref:`dnnl_wigo<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da27409507462c2e1f45deb987623de29c>`             = dnnl_dcab,
	    :ref:`dnnl_goihw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daee332d306e012c0614dc32e0ed66072f>`            = dnnl_abcde,
	    :ref:`dnnl_gohwi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da28265c5d27d555da689f63472db852cf>`            = dnnl_abdec,
	    :ref:`dnnl_hwigo<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da059c27962618c1f7670f5389bb7fb7ed>`            = dnnl_decab,
	    :ref:`dnnl_giohw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daecd143f4e88930c11ab4348bb5c24dc4>`            = dnnl_acbde,
	    :ref:`dnnl_goidhw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da621b8c724d93d2a5444feab224b2ce3e>`           = dnnl_abcdef,
	    :ref:`dnnl_godhwi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3f6895614839ab01bd55b85f81452293>`           = dnnl_abdefc,
	    :ref:`dnnl_giodhw<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da44de40f7cc576af9bff1d0c95fe6bf0f>`           = dnnl_acbdef,
	    :ref:`dnnl_dhwigo<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da81de66a3883f148984ed2419c794d5ce>`           = dnnl_defcab,
	    :ref:`dnnl_tnc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2a9735ec024c9362b717304edbfe2237>`              = dnnl_abc,
	    :ref:`dnnl_ntc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5d73ca7a68559ef44241be5a096e6bff>`              = dnnl_bac,
	    :ref:`dnnl_ldnc<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4fd1cf9fdb67c554bcd8281695b65b3c>`             = dnnl_abcd,
	    :ref:`dnnl_ldigo<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da96a227ab1a1be1825c1fa596c38847fc>`            = dnnl_abcde,
	    :ref:`dnnl_ldgoi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5fc9120d8f52d7d7fa853aa79bf654fe>`            = dnnl_abdec,
	    :ref:`dnnl_ldio<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2d5a8e24d6b4904b4e8986d9b0fb4613>`             = dnnl_abcd,
	    :ref:`dnnl_ldoi<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da475da8ead8c761bac894e6c87042355d>`             = dnnl_abdc,
	    :ref:`dnnl_ldgo<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2bc162b37fd0049dceab3b12300a26c7>`             = dnnl_abcd,
	    :ref:`dnnl_ldOi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8d352f148c426ccaa7dae070f50f24b4>`          = dnnl_abDc16d,
	    :target:`dnnl_ldOi32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2c3de22966fb44f0d70f62b5b4e943e7>`          = dnnl_abDc32d,
	    :target:`dnnl_ldOI32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa7361baa721c74fab976b16e2c72dc04>`        = dnnl_abDC32d4c,
	    :target:`dnnl_ldIo32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da597abd3a3b364f5eb9101652629f9f95>`          = dnnl_abCd32c,
	    :ref:`dnnl_ldgOi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4017f8ca50acb2e22e3e5a31e4aae0ee>`         = dnnl_abdEc16e,
	    :target:`dnnl_ldgOi32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da17e0124339e85b51347d652ccdc3351e>`         = dnnl_abdEc32e,
	    :target:`dnnl_ldgOI32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab18f63a8ec1ca040b4f7f82e94329e28>`       = dnnl_abdEC32e2c,
	    :target:`dnnl_ldgOI32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da797f3876262fcc5eb265d6039acb57c4>`       = dnnl_abdEC32e4c,
	    :target:`dnnl_ldgOI64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da728597fd53cfe690997ad15a72d19a56>`       = dnnl_abdEC64e2c,
	    :target:`dnnl_ldgOI64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadd10607fdb1ed9d7afd344086012c296>`       = dnnl_abdEC64e4c,
	    :target:`dnnl_ldgIo16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da340bc31e9b39a7af629c7553d77a6ea5>`         = dnnl_abdCe16c,
	    :target:`dnnl_ldgIo32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1a6a323b31a1f92582fd41a9a8acda7d>`         = dnnl_abdCe32c,
	    :target:`dnnl_ldgIO32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab586283172711dc4ff86141de323baf6>`       = dnnl_abdCE32c2e,
	    :ref:`dnnl_nCdhw32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da773408e10d72e338fb417ba0634a7169>`         = dnnl_aBcde32b,
	    :ref:`dnnl_nCdhw16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2fd2bec64ddb6031730c180acf1a894f>`         = dnnl_aBcde16b,
	    :ref:`dnnl_nCdhw4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da72cb744b96d7a9feb19fbc488dfcdc90>`          = dnnl_aBcde4b,
	    :ref:`dnnl_nCdhw8c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabacffa20b5188cda4d5f86e2e10d2572>`          = dnnl_aBcde8b,
	    :ref:`dnnl_nChw32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da18b4ebf475d956c82508de954e38adf2>`          = dnnl_aBcd32b,
	    :ref:`dnnl_nChw16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa7847819b4fb840d2db20796bc607a5c>`          = dnnl_aBcd16b,
	    :ref:`dnnl_nChw4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da80c1478a2fb2bb2095d1866e1d04908e>`           = dnnl_aBcd4b,
	    :ref:`dnnl_nChw8c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da775389dbdcda91ea12906210c309746f>`           = dnnl_aBcd8b,
	    :ref:`dnnl_nCw32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5b32114a6ae17691ce4c81869d6aebd8>`           = dnnl_aBc32b,
	    :ref:`dnnl_nCw16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf996bf5a22fc850b0d4cf5d2533aa134>`           = dnnl_aBc16b,
	    :ref:`dnnl_nCw4c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa487c7e7aad196e5dbe2d1b22da5b045>`            = dnnl_aBc4b,
	    :ref:`dnnl_nCw8c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad691bfd0677cab219ce336afa6023a59>`            = dnnl_aBc8b,
	    :target:`dnnl_NCw16n16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3263c4ee7abc6321442a231e21a21c54>`        = dnnl_ABc16a16b,
	    :target:`dnnl_NCdhw16n16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da097fbf94e1c245521fcec98f26cef805>`      = dnnl_ABcde16a16b,
	    :target:`dnnl_NChw16n16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da483ea1ba0d29518fd59adefafcaad284>`       = dnnl_ABcd16a16b,
	    :target:`dnnl_NCw32n16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1e5c2cdc89d1a4594dbd6ba7c56112b7>`        = dnnl_ABc32a16b,
	    :target:`dnnl_NChw32n16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8e784a25c78acf17bc2dafd51f47d283>`       = dnnl_ABcd32a16b,
	    :target:`dnnl_NChw16n32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1c133cf487c88bb2abc2311ac08c0310>`       = dnnl_ABcd16a32b,
	    :target:`dnnl_NCdhw32n16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa6948693b65c40d57b3a2ce40103cf3f>`      = dnnl_ABcde32a16b,
	    :target:`dnnl_NCw32n32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae6cecc59e9e98488f6567a7762f8d50f>`        = dnnl_ABc32a32b,
	    :target:`dnnl_NChw32n32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0140dd8908a849f3c207359a61e53e23>`       = dnnl_ABcd32a32b,
	    :target:`dnnl_NCdhw32n32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa849bf31997dbd878cc29be9d892ee26>`      = dnnl_ABcde32a32b,
	    :target:`dnnl_OI16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da70ae08d1472cf95e88e9cea1e589b487>`         = dnnl_AB16b16a,
	    :target:`dnnl_OI16i32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3e0514046979899a2464a164998140be>`         = dnnl_AB16b32a,
	    :target:`dnnl_OI16i64o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad5c4147c90a89b045239307a416e31a0>`         = dnnl_AB16b64a,
	    :target:`dnnl_OI8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da953e1e7ea3354d0d06dcc2e3a5350d33>`         = dnnl_AB8b8a2b,
	    :target:`dnnl_OI8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad32226d92ce1f9abbb8a0b49da2aea61>`        = dnnl_AB8b16a2b,
	    :target:`dnnl_OI8i24o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac9a04125a9cccd8b033af2e9ae328e23>`        = dnnl_AB8b24a2b,
	    :target:`dnnl_OI8i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac37e806f9c6715a8cc6f7322eea5154d>`        = dnnl_AB8b32a2b,
	    :target:`dnnl_OI8i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac1a73794590a4b94882059af400f5964>`        = dnnl_AB8b64a2b,
	    :target:`dnnl_OI4i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da126345cccbd3e7151bed91e83cb29480>`         = dnnl_AB4b8a4b,
	    :target:`dnnl_OI4i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da329e0032ad40264f7dff3af6c629e9af>`        = dnnl_AB4b16a4b,
	    :target:`dnnl_OI4i24o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab1ff947e8ba8e45ccd0dd9fae099d3e4>`        = dnnl_AB4b24a4b,
	    :target:`dnnl_OI4i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadb9ccc19712f4253da97206cfd3b372a>`        = dnnl_AB4b32a4b,
	    :target:`dnnl_OI4i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab7248a2034cb393a203afb495f4a7d93>`        = dnnl_AB4b64a4b,
	    :target:`dnnl_OI16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7b0ac0a10a976c18cabea7f77cbc852b>`       = dnnl_AB16b16a4b,
	    :target:`dnnl_OI8i32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab7848f1e49d7c8371ae3d8ddbc16c833>`          = dnnl_AB8b32a,
	    :target:`dnnl_OI8i24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da84aaca7609aae6e3d056067b9cfcff0b>`          = dnnl_AB8b24a,
	    :target:`dnnl_OI8i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadba3624f83714937ea07d73c50625ed2>`          = dnnl_AB8b16a,
	    :target:`dnnl_OI8i8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8b7401712f554efc7d191891a32fc0cd>`           = dnnl_AB8b8a,
	    :target:`dnnl_IOw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9fa40d9f4ff8088d342f64853caebcc3>`        = dnnl_BAc16a16b,
	    :target:`dnnl_IOw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab7724db5808bf70860eb4e5dc2276ed8>`        = dnnl_BAc16b16a,
	    :target:`dnnl_OIw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3114a77fa7e684077d9befdebbb2960f>`        = dnnl_ABc16b16a,
	    :target:`dnnl_OIw16i32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daac8f4ec7ff9297816b48d5d901ca4e4e>`        = dnnl_ABc16b32a,
	    :target:`dnnl_OIw16i64o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da833eccd4eb04d13eff7ea96e0dce7c13>`        = dnnl_ABc16b64a,
	    :target:`dnnl_OIw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da64e2e3926f5b52386c8c7455391f4361>`        = dnnl_ABc16a16b,
	    :target:`dnnl_Oiw16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab4adb356b9d272b0a8d33d9453f6654b>`           = dnnl_Abc16a,
	    :target:`dnnl_OIw4i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0554af70de5fe5db5bde03030c3348b0>`        = dnnl_ABc4b8a4b,
	    :target:`dnnl_OIw4i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8c6683cb23dfb280b086140df1b8dd3d>`       = dnnl_ABc4b16a4b,
	    :target:`dnnl_OIw4i24o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacf634c773f92769a72378cb9957f3293>`       = dnnl_ABc4b24a4b,
	    :target:`dnnl_OIw4i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da59ff4b23ce132ed0737617c5b33bef7c>`       = dnnl_ABc4b32a4b,
	    :target:`dnnl_OIw4i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5ffabaa439f8b80f6f4dfeb1f2029b7e>`       = dnnl_ABc4b64a4b,
	    :target:`dnnl_OIw2i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da75cbe5ac5b0a90d0e89f01f1996b002f>`        = dnnl_ABc2b8a4b,
	    :target:`dnnl_OIw16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2aa9a0656dfe4546352c1559104519ca>`      = dnnl_ABc16b16a4b,
	    :target:`dnnl_OIw16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf37066eb0599edef9b96ad7e85176e1b>`      = dnnl_ABc16b16a2b,
	    :target:`dnnl_OIw16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daee3dd9d1a057b609853ca55b6854838f>`      = dnnl_ABc16a16b2a,
	    :target:`dnnl_OIw4i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadac1acfcf7555c201a05b2ebb056f87c>`          = dnnl_ABc4b4a,
	    :target:`dnnl_OIw4o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7b1585da5b11f9dae89668147630c549>`          = dnnl_ABc4a4b,
	    :target:`dnnl_Oiw4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da451b08d579211bf061bca20dcd9689a2>`            = dnnl_Abc4a,
	    :target:`dnnl_OIw8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9db6ead9b599ead33d342ba1b1f6c1f1>`        = dnnl_ABc8b8a2b,
	    :target:`dnnl_OIw8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac71a95553d16c91fb5da68f17750cfdd>`       = dnnl_ABc8b16a2b,
	    :target:`dnnl_OIw8i24o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da28cc52fc8cc6e9f2ee869557d7cf8d83>`       = dnnl_ABc8b24a2b,
	    :target:`dnnl_OIw8i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da71af48730ab3adc014a01533b736b7c8>`       = dnnl_ABc8b32a2b,
	    :target:`dnnl_OIw8i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da676ff324bb14520f8802ad5ab6a243f0>`       = dnnl_ABc8b64a2b,
	    :target:`dnnl_OIw8i8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da086b93d2f9b9d012b296c414922f291a>`          = dnnl_ABc8b8a,
	    :target:`dnnl_OIw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab28e117f612635b15f24026739a267f>`       = dnnl_ABc8a16b2a,
	    :target:`dnnl_IOw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9bef70d4d56a3497446969d39092f336>`       = dnnl_BAc8a16b2a,
	    :target:`dnnl_OIw8o8i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9bcd61191ce81116de23863c0969de90>`          = dnnl_ABc8a8b,
	    :target:`dnnl_OIw8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da56b42624d5a72bd6796f652305a0e3f3>`          = dnnl_ABc8a4b,
	    :target:`dnnl_Owi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0852f5b60cedf32291274ece1dfd0d52>`           = dnnl_Acb16a,
	    :target:`dnnl_OwI16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da476ae766d01f68357dd960afc312a691>`         = dnnl_AcB16a2b,
	    :target:`dnnl_OwI16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad42395dc7ed898c90ed94376f64b39fa>`         = dnnl_AcB16a4b,
	    :target:`dnnl_Iwo16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da59aeb5086525351edf04b8279c563383>`           = dnnl_Bca16b,
	    :target:`dnnl_IwO16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac0203714ecdbf050b8bda69a793f0185>`         = dnnl_BcA16b2a,
	    :target:`dnnl_IwO16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da04ebaaca9054c1258793431aaaf29e47>`         = dnnl_BcA16b4a,
	    :target:`dnnl_Owi4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7903b928a681ab536a5ba6f17e9948cb>`            = dnnl_Acb4a,
	    :target:`dnnl_Owi8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac89d7bc5e725e012d3e6af162ac48eac>`            = dnnl_Acb8a,
	    :target:`dnnl_OwI8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1dadb702b28356ed0028b2f10a6dd95c>`          = dnnl_AcB8a2b,
	    :target:`dnnl_OIw8i32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafbce184bae694ebf16a7513f367f107d>`         = dnnl_ABc8b32a,
	    :target:`dnnl_OIw8i24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da680831b4b7c67e1eee543b137b539e20>`         = dnnl_ABc8b24a,
	    :target:`dnnl_OIw8i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7565994b49627bf63d555b3b14bdc6c9>`         = dnnl_ABc8b16a,
	    :target:`dnnl_IOhw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8425f2db8faf2705441568334df6d3cf>`       = dnnl_BAcd16b16a,
	    :target:`dnnl_IOhw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da627f9f2ee495860635e6ba587bde3606>`       = dnnl_BAcd16a16b,
	    :target:`dnnl_Ohwi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da271497959df30646ed1b558e5785cd45>`          = dnnl_Acdb16a,
	    :target:`dnnl_OhwI16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad91e4760ccfce3b717069190514ca1c9>`        = dnnl_AcdB16a2b,
	    :target:`dnnl_OhwI16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da06ce9adc2598f05375cb0d87588345d0>`        = dnnl_AcdB16a4b,
	    :target:`dnnl_Ihwo16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafab5e755463371e82bdaf45875891fda>`          = dnnl_Bcda16b,
	    :target:`dnnl_IhwO16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1005a0650a5663205b78f55d79cd9921>`        = dnnl_BcdA16b2a,
	    :target:`dnnl_IhwO16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8baf4382e74a0d69d1aa71cb136695b8>`        = dnnl_BcdA16b4a,
	    :target:`dnnl_Ohwi24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadb160f9a0500c0ae0cd535e467c4b786>`          = dnnl_Acdb24a,
	    :target:`dnnl_Ohwi32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da79298fe723dee9417a0415c1c400ea86>`          = dnnl_Acdb32a,
	    :target:`dnnl_Ohwi4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da56ead2126922096286db563d59bad65c>`           = dnnl_Acdb4a,
	    :target:`dnnl_Ohwi8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8984656af25433160cade03629a9167c>`           = dnnl_Acdb8a,
	    :target:`dnnl_OhwI8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cb945d27d7eda3397b734e8061a61a0>`         = dnnl_AcdB8a2b,
	    :target:`dnnl_OIhw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa46c6aad73c2004f1dcec6a9e70c65b6>`       = dnnl_ABcd16b16a,
	    :target:`dnnl_OIhw16i32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabdac46f0103c462b4271e6040b089f98>`       = dnnl_ABcd16b32a,
	    :target:`dnnl_OIhw16i64o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad9624ae00240b4652d1f41fed42a6fe1>`       = dnnl_ABcd16b64a,
	    :target:`dnnl_OIhw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a3d90df236de2f968c0d5b36a5c1903>`       = dnnl_ABcd16a16b,
	    :target:`dnnl_Oihw16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2310c35aba3a0f86b3df22227c296f0b>`          = dnnl_Abcd16a,
	    :target:`dnnl_OIhw4i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa68eaeaeddfe2f638cd65c004522d33b>`       = dnnl_ABcd4b8a4b,
	    :target:`dnnl_OIhw4i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4a55807db32042ae007bc31e5130eaf1>`      = dnnl_ABcd4b16a4b,
	    :target:`dnnl_OIhw4i24o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da89e29583b628c9c9ee45864985f1f190>`      = dnnl_ABcd4b24a4b,
	    :target:`dnnl_OIhw4i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5473792055784b280bde46a4df52ff75>`      = dnnl_ABcd4b32a4b,
	    :target:`dnnl_OIhw4i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4beed19205cf5c842525f1648003fbba>`      = dnnl_ABcd4b64a4b,
	    :target:`dnnl_OIhw16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3e0c37627fb53cb9d1f237723d07ab78>`     = dnnl_ABcd16b16a4b,
	    :target:`dnnl_OIhw16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1e0409e6338af20bf8e3e93d54d51d88>`     = dnnl_ABcd16b16a2b,
	    :target:`dnnl_OIhw16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac11b47c741b0fa1aec4e1f2943d429a6>`     = dnnl_ABcd16a16b2a,
	    :target:`dnnl_OIhw4i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae10f42f5bfdc16968a3de458a7e90498>`         = dnnl_ABcd4b4a,
	    :target:`dnnl_OIhw4o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da116feda55d4d9769c75f146c5511a80f>`         = dnnl_ABcd4a4b,
	    :target:`dnnl_Oihw4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaa1eb3a28d52a82f80b234d720610ef4>`           = dnnl_Abcd4a,
	    :target:`dnnl_OIhw8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2d2a1b05a758c55d8a2f6fbc4e120eae>`       = dnnl_ABcd8b8a2b,
	    :target:`dnnl_OIhw8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5764874de729b6a42d965aec5d979299>`      = dnnl_ABcd8b16a2b,
	    :target:`dnnl_OIhw8i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac77ef425f2b62ecd8e1e97e8cd3fd8a0>`      = dnnl_ABcd8b32a2b,
	    :target:`dnnl_OIhw8i24o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da43c1be56e5f2c4191757b0dbbe3a5de5>`      = dnnl_ABcd8b24a2b,
	    :target:`dnnl_OIhw8i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa6a0285dc514f0e9e6652cb44d06021a>`      = dnnl_ABcd8b64a2b,
	    :target:`dnnl_OIhw8i8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da33339a60b782ebef804cbef7a70b4515>`         = dnnl_ABcd8b8a,
	    :target:`dnnl_OIhw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da266245277a3c7d6f241886759f29a7e5>`      = dnnl_ABcd8a16b2a,
	    :target:`dnnl_OIhw2i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da025e5aabf36f5cdadb86232c45c947cc>`       = dnnl_ABcd2b8a4b,
	    :target:`dnnl_IOhw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad16b5b8d6b71e5c29050af798ffc2934>`      = dnnl_BAcd8a16b2a,
	    :target:`dnnl_OIhw8o8i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da842592faef431a425bea451d0d1454e0>`         = dnnl_ABcd8a8b,
	    :target:`dnnl_OIhw8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacfd86a2a58c0c89e5464c738b588ead2>`         = dnnl_ABcd8a4b,
	    :target:`dnnl_Owhi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da60e21280940714d1d5e5420eff7c54d1>`          = dnnl_Adcb16a,
	    :target:`dnnl_OIhw8i32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad265f69ec2f123dd3c0eb187cd822ef8>`        = dnnl_ABcd8b32a,
	    :target:`dnnl_OIhw8i24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5b546153c8bc08486af7790eaf925c74>`        = dnnl_ABcd8b24a,
	    :target:`dnnl_OIhw8i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac487392b239239c5aa985d13b69a4ae5>`        = dnnl_ABcd8b16a,
	    :target:`dnnl_Odhwi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9338a3e43b1343762ce8b33404d4c1df>`         = dnnl_Acdeb16a,
	    :target:`dnnl_OdhwI16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf41deee354999e4ed1a95efd2f49a877>`       = dnnl_AcdeB16a2b,
	    :target:`dnnl_OdhwI16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da01a271945552a81e3c2a0e940ab66dfc>`       = dnnl_AcdeB16a4b,
	    :target:`dnnl_Idhwo16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad90dd2711735b30b77c00614a7ede126>`         = dnnl_Bcdea16b,
	    :target:`dnnl_IdhwO16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac534498a21e8d94c0b0de3e2d317b26b>`       = dnnl_BcdeA16b2a,
	    :target:`dnnl_IdhwO16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7afa937909cc9e48b8320d0837d5ced3>`       = dnnl_BcdeA16b4a,
	    :target:`dnnl_Odhwi4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da261f4d021b8b27b0401929ab0e96bfe8>`          = dnnl_Acdeb4a,
	    :target:`dnnl_Odhwi8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daadc6ba7f3e8ac5440150080f33a48ba0>`          = dnnl_Acdeb8a,
	    :target:`dnnl_OdhwI8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7ae87ec041c6ca91391f8eca1772c17e>`        = dnnl_AcdeB8a2b,
	    :target:`dnnl_Odwhi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab17d0442cbb77dfb8d78517e3aa34d37>`         = dnnl_Acedb16a,
	    :target:`dnnl_OIdhw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1da50b8a6e1d61d2e43605a2ae61e610>`      = dnnl_ABcde16b16a,
	    :target:`dnnl_OIdhw16i32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa9370fbcf42404a5906dcd4fee1c426a>`      = dnnl_ABcde16b32a,
	    :target:`dnnl_OIdhw16i64o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da97a83d75c9013fe9e2b4c60b2a8cffcf>`      = dnnl_ABcde16b64a,
	    :target:`dnnl_OIdhw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6f5cfe739fb1cefeea9a72d0c931837d>`      = dnnl_ABcde16a16b,
	    :target:`dnnl_Oidhw16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4099135862f4a8ecfccf44ba1a33e9ad>`         = dnnl_Abcde16a,
	    :target:`dnnl_OIdhw4i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae5de15c2eabdbc703d03d9b90377be3e>`        = dnnl_ABcde4b4a,
	    :target:`dnnl_OIdhw4o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0e193cb7ff2019ae8edd396d271cd694>`        = dnnl_ABcde4a4b,
	    :target:`dnnl_Oidhw4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8035cedead05e7048ce718d04009d03a>`          = dnnl_Abcde4a,
	    :target:`dnnl_OIdhw8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da20e4a8f77003bac5603b031ff065051b>`      = dnnl_ABcde8b8a2b,
	    :target:`dnnl_OIdhw8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa9d131f0c062e1158e2ecd5bec452030>`     = dnnl_ABcde8b16a2b,
	    :target:`dnnl_OIdhw8i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da274bd12ddb658234598384f426adcdeb>`     = dnnl_ABcde8b32a2b,
	    :target:`dnnl_OIdhw8i24o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae72ded6d0617e36d52baa0afee2ad885>`     = dnnl_ABcde8b24a2b,
	    :target:`dnnl_OIdhw8i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac9d8fca34195b16b7dcbc856bb41ad08>`     = dnnl_ABcde8b64a2b,
	    :target:`dnnl_OIdhw8i8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa4896403ee259dd66afe7fb73bc7b005>`        = dnnl_ABcde8b8a,
	    :target:`dnnl_OIdhw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da48892a878623d036d7696831831fa723>`     = dnnl_ABcde8a16b2a,
	    :target:`dnnl_IOdhw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da265621138abc7031d10323b8ab16d03a>`     = dnnl_BAcde8a16b2a,
	    :target:`dnnl_OIdhw4i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da70ceeac28938df6eb4cc960cd9863e32>`      = dnnl_ABcde4b8a4b,
	    :target:`dnnl_OIdhw4i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da31c9710dad940245fe60124ce1f83ce5>`     = dnnl_ABcde4b16a4b,
	    :target:`dnnl_OIdhw4i24o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad167998e43ac8bbacb49cb6cb9985805>`     = dnnl_ABcde4b24a4b,
	    :target:`dnnl_OIdhw4i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf5083b7f1c1fd5281b041a21e810fe6a>`     = dnnl_ABcde4b32a4b,
	    :target:`dnnl_OIdhw4i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da91b3ed9a3884831dc521fb4c28957c7f>`     = dnnl_ABcde4b64a4b,
	    :target:`dnnl_OIdhw16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf6d029a6d6c527fad85bbe8dbee2a2c9>`    = dnnl_ABcde16b16a4b,
	    :target:`dnnl_OIdhw16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da415b663bc6b2e7ce4569815492bae77f>`    = dnnl_ABcde16b16a2b,
	    :target:`dnnl_OIdhw2i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9d9717d5c0f9429a78380687db7bb5fd>`      = dnnl_ABcde2b8a4b,
	    :target:`dnnl_OIdhw8o8i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da685b7698f92fee657521d4900a314745>`        = dnnl_ABcde8a8b,
	    :target:`dnnl_OIdhw8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da643c12e5a419c28a2f10abfabf93f7df>`        = dnnl_ABcde8a4b,
	    :target:`dnnl_IOdhw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf69570588ad0d5d22ba33bbdb9f487c7>`      = dnnl_BAcde16b16a,
	    :target:`dnnl_OIdhw4o8i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5b68d14043aa631f9e2c0a847a47e613>`    = dnnl_ABcde4a8b8a4b,
	    :target:`dnnl_IOdhw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae66fcb6e7e60c87b5c41ffe04882549b>`      = dnnl_BAcde16a16b,
	    :target:`dnnl_OIdhw16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7186c442420b7133c7df69a8f9c38c37>`    = dnnl_ABcde16a16b2a,
	    :target:`dnnl_OIdhw8i32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab7f7dcdd8dcd8549105c432f5b7a1b76>`       = dnnl_ABcde8b32a,
	    :target:`dnnl_OIdhw8i24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad6970960fd74582978278176db294a09>`       = dnnl_ABcde8b24a,
	    :target:`dnnl_OIdhw8i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4c6782742a902d0b76faa3a7f0a91cae>`       = dnnl_ABcde8b16a,
	    :target:`dnnl_Goiw16g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daecafe7d7f9121924be14d0b3c8207160>`          = dnnl_Abcd16a,
	    :target:`dnnl_Goiw8g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab9a1500c0a3146f6999140ad811c18f2>`           = dnnl_Abcd8a,
	    :target:`dnnl_Goiw4g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da545b4525c26cf8263d1761591585c119>`           = dnnl_Abcd4a,
	    :target:`dnnl_gIOw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da16c730ca0413a66f2d8b0fa56a8e2b1b>`       = dnnl_aCBd16b16c,
	    :target:`dnnl_gIOw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da89b7993ddf9866d6275990b43ca714e0>`       = dnnl_aCBd16c16b,
	    :target:`dnnl_gOIw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da278e39205d04f12cb2a627e32d24b799>`       = dnnl_aBCd16c16b,
	    :target:`dnnl_gOIw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadd1e10a13d39a3fb7eba45733b361167>`       = dnnl_aBCd16b16c,
	    :target:`dnnl_gOiw16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacf4081975ba58dd49482c57457f41676>`          = dnnl_aBcd16b,
	    :target:`dnnl_gOIw4i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac670126c44cea5d7c7cf0425c42f3229>`      = dnnl_aBCd4c16b4c,
	    :target:`dnnl_gOIw2i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9cc32ddec948b8117eeed3f8fd16afa3>`       = dnnl_aBCd2c8b4c,
	    :target:`dnnl_gOIw16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da03c1f9ff1b10cf3d13d0dbe1136dd497>`     = dnnl_aBCd16c16b4c,
	    :target:`dnnl_gOIw16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da56ad0cdaf350e5eb4a34d3f5cd3c1577>`     = dnnl_aBCd16c16b2c,
	    :target:`dnnl_gOIw16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacf3890c2df03f05a8e2111bd24879ad6>`     = dnnl_aBCd16b16c2b,
	    :target:`dnnl_gOIw4i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1c9fe22d796db0f2637f649f10d3519c>`         = dnnl_aBCd4c4b,
	    :target:`dnnl_gOIw4o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab4c266fdd09396a5a388ec33f17ef2c5>`         = dnnl_aBCd4b4c,
	    :target:`dnnl_gOiw4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab225574b0fdfcc8c98b810052e95f14e>`           = dnnl_aBcd4b,
	    :target:`dnnl_gOIw8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0638bc6415780c95f6743abc380def6d>`      = dnnl_aBCd8c16b2c,
	    :target:`dnnl_gOIw8i8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1c044c2a460349d3aff6ce8c49dcadfe>`         = dnnl_aBCd8c8b,
	    :target:`dnnl_gOIw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5a482fd0f6e4e80e424a60935c98d26e>`      = dnnl_aBCd8b16c2b,
	    :target:`dnnl_gIOw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da64f455ed3177d7512f301ca58f781faa>`      = dnnl_aCBd8b16c2b,
	    :target:`dnnl_gOIw8o8i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa6530ed0e9636623d0d963c74def8c80>`         = dnnl_aBCd8b8c,
	    :target:`dnnl_gOIw8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad7a8f25890193bdb06012c79ec7a99ea>`         = dnnl_aBCd8b4c,
	    :target:`dnnl_gOwi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabe0a7effb533bb77f63d2eb1574e7500>`          = dnnl_aBdc16b,
	    :target:`dnnl_gOwI16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da40a8a43451d43f7a577fefce4e10ff0f>`        = dnnl_aBdC16b2c,
	    :target:`dnnl_gOwI16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac4a86673a8cf4708d415e80e7e12fff4>`        = dnnl_aBdC16b4c,
	    :target:`dnnl_gIwo16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da212083cbad57ef9baf9ee54f1de30441>`          = dnnl_aCdb16c,
	    :target:`dnnl_gIwO16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4966e0899864809ee8c2966fc907dd7b>`        = dnnl_aCdB16c2b,
	    :target:`dnnl_gIwO16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da12bb4b5904a47e01129bf5a3ecf1956d>`        = dnnl_aCdB16c4b,
	    :target:`dnnl_gOwi4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da632a70bffa0c9536f66df3f66b3bcf33>`           = dnnl_aBdc4b,
	    :target:`dnnl_gOwi8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac931b080392e2e243b165c2c132b17df>`           = dnnl_aBdc8b,
	    :target:`dnnl_gOwI8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac6f3ea52d97141a517603061facdb3e6>`         = dnnl_aBdC8b2c,
	    :target:`dnnl_Goiw32g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadc9cab77e04f5799dd64ba1720a0c42f>`          = dnnl_Abcd32a,
	    :target:`dnnl_gOIw2i4o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab234dd41d9dd4201adf26c13d004570>`       = dnnl_aBCd2c4b2c,
	    :target:`dnnl_gOIw2o4i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da217d8af2928dc7e10fd1699281769a55>`       = dnnl_aBCd2b4c2b,
	    :target:`dnnl_gOIw4i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da977d21e701781261fae252015f819d07>`       = dnnl_aBCd4c8b2c,
	    :target:`dnnl_gOIw4o8i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da32f6b70cb3de1b0ba68a5e66fd5a0ca9>`       = dnnl_aBCd4b8c2b,
	    :target:`dnnl_goIw4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1780497d5a9868d7fcbf3583c020efa1>`           = dnnl_abCd4c,
	    :target:`dnnl_goIw32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3d4ff441bce71885518aecb298061dd4>`          = dnnl_abCd32c,
	    :target:`dnnl_gIOhw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5747cf4222fdd208ccfbeb6159c85ac9>`      = dnnl_aCBde16c16b,
	    :target:`dnnl_gIOhw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da26635131082765c96ef904501778acf5>`      = dnnl_aCBde16b16c,
	    :target:`dnnl_gOhwi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4ea931d8b37b51814582561d7b88713c>`         = dnnl_aBdec16b,
	    :target:`dnnl_gOhwI16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da22e32dea39cc9918ff5e8be91f6d7a15>`       = dnnl_aBdeC16b2c,
	    :target:`dnnl_gOhwI16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da76164078730cd69851e77b7a7480e1cd>`       = dnnl_aBdeC16b4c,
	    :target:`dnnl_gIhwo16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da76dc46ebde8eb59b61aa38b3e71ad5aa>`         = dnnl_aCdeb16c,
	    :target:`dnnl_gIhwO16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa6df0db9cdb3ad43691f7243086c026c>`       = dnnl_aCdeB16c2b,
	    :target:`dnnl_gIhwO16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da196ad57a50e4b48890606a6349f3b956>`       = dnnl_aCdeB16c4b,
	    :target:`dnnl_gOhwi32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da459a7020793cc4d23d5dd5b3b14ba255>`         = dnnl_aBdec32b,
	    :target:`dnnl_gOhwi24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa9738f55194289732489f7775899a26b>`         = dnnl_aBdec24b,
	    :target:`dnnl_gOhwI24o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a21effa06f86813272e0f67c9954370>`       = dnnl_aBdeC24b2c,
	    :target:`dnnl_gOhwi4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0bb61767ca7b1b96a93c9256aa29225d>`          = dnnl_aBdec4b,
	    :target:`dnnl_gOhwi8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3a9c0a66b7c774abbb3e1871d3e75719>`          = dnnl_aBdec8b,
	    :target:`dnnl_gOhwI8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa3eb333d4f1fef06b798e26e880d7d7c>`        = dnnl_aBdeC8b2c,
	    :target:`dnnl_Goihw16g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab6c951dded8af9361c0889c44acd0b3c>`         = dnnl_Abcde16a,
	    :target:`dnnl_gOIhw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab97b18f67b0feb175fcbdbe06b6710ff>`      = dnnl_aBCde16c16b,
	    :target:`dnnl_gOIhw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5c8fbc96e3d77a4f6162d9ebc96a167a>`      = dnnl_aBCde16b16c,
	    :target:`dnnl_gOihw16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaa9faf9a58ef7f07602df749d222c53e>`         = dnnl_aBcde16b,
	    :target:`dnnl_gOIhw2i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1fc7a4eb1b4787c65687fd1ad3e6c541>`      = dnnl_aBCde2c8b4c,
	    :target:`dnnl_gOIhw4i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf984f1a1a85663bf633d82fda1344855>`     = dnnl_aBCde4c16b4c,
	    :target:`dnnl_gOIhw16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daec88a861bcb48bc8397d5b5b2553cfbf>`    = dnnl_aBCde16c16b4c,
	    :target:`dnnl_gOIhw16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa2400634af11075a10ce0ed432ada402>`    = dnnl_aBCde16c16b2c,
	    :target:`dnnl_gOIhw16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad027df0a8cadfe6bf96fd045e97f2acf>`    = dnnl_aBCde16b16c2b,
	    :target:`dnnl_gOIhw4i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8d28af3ee5089357619970e651286a40>`        = dnnl_aBCde4c4b,
	    :target:`dnnl_gOIhw4o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da908f00b7f2da7a34fcf1d7f7ab2d921d>`        = dnnl_aBCde4b4c,
	    :target:`dnnl_gOihw4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7dbf2585dc4a904e816f6129413840bb>`          = dnnl_aBcde4b,
	    :target:`dnnl_Goihw8g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da84ef626b73f965835064d5c7032988e3>`          = dnnl_Abcde8a,
	    :target:`dnnl_Goihw4g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0e04c060a560d50c6c5f0d5cefac700b>`          = dnnl_Abcde4a,
	    :target:`dnnl_gOIhw8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da35698b03513a04eeb013e8986bce3e20>`     = dnnl_aBCde8c16b2c,
	    :target:`dnnl_gOIhw8i8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3cb067d573b830e463edb7b571291270>`        = dnnl_aBCde8c8b,
	    :target:`dnnl_gOIhw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3c0e8331efdf988393db7d38dc15f3f7>`     = dnnl_aBCde8b16c2b,
	    :target:`dnnl_gIOhw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da32d155351a632081e8a137e1b0d53ec7>`     = dnnl_aCBde8b16c2b,
	    :target:`dnnl_gOIhw8o8i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa74cf89a288d4010966c4a341bf3faeb>`        = dnnl_aBCde8b8c,
	    :target:`dnnl_gOIhw8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2a8c7cdad23e5c4dcdd4c72b0e2f8c19>`        = dnnl_aBCde8b4c,
	    :target:`dnnl_Goihw32g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da843d7421adb0792de575ad7150b3c02d>`         = dnnl_Abcde32a,
	    :target:`dnnl_gOwhi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3eec91ff9fcabf5d3f10bc5d6717aa3c>`         = dnnl_aBedc16b,
	    :target:`dnnl_goIhw4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da411895d4c26d52d32450b7abf8b6a9a9>`          = dnnl_abCde4c,
	    :target:`dnnl_goIhw32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae2d6b384acc9311f7c2c562df25da190>`         = dnnl_abCde32c,
	    :target:`dnnl_OIw4o8i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da935f85c435ffb862e3aabb05bd7982bd>`      = dnnl_ABc4a8b8a4b,
	    :target:`dnnl_OIhw4o8i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadc278583043f76a3767bf28a3f9e5ec0>`     = dnnl_ABcd4a8b8a4b,
	    :target:`dnnl_IOw4i8o8i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac2ed61e68ed0b406a93cafb01b6e45e0>`      = dnnl_BAc4b8a8b4a,
	    :target:`dnnl_IOhw4i8o8i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daddfb0427041ca3ea227f868281db02f8>`     = dnnl_BAcd4b8a8b4a,
	    :target:`dnnl_IOdhw4i8o8i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da812548bd5cae45ec043c84ca5a0ff4cf>`    = dnnl_BAcde4b8a8b4a,
	    :target:`dnnl_OIhw2o8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad51153998d59fd8696c9e70fc4000959>`     = dnnl_ABcd2a8b8a2b,
	    :target:`dnnl_gOIw4o8i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da36650e484fb735030be73cb692ffc9e2>`     = dnnl_aBCd4b8c8b4c,
	    :target:`dnnl_gOIhw4o8i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0001ca847ed83e3049d2450b4d9c2572>`    = dnnl_aBCde4b8c8b4c,
	    :target:`dnnl_gOIdhw4o8i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8d317b334963e6807e13e9bbe33790ba>`   = dnnl_aBCdef4b8c8b4c,
	    :target:`dnnl_gIOw4i8o8i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da926cb89aa2693a25fc1f51481874b3d2>`     = dnnl_aCBd4c8b8c4b,
	    :target:`dnnl_gIOhw4i8o8i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da759402bc4a3c9406c0cadc597a9ed4e3>`    = dnnl_aCBde4c8b8c4b,
	    :target:`dnnl_gIOdhw4i8o8i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da414911abc93084b812753ed6beb56582>`   = dnnl_aCBdef4c8b8c4b,
	    :target:`dnnl_gOIhw2o8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae94d09dd8450bf531c23945762515a52>`    = dnnl_aBCde2b8c8b2c,
	    :target:`dnnl_gOIhw2i4o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da65198f43293eaa5ca248cffff7a6d032>`      = dnnl_aBCde2c4b2c,
	    :target:`dnnl_gOIhw2o4i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da96855d419a05fc2aec51a3983bf68862>`      = dnnl_aBCde2b4c2b,
	    :target:`dnnl_gOIhw4i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da47f0b0abf6e0d9d794bc46cb422738e6>`      = dnnl_aBCde4c8b2c,
	    :target:`dnnl_gOIhw4o8i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1a89c1e9f7c07b56dda42b5a0b84bfbb>`      = dnnl_aBCde4b8c2b,
	    :target:`dnnl_gIOdhw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7ff18a13c1a94f70d70f5d38fabcd576>`     = dnnl_aCBdef16c16b,
	    :target:`dnnl_gIOdhw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0bea6075cb318ea108b2789fe606ccde>`     = dnnl_aCBdef16b16c,
	    :target:`dnnl_gOdhwi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0e96fb2287dc1ce149ea66b0360e203c>`        = dnnl_aBdefc16b,
	    :target:`dnnl_gOdhwI16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da85a76a8da045aede2bb7523b1c19c2af>`      = dnnl_aBdefC16b2c,
	    :target:`dnnl_gOdhwI16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae72edc5a6aa0bca8e6dc119bd560a93e>`      = dnnl_aBdefC16b4c,
	    :target:`dnnl_gIdhwo16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da14528ac92d6a7726313b4e1898beb4b6>`        = dnnl_aCdefb16c,
	    :target:`dnnl_gIdhwO16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3eb2c678d1fafe1d2510c9b03d2c7c54>`      = dnnl_aCdefB16c2b,
	    :target:`dnnl_gIdhwO16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4dc396686a8a9e34b64f26ff290fcbb0>`      = dnnl_aCdefB16c4b,
	    :target:`dnnl_gOdhwi4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac09596c7a19f4e84b6bf01047272cb56>`         = dnnl_aBdefc4b,
	    :target:`dnnl_gOdhwi8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da036eb2f586dd5be7ae631ed01dfa9b45>`         = dnnl_aBdefc8b,
	    :target:`dnnl_gOdhwI8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3c4bdbde76dfb668394079655fe16a60>`       = dnnl_aBdefC8b2c,
	    :target:`dnnl_gOdwhi16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf586fda9c9ac32cc354d224cdce7e7ed>`        = dnnl_aBdfec16b,
	    :target:`dnnl_gOIdhw16i16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae39149ba5089aaac48e910c652528b24>`     = dnnl_aBCdef16c16b,
	    :target:`dnnl_gOIdhw4i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4b292c23aa1ccea7fea8b4b500ea196e>`    = dnnl_aBCdef4c16b4c,
	    :target:`dnnl_gOIdhw16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf713505134ec2fe0e3510b67094268c0>`   = dnnl_aBCdef16c16b4c,
	    :target:`dnnl_gOIdhw2i8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da84c8b247473eebeaa8e1b0ce06e6dcb7>`     = dnnl_aBCdef2c8b4c,
	    :target:`dnnl_gOIdhw16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac88bcecc1aa64ae88f760f9ebd3b64ec>`   = dnnl_aBCdef16c16b2c,
	    :target:`dnnl_gOIdhw16o16i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bc59dcd9b5d04ddd5c2fab2732a7d64>`     = dnnl_aBCdef16b16c,
	    :target:`dnnl_gOIdhw16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7f6797416b950720623d108d759cf938>`   = dnnl_aBCdef16b16c2b,
	    :target:`dnnl_gOidhw16o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da85e7f54814f76b198853e8a40e71323f>`        = dnnl_aBcdef16b,
	    :target:`dnnl_gOIdhw4i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da591887ae0872bac1a6c357f1ca2294bf>`       = dnnl_aBCdef4c4b,
	    :target:`dnnl_gOIdhw4o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadb7e21a4352d01efb768c6ad266238bf>`       = dnnl_aBCdef4b4c,
	    :target:`dnnl_gOidhw4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7551cbe5d7cf96b627accfcaf3af8126>`         = dnnl_aBcdef4b,
	    :target:`dnnl_gOIdhw8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da04d266ad05f4974831dcd2c8b3ed78d2>`    = dnnl_aBCdef8c16b2c,
	    :target:`dnnl_gOIdhw8i8o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad2597d2f09c11d2cbd44ee6849f4f7df>`       = dnnl_aBCdef8c8b,
	    :target:`dnnl_gOIdhw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9789b25ae029f7420c0a1a2e3720cc79>`    = dnnl_aBCdef8b16c2b,
	    :target:`dnnl_gIOdhw8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1ebf48f2d16752b1a1806ae7b69829d7>`    = dnnl_aCBdef8b16c2b,
	    :target:`dnnl_gOIdhw8o8i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da31895e424901dc014b5b07bf8f11db16>`       = dnnl_aBCdef8b8c,
	    :target:`dnnl_gOIdhw8o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da783bcc80e7c9f8bcb60d78bdc39a18fa>`       = dnnl_aBCdef8b4c,
	    :target:`dnnl_Goidhw16g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab2432b83c5df88b8e5d6b7db121cd2eb>`        = dnnl_Abcdef16a,
	    :target:`dnnl_Goidhw32g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da02fccbcb8f60c117c51d8395bfae94f8>`        = dnnl_Abcdef32a,
	    :target:`dnnl_gOIdhw2i4o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8bdd35d54710022f355c8ec1fcc0d1ff>`     = dnnl_aBCdef2c4b2c,
	    :target:`dnnl_gOIdhw4i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacdf609e66f7e24e64904476d753574a3>`     = dnnl_aBCdef4c8b2c,
	    :target:`dnnl_gOIdhw2o4i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadf1e1a0f916ae2be72e4612943ba7f3f>`     = dnnl_aBCdef2b4c2b,
	    :target:`dnnl_gOIdhw4o8i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf85d199dcfd5dfc339db8332eabc4dcd>`     = dnnl_aBCdef4b8c2b,
	    :target:`dnnl_goIdhw4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac0059581fae312b1a88223cbad59b56c>`         = dnnl_abCdef4c,
	    :target:`dnnl_goIdhw32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4630c35b45b18b831a70d5760094cbd5>`        = dnnl_abCdef32c,
	    :target:`dnnl_Owi24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2255f0099efa9d7fd0cfdfa6ac764362>`           = dnnl_Acb24a,
	    :target:`dnnl_OwI24o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da524c873ad31ca454ae78eb1a6aa1ad69>`         = dnnl_AcB24a2b,
	    :target:`dnnl_Owi32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7af701159b7ce7bdf9c01176c1837b13>`           = dnnl_Acb32a,
	    :target:`dnnl_OwI32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad269b81372724e315532dbd2ac295286>`         = dnnl_AcB32a2b,
	    :target:`dnnl_OwI32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da33b25ec1c076544ebb6c34f546bed93e>`         = dnnl_AcB32a4b,
	    :target:`dnnl_Owi48o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3816336ef4834c7b94e4b436546e63a0>`           = dnnl_Acb48a,
	    :target:`dnnl_OwI48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da44fd5129e0a36768acc8b5151b7a0578>`         = dnnl_AcB48a2b,
	    :target:`dnnl_OwI48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf25ef15a9615a6e2158133e2557961df>`         = dnnl_AcB48a4b,
	    :target:`dnnl_Owi64o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1548f559ac5b2d0d71dad965bd96d9be>`           = dnnl_Acb64a,
	    :target:`dnnl_OwI64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8cdefe74cde6142acd6300078f176cc2>`         = dnnl_AcB64a2b,
	    :target:`dnnl_OwI64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3a16c4d2b568f86bc4e394539449f26e>`         = dnnl_AcB64a4b,
	    :target:`dnnl_Iwo32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3f49a7b884952f2424d6e8e9697f97eb>`           = dnnl_Bca32b,
	    :target:`dnnl_IwO32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dada2d169ab674f5622457c3e0970bcd7e>`         = dnnl_BcA32b2a,
	    :target:`dnnl_IwO32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa8355ab2c62444f52c9ac1d2a8951d06>`         = dnnl_BcA32b4a,
	    :target:`dnnl_Iwo48i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da88af4064b4a650ac5fae10e118af8267>`           = dnnl_Bca48b,
	    :target:`dnnl_IwO48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da58b6bffc2b5172900287a075634ed31c>`         = dnnl_BcA48b2a,
	    :target:`dnnl_IwO48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da626a62f73d2794051f6f3ccdfd2d72a8>`         = dnnl_BcA48b4a,
	    :target:`dnnl_Iwo64i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9fa99b092949ea35232a4d4c06e566ec>`           = dnnl_Bca64b,
	    :target:`dnnl_IwO64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2b743cfb600b79f7ee79c68071999663>`         = dnnl_BcA64b2a,
	    :target:`dnnl_IwO64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacbd7bf538b5ba3dbf06614f5795c402e>`         = dnnl_BcA64b4a,
	    :target:`dnnl_wIo2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daeb86e353ca78ce20d5e194706d3b9cc2>`            = dnnl_cBa2b,
	    :target:`dnnl_wIo4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daacc1f1177fd30cfd38e3b42af1fdee6b>`            = dnnl_cBa4b,
	    :target:`dnnl_gOwi24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daef0257e876d9496e8fe49f0900816677>`          = dnnl_aBdc24b,
	    :target:`dnnl_gOwI24o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0d301b66edc8c630a3999f085a832957>`        = dnnl_aBdC24b2c,
	    :target:`dnnl_gOwi32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da360f70f64f5ca7c8114f5862d2b1eab3>`          = dnnl_aBdc32b,
	    :target:`dnnl_gOwI32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa7962f734b7b5670d2cf8c27839f559f>`        = dnnl_aBdC32b2c,
	    :target:`dnnl_gOwI32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0be31a3d2abcc271d94b41a78fb9084b>`        = dnnl_aBdC32b4c,
	    :target:`dnnl_gOwi48o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad73e793681274b59a1514341e8f39b1e>`          = dnnl_aBdc48b,
	    :target:`dnnl_gOwI48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacde02351c1b396278430db364935c93e>`        = dnnl_aBdC48b2c,
	    :target:`dnnl_gOwI48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf7e759ed01a4974dbf125d9e92aa9da7>`        = dnnl_aBdC48b4c,
	    :target:`dnnl_gOwi64o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da17da5a079028bbafae56ca8c6f287aee>`          = dnnl_aBdc64b,
	    :target:`dnnl_gOwI64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da904c7663670b062929c21437a7bb30ba>`        = dnnl_aBdC64b2c,
	    :target:`dnnl_gOwI64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0d42658fe5688656d7b89d1bbb86da20>`        = dnnl_aBdC64b4c,
	    :target:`dnnl_gIwo32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da06e9722030d7fe591c5a5abeafdf25b9>`          = dnnl_aCdb32c,
	    :target:`dnnl_gIwO32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab386d0eb85d8ff3ad921fcb12a8e4ca6>`        = dnnl_aCdB32c2b,
	    :target:`dnnl_gIwO32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da536e4f38df115985a63f1d484250021b>`        = dnnl_aCdB32c4b,
	    :target:`dnnl_gIwo48i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da08e845dcefee405dec2bf257e90fc242>`          = dnnl_aCdb48c,
	    :target:`dnnl_gIwO48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da839dae8b8d79007115547c656a41cf50>`        = dnnl_aCdB48c2b,
	    :target:`dnnl_gIwO48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da54138cb2d22a4a4a54407695224644b7>`        = dnnl_aCdB48c4b,
	    :target:`dnnl_gIwo64i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae6dbd70fcfd5c6f8f062ee0e3e8d7d75>`          = dnnl_aCdb64c,
	    :target:`dnnl_gIwO64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae1f8b1e28a8f6ecf35d6fdd34f19268b>`        = dnnl_aCdB64c2b,
	    :target:`dnnl_gIwO64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5392f39a6ca63397a23dd269d6d17518>`        = dnnl_aCdB64c4b,
	    :target:`dnnl_gwio<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6edcd9597a325899813a1fb33e438ac0>`             = dnnl_adcb,
	    :target:`dnnl_gwIo2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da795878a1eabe3e85d4a82a56b2409d85>`           = dnnl_adCb2c,
	    :target:`dnnl_gwIo4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0323884153adcfa8ad62c9994561e49a>`           = dnnl_adCb4c,
	    :target:`dnnl_OhwI24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daeccaff890f9ce99b8cbb68b5be1f2b52>`          = dnnl_Acdb24a,
	    :target:`dnnl_OhwI24o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8479f4f94346a13f1333b70932a7338b>`        = dnnl_AcdB24a2b,
	    :target:`dnnl_OhwI32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacb9dfe768517ca9955f2a68f089efcb3>`          = dnnl_Acdb32a,
	    :target:`dnnl_OhwI32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab4cdc23e83cd2e8ff40257067359339b>`        = dnnl_AcdB32a2b,
	    :target:`dnnl_OhwI32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8bbca98f6100216c9c198cc4cbead0f6>`        = dnnl_AcdB32a4b,
	    :target:`dnnl_Ohwi48o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da37b03848aa1a4df8cec6ad873b72c806>`          = dnnl_Acdb48a,
	    :target:`dnnl_OhwI48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0e0c725aef60342d6b429e658d285e85>`        = dnnl_AcdB48a2b,
	    :target:`dnnl_OhwI48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5eec9b99bc6b452f87ba37a3aff93baf>`        = dnnl_AcdB48a4b,
	    :target:`dnnl_Ohwi64o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8788a48994f51872fa87e036af5cd8f2>`          = dnnl_Acdb64a,
	    :target:`dnnl_OhwI64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da358761293681eb2954fbf580c4846399>`        = dnnl_AcdB64a2b,
	    :target:`dnnl_OhwI64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1cb7aaac08ea9ec6562460f89c1f2824>`        = dnnl_AcdB64a4b,
	    :target:`dnnl_Ihwo32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da61e979774202bfde7fa65d23ce845eb6>`          = dnnl_Bcda32b,
	    :target:`dnnl_IhwO32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da26c3dd032676a2a6b9acafa8c4777f2a>`        = dnnl_BcdA32b2a,
	    :target:`dnnl_IhwO32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf0b0e6f1c1304278fd1499e7595142d7>`        = dnnl_BcdA32b4a,
	    :target:`dnnl_Ihwo48i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4c5031e549d707523b6eac5b314e31c9>`          = dnnl_Bcda48b,
	    :target:`dnnl_IhwO48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da326dad97c386080a22d29eeb0b2aae86>`        = dnnl_BcdA48b2a,
	    :target:`dnnl_IhwO48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae5c967472c3e1c6d61a69e926721ce10>`        = dnnl_BcdA48b4a,
	    :target:`dnnl_Ihwo64i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da67f057870b3dd1f8f225279037b5dd42>`          = dnnl_Bcda64b,
	    :target:`dnnl_IhwO64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da44af869eabb7aa998fcfd288222df6b0>`        = dnnl_BcdA64b2a,
	    :target:`dnnl_IhwO64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0f06c0fff745d8b3f1472e298ab3ba11>`        = dnnl_BcdA64b4a,
	    :target:`dnnl_hwIo2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6d369f077e125b22c3217c422facbbdc>`           = dnnl_cdBa2b,
	    :target:`dnnl_hwIo4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da134bb2a2cd0ffb24cfcca9559256beda>`           = dnnl_cdBa4b,
	    :target:`dnnl_gOhwI24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab475466ef187053f016f218c1f3b48d9>`         = dnnl_aBdec24b,
	    :target:`dnnl_gOhwI32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da50b5c16e391d265796f5f8456bdeb829>`         = dnnl_aBdec32b,
	    :target:`dnnl_gOhwI32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da92c8ff580cb71ba896d021a02f60c335>`       = dnnl_aBdeC32b2c,
	    :target:`dnnl_gOhwI32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da35ef5cf8f4b59e8c190a9caefd34db01>`       = dnnl_aBdeC32b4c,
	    :target:`dnnl_gOhwi48o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa5dd27966059b3c47dd6f05689a0ca31>`         = dnnl_aBdec48b,
	    :target:`dnnl_gOhwI48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacc64099b8a54902fccbdf8b4f976f4f8>`       = dnnl_aBdeC48b2c,
	    :target:`dnnl_gOhwI48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2209f3466b3d0076a01c0b31ac1ce5b8>`       = dnnl_aBdeC48b4c,
	    :target:`dnnl_gOhwi64o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae47423d7843f013218a458fa01c2f72d>`         = dnnl_aBdec64b,
	    :target:`dnnl_gOhwI64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8b3358ea6d5e04455f87a584daf22cea>`       = dnnl_aBdeC64b2c,
	    :target:`dnnl_gOhwI64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa097bc366a7e03556d821676d4e2f62f>`       = dnnl_aBdeC64b4c,
	    :target:`dnnl_gIhwo32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1160813fce3e9e4f84e73e0d37dd82e2>`         = dnnl_aCdeb32c,
	    :target:`dnnl_gIhwO32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da576f684622787e3b513bfd9065345c32>`       = dnnl_aCdeB32c2b,
	    :target:`dnnl_gIhwO32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1023031a3fb95d5c5a037cf9ef351e6d>`       = dnnl_aCdeB32c4b,
	    :target:`dnnl_gIhwo48i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad6d0632deb0d816b7059d115fb28ec96>`         = dnnl_aCdeb48c,
	    :target:`dnnl_gIhwO48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf01519c5856843e9e9b751731f2ab623>`       = dnnl_aCdeB48c2b,
	    :target:`dnnl_gIhwO48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da716c6dd4e4d52ec4ddaac100119c78e6>`       = dnnl_aCdeB48c4b,
	    :target:`dnnl_gIhwo64i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac6519be7d059489d34adbd61a5607224>`         = dnnl_aCdeb64c,
	    :target:`dnnl_gIhwO64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac5accd9d09197cf6570d63e868e378fb>`       = dnnl_aCdeB64c2b,
	    :target:`dnnl_gIhwO64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da69b7a60105ff9f4774cc8eb279fcf713>`       = dnnl_aCdeB64c4b,
	    :target:`dnnl_ghwio<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2e692bb5c871d3717c7e25630e33c607>`            = dnnl_adecb,
	    :target:`dnnl_ghwIo2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daacfceaf2b4230b92ab61f4b3472ce3eb>`          = dnnl_adeCb2c,
	    :target:`dnnl_ghwIo4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da85047ee28d499d3bb3c001d78f755a26>`          = dnnl_adeCb4c,
	    :target:`dnnl_Odhwi24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7425afd30c749375dea1b808b57c3e4a>`         = dnnl_Acdeb24a,
	    :target:`dnnl_OdhwI24o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabd461b8a4680e4521ab8641229667a39>`       = dnnl_AcdeB24a2b,
	    :target:`dnnl_Odhwi32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab58290f0428318896f5536dff2f623fa>`         = dnnl_Acdeb32a,
	    :target:`dnnl_OdhwI32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadd233459e0f7c30de7fcda45a230356d>`       = dnnl_AcdeB32a2b,
	    :target:`dnnl_OdhwI32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1619212689caade2c03615329c481153>`       = dnnl_AcdeB32a4b,
	    :target:`dnnl_Odhwi48o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab3c242e00b8667271ab97e3debc0a1e6>`         = dnnl_Acdeb48a,
	    :target:`dnnl_OdhwI48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad114d829267c462933c6a45feec6d0fb>`       = dnnl_AcdeB48a2b,
	    :target:`dnnl_OdhwI48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da880710c3c5ba84b5133cb06d221c71a3>`       = dnnl_AcdeB48a4b,
	    :target:`dnnl_Odhwi64o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab762c3979ea6dbe0c4d52c8e9caa1c6>`         = dnnl_Acdeb64a,
	    :target:`dnnl_OdhwI64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6b6ff475f6d7eb5546ae6b693e7a94fa>`       = dnnl_AcdeB64a2b,
	    :target:`dnnl_OdhwI64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da38f927746e529d3af3f179431f8d4cf6>`       = dnnl_AcdeB64a4b,
	    :target:`dnnl_Idhwo32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabc7766fdcfc44944fd07a7c05d4c4433>`         = dnnl_Bcdea32b,
	    :target:`dnnl_IdhwO32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da14f74bdd8ac07a7f3be37c2c59a42bec>`       = dnnl_BcdeA32b2a,
	    :target:`dnnl_IdhwO32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab26a881f4e0599e2fbec8007535810a6>`       = dnnl_BcdeA32b4a,
	    :target:`dnnl_Idhwo48i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da692408fa292b308405eb1664f1aa2897>`         = dnnl_Bcdea48b,
	    :target:`dnnl_IdhwO48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa1d85c8477f14cb80df37a7a2d90d2e8>`       = dnnl_BcdeA48b2a,
	    :target:`dnnl_IdhwO48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da961f7a2f3ca4d0737f156eb10a9610d3>`       = dnnl_BcdeA48b4a,
	    :target:`dnnl_Idhwo64i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3f70db622fd849c097f26fc369c85d59>`         = dnnl_Bcdea64b,
	    :target:`dnnl_IdhwO64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da246a31bf74fb8e0e8f74a704430b9d0c>`       = dnnl_BcdeA64b2a,
	    :target:`dnnl_IdhwO64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da474b2d4ae997a5c624694b4a67024035>`       = dnnl_BcdeA64b4a,
	    :target:`dnnl_dhwIo2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da48d77b64e407b92a019f6598b7887da0>`          = dnnl_cdeBa2b,
	    :target:`dnnl_dhwIo4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dada1f79334385dcff24d1c04e0edd7e56>`          = dnnl_cdeBa4b,
	    :target:`dnnl_gOdhwi24o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6fa60f74d0708f0370e892ea4239ae46>`        = dnnl_aBdefc24b,
	    :target:`dnnl_gOdhwI24o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad0517cba7cc6dcc16d5e1e8a67c7b773>`      = dnnl_aBdefC24b2c,
	    :target:`dnnl_gOdhwi32o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a18b5a0399e6394935305932235cede>`        = dnnl_aBdefc32b,
	    :target:`dnnl_gOdhwI32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4c6e6d105100fa9eea2536b56aa1cb2e>`      = dnnl_aBdefC32b2c,
	    :target:`dnnl_gOdhwI32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9591532d11f0b06f549765325606e6b8>`      = dnnl_aBdefC32b4c,
	    :target:`dnnl_gOdhwi48o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3211308854a8e27267b1e2bdac9cc267>`        = dnnl_aBdefc48b,
	    :target:`dnnl_gOdhwI48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacfdc91c56ae0233939e4d0e34ece3b97>`      = dnnl_aBdefC48b2c,
	    :target:`dnnl_gOdhwI48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf11c56c51ef11b0367e2250283e1ddf9>`      = dnnl_aBdefC48b4c,
	    :target:`dnnl_gOdhwi64o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad88b5e1ea89be0926989324153406743>`        = dnnl_aBdefc64b,
	    :target:`dnnl_gOdhwI64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac0e436c4c094fb46dc989a7065d8b3cf>`      = dnnl_aBdefC64b2c,
	    :target:`dnnl_gOdhwI64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadf5815b1a5886f8adabb3f886a32cc47>`      = dnnl_aBdefC64b4c,
	    :target:`dnnl_gIdhwo32i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da38c30a969ae21d7f65b7508863745f1f>`        = dnnl_aCdefb32c,
	    :target:`dnnl_gIdhwO32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4269dfdf5ef5cbc5215f7ebece0695eb>`      = dnnl_aCdefB32c2b,
	    :target:`dnnl_gIdhwO32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da25e63933226a98a283a4040ed7a7d374>`      = dnnl_aCdefB32c4b,
	    :target:`dnnl_gIdhwo48i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4ad61638d7915ddf7614cec81c343538>`        = dnnl_aCdefb48c,
	    :target:`dnnl_gIdhwO48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabaa2a7cd79b914929938609d83c48816>`      = dnnl_aCdefB48c2b,
	    :target:`dnnl_gIdhwO48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf273c1deac169142995e83b7314c9113>`      = dnnl_aCdefB48c4b,
	    :target:`dnnl_gIdhwo64i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafae71ccc778856e9570bec92590563c8>`        = dnnl_aCdefb64c,
	    :target:`dnnl_gIdhwO64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da01a02c215eeeddf1b3218583953a6dbb>`      = dnnl_aCdefB64c2b,
	    :target:`dnnl_gIdhwO64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0bb692ff8c77dc7b91c1187bd2e06db>`      = dnnl_aCdefB64c4b,
	    :target:`dnnl_gdhwio<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da64c10329973a28d1f6ae2773977ff639>`           = dnnl_adefcb,
	    :target:`dnnl_gdhwIo2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dada3a14ece34b137a95b90cf6361d721f>`         = dnnl_adefCb2c,
	    :target:`dnnl_gdhwIo4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2bfae9bae9f309c050c3c2c95d518247>`         = dnnl_adefCb4c,
	    :target:`dnnl_OI16i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7e253d67eecd50a2831a3d4738791bb5>`       = dnnl_AB16b32a4b,
	    :target:`dnnl_OI16i48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3227b1bad81281afeba3ef954dad8d51>`       = dnnl_AB16b48a4b,
	    :target:`dnnl_OI16i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8d1874a057c557a2ac79f72bab9c3a2c>`       = dnnl_AB16b64a4b,
	    :target:`dnnl_OI16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadf2949017a22be3b636db141b954ead2>`       = dnnl_AB16b16a2b,
	    :target:`dnnl_OI16i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daabf0b68a6dfeeedf9d8ceec7439fa691>`       = dnnl_AB16b32a2b,
	    :target:`dnnl_OI16i48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daeeca9771700606275009487775f39f15>`       = dnnl_AB16b48a2b,
	    :target:`dnnl_OI16i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab2a475355f2818a02f59f3dec26413ac>`       = dnnl_AB16b64a2b,
	    :target:`dnnl_OIw16i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1721a0c6100f6a9da41ad0d5353dc4fb>`      = dnnl_ABc16b32a4b,
	    :target:`dnnl_OIw16i48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da253a5d93e113034023e2c15af864602f>`      = dnnl_ABc16b48a4b,
	    :target:`dnnl_OIw16i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5a0cf714ad978a007c24cc64c05168de>`      = dnnl_ABc16b64a4b,
	    :target:`dnnl_OIw16i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4967e98b5be7ee9a208271eb3c8962c8>`      = dnnl_ABc16b32a2b,
	    :target:`dnnl_OIw16i48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad5108cc231946ce6fc91bcb2d92f1217>`      = dnnl_ABc16b48a2b,
	    :target:`dnnl_OIw16i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0c32d1dc1a169408d9973a8b1e0a0d65>`      = dnnl_ABc16b64a2b,
	    :target:`dnnl_OIhw16i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad5463f9b7cd128dadbf3add82c39842d>`     = dnnl_ABcd16b32a4b,
	    :target:`dnnl_OIhw16i48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae950c9fc7b3e090f2b50cf464d54e5a1>`     = dnnl_ABcd16b48a4b,
	    :target:`dnnl_OIhw16i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da48f8bed14dd4091de616d4dbcf4aa71a>`     = dnnl_ABcd16b64a4b,
	    :target:`dnnl_OIhw16i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac70e9ed6195b9fa925e46f44ad8b6f17>`     = dnnl_ABcd16b32a2b,
	    :target:`dnnl_OIhw16i48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8825fc8c096d650a4e59a8d361f3fe06>`     = dnnl_ABcd16b48a2b,
	    :target:`dnnl_OIhw16i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da41aa3d0a356c5456557ec6cdc5a63acb>`     = dnnl_ABcd16b64a2b,
	    :target:`dnnl_OIdhw16i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dada6626186b07da8f759e8bd1770b886c>`    = dnnl_ABcde16b32a4b,
	    :target:`dnnl_OIdhw16i48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae193f800ee77cde629bd8b3ddb578498>`    = dnnl_ABcde16b48a4b,
	    :target:`dnnl_OIdhw16i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da34e389b0a8e863f8e614146c75edea20>`    = dnnl_ABcde16b64a4b,
	    :target:`dnnl_OIdhw16i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8c0331401cf7099bedee5c8b95da043b>`    = dnnl_ABcde16b32a2b,
	    :target:`dnnl_OIdhw16i48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad92c34291bcdd2a43a93069f32c74c94>`    = dnnl_ABcde16b48a2b,
	    :target:`dnnl_OIdhw16i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae39934e74b188cee1e7323ef526ce1dd>`    = dnnl_ABcde16b64a2b,
	    :target:`dnnl_OwI16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacffcb2280f1086418a0d7302f358d184>`      = dnnl_AcB16b16a2b,
	    :target:`dnnl_OwI16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0f38c02714d8a5cadd27fabdf3a12f89>`      = dnnl_AcB16b16a4b,
	    :target:`dnnl_OhwI16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad7a38fa50923d4c72df52e6a126d07aa>`     = dnnl_AcdB16b16a2b,
	    :target:`dnnl_OhwI16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da79029121cddd3b25582e3bc11419965c>`     = dnnl_AcdB16b16a4b,
	    :target:`dnnl_OdhwI16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadfe6b9d730bfc1ef4243a360f4b942a1>`    = dnnl_AcdeB16b16a2b,
	    :target:`dnnl_OdhwI16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafb12e6e3ce471cb9707aa7d1c3beea87>`    = dnnl_AcdeB16b16a4b,
	    :target:`dnnl_IwO16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da579231726469034b77cba768892f2710>`      = dnnl_BcA16a16b2a,
	    :target:`dnnl_IwO16o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab05d74d53b9a598e06384811daf21ba6>`      = dnnl_BcA16a16b4a,
	    :target:`dnnl_IhwO16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7fb7edcd36c8a3543df1def378bfe537>`     = dnnl_BcdA16a16b2a,
	    :target:`dnnl_IhwO16o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da036d4c17c575ddc3273fbaa2064afd11>`     = dnnl_BcdA16a16b4a,
	    :target:`dnnl_IdhwO16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf01433866e1bb3afd25b851ff76ac153>`    = dnnl_BcdeA16a16b2a,
	    :target:`dnnl_IdhwO16o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadcf355e894bc48b197ee8c301c9e8cd3>`    = dnnl_BcdeA16a16b4a,
	    :target:`dnnl_gOwI16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0038619db08a57fba1c2394e3e2166bb>`     = dnnl_aBdC16c16b2c,
	    :target:`dnnl_gOwI16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da29d0ac398b572cd559abfed221a77e81>`     = dnnl_aBdC16c16b4c,
	    :target:`dnnl_gOhwI16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da89ba9cc806b557de98750e8b4a3eedbf>`    = dnnl_aBdeC16c16b2c,
	    :target:`dnnl_gOhwI16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da051951b78884d6bd880c1fada8f2d7c1>`    = dnnl_aBdeC16c16b4c,
	    :target:`dnnl_gOdhwI16i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da234d5c78ea247b93606f322d7a6c3100>`   = dnnl_aBdefC16c16b2c,
	    :target:`dnnl_gOdhwI16i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daac63353a4e1d9a7d51266ecacb85062e>`   = dnnl_aBdefC16c16b4c,
	    :target:`dnnl_gIwO16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6216568609444d04b6083c3d2b5bde0d>`     = dnnl_aCdB16b16c2b,
	    :target:`dnnl_gIwO16o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da483d725f0bacb716101dceae51d78bd6>`     = dnnl_aCdB16b16c4b,
	    :target:`dnnl_gIhwO16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da46e88835b5da0fe2f651b9522e45694f>`    = dnnl_aCdeB16b16c2b,
	    :target:`dnnl_gIhwO16o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae5873daecbb752299103ffdb31581851>`    = dnnl_aCdeB16b16c4b,
	    :target:`dnnl_gIdhwO16o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabd2444d6336542f8a408b472c463b1a9>`   = dnnl_aCdefB16b16c2b,
	    :target:`dnnl_gIdhwO16o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3777e8211cd4221d5b99f22d664f8c23>`   = dnnl_aCdefB16b16c4b,
	    :target:`dnnl_OwI16i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da001b7adfc7badbaec0d50a00ac45c089>`      = dnnl_AcB16b32a2b,
	    :target:`dnnl_OwI16i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfb92d38ee95ef3f7fd812c51021731>`      = dnnl_AcB16b32a4b,
	    :target:`dnnl_OwI16i48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafef671a24a593d841729a60a62ff9383>`      = dnnl_AcB16b48a2b,
	    :target:`dnnl_OwI16i48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1d5162231344f8526354cf1a86655ce0>`      = dnnl_AcB16b48a4b,
	    :target:`dnnl_OwI16i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da44bc6de355b05b8a012782276133efba>`      = dnnl_AcB16b64a2b,
	    :target:`dnnl_OwI16i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e7ed0f905e41565c3a80f03b2e5e513>`      = dnnl_AcB16b64a4b,
	    :target:`dnnl_IwO16o32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf15c4eee6e273f6a754f45945ebd3603>`      = dnnl_BcA16a32b2a,
	    :target:`dnnl_IwO16o32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da898d7352197dc7b6cd3bd086e08cb9d8>`      = dnnl_BcA16a32b4a,
	    :target:`dnnl_IwO16o48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadfd9f5341fcf5770af7bcc9d7c2fa8a5>`      = dnnl_BcA16a48b2a,
	    :target:`dnnl_IwO16o48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafc85ce82807628736929f98056b30275>`      = dnnl_BcA16a48b4a,
	    :target:`dnnl_IwO16o64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad64ccd70339be4aba1674ff576b15709>`      = dnnl_BcA16a64b2a,
	    :target:`dnnl_IwO16o64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf6af14d737065b4c534b80f8263f701c>`      = dnnl_BcA16a64b4a,
	    :target:`dnnl_gOwI16i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad8b94c15e775fd65dc3c53bfb64f3ba8>`     = dnnl_aBdC16c32b2c,
	    :target:`dnnl_gOwI16i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da295db45ed8d6acb763c5bd37903e7e1a>`     = dnnl_aBdC16c32b4c,
	    :target:`dnnl_gOwI16i48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da879c0244693e251e798267d75d38a6e9>`     = dnnl_aBdC16c48b2c,
	    :target:`dnnl_gOwI16i48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da93d6990b06abe48901e850f8d79069b7>`     = dnnl_aBdC16c48b4c,
	    :target:`dnnl_gOwI16i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad09088dedcac13d2a71ad4c51e76859e>`     = dnnl_aBdC16c64b2c,
	    :target:`dnnl_gOwI16i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8b646f3c6a09ebc566aafbcb006bf023>`     = dnnl_aBdC16c64b4c,
	    :target:`dnnl_gIwO16o32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9bafaa2ec3536cb1c7e12108545143be>`     = dnnl_aCdB16b32c2b,
	    :target:`dnnl_gIwO16o32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa9acd53b8bc311ac3de162c595f6c027>`     = dnnl_aCdB16b32c4b,
	    :target:`dnnl_gIwO16o48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4a242f7d344d64d0efb5b050211ab856>`     = dnnl_aCdB16b48c2b,
	    :target:`dnnl_gIwO16o48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1eb5d58c75bc8cfdb4c65ba5602f8f78>`     = dnnl_aCdB16b48c4b,
	    :target:`dnnl_gIwO16o64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da284d2e9413709d16fb1e8edbfe9d6c49>`     = dnnl_aCdB16b64c2b,
	    :target:`dnnl_gIwO16o64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa10a46666acd6fba8f73da63476b241f>`     = dnnl_aCdB16b64c4b,
	    :target:`dnnl_OhwI16i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7eba23961cf5d29d0c5575649f8eb1af>`     = dnnl_AcdB16b32a2b,
	    :target:`dnnl_OhwI16i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafa8244145b3f2bded809d51ba718e180>`     = dnnl_AcdB16b32a4b,
	    :target:`dnnl_OhwI16i48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabf7f0c855605ff0c975e9803176cff8f>`     = dnnl_AcdB16b48a2b,
	    :target:`dnnl_OhwI16i48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae93c5aeb688c60182deeedd5b9448659>`     = dnnl_AcdB16b48a4b,
	    :target:`dnnl_OhwI16i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaf319e779070affec7f25a5d47673ef5>`     = dnnl_AcdB16b64a2b,
	    :target:`dnnl_OhwI16i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5d6722bf5d3f3f7341dc0741f2755ca2>`     = dnnl_AcdB16b64a4b,
	    :target:`dnnl_IhwO16o32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaf5cfd8e6364cf36f1b6bed4e1dc9327>`     = dnnl_BcdA16a32b2a,
	    :target:`dnnl_IhwO16o32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac6d701038f3e1c6e63bfb7a70075e460>`     = dnnl_BcdA16a32b4a,
	    :target:`dnnl_IhwO16o48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da74bbd21aefdaee55ab3eecb0c4ea9507>`     = dnnl_BcdA16a48b2a,
	    :target:`dnnl_IhwO16o48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da86632b195cbe3e3d5dc60ba899ccf173>`     = dnnl_BcdA16a48b4a,
	    :target:`dnnl_IhwO16o64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac97f4ee2b32617b851be0b94dbe0d2e4>`     = dnnl_BcdA16a64b2a,
	    :target:`dnnl_IhwO16o64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6aab396e402ab624ef40e139054ed57c>`     = dnnl_BcdA16a64b4a,
	    :target:`dnnl_gOhwI16i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da88b183c1819af8ea25c32d7f0f41fe14>`    = dnnl_aBdeC16c32b2c,
	    :target:`dnnl_gOhwI16i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7c089f071522add23990d60c959df50b>`    = dnnl_aBdeC16c32b4c,
	    :target:`dnnl_gOhwI16i48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab55873363105271ab0bd9f060388498>`    = dnnl_aBdeC16c48b2c,
	    :target:`dnnl_gOhwI16i48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac67818c7d8bab4bdccb2b3d16dc5ddad>`    = dnnl_aBdeC16c48b4c,
	    :target:`dnnl_gOhwI16i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab44c55fa0eca34872560954a8ab34f02>`    = dnnl_aBdeC16c64b2c,
	    :target:`dnnl_gOhwI16i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6d26cd6c35752eb7c30d1dab63e0f648>`    = dnnl_aBdeC16c64b4c,
	    :target:`dnnl_gIhwO16o32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daecf92b4dff477cd4a8c6b770cc70e4ec>`    = dnnl_aCdeB16b32c2b,
	    :target:`dnnl_gIhwO16o32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac3a23c38401a8e748ce62739d7d9f983>`    = dnnl_aCdeB16b32c4b,
	    :target:`dnnl_gIhwO16o48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6fb388d4886b5d44e05e6db9f9a54867>`    = dnnl_aCdeB16b48c2b,
	    :target:`dnnl_gIhwO16o48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabbba05a6799f3ab24846bc31200547b8>`    = dnnl_aCdeB16b48c4b,
	    :target:`dnnl_gIhwO16o64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac4ab5290872cecdcb376d2cd3f1c0068>`    = dnnl_aCdeB16b64c2b,
	    :target:`dnnl_gIhwO16o64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8d50da28b97c0e023c848d9d838d734a>`    = dnnl_aCdeB16b64c4b,
	    :target:`dnnl_OdhwI16i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5c42dc632e8787e0e920191edb4aa697>`    = dnnl_AcdeB16b32a2b,
	    :target:`dnnl_OdhwI16i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daed40051ab1ddf16b7710b99eadb37f8a>`    = dnnl_AcdeB16b32a4b,
	    :target:`dnnl_OdhwI16i48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2b8f12b998649b6377b4197d30a04445>`    = dnnl_AcdeB16b48a2b,
	    :target:`dnnl_OdhwI16i48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6dc7bc8a19aac8126acd1dc06172ab7c>`    = dnnl_AcdeB16b48a4b,
	    :target:`dnnl_OdhwI16i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4b621878bc13907e51d31059852dcc39>`    = dnnl_AcdeB16b64a2b,
	    :target:`dnnl_OdhwI16i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae18f903ad19c296821b3a14c1abd466b>`    = dnnl_AcdeB16b64a4b,
	    :target:`dnnl_IdhwO16o32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae1edb1fcdcbc0e9b50a4304a15a7a8b9>`    = dnnl_BcdeA16a32b2a,
	    :target:`dnnl_IdhwO16o32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadcda45e29fab98e83b863216f4ed9996>`    = dnnl_BcdeA16a32b4a,
	    :target:`dnnl_IdhwO16o48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da441cc5bca9ff97e5b4d3d63f14efd779>`    = dnnl_BcdeA16a48b2a,
	    :target:`dnnl_IdhwO16o48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da75531d507cd55d590f243a3a77958752>`    = dnnl_BcdeA16a48b4a,
	    :target:`dnnl_IdhwO16o64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab819ccd0aba669584aa42e638634b7e>`    = dnnl_BcdeA16a64b2a,
	    :target:`dnnl_IdhwO16o64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da54ba61d01d877017272f33123e4522f9>`    = dnnl_BcdeA16a64b4a,
	    :target:`dnnl_gOdhwI16i32o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5464fb0b65589fbe0e8ed54d5a3eda4c>`   = dnnl_aBdefC16c32b2c,
	    :target:`dnnl_gOdhwI16i32o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da201fc5b1e58d5a25b7055724525cb017>`   = dnnl_aBdefC16c32b4c,
	    :target:`dnnl_gOdhwI16i48o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad980512a2990b6931b9df1dbcc3ab256>`   = dnnl_aBdefC16c48b2c,
	    :target:`dnnl_gOdhwI16i48o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f43fcd92dfb95ecf6cab71f071f1940>`   = dnnl_aBdefC16c48b4c,
	    :target:`dnnl_gOdhwI16i64o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac4a53b158c19934611a802e2488b478d>`   = dnnl_aBdefC16c64b2c,
	    :target:`dnnl_gOdhwI16i64o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafdb81f4efb6debf6dcfc704734c294ba>`   = dnnl_aBdefC16c64b4c,
	    :target:`dnnl_gIdhwO16o32i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da02c8470f15c8594fe43296069b75bd2d>`   = dnnl_aCdefB16b32c2b,
	    :target:`dnnl_gIdhwO16o32i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daba5eb4a00f14e2ee6e13d14fa85e4d40>`   = dnnl_aCdefB16b32c4b,
	    :target:`dnnl_gIdhwO16o48i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6365f4ce8084142e0340399b7aa258ac>`   = dnnl_aCdefB16b48c2b,
	    :target:`dnnl_gIdhwO16o48i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad9f5c8e4b06a21464976bc7aeaeb1d56>`   = dnnl_aCdefB16b48c4b,
	    :target:`dnnl_gIdhwO16o64i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da63c96afb79af43301db00e43e0098f09>`   = dnnl_aCdefB16b64c2b,
	    :target:`dnnl_gIdhwO16o64i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daaff2fe731586926300ad0901081dcf33>`   = dnnl_aCdefB16b64c4b,
	    :target:`dnnl_hwioG16g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacbdffd937f2102f817aaf081e92d1f17>`         = dnnl_decbA16a,
	    :target:`dnnl_hwioG8g<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9d54994fc41a3c97a42ed1c9f660e06a>`          = dnnl_decbA8a,
	    :target:`dnnl_NCdhw40n16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae3bd5ffecd8b0534b4ceb5d0c4203432>`      = dnnl_ABcde40a16b,
	    :target:`dnnl_NCw40n16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da04cf6cfd69fe294d0715ea2a9118f95f>`        = dnnl_ABc40a16b,
	    :target:`dnnl_NChw40n16c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da655446d73e10a2acc23283406f1e3912>`       = dnnl_ABcd40a16b,
	    :target:`dnnl_NCw40n32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daea50b572fb9a28601e8f86cc0f9875ff>`        = dnnl_ABc40a32b,
	    :target:`dnnl_NChw40n32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa04fe2bb8251a3b6734e4746766b04b9>`       = dnnl_ABcd40a32b,
	    :target:`dnnl_NCdhw40n32c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7fdaebb7adc7ba4343073cc404cdf401>`      = dnnl_ABcde40a32b,
	    :target:`dnnl_OIdhw4o8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafdb25557c0a20cf01cdfc96cd5edf52d>`    = dnnl_ABcde4a8b8a2b,
	    :target:`dnnl_OIhw4o8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da36d983fdd8659b1f22bba1b16392743c>`     = dnnl_ABcd4a8b8a2b,
	    :target:`dnnl_OIw4o8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dade6d4e0ce2c15f2145c1710fa2071f77>`      = dnnl_ABc4a8b8a2b,
	    :target:`dnnl_gOIdhw4o8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7f216b74496f6e26a560db0953ec8ca0>`   = dnnl_aBCdef4b8c8b2c,
	    :target:`dnnl_gOIhw4o8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da48d29c73cbeadcb3a32ad140d7e65168>`    = dnnl_aBCde4b8c8b2c,
	    :target:`dnnl_gOIw4o8i8o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daebea6248af2a67f1e9054c1633c8701e>`     = dnnl_aBCd4b8c8b2c,
	    :target:`dnnl_IOdhw4i8o8i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da84973ec57f342f35d22779767b3c337f>`    = dnnl_BAcde4b8a8b2a,
	    :target:`dnnl_IOhw4i8o8i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da887e46d2bae98ec4d794b5785b200fb6>`     = dnnl_BAcd4b8a8b2a,
	    :target:`dnnl_IOw4i8o8i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab44308a1f0f205e87301441c19fcc7d9>`      = dnnl_BAc4b8a8b2a,
	    :target:`dnnl_gIOdhw4i8o8i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da550dda0209a95d70a502e942ffc28831>`   = dnnl_aCBdef4c8b8c2b,
	    :target:`dnnl_gIOhw4i8o8i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafb12c45ff19ad995aa9fd16f016c871c>`    = dnnl_aCBde4c8b8c2b,
	    :target:`dnnl_gIOw4i8o8i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8bdd59bc22ce5cf549b76215445bb4fe>`     = dnnl_aCBd4c8b8c2b,
	    :target:`dnnl_NCw2c32n8c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9e164487a8f06493c3ea97f68738fd44>`       = dnnl_ABc2b32a8b,
	    :target:`dnnl_NChw2c32n8c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9ac92901a8baa63417d2d3c69312a901>`      = dnnl_ABcd2b32a8b,
	    :target:`dnnl_NCdhw2c32n8c<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1b39276d64b753081f4b5a977f96e9da>`     = dnnl_ABcde2b32a8b,
	    :target:`dnnl_OIw2i8o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2fcec424c2e71ade0a03da43dd1d8a39>`     = dnnl_ABc2b8a16b4a,
	    :target:`dnnl_OIhw2i8o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf6f8283b4b98a172f69fc8d838bcaea7>`    = dnnl_ABcd2b8a16b4a,
	    :target:`dnnl_OIdhw2i8o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2c7cde1f180979255568cfeb35d53bc6>`   = dnnl_ABcde2b8a16b4a,
	    :target:`dnnl_OIw2o8i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da10caba25da8eab2caa2ab1f3e10dd6e0>`     = dnnl_ABc2a8b16a4b,
	    :target:`dnnl_OIw2o8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f55acf7f1c4407f76d5c309aca0c938>`     = dnnl_ABc2a8b16a2b,
	    :target:`dnnl_IOw2i8o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae0afd370dd3a95f8fb53e64b4ff97a93>`     = dnnl_BAc2b8a16b4a,
	    :target:`dnnl_IOw2i8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6921fd6f9a71823dbaf8eb57c0f347d1>`     = dnnl_BAc2b8a16b2a,
	    :target:`dnnl_OIhw2o8i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4be4e32406fd7f83f8f275a61cd3ba04>`    = dnnl_ABcd2a8b16a4b,
	    :target:`dnnl_OIhw2o8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad39186f171dec717809d8bf7c7a164a0>`    = dnnl_ABcd2a8b16a2b,
	    :target:`dnnl_IOhw2i8o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacd3f9a3d22c7653a0b3c24b72b9d6806>`    = dnnl_BAcd2b8a16b4a,
	    :target:`dnnl_IOhw2i8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daea40786a39e02e45e4e9cb74a8fc65f2>`    = dnnl_BAcd2b8a16b2a,
	    :target:`dnnl_OIdhw2o8i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da23bf86f49572eee01a3e3e909c7ead92>`   = dnnl_ABcde2a8b16a4b,
	    :target:`dnnl_OIdhw2o8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da14bf24a68e373079f676afe93ac8169b>`   = dnnl_ABcde2a8b16a2b,
	    :target:`dnnl_IOdhw2i8o16i4o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dacc3c2c94cb0f18b97e30d7520b68a3ba>`   = dnnl_BAcde2b8a16b4a,
	    :target:`dnnl_IOdhw2i8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7b08f5cf15027610798a8cee3b072f2f>`   = dnnl_BAcde2b8a16b2a,
	    :target:`dnnl_gOIw2o8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2d3e0bbedc4d6ba213d7a77c016cf922>`    = dnnl_aBCd2b8c16b2c,
	    :target:`dnnl_gIOw2i8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7b4a9dee4d007fd03b8381995e2b0833>`    = dnnl_aCBd2c8b16c2b,
	    :target:`dnnl_gIOhw2i8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab5b56c2c4335bf73fc8b74fde97c7d83>`   = dnnl_aBCde2c8b16c2b,
	    :target:`dnnl_gIOdhw2i8o16i2o<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da31433e76b3191b628f9ba7a8f28b9263>`  = dnnl_aBCdef2c8b16c2b,
	    :target:`dnnl_gOIhw2o8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0dd30c27f980cfb399c5159925e0ba65>`   = dnnl_aBCde2b8c16b2c,
	    :target:`dnnl_gOIdhw2o8i16o2i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da97b5469c9dd1cc5422cc05fa2923b731>`  = dnnl_aBCdef2b8c16b2c,
	    :target:`dnnl_gOIw2o8i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf54d1d8f533f54e10ce818b9b4a30966>`    = dnnl_aBCd2b8c16b4c,
	    :target:`dnnl_gOIhw2o8i16o4i<doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da839316c03df0a0a581f9e25d4302a156>`   = dnnl_aBCde2b8c16b4c,
	};

.. _details-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Memory format tag specification.

oneDNN formats describe physical data layout. The physical layout is described as a sequence of the dimensions as they are laid out in the memory (from the outer-most to the inner-most). Note that this order doesn't affect the logical order of the dimensions that is kept in the ``dims`` field of the dnnl_memory_desc_t structure. The logical order of the dimensions is specified by the primitive that uses the tensor.

For example, CNN 5D tensor always has its logical dimensions in the order ``(batch, channels, depth, height, width)``, while the physical layout might be ``NCDHW`` (corresponds to :ref:`dnnl_ncdhw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464>` format tag) or ``NDHWC`` (corresponds to :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>` format tag).

.. ref-code-block:: cpp

	int batch = 2, channels = 16, depth = 13, height = 13, width = 13;
	
	int ndims = 5; // 5D tensor
	:ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>` = {batch, channels, depth, height, width};
	:ref:`dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` data_in_ncdhw;
	:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(
	     &data_in_ncdhw, 5, dims, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_ncdhw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464>`);
	
	// note that in both cases dims passed are the same
	:ref:`dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` data_in_ndhwc;
	:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(
	     &data_in_ndhwc, 5, dims, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>`);
	
	:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(data_in_ncdhw);
	:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(data_in_ndhwc);

Memory format tags can be further divided into two categories:

* Domain-agnostic names, i.e. names the do not depend on the tensor usage in the specific primitive. These names use letters from ``a`` to ``l`` to denote logical dimension from 1 to 12, and form the order in which the dimensions are laid in memory. For instance, :ref:`dnnl_ab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53>` is used to denote 2D tensor where the second logical dimension (aka ``b``) is the innermost, i.e. has stride = 1, and the first logical dimension (``a``) laid out in memory with stride equal to the size of second dimension. On the other hand, :ref:`dnnl_ba <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6a6dbc0b30468d92e32a9cb3f6615c43>` is just transposed version of the same tensor: the first dimension (``a``) becomes the innermost one.

* Domain-specific names, i.e. names that make sense only in the context of a certain domain, such as CNN. This names are just aliases to the corresponding domain-agnostic tags and used mostly for the convenience. For example, :ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>` is used to denote 2D CNN activations tensor memory format, where channels are the innermost dimension and batch is an outermost one. Moreover, :ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>` is just an alias to :ref:`dnnl_ab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53>`, since for oneDNN CNN primitives the logical dimensions of activations tensors come in order: batch, channels, spatial. In other words, batch corresponds to the first logical dimension (``a``), channels correspond to the second one (``b``).

The following domain-specific notation applies to memory format tags:

* ``'n'`` denotes the mini-batch dimension

* ``'c'`` denotes a channels dimension

* When there are multiple channel dimensions (for example, in convolution weights tensor), ``'i'`` and ``'o'`` denote dimensions of input and output channels

* ``'d'``, ``'h'``, and ``'w'`` denote spatial depth, height, and width respectively

Upper-case letters indicate that the data is laid out in blocks for a particular dimension. In such cases, the format name contains both upper- and lower-case letters for that dimension with a lower-case letter preceded by the block size. For example: :ref:`dnnl_nChw8c <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da775389dbdcda91ea12906210c309746f>` describes a format where the outermost dimension is mini-batch, followed by the channel block number, followed by the spatial height and width, and finally followed by 8-element channel blocks.



.. rubric:: See also:

:ref:`Understanding Memory Formats <doxid-dev_guide_understanding_memory_formats>`

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_format_tag_undef
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6f473b40f7e7470ba886ce3220ec69a7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_format_tag_undef

Undefined memory format tag.

.. index:: pair: enumvalue; dnnl_format_tag_any
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_format_tag_any

Undefined memory format tag.

The primitive selects a format automatically.

.. index:: pair: enumvalue; dnnl_a
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a72c401669bf1737439d6c4af17d0be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_a

plain 1D tensor

.. index:: pair: enumvalue; dnnl_ab
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ab

plain 2D tensor

.. index:: pair: enumvalue; dnnl_abc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadff5ea69392d7e4da23179dc0ba7cbc2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abc

plain 3D tensor

.. index:: pair: enumvalue; dnnl_abcd
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcd

plain 4D tensor

.. index:: pair: enumvalue; dnnl_abcde
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da30d5d3c9de2931f06d265af81787ada3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcde

plain 5D tensor

.. index:: pair: enumvalue; dnnl_abcdef
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da74b2db34e6253e56f441764d16e6ec42:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdef

plain 6D tensor

.. index:: pair: enumvalue; dnnl_abcdefg
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa6db6b10c9279959f5de845d454c19f0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefg

plain 7D tensor

.. index:: pair: enumvalue; dnnl_abcdefgh
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da12cecc4b1febc6f7bbc4ddced6482155:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefgh

plain 8D tensor

.. index:: pair: enumvalue; dnnl_abcdefghi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0a0796a36ef631a3f1bdd9210e59684e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefghi

plain 9D tensor

.. index:: pair: enumvalue; dnnl_abcdefghij
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafaaf9a62cda825a376f1e35936dbf5b1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefghij

plain 10D tensor

.. index:: pair: enumvalue; dnnl_abcdefghijk
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da571cdcd3d6fd3ae211377f7c2068846a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefghijk

plain 11D tensor

.. index:: pair: enumvalue; dnnl_abcdefghijkl
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da24084d001c72ff2e0543fe09d543678f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefghijkl

plain 12D tensor

.. index:: pair: enumvalue; dnnl_ba
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6a6dbc0b30468d92e32a9cb3f6615c43:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ba

permuted 2D tensor

.. index:: pair: enumvalue; dnnl_acb
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8537ed269eb5d0586456db114039c00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_acb

permuted 3D tensor

.. index:: pair: enumvalue; dnnl_bac
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae31cc3d3a0bfbd4936f7b503cf6dc9de:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_bac

permuted 3D tensor

.. index:: pair: enumvalue; dnnl_bca
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da37fcc057fe063d378bb5f1d5111e0e60:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_bca

permuted 3D tensor

.. index:: pair: enumvalue; dnnl_cab
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4a42ef47bea9ef571f64944b62fddea3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cab

permuted 3D tensor

.. index:: pair: enumvalue; dnnl_cba
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5a8de15eee2e5bcc2515eb7b01965789:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cba

permuted 3D tensor

.. index:: pair: enumvalue; dnnl_abdc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad7bcb765b861ed46b2b752e912989217:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abdc

permuted 4D tensor

.. index:: pair: enumvalue; dnnl_acbd
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6a81c235d5a08451ac61edb1681bd925:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_acbd

permuted 4D tensor

.. index:: pair: enumvalue; dnnl_acdb
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fcce5dd7260b5b0740e3b37b1e9ad41:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_acdb

permuted 4D tensor

.. index:: pair: enumvalue; dnnl_adbc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf3800b4f18027abd248a61ae7c5fc8e0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_adbc

permuted 4D tensor

.. index:: pair: enumvalue; dnnl_adcb
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac3b46f2409584aab3c10f8189a9fb47f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_adcb

permuted 4D tensor

.. index:: pair: enumvalue; dnnl_bacd
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab1a39583c90f536df19e47d8262ef6f6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_bacd

permuted 4D tensor

.. index:: pair: enumvalue; dnnl_bcda
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac42f365ea58d5b73c141c0788641f74a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_bcda

permuted 4D tensor

.. index:: pair: enumvalue; dnnl_cdab
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e99e668dcd118f6f576c7be57d5c43d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cdab

permuted 4D tensor

.. index:: pair: enumvalue; dnnl_cdba
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2368e1a4ce9d9954fa10c242569eacb4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cdba

permuted 4D tensor

.. index:: pair: enumvalue; dnnl_dcab
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae6c549f1a265f8d61bffb24c24925a1d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_dcab

permuted 4D tensor

.. index:: pair: enumvalue; dnnl_abced
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac2006f669dbddbc93e5fd8cf2abbe005:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abced

permuted 5D tensor

.. index:: pair: enumvalue; dnnl_abdec
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab417c26841486da2a0da84730d0874d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abdec

permuted 5D tensor

.. index:: pair: enumvalue; dnnl_acbde
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac5f398cc58f5625915719ec541bb152e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_acbde

permuted 5D tensor

.. index:: pair: enumvalue; dnnl_acdeb
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfe86402763786b9b4d73062cfd2f05:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_acdeb

permuted 5D tensor

.. index:: pair: enumvalue; dnnl_adecb
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf87ca2ebe0be8da766b84f3a6bdf72ff:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_adecb

permuted 5D tensor

.. index:: pair: enumvalue; dnnl_bacde
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da57c548e503e61b8c0f9a70a0a9acce1c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_bacde

permuted 5D tensor

.. index:: pair: enumvalue; dnnl_bcdea
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad219c123f120d32a5f26d584b334af41:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_bcdea

permuted 5D tensor

.. index:: pair: enumvalue; dnnl_cdeab
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf4870a90b840aed0c8162f0fa5822967:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cdeab

permuted 5D tensor

.. index:: pair: enumvalue; dnnl_cdeba
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae7dd2b81eb2b502efbf65c888665b358:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cdeba

permuted 5D tensor

.. index:: pair: enumvalue; dnnl_decab
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da31fa5f2eb4a236664583afc291310ec6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_decab

permuted 5D tensor

.. index:: pair: enumvalue; dnnl_abcdfe
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2e7f7dc1fa5cc578ee70b018ae78d63d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdfe

permuted 6D tensor

.. index:: pair: enumvalue; dnnl_abdefc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad23172b9338edbd306f84f24fd7b00be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abdefc

permuted 6D tensor

.. index:: pair: enumvalue; dnnl_abdfce
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa9483a5b4d17c8aebd14a9894ff67fe5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abdfce

permuted 6D tensor

.. index:: pair: enumvalue; dnnl_acbdef
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8463916a68cb816050fbaf71817093ae:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_acbdef

permuted 6D tensor

.. index:: pair: enumvalue; dnnl_adefcb
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da934ab2ceb3d0f6b358f3724b407811a0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_adefcb

permuted 6D tensor

.. index:: pair: enumvalue; dnnl_defcab
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da73502e9308b365e74da295eaa83703ad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_defcab

permuted 6D tensor

.. index:: pair: enumvalue; dnnl_abcdegf
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da28fa9c38c8aba3f7366a40c5f28f5894:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdegf

permuted 7D tensor

.. index:: pair: enumvalue; dnnl_abcdefhg
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da013495943a14ba43871723e2a1f29de8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefhg

permuted 8D tensor

.. index:: pair: enumvalue; dnnl_abcdefgih
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad1ce4f5554fe6c5338ca3de649477545:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefgih

permuted 9D tensor

.. index:: pair: enumvalue; dnnl_abcdefghji
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8dc3817d010338ab9150378f8fded79:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefghji

permuted 10D tensor

.. index:: pair: enumvalue; dnnl_abcdefghikj
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da190b0571914d2675e1bc51d154e0de48:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefghikj

permuted 11D tensor

.. index:: pair: enumvalue; dnnl_abcdefghijlk
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadeb66e32bfc557ea52fbeeb532287b45:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_abcdefghijlk

permuted 12D tensor

.. index:: pair: enumvalue; dnnl_aBc16b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3e2da6258e387e0eb149b22f57209056:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBc16b

3D tensor blocked by 2nd dimension with block size 16

.. index:: pair: enumvalue; dnnl_aBc32b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8c6671d24cda4d6ea0fa7d8613f686c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBc32b

3D tensor blocked by 2nd dimension with block size 32

.. index:: pair: enumvalue; dnnl_aBc4b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4df9991897c2031b5fbdf97a4b3d2a02:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBc4b

3D tensor blocked by 2nd dimension with block size 4

.. index:: pair: enumvalue; dnnl_aBc8b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da96fcac28a789530765def2cfae2b61f6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBc8b

3D tensor blocked by 2nd dimension with block size 8

.. index:: pair: enumvalue; dnnl_aBcd16b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da590d748ac3761a51b407555a9a0d5efe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBcd16b

4D tensor blocked by 2nd dimension with block size 16

.. index:: pair: enumvalue; dnnl_aBcd32b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadaa7aeb5746c398a1c8ee1b42362bc58:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBcd32b

4D tensor blocked by 2nd dimension with block size 32

.. index:: pair: enumvalue; dnnl_aBcd4b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2ec2a0d2900e5ddb7f88783e2d8a5e66:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBcd4b

4D tensor blocked by 2nd dimension with block size 4

.. index:: pair: enumvalue; dnnl_aBcd8b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9523ccda38e86a8f7f58f8f780ef6eac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBcd8b

4D tensor blocked by 2nd dimension with block size 8

.. index:: pair: enumvalue; dnnl_ABcd8b8a
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2062c0db6b460fd8ac254684bbc15c36:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ABcd8b8a

4D tensor blocked by 1st and 2nd dimension with block size 8

.. index:: pair: enumvalue; dnnl_aBCd2b4c2b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac8768b971e97816930a09b6199f5bc05:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBCd2b4c2b

4D tensor blocked by 3rd dimension with block size 4

.. index:: pair: enumvalue; dnnl_ABcde4b16a4b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad8e56dc7af8c48b5d4e5d11650a19eab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ABcde4b16a4b

5D tensor blocked by 1st dimension with block size 16

.. index:: pair: enumvalue; dnnl_ABcde2b8a4b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6dc96fd5cf79d2d4395cd129830f17a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ABcde2b8a4b

5D tensor blocked by 1st dimension with block size 8

.. index:: pair: enumvalue; dnnl_aBcde16b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad4357b1ef924c1331e9e7e25facc3827:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBcde16b

5D tensor blocked by 2nd dimension with block size 16

.. index:: pair: enumvalue; dnnl_aBcde32b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabdf6daaaaa7e79356fb5b4ec7ae90e5a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBcde32b

5D tensor blocked by 2nd dimension with block size 32

.. index:: pair: enumvalue; dnnl_aBcde4b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3b841f4fc9008e7a27f4dcc4b130d819:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBcde4b

5D tensor blocked by 2nd dimension with block size 4

.. index:: pair: enumvalue; dnnl_aBcde8b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa778e377f1b0afdee9a6bcbc63ae76bd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBcde8b

5D tensor blocked by 2nd dimension with block size 8

.. index:: pair: enumvalue; dnnl_aBCde2b4c2b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da451386bf211128eaa11a8863a9d97e2b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBCde2b4c2b

5D tensor blocked by 3rd dimension with block size 4

.. index:: pair: enumvalue; dnnl_aBcdef16b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da431c182a4f2665030f29973efb556c94:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBcdef16b

6D tensor blocked by 2nd dimension with block size 16

.. index:: pair: enumvalue; dnnl_aBCdef2c8b4c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf9d8eabf65635e7a932e5f5b3339cdd8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBCdef2c8b4c

6D tensor blocked by 2nd dimension with block size 8

.. index:: pair: enumvalue; dnnl_aBCdef2b4c2b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da06fe4d629df62b68f499416b791da96b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBCdef2b4c2b

6D tensor blocked by 3rd dimension with block size 4

.. index:: pair: enumvalue; dnnl_aBcdef4b
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2735d5d034a5abe998df55be8e47e5e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_aBcdef4b

6D tensor blocked by 2nd dimension with block size 4

.. index:: pair: enumvalue; dnnl_format_tag_last
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadece229e4705088e29d6cbfb94c64756:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_format_tag_last

Just a sentinel, not real memory format tag.

Must be changed after new format tag is added.

.. index:: pair: enumvalue; dnnl_x
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9ccb37bb1a788f0245efbffbaf81e145:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_x

1D tensor, an alias to :ref:`dnnl_a <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a72c401669bf1737439d6c4af17d0be>`

.. index:: pair: enumvalue; dnnl_nc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nc

2D CNN activations tensor, an alias to :ref:`dnnl_ab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53>`

.. index:: pair: enumvalue; dnnl_cn
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da11b661fe7ab82f99b34bcaf460a8dc66:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cn

2D CNN activations tensor, an alias to :ref:`dnnl_ba <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6a6dbc0b30468d92e32a9cb3f6615c43>`

.. index:: pair: enumvalue; dnnl_tn
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac1ee747f881fc8cae805ae4051e671cf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_tn

2D RNN statistics tensor, an alias to :ref:`dnnl_ab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53>`

.. index:: pair: enumvalue; dnnl_nt
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7ede27a46172c1ab2775593ba4b118dc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nt

2D RNN statistics tensor, an alias to :ref:`dnnl_ba <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6a6dbc0b30468d92e32a9cb3f6615c43>`

.. index:: pair: enumvalue; dnnl_ncw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab55cb1d54480dd7f796bf66eea3ad32f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ncw

3D CNN activations tensor, an alias to :ref:`dnnl_abc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadff5ea69392d7e4da23179dc0ba7cbc2>`

.. index:: pair: enumvalue; dnnl_nwc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f756dbdc1e949646c95f83e0f51bc43:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nwc

3D CNN activations tensor, an alias to :ref:`dnnl_acb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8537ed269eb5d0586456db114039c00>`

.. index:: pair: enumvalue; dnnl_nchw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nchw

4D CNN activations tensor, an alias to :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>`

.. index:: pair: enumvalue; dnnl_nhwc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nhwc

4D CNN activations tensor, an alias to :ref:`dnnl_acdb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fcce5dd7260b5b0740e3b37b1e9ad41>`

.. index:: pair: enumvalue; dnnl_chwn
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab65a38658838cec19e718ba048cd459:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_chwn

4D CNN activations tensor, an alias to :ref:`dnnl_bcda <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac42f365ea58d5b73c141c0788641f74a>`

.. index:: pair: enumvalue; dnnl_ncdhw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ncdhw

5D CNN activations tensor, an alias to :ref:`dnnl_abcde <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da30d5d3c9de2931f06d265af81787ada3>`

.. index:: pair: enumvalue; dnnl_ndhwc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ndhwc

5D CNN activations tensor, an alias to :ref:`dnnl_acdeb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfe86402763786b9b4d73062cfd2f05>`

.. index:: pair: enumvalue; dnnl_oi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daee91ff6fadfe8c5494be1595fd253c4e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_oi

2D CNN weights tensor, an alias to :ref:`dnnl_ab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53>`

.. index:: pair: enumvalue; dnnl_io
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da46de139a50746a1a22d8d33c60d9081b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_io

2D CNN weights tensor, an alias to :ref:`dnnl_ba <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6a6dbc0b30468d92e32a9cb3f6615c43>`

.. index:: pair: enumvalue; dnnl_oiw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae6bd836b8515857585336a0921e35663:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_oiw

3D CNN weights tensor, an alias to :ref:`dnnl_abc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadff5ea69392d7e4da23179dc0ba7cbc2>`

.. index:: pair: enumvalue; dnnl_owi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da250d14a2373c43ea403098b4758152ab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_owi

3D CNN weights tensor, an alias to :ref:`dnnl_acb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8537ed269eb5d0586456db114039c00>`

.. index:: pair: enumvalue; dnnl_wio
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da93eecc25f8ab1b07604b632401aa28e5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_wio

3D CNN weights tensor, an alias to :ref:`dnnl_cba <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5a8de15eee2e5bcc2515eb7b01965789>`

.. index:: pair: enumvalue; dnnl_woi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da16fac91932b20b2f38776916a59765fb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_woi

3D CNN weights tensor, an alias to :ref:`dnnl_cab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4a42ef47bea9ef571f64944b62fddea3>`

.. index:: pair: enumvalue; dnnl_iwo
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7811cefee328570707df45965b63844e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_iwo

3D CNN weights tensor, an alias to :ref:`dnnl_bca <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da37fcc057fe063d378bb5f1d5111e0e60>`

.. index:: pair: enumvalue; dnnl_oihw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da11176ff202375dcd0d06e2fba5f8a8e0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_oihw

4D CNN weights tensor, an alias to :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>`

.. index:: pair: enumvalue; dnnl_hwio
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4f4c7bd98c6d53fb3b69e1c8df0a80f6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_hwio

4D CNN weights tensor, an alias to :ref:`dnnl_cdba <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2368e1a4ce9d9954fa10c242569eacb4>`

.. index:: pair: enumvalue; dnnl_hwoi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa7f07aa691f07f8b4d04422537b82202:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_hwoi

4D CNN weights tensor, an alias to :ref:`dnnl_cdab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e99e668dcd118f6f576c7be57d5c43d>`

.. index:: pair: enumvalue; dnnl_ohwi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7543e08a69ca4f7770124d6424314d4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ohwi

4D CNN weights tensor, an alias to :ref:`dnnl_acdb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fcce5dd7260b5b0740e3b37b1e9ad41>`

.. index:: pair: enumvalue; dnnl_ihwo
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf1345b7d3d7aef806d9716287104dc96:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ihwo

4D CNN weights tensor, an alias to :ref:`dnnl_bcda <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac42f365ea58d5b73c141c0788641f74a>`

.. index:: pair: enumvalue; dnnl_iohw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da17b2a778caa0f78f3387b09611ab9ceb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_iohw

4D CNN weights tensor, an alias to :ref:`dnnl_bacd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab1a39583c90f536df19e47d8262ef6f6>`

.. index:: pair: enumvalue; dnnl_oidhw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da18e605a5f6afe9995961fc21fddf314e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_oidhw

5D CNN weights tensor, an alias to :ref:`dnnl_abcde <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da30d5d3c9de2931f06d265af81787ada3>`

.. index:: pair: enumvalue; dnnl_iodhw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9681b8a3bb79ce73a21b5bb9c682bfde:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_iodhw

5D CNN weights tensor, an alias to :ref:`dnnl_bacde <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da57c548e503e61b8c0f9a70a0a9acce1c>`

.. index:: pair: enumvalue; dnnl_dhwio
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae4885779f955beeddc25443a3f8c2a63:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_dhwio

5D CNN weights tensor, an alias to :ref:`dnnl_cdeba <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae7dd2b81eb2b502efbf65c888665b358>`

.. index:: pair: enumvalue; dnnl_dhwoi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daba857ec496e1489ed2692f6441e7ece5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_dhwoi

5D CNN weights tensor, an alias to :ref:`dnnl_cdeab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf4870a90b840aed0c8162f0fa5822967>`

.. index:: pair: enumvalue; dnnl_odhwi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadf48b08738673a6746907b934d82248d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_odhwi

5D CNN weights tensor, an alias to :ref:`dnnl_acdeb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfe86402763786b9b4d73062cfd2f05>`

.. index:: pair: enumvalue; dnnl_idhwo
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6fdbc8d31779a5548bc39c23d41466b2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_idhwo

5D CNN weights tensor, an alias to :ref:`dnnl_bcdea <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad219c123f120d32a5f26d584b334af41>`

.. index:: pair: enumvalue; dnnl_goiw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a95a9ab4d0be153fe304d94a4f96138:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_goiw

4D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>`

.. index:: pair: enumvalue; dnnl_gowi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4ff8c44741175d152711815fb15a9da5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_gowi

4D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_abdc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad7bcb765b861ed46b2b752e912989217>`

.. index:: pair: enumvalue; dnnl_wigo
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da27409507462c2e1f45deb987623de29c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_wigo

4D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_dcab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae6c549f1a265f8d61bffb24c24925a1d>`

.. index:: pair: enumvalue; dnnl_goihw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daee332d306e012c0614dc32e0ed66072f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_goihw

5D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_abcde <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da30d5d3c9de2931f06d265af81787ada3>`

.. index:: pair: enumvalue; dnnl_gohwi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da28265c5d27d555da689f63472db852cf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_gohwi

5D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_abdec <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab417c26841486da2a0da84730d0874d>`

.. index:: pair: enumvalue; dnnl_hwigo
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da059c27962618c1f7670f5389bb7fb7ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_hwigo

5D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_decab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da31fa5f2eb4a236664583afc291310ec6>`

.. index:: pair: enumvalue; dnnl_giohw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daecd143f4e88930c11ab4348bb5c24dc4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_giohw

5D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_acbde <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac5f398cc58f5625915719ec541bb152e>`

.. index:: pair: enumvalue; dnnl_goidhw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da621b8c724d93d2a5444feab224b2ce3e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_goidhw

6D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_abcdef <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da74b2db34e6253e56f441764d16e6ec42>`

.. index:: pair: enumvalue; dnnl_godhwi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3f6895614839ab01bd55b85f81452293:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_godhwi

6D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_abdefc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad23172b9338edbd306f84f24fd7b00be>`

.. index:: pair: enumvalue; dnnl_giodhw
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da44de40f7cc576af9bff1d0c95fe6bf0f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_giodhw

6D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_acbdef <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8463916a68cb816050fbaf71817093ae>`

.. index:: pair: enumvalue; dnnl_dhwigo
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da81de66a3883f148984ed2419c794d5ce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_dhwigo

6D CNN weights tensor (incl. groups), an alias to :ref:`dnnl_defcab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da73502e9308b365e74da295eaa83703ad>`

.. index:: pair: enumvalue; dnnl_tnc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2a9735ec024c9362b717304edbfe2237:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_tnc

3D RNN data tensor in the format (seq_length, batch, input channels), an alias to :ref:`dnnl_abc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadff5ea69392d7e4da23179dc0ba7cbc2>`.

.. index:: pair: enumvalue; dnnl_ntc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5d73ca7a68559ef44241be5a096e6bff:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ntc

3D RNN data tensor in the format (batch, seq_length, input channels), an alias to :ref:`dnnl_bac <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae31cc3d3a0bfbd4936f7b503cf6dc9de>`.

.. index:: pair: enumvalue; dnnl_ldnc
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4fd1cf9fdb67c554bcd8281695b65b3c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ldnc

4D RNN states tensor in the format (num_layers, num_directions, batch, state channels), an alias to :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>`.

.. index:: pair: enumvalue; dnnl_ldigo
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da96a227ab1a1be1825c1fa596c38847fc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ldigo

5D RNN weights tensor in the format (num_layers, num_directions, input_channels, num_gates, output_channels), an alias to :ref:`dnnl_abcde <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da30d5d3c9de2931f06d265af81787ada3>`.

* For LSTM cells, the gates order is input, forget, candidate and output gate.

* For GRU cells, the gates order is update, reset and output gate.

.. index:: pair: enumvalue; dnnl_ldgoi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5fc9120d8f52d7d7fa853aa79bf654fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ldgoi

5D RNN weights tensor in the format (num_layers, num_directions, num_gates, output_channels, input_channels), an alias to :ref:`dnnl_abdec <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daab417c26841486da2a0da84730d0874d>`.

* For LSTM cells, the gates order is input, forget, candidate and output gate.

* For GRU cells, the gates order is update, reset and output gate.

.. index:: pair: enumvalue; dnnl_ldio
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2d5a8e24d6b4904b4e8986d9b0fb4613:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ldio

4D LSTM projection tensor in the format (num_layers, num_directions, num_channels_in_hidden_state, num_channels_in_recurrent_projection), an alias to :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>`.

.. index:: pair: enumvalue; dnnl_ldoi
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da475da8ead8c761bac894e6c87042355d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ldoi

4D LSTM projection tensor in the format (num_layers, num_directions, num_channels_in_recurrent_projection, num_channels_in_hidden_state), an alias to :ref:`dnnl_abdc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad7bcb765b861ed46b2b752e912989217>`.

.. index:: pair: enumvalue; dnnl_ldgo
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2bc162b37fd0049dceab3b12300a26c7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ldgo

4D RNN bias tensor in the format (num_layers, num_directions, num_gates, output_channels), an alias to :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>`.

* For LSTM cells, the gates order is input, forget, candidate and output gate.

* For GRU cells, the gates order is update, reset and output gate.

.. index:: pair: enumvalue; dnnl_ldOi16o
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8d352f148c426ccaa7dae070f50f24b4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ldOi16o

5D LSTM projection tensor

.. index:: pair: enumvalue; dnnl_ldgOi16o
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4017f8ca50acb2e22e3e5a31e4aae0ee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ldgOi16o

6D RNN weights tensor

.. index:: pair: enumvalue; dnnl_nCdhw32c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da773408e10d72e338fb417ba0634a7169:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nCdhw32c

5D CNN activations tensor blocked by channels with block size 32, an alias to :ref:`dnnl_aBcde32b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabdf6daaaaa7e79356fb5b4ec7ae90e5a>`

.. index:: pair: enumvalue; dnnl_nCdhw16c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2fd2bec64ddb6031730c180acf1a894f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nCdhw16c

5D CNN activations tensor blocked by channels with block size 16, an alias to :ref:`dnnl_aBcde16b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad4357b1ef924c1331e9e7e25facc3827>`

.. index:: pair: enumvalue; dnnl_nCdhw4c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da72cb744b96d7a9feb19fbc488dfcdc90:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nCdhw4c

5D CNN activations tensor blocked by channels with block size 4, an alias to :ref:`dnnl_aBcde4b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3b841f4fc9008e7a27f4dcc4b130d819>`

.. index:: pair: enumvalue; dnnl_nCdhw8c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabacffa20b5188cda4d5f86e2e10d2572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nCdhw8c

5D CNN activations tensor blocked by channels with block size 8, an alias to :ref:`dnnl_aBcde8b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa778e377f1b0afdee9a6bcbc63ae76bd>`

.. index:: pair: enumvalue; dnnl_nChw32c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da18b4ebf475d956c82508de954e38adf2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nChw32c

4D CNN activations tensor blocked by channels with block size 32, an alias to :ref:`dnnl_aBcd32b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadaa7aeb5746c398a1c8ee1b42362bc58>`

.. index:: pair: enumvalue; dnnl_nChw16c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa7847819b4fb840d2db20796bc607a5c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nChw16c

4D CNN activations tensor blocked by channels with block size 16, an alias to :ref:`dnnl_aBcd16b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da590d748ac3761a51b407555a9a0d5efe>`

.. index:: pair: enumvalue; dnnl_nChw4c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da80c1478a2fb2bb2095d1866e1d04908e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nChw4c

4D CNN activations tensor blocked by channels with block size 4, an alias to :ref:`dnnl_aBcd4b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2ec2a0d2900e5ddb7f88783e2d8a5e66>`

.. index:: pair: enumvalue; dnnl_nChw8c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da775389dbdcda91ea12906210c309746f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nChw8c

4D CNN activations tensor blocked by channels with block size 8, an alias to :ref:`dnnl_aBcd8b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9523ccda38e86a8f7f58f8f780ef6eac>`

.. index:: pair: enumvalue; dnnl_nCw32c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5b32114a6ae17691ce4c81869d6aebd8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nCw32c

3D CNN activations tensor blocked by channels with block size 32, an alias to :ref:`dnnl_aBc32b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8c6671d24cda4d6ea0fa7d8613f686c>`

.. index:: pair: enumvalue; dnnl_nCw16c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf996bf5a22fc850b0d4cf5d2533aa134:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nCw16c

3D CNN activations tensor blocked by channels with block size 16, an alias to :ref:`dnnl_aBc16b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da3e2da6258e387e0eb149b22f57209056>`

.. index:: pair: enumvalue; dnnl_nCw4c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa487c7e7aad196e5dbe2d1b22da5b045:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nCw4c

3D CNN activations tensor blocked by channels with block size 4, an alias to :ref:`dnnl_aBc4b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4df9991897c2031b5fbdf97a4b3d2a02>`

.. index:: pair: enumvalue; dnnl_nCw8c
.. _doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dad691bfd0677cab219ce336afa6023a59:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_nCw8c

3D CNN activations tensor blocked by channels with block size 8, an alias to :ref:`dnnl_aBc8b <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da96fcac28a789530765def2cfae2b61f6>`

