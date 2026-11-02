.. index:: pair: enum; dnnl_graph_partition_policy_t
.. _doxid-group__dnnl__graph__api__partition_1ga7e24b277b64600ef3a83dac2e8dfa83b:

enum dnnl_graph_partition_policy_t
==================================

Overview
~~~~~~~~

Policy specifications for partitioning. :ref:`More...<details-group__dnnl__graph__api__partition_1ga7e24b277b64600ef3a83dac2e8dfa83b>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph_types.h>

	enum dnnl_graph_partition_policy_t
	{
	    :ref:`dnnl_graph_partition_policy_fusion<doxid-group__dnnl__graph__api__partition_1gga7e24b277b64600ef3a83dac2e8dfa83ba1489f3f9bb1262b58fa0bf0524d883a4>` = 1,
	    :ref:`dnnl_graph_partition_policy_debug<doxid-group__dnnl__graph__api__partition_1gga7e24b277b64600ef3a83dac2e8dfa83ba5f841a05c0fc7df9bc023359f49ca3a0>`  = 2,
	};

.. _details-group__dnnl__graph__api__partition_1ga7e24b277b64600ef3a83dac2e8dfa83b:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Policy specifications for partitioning.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_graph_partition_policy_fusion
.. _doxid-group__dnnl__graph__api__partition_1gga7e24b277b64600ef3a83dac2e8dfa83ba1489f3f9bb1262b58fa0bf0524d883a4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_partition_policy_fusion

Fusion policy returns partitions with typical post-op fusions, eg.

Convolution + ReLU or other element-wise operations or a chian of post-ops.

.. index:: pair: enumvalue; dnnl_graph_partition_policy_debug
.. _doxid-group__dnnl__graph__api__partition_1gga7e24b277b64600ef3a83dac2e8dfa83ba5f841a05c0fc7df9bc023359f49ca3a0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_partition_policy_debug

Debug policy doesn't not apply any fusions.

It returns partitions with single operation in each partition. The policy is useful when users notice any bug or correctness issue in fusion policy.

