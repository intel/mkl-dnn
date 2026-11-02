.. index:: pair: enum; policy
.. _doxid-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3c:

enum dnnl::graph::partition::policy
===================================

Overview
~~~~~~~~

Policy specifications for partitioning. :ref:`More...<details-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3c>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>

	enum policy
	{
	    :ref:`fusion<doxid-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3ca051de32597041e41f73b97d61c67a13b>` = dnnl_graph_partition_policy_fusion,
	    :ref:`debug<doxid-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3caad42f6697b035b7580e4fef93be20b4d>`  = dnnl_graph_partition_policy_debug,
	};

.. _details-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3c:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Policy specifications for partitioning.

Enum Values
-----------

.. index:: pair: enumvalue; fusion
.. _doxid-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3ca051de32597041e41f73b97d61c67a13b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	fusion

Fusion policy returns partitions with typical post-op fusions, eg.

Convolution + ReLU or other element-wise operations or a chian of post-ops.

.. index:: pair: enumvalue; debug
.. _doxid-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3caad42f6697b035b7580e4fef93be20b4d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	debug

Debug policy doesn't not apply any fusions.

It returns partitions with single operations in each partition. The policy is useful when users notice any bug or correctness issue in fusion policy.

