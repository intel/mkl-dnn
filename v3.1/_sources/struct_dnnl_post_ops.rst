.. index:: pair: struct; dnnl_post_ops
.. _doxid-structdnnl__post__ops:

struct dnnl_post_ops
====================

.. toctree::
	:hidden:

Overview
~~~~~~~~

An opaque structure for a chain of post operations. :ref:`More...<details-structdnnl__post__ops>`

.. _details-structdnnl__post__ops:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

An opaque structure for a chain of post operations.

:ref:`dnnl_post_ops <doxid-structdnnl__post__ops>` can be used to perform some (trivial) operations like accumulation or eltwise after certain primitives like convolution.

Post operations might be combined together, making a chain of post operations. For instance one can configure convolution followed by accumulation followed by eltwise. This might be especially beneficial for residual learning blocks.

.. warning:: 

   Of course not all combinations are supported, so the user should handle errors accordingly.
   
   
Supported post operations:

* accumulation (base primitive: convolution)

* eltwise (base primitive: convolution)

