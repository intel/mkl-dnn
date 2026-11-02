.. index:: pair: page; Constant Tensor Cache
.. _doxid-dev_guide_constant_tensor_cache:

Constant Tensor Cache
=====================

The oneDNN Graph component supports the constant tensor cache feature, which is used to cache processed constant tensors such as reordered constant weights and folded constant scales to reduce redundant computation and improve performance. The feature is disabled by default. Users can use the graph API or environment variable to set or get specific cache capacity for different engine kinds (CPU and GPU).

Build-Time Controls
~~~~~~~~~~~~~~~~~~~

Build-time controls to enable or disable the constant tensor cache feature are not supported. Only run-time controls through the graph API or environment variables are supported. Refer to the following section.

Run-Time Controls
~~~~~~~~~~~~~~~~~

Constant Tensor Cache Capacity Control API
------------------------------------------

oneDNN Graph provides users with a pair of APIs to control the constant tensor cache feature. To enable the constant tensor cache and set the capacity to a specific engine kind, call the ``setter`` API. The unit of ``setter`` capacity API is megabytes (MB). New tensors won't be cached when capacity is reached. To query the current capacity for a specific engine kind, call the ``getter`` API.

.. ref-code-block:: cpp

	// setter API
	@ref :ref:`dnnl_graph_set_constant_tensor_cache_capacity <doxid-group__dnnl__graph__api__constant__tensor__cache_1gac9088e72c59a66d02c68e7200ae59b2d>`
	
	// getter API
	@ref :ref:`dnnl_graph_get_constant_tensor_cache_capacity <doxid-group__dnnl__graph__api__constant__tensor__cache_1gad6822ef150b8de02dec49ffd49a83f87>`

Environment Variable
--------------------

In addition to a programmable API, oneDNN Graph also provides users with an environment variable named ``ONEDNN_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY`` to control the capacity. It accepts values in the form ``engine_kind:size`` or ``engine_kind1:size1;engine_kind2:size2``. The first example below means the user can set capacity for one engine kind (``cpu``). The second example is that the capacity of ``cpu`` and ``gpu`` are set to 1024 MB and 2048 MB separately.

============================================  ======================  ===============================================================  
Environment variable                          Value(string)           Description                                                      
============================================  ======================  ===============================================================  
ONEDNN_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY   "cpu:size1;gpu:size2"   Set cpu constant cache capacity size to size1 and gpu to size2   
============================================  ======================  ===============================================================

.. ref-code-block:: cpp

	export ONEDNN_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY="cpu:1024"
	export ONEDNN_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY="cpu:1024;gpu:2048"

.. note:: 

   The environment variable API should be set only once before the application starts; the library will read the variable and cache it inside to reduce string parsing overhead. Re-setting the environment variable at runtime will not take effect. Functional APIs have higher priority than environment variables. If users call the functional APIs, it will overwrite the capacity values specified through the environment variable.

