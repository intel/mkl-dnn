.. index:: pair: page; Host-Side Scalars Support
.. _doxid-dev_guide_host_side_scalars:

Host-Side Scalars Support
=========================

oneDNN supports a special memory object for host-side scalar values. Creating such an object is lightweight and does not require specifying an engine.

To create a host-side scalar memory object, first create a memory descriptor with the scalar data type. Then, use this descriptor and a scalar value to create the memory object. The scalar value is copied into the memory object, so its lifetime does not need to be managed by the user.

Using the C++ API:

.. ref-code-block:: cpp

	float alpha = 1.0f;
	
	// Create a memory object for a host scalar of type float
	:ref:`dnnl::memory <doxid-structdnnl_1_1memory>` alpha_mem(memory::desc::host_scalar(memory::data_type::f32), alpha);

Using the C API:

.. ref-code-block:: cpp

	float alpha = 1.0f;
	
	// Create a memory descriptor for a host scalar of type float
	:ref:`dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` alpha_md;
	:ref:`dnnl_memory_desc_create_host_scalar <doxid-group__dnnl__api__memory_1gae52f3003429b2a60bd15b710977fe361>`(&alpha_md, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`);
	
	// Create a memory object for the scalar
	:ref:`dnnl_memory_t <doxid-structdnnl__memory>` alpha_mem;
	:ref:`dnnl_memory_create_host_scalar <doxid-group__dnnl__api__memory_1gabe53ba0e17bcb92509c3bbbd250e520b>`(&alpha_mem, alpha_md, &alpha);

The memory object can then be used in primitives just like any other memory object.

If at any point the user needs to access or update the scalar value, they can do so using :ref:`dnnl_memory_get_host_scalar_value <doxid-group__dnnl__api__memory_1ga1c9c615901b76e4493d7c9c8f299e7ed>` and :ref:`dnnl_memory_set_host_scalar_value <doxid-group__dnnl__api__memory_1ga6be28e263bcee8d85684ed218902f995>`, or using the C++ API:

.. ref-code-block:: cpp

	// Get the scalar value from the memory object
	float alpha_value = alpha_mem.get_host_scalar_value();
	
	float new_alpha_value = 2.0f;
	// Update the scalar value in the memory object
	alpha_mem.set_host_scalar_value(&new_alpha_value);

