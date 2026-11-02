.. index:: pair: page; Basic Concepts
.. _doxid-dev_guide_graph_basic_concepts:

Basic Concepts
==============

Introduction
~~~~~~~~~~~~

In oneDNN Graph API programming model, a computation graph is passed to library and then optimized sub-graphs which are called ``partitions`` are returned by the library. ``Partition`` is decided by oneDNN Graph API implementation. It is the key concept to satisfy the different needs of AI hardware classes by using a unified API. Typically can compile ``partitions``, bind ``tensor`` data, and execute ``compiled partitions``.

The key concepts in oneDNN Graph API include ``logical tensor``, ``op``, ``graph``, ``partition``, ``compiled partition``, and ``tensor``. Here is the relationship between these entities. Besides, oneDNN Graph API shares the common ``engine`` and ``stream`` concepts of oneDNN primitive API.

Check the API documentation for detailed usage of each API concept.

.. image:: img_graph_programming_model.png
	:alt: Figure 1: Overview of Graph API programming model. Blue rectangles denote oneDNN objects, and red lines denote dependencies between objects.



Logical Tensor
~~~~~~~~~~~~~~

``Logical tensor`` (:ref:`dnnl::graph::logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>`) describes the metadata of the input and output tensors, like data type, number of dimensions, size for each dimension, tensor layout and property. Each logical tensor has a unique ID which is immutable during the lifetime of a logical tensor. Shape information for the input logical tensor will be required at the partition compilation stage. Logical tensor is not mutable. Users must create a new logical tensor with the same ID to pass any new additional information to oneDNN Graph API.

Op
~~

``Op`` (:ref:`dnnl::graph::op <doxid-classdnnl_1_1graph_1_1op>`) represents an operation as part of a computation graph. An operation has kind, attribute, and input and output logical tensors. Operations are added into a graph object to construct a computation graph. As both operation and logical tensor contain a unique ID, the graph object knows how to connect a producer operation to a consumer operation through a logical tensor as the edge between them.

Graph
~~~~~

``Graph`` (:ref:`dnnl::graph::graph <doxid-classdnnl_1_1graph_1_1graph>`) contains a set of operations. A graph object is associated to a specific engine kind (:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>`). Multiple operations can be added (:ref:`dnnl::graph::graph::add_op <doxid-classdnnl_1_1graph_1_1graph_1a1cdf41276f953ecc482df858408c9ff0>`) along with input and output logical tensors to a graph. After finishing adding operations, finalization API (:ref:`dnnl::graph::graph::finalize <doxid-classdnnl_1_1graph_1_1graph_1a4454e3093a112022cd607c4c3cc66ee2>`) can be called to indicate that the graph is ready for partitioning. By calling partitioning API (:ref:`dnnl::graph::graph::get_partitions <doxid-classdnnl_1_1graph_1_1graph_1a116d3552e3b0e6c739a1564329bde014>`), a group of partitions from the graph will be returned .

Partition
~~~~~~~~~

``Partition`` (:ref:`dnnl::graph::partition <doxid-classdnnl_1_1graph_1_1partition>`) represents a collection of operations identified by library implementation as the basic unit for compilation and execution. A partition is a connected subgraph within the source graph. The partitions returned from the library must not form a dependence cycle.

A partition needs to be compiled (:ref:`dnnl::graph::partition::compile <doxid-classdnnl_1_1graph_1_1partition_1a5c2af93c65a09c9d0a1507571ada0318>`) before execution. The compilation lowers down the computation logic to hardware ISA level and generates binary code. The generated code is specialized for the input and output logical tensors and engine (:ref:`dnnl::engine <doxid-structdnnl_1_1engine>`).

The output logical tensors can have unknown dimensions during compilation. In this case, the compilation procedure should deduce the output shapes according to the input shapes and will return an error if the output shapes cannot be deduced deterministically. The input logical tensors should have either the ``strided`` or ``opaque`` layout type (:ref:`dnnl::graph::logical_tensor::layout_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>`). Additionally, the output logical tensors can have layout type ``any``. It means that the compilation procedure can choose the optimal layouts for the output tensors. Optimal layouts are represented as opaque layout IDs and saved in the corresponding output logical tensors.

A partition may contains many logical tensors with part of them are internal intermediate results connecting two operations inside the partition. The required inputs and outputs of a partition are also called ``ports`` of a partition. Two APIs ``get_input_ports`` (:ref:`dnnl::graph::partition::get_input_ports <doxid-classdnnl_1_1graph_1_1partition_1a415319dcb89d9e1d77bd4b7b0058df52>`) and ``get_output_ports`` (:ref:`dnnl::graph::partition::get_output_ports <doxid-classdnnl_1_1graph_1_1partition_1aaa4abecc6e09f417742402ab207a1e6d>`) are provided to query the ports and help understand which input logical tensors and output logical tensors are needed to compile a partition. The input logical tensors and output logical tensors must match IDs with ports. These in ports and out ports can also be used to track the producer and consumer of a partitions through logical tensor IDs and for framework integration, connect the partition back to the framework graph as a custom node.

Compiled Partition
~~~~~~~~~~~~~~~~~~

``Compiled partition`` (:ref:`dnnl::graph::compiled_partition <doxid-classdnnl_1_1graph_1_1compiled__partition>`) represents the generated code specialized for a target hardware and tensor metadata passed through compilation API. To execute a compiled partition (:ref:`dnnl::graph::compiled_partition::execute <doxid-classdnnl_1_1graph_1_1compiled__partition_1a558ed47b3cbc5cc2167001da3faa0339>`), both input and output tensors, and a stream (:ref:`dnnl::stream <doxid-structdnnl_1_1stream>`) are required to pass. Input and output tensors must bind data buffers to the input and output logical tensors respectively.

An API (:ref:`dnnl::graph::compiled_partition::query_logical_tensor <doxid-classdnnl_1_1graph_1_1compiled__partition_1a85962826e94cc3cefb3c19c0fadc4e09>`) is provided to query output logical tensors from a compiled partition. It allows to know the output layout and memory size (:ref:`dnnl::graph::logical_tensor::get_mem_size <doxid-classdnnl_1_1graph_1_1logical__tensor_1a12b73d1201259d4260de5603f62c7f15>`) when they specify output logical tensor with ``any`` layout type during compilation.

Tensor
~~~~~~

``Tensor`` (:ref:`dnnl::graph::tensor <doxid-classdnnl_1_1graph_1_1tensor>`) is an abstraction for multi-dimensional input and output data which is needed in the execution of a compiled partition. A tensor contains a logical tensor, an engine (:ref:`dnnl::engine <doxid-structdnnl_1_1engine>`), and a data handle. The application is responsible for managing the data handle's lifecycle, for example free the memory resource when it is not used anymore.

