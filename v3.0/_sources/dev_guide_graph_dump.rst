.. index:: pair: page; Graph Dump
.. _doxid-dev_guide_graph_dump:

Graph Dump
==========

oneDNN supports dumping the computation graph constructed with graph API. The dumped graph can be used for visualization and performance benchmark.

Build-Time Controls
~~~~~~~~~~~~~~~~~~~

The graph dumping feature only works when ``ONEDNN_BUILD_GRAPH`` is ON.

=========================  ====================================  ====================================================================================  
CMake Option               Supported values (defaults in bold)   Desc                                                                                  
=========================  ====================================  ====================================================================================  
ONEDNN_ENABLE_GRAPH_DUMP   ON, **OFF**                           Controls dumping ( :ref:`Graph Dump <doxid-dev_guide_graph_dump>` ) graph artifacts   
=========================  ====================================  ====================================================================================

Run-Time Controls
~~~~~~~~~~~~~~~~~

When the feature is enabled at build time, the environment variable ``ONEDNN_GRAPH_DUMP`` can be used to control the serialization level. This option accepts setting flags. These flags can be combined together to make the library dumping different files. For example, the below setting will generate files containing library graph and subgraphs in each partition.

==================  ==========  ==========================================  
Variable            Flags       Des                                         
==================  ==========  ==========================================  
ONEDNN_GRAPH_DUMP   (default)   No graph or subgraph dumped                 
                    graph       Library graph                               
                    subgraph    Library subgraph contained in a partition   
==================  ==========  ==========================================

.. ref-code-block:: cpp

	ONEDNN_GRAPH_DUMP=graph,subgraph ./application

This may produce graph JSON files as follows:

.. ref-code-block:: cpp

	onednn_graph_verbose,info,serialize graph to a json file graph-100001.json
	onednn_graph_verbose,info,serialize graph to a json file graph-100001-partitioning.json
	onednn_graph_verbose,info,serialize graph to a json file graph-100002-1313609102600373579.json
	onednn_graph_verbose,info,serialize graph to a json file graph-100003-12829238476173481280.json

