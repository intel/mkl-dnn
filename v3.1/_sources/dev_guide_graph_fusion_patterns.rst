.. index:: pair: page; Supported Fusion Patterns
.. _doxid-dev_guide_graph_fusion_patterns:

Supported Fusion Patterns
=========================

Fusion Patterns
~~~~~~~~~~~~~~~

The following fusion patterns are subgraphs that the oneDNN Graph API recognizes as candidate for fusion. The patterns are described using oneDNN Graph operation (op) names with the following convention.

.. note:: 

   oneDNN Graph performs limited input validation to minimize the performance overheads. The application is responsible for sanitizing inputs passed to the library. For large u8 or s8 inputs may lead to accumulator overflow, you can use floating point patterns instead of quantized patterns.
   
   
``"+"`` describes a chain of two ops. The preceding op produces an output tensor, which is consumed by the following op as its first operand.

``"[]"`` describes a component of the overall pattern description. For example, it could include a subgraph or all the op choices within the bracket.

``"|"`` describes choices of multiple operations, say A+[B\|C] means the graph partition contains A followed by B or C.

``","`` describes a graph composed of multiple subgraphs, each subgraph marks its output tensor explicitly, which is consumed by other subgraphs.

``Superscript`` denotes the numbers of repetition pattern. For example, A+[B\|C] :math:`^{3}` means the graph partition contains A followed by three ops, each of them is either B or C. The superscript could be a range of number meaning allowing a range of repetition. If the range is between 0 and 1, we use superscript ``"?"``.

``Subscript`` denotes the input and output tensors which need to explicitly mark the producer and consumer relation within one graph partition. For example, A :math:`_{>t1}` +B+C :math:`_{<t1}` refers to the pattern started with A followed by B and C, and C takes an implicit input tensor from B and an extra tensor t1 output from A. ``">"`` refers to the output tensor, and ``"<"`` for input tensor. Input and output tensor between neighbor ops are not explicitly marked, for example, B consumes t1 implicitly in the example above.

Subscript ``"out"`` marks the output tensor of a certain op to be the output of a graph partition. For example, in A :math:`_{>t1}` +B :math:`_{>out}` +C :math:`_{<t1,>out}`, B's output and C's output are marked as output tensors.

Subscript ``"in"`` marks the input tensor of a certain op to be the input of a graph partition. For example, in A :math:`_{<in1}` +B :math:`_{<in1}` A's input and B's second input are graph partition input, and they share the same input tensor in1. Most input tensors of a graph partition are not explicitly marked. For example, the input tensors of the first op are implicitly regarded as graph partition inputs. Besides, for input tensors of other ops, if they are not produced by any proceeding ops, they are regarded as implicit graph partition inputs. In the example A :math:`_{>t1}` +B+C :math:`_{<t1}`, A's inputs are regarded as implicit graph partition inputs, and if B is a binary operation, the second input tensor is an implicit graph partition input.

The following categories will be used in describing fusion pattern.

Unary = [Abs \| Clamp \| Elu \| Exp \| GELU \| HardSwish \| LeakyReLU \| Log \| Sigmoid \| SoftPlus \| Pow \| ReLU \| Round \| Sqrt \| Square \| Tanh]

Binary = [Add \| Divide \| Maximum \| Minimum \| Multiply \| Subtract]

Reduction = [ReduceL1 \| ReduceL2 \| ReduceMax \| ReduceMean \| ReduceMin \| ReduceProd \| ReduceSum]

Inference
---------

Floating Point Patterns
+++++++++++++++++++++++

===================================================================================================================  =======================================================================================================  
Pattern                                                                                                                                                                                                                       
===================================================================================================================  =======================================================================================================  
Convolution + BiasAdd :math:`^?` + BatchNormInference :math:`^?` + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`   This pattern is widely used in Convolution Neural Networks, for example ResNet, ResNext, SSD, etc.       
ConvTranspose + BiasAdd :math:`^?` + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`                                 This pattern is widely used in Generative Adversarial Networks.                                          
Interpolate + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`                                                        This pattern is widely used for image processing.                                                        
MatMul + BiasAdd :math:`^?` + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`                                        This pattern is widely used in language models and recommendation models, for example BERT, DLRM, etc.   
Reduction + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`                                                          This pattern is widely used for data processing, for example loss reduction.                             
Unary + Binary :math:`^{0-3}` :math:`_{>out}`                                                                        This pattern is widely used in Convolution Neural Networks.                                              
Binary + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`                                                             This pattern is widely used in Generative Adversarial Networks, for example ParallelWaveGAN.             
[AvgPool | MaxPool] + Binary :math:`^{0-3}` :math:`_{>out}`                                                          This pattern is widely used in Convolution Neural Networks.                                              
BatchNormInference + ReLU :math:`_{>out}`                                                                            This pattern is widely used in Convolution Neural Networks, for example DenseNet.                        
Reciprocal + Multiply :math:`_{>out}`                                                                                N/A                                                                                                      
Reorder + Add :math:`_{>out}`                                                                                        N/A                                                                                                      
===================================================================================================================  =======================================================================================================

Quantized Patterns
++++++++++++++++++

====================================================================================================================================================================================================================================================  ====  
Pattern                                                                                                                                                                                                                                                     
====================================================================================================================================================================================================================================================  ====  
Quantize :math:`^?` + Dequantize :math:`_{>t1}` , Dequantize :math:`_{>t2}` :math:`^{0-3}` , Dequantize + Convolution :math:`_{<t1}` + BiasAdd :math:`^?` + [Unary | Binary :math:`_{<t2}` ] :math:`^{0-3}` + Quantize :math:`^?` :math:`_{>out}`     N/A   
Quantize :math:`^?` + Dequantize :math:`_{>t1}` , Dequantize :math:`_{>t2}` :math:`^{0-3}` , Dequantize + ConvTranspose :math:`_{<t1}` + BiasAdd :math:`^?` + [Unary | Binary :math:`_{<t2}` ] :math:`^{0-3}` + Quantize :math:`^?` :math:`_{>out}`   N/A   
Quantize :math:`^?` + Dequantize :math:`_{>t1}` , Dequantize :math:`_{>t2}` :math:`^{0-3}` , Dequantize + MatMul :math:`_{<t1}` + BiasAdd :math:`^?` + [Unary | Binary :math:`_{<t2}` ] :math:`^{0-3}` + Quantize :math:`^?` :math:`_{>out}`          N/A   
Dequantize + [AvgPool | MaxPool] + Quantize :math:`_{>out}`                                                                                                                                                                                           N/A   
Dequantize :math:`_{>t1}` , Dequantize + [AvgPool | MaxPool] + Add :math:`_{<t1}` + Quantize :math:`_{>out}`                                                                                                                                          N/A   
Dequantize + Reorder + Quantize :math:`_{>out}`                                                                                                                                                                                                       N/A   
Dequantize :math:`_{>t1}` , Dequantize + Reorder + Add :math:`_{<t1}` + Quantize :math:`_{>out}`                                                                                                                                                      N/A   
====================================================================================================================================================================================================================================================  ====

Training
++++++++

=============================================================  ====  
Pattern                                                              
=============================================================  ====  
ConvolutionBackwardWeights + BiasAddBackward :math:`_{>out}`   N/A   
ReLUBackward + BatchNormTrainingBackward :math:`_{>out}`       N/A   
=============================================================  ====

All the above fusion patterns are supported by default.

