.. index:: pair: page; Supported Fusion Patterns
.. _doxid-dev_guide_graph_fusion_patterns:

Supported Fusion Patterns
=========================

:target:`doxid-dev_guide_graph_fusion_patterns_1fusion_patterns`

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
Pattern                                                                                                              Description                                                                                              
===================================================================================================================  =======================================================================================================  
Convolution + BiasAdd :math:`^?` + BatchNormInference :math:`^?` + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`   This pattern is widely used in Convolution Neural Networks, for example ResNet, ResNext, SSD, etc.       
ConvTranspose + BiasAdd :math:`^?` + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`                                 This pattern is widely used in Generative Adversarial Networks.                                          
Interpolate + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`                                                        This pattern is widely used for image processing.                                                        
MatMul + BiasAdd :math:`^?` + [Unary | Binary] :math:`^{0-3}` + Select :math:`^?` :math:`_{>out}`                    This pattern is widely used in language models and recommendation models, for example BERT, DLRM, etc.   
Reduction + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`                                                          This pattern is widely used for data processing, for example loss reduction.                             
Unary + Binary :math:`^{0-3}` :math:`_{>out}`                                                                        This pattern is widely used in Convolution Neural Networks.                                              
Binary + [Unary | Binary] :math:`^{0-3}` :math:`_{>out}`                                                             This pattern is widely used in Generative Adversarial Networks, for example ParallelWaveGAN.             
[AvgPool | MaxPool] + Binary :math:`^{0-3}` :math:`_{>out}`                                                          This pattern is widely used in Convolution Neural Networks.                                              
BatchNormInference + ReLU :math:`_{>out}`                                                                            This pattern is widely used in Convolution Neural Networks, for example DenseNet.                        
Reciprocal + Multiply :math:`_{>out}`                                                                                N/A                                                                                                      
Reorder + Add :math:`_{>out}`                                                                                        N/A                                                                                                      
Scaled Dot-Product Attention                                                                                         Refer to :ref:`Scaled Dot-Product Attention (SDPA) <doxid-dev_guide_graph_sdpa>` for more details.       
===================================================================================================================  =======================================================================================================

Quantized Patterns
++++++++++++++++++

=================================================================================================================================================================================================================================================================  =========================================================================================  
Pattern                                                                                                                                                                                                                                                            Description                                                                                
=================================================================================================================================================================================================================================================================  =========================================================================================  
Quantize :math:`^?` + Dequantize :math:`_{>t1}` , Dequantize :math:`_{>t2}` :math:`^{0-3}` , Dequantize + Convolution :math:`_{<t1}` + BiasAdd :math:`^?` + [Unary | Binary :math:`_{<t2}` ] :math:`^{0-3}` + Quantize :math:`^?` :math:`_{>out}`                  N/A                                                                                        
Quantize :math:`^?` + Dequantize :math:`_{>t1}` , Dequantize :math:`_{>t2}` :math:`^{0-3}` , Dequantize + ConvTranspose :math:`_{<t1}` + BiasAdd :math:`^?` + [Unary | Binary :math:`_{<t2}` ] :math:`^{0-3}` + Quantize :math:`^?` :math:`_{>out}`                N/A                                                                                        
Quantize :math:`^?` + Dequantize :math:`_{>t1}` , Dequantize :math:`_{>t2}` :math:`^{0-3}` , Dequantize + MatMul :math:`_{<t1}` + BiasAdd :math:`^?` + [Unary | Binary :math:`_{<t2}` ] :math:`^{0-3}` + Select :math:`^?` + Quantize :math:`^?` :math:`_{>out}`   N/A                                                                                        
Dequantize + [AvgPool | MaxPool] + Quantize :math:`_{>out}`                                                                                                                                                                                                        N/A                                                                                        
Dequantize :math:`_{>t1}` , Dequantize + [AvgPool | MaxPool] + Add :math:`_{<t1}` + Quantize :math:`_{>out}`                                                                                                                                                       N/A                                                                                        
Dequantize + Reorder + Quantize :math:`_{>out}`                                                                                                                                                                                                                    N/A                                                                                        
Dequantize :math:`_{>t1}` , Dequantize + Reorder + Add :math:`_{<t1}` + Quantize :math:`_{>out}`                                                                                                                                                                   N/A                                                                                        
[SoftMax | LayerNorm | GroupNorm] + [Unary | Binary :math:`_{<t2}` ] :math:`^{0-3}` + Quantize :math:`^?` :math:`_{>out}`                                                                                                                                          This pattern is used in SmoothQuant to fuse scales and quantization into previous layers   
=================================================================================================================================================================================================================================================================  =========================================================================================

Training
--------

=============================================================  ============  
Pattern                                                        Description   
=============================================================  ============  
ConvolutionBackwardWeights + BiasAddBackward :math:`_{>out}`   N/A           
ReLUBackward + BatchNormTrainingBackward :math:`_{>out}`       N/A           
=============================================================  ============

All the above fusion patterns are supported by default.

Aggressive Fusion Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggressive fusion patterns also follow the pattern description convention defined in the :ref:`Fusion Patterns <doxid-dev_guide_graph_fusion_patterns_1fusion_patterns>` section.

.. note:: 

   Aggressive fusion patterns are only supported when :ref:`Graph Compiler <doxid-dev_guide_graph_compiler>` is enabled.
   
   
The following categories will also be used to describe aggressive fusion patterns.

* ReshapeTranspose = [StaticReshape + StaticTranspose :math:`^{1-2}`]

* Activation = [ReLU \| Sigmoid \| GELU]

* ActivationBackward = [ReLUBackward \| SigmoidBackward \| GELUBackward]

Inference
---------

Floating Point Patterns
+++++++++++++++++++++++

=============================================================================================================================================================================================================================  ==========================================================================================================================================================  
Pattern                                                                                                                                                                                                                        Description                                                                                                                                                 
=============================================================================================================================================================================================================================  ==========================================================================================================================================================  
MatMul + [Multiply | Divide] + Add + Softmax + MatMul + StaticTranspose + Reorder :math:`_{>out}`                                                                                                                              Multi-head Attention. This pattern is widely used in models containing encoder-decoder structures, for example BERT.                                        
ReshapeTranspose :math:`_{>t1}` , ReshapeTranspose :math:`_{>t2}` , ReshapeTranspose + MatMul :math:`_{<t1}` + [Multiply | Divide] + Add + Softmax + MatMul :math:`_{<t2}` + StaticTranspose + StaticReshape :math:`_{>out}`   Multi-head Attention.                                                                                                                                       
MatMul + Activation :math:`_{>t1}` , [MatMul :math:`_{<t1}` + Activation :math:`_{>t1}` ] :math:`^{0-4}` , MatMul :math:`_{<t1}` + Activation :math:`_{>out}`                                                                  Multi-layer Perceptron. This pattern is widely used in recommendation models, for example DLRM.                                                             
[Convolution + BiasAdd :math:`^{?}` + ReLU] :math:`^{1-3}` + Convolution + BiasAdd :math:`^{?}` + Add + ReLU :math:`_{>out}`                                                                                                   Identical Bottleneck. Enabled only in single thread runtime scenario. This pattern is widely used in Convolution Neural Networks, for example ResNet.       
Convolution + BiasAdd :math:`^{?}` :math:`_{>t1}` , [Convolution + BiasAdd :math:`^{?}` + ReLU] :math:`^{1-3}` + Convolution + BiasAdd :math:`^{?}` + Add :math:`_{<t1}` + ReLU :math:`_{>out}`                                Convolutional Bottleneck. Enabled only in single thread runtime scenario. This pattern is widely used in Convolution Neural Networks, for example ResNet.   
=============================================================================================================================================================================================================================  ==========================================================================================================================================================

Quantized Patterns
++++++++++++++++++

========================================================================================================================================================================================================================================================================================================================================================================================================================  ====================================================================================  
Pattern                                                                                                                                                                                                                                                                                                                                                                                                                   Description                                                                           
========================================================================================================================================================================================================================================================================================================================================================================================================================  ====================================================================================  
Dequantize :math:`_{>t1}` , Dequantize :math:`_{>t2}` , Dequantize + MatMul :math:`_{<t1}` + [Multiply | Divide] + Add + Softmax + Quantize + Dequantize + MatMul :math:`_{<t2}` + StaticTranspose + Reorder + Quantize :math:`_{>out}`                                                                                                                                                                                   Quantized Multi-head Attention.                                                       
Dequantize + ReshapeTranspose :math:`_{>t1}` , Dequantize + ReshapeTranspose :math:`_{>t2}` , Dequantize + MatMul :math:`_{<t1}` + [Multiply | Divide] + Add + Softmax + Quantize + Dequantize + MatMul :math:`_{<t2}` + StaticTranspose + StaticReshape + Quantize :math:`_{>out}`                                                                                                                                       Quantized Multi-head Attention.                                                       
Dequantize :math:`_{>t1}` , Dequantize + MatMul :math:`_{<t1}` + Activation + Quantize :math:`_{>t2}` , [Dequantize :math:`_{>t3}` , Dequantize :math:`_{<t2}` + MatMul :math:`_{<t3}` + Activation + Quantize :math:`_{>t2}` ] :math:`^{0-4}` , Dequantize :math:`_{>t4}` , Dequantize :math:`_{<t2}` + MatMul :math:`_{<t4}` + Activation + Quantize :math:`_{>out}`                                                    Quantized Multi-layer Perceptron.                                                     
Dequantize :math:`_{>t2}` , Dequantize :math:`_{>t3}` , [Dequantize :math:`_{>t1}` , Dequantize + Convolution :math:`_{<t1}` + BiasAdd :math:`^{?}` + ReLU + Quantize] :math:`^{1-3}` + Dequantize + Convolution :math:`_{<t2}` + BiasAdd :math:`^{?}` + Add :math:`_{<t3}` + ReLU + Quantize :math:`_{>out}`                                                                                                             Quantized Identical Bottleneck. Enabled only in single thread runtime scenario.       
[Dequantize :math:`_{>t1}` , Dequantize + Convolution :math:`_{<t1}` + BiasAdd :math:`^{?}` + Quantize + Dequantize] :math:`_{>t2}` , Dequantize :math:`_{>t4}` , [Dequantize :math:`_{>t3}` , Dequantize + Convolution :math:`_{<t3}` + BiasAdd :math:`^{?}` + ReLU + Quantize] :math:`^{1-3}` + Dequantize + Convolution :math:`_{<t4}` + BiasAdd :math:`^{?}` + Add :math:`_{<t2}` + ReLU + Quantize :math:`_{>out}`   Quantized Convolutional Bottleneck. Enabled only in single thread runtime scenario.   
========================================================================================================================================================================================================================================================================================================================================================================================================================  ====================================================================================

Training
--------

=====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ====================================================  
Pattern                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Description                                           
=====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ====================================================  
Dequantize :math:`_{>t1}` , Dequantize :math:`_{>t2}` , Dequantize + MatMul :math:`_{<t1}` + [Multiply | Divide] + Add + Softmax + Quantize + Dequantize + MatMul :math:`_{<t2}` + StaticTranspose + Reorder + Quantize :math:`_{>out}`                                                                                                                                                                                                                                                                                                                                                                                                                                                                Multi-head Attention Training Forward Pattern.        
StaticReshape + StaticTranspose :math:`_{>t1}` + MatMul + Multiply :math:`_{>t2}` + Subtract :math:`_{<t3}` + Multiply :math:`^{?}` + [Multiply | Divide] :math:`_{>t4}` + MatMul :math:`_{>out1}` , Multiply :math:`_{<t2}` + ReduceSum :math:`_{>t3}` , MatMul :math:`_{<t1,>out2}` , MatMul :math:`_{<t4,>out3}`                                                                                                                                                                                                                                                                                                                                                                                    Multi-head Attention Training Backward Pattern.       
MatMul :math:`_{>out1}` + Activation :math:`_{>t1,>out2}` , [MatMul :math:`_{<t1,>out3}` + Activation :math:`_{>t1,>out4}` ] :math:`^{0-4}` , MatMul :math:`_{<t1,>out5}` + Activation :math:`_{>out6}`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Multi-layer Perceptron Training Forward Pattern.      
StaticTranspose :math:`^{?}` :math:`_{>t0}` , ActivationBackward :math:`_{>t2}` + MatMul :math:`_{<t0,>t1}` , ReduceSum :math:`^{?}` :math:`_{<t2,>out1}` , StaticTranspose :math:`^{?}` + MatMul :math:`_{<t2,>out2}` , [StaticTranspose :math:`^{?}` :math:`_{>t3}` , ActivationBackward :math:`_{>t4,<t1}` + MatMul :math:`_{<t3,>t1}` , ReduceSum :math:`^{?}` :math:`_{<t4,>out3}` , StaticTranspose :math:`^{?}` + MatMul :math:`_{<t4,>out4}` ] :math:`^{0-4}` , StaticTranspose :math:`^{?}` :math:`_{>t5}` , ActivationBackward :math:`_{>t6,<t1}` + MatMul :math:`_{<t5,>out5}` , ReduceSum :math:`^{?}` :math:`_{<t6,>out6}` , StaticTranspose :math:`^{?}` + MatMul :math:`_{<t6,>out7}`   Multi-layer Perceptron Training Backward Pattern.     
Convolution :math:`_{>out1}` + BatchNormForwardTraining :math:`_{>out2}` + ReLU :math:`_{>out3}` + Convolution :math:`_{>out4}` + BatchNormForwardTraining :math:`_{>out5}` + ReLU :math:`_{>out6}` + Convolution :math:`_{>out7}` + BatchNormForwardTraining :math:`_{>out8}` + Add + ReLU :math:`_{>out9}`                                                                                                                                                                                                                                                                                                                                                                                           Identical Bottleneck Training Forward Pattern.        
Convolution :math:`_{>out1}` + BatchNormForwardTraining :math:`_{>t1,>out2}` , Convolution :math:`_{>out3}` + BatchNormForwardTraining :math:`_{>out4}` + ReLU :math:`_{>out5}` + Convolution :math:`_{>out6}` + BatchNormForwardTraining :math:`_{>out7}` + ReLU :math:`_{>out8}` + Convolution :math:`_{>out9}` + BatchNormForwardTraining :math:`_{>out10}` + Add :math:`_{<t1}` + ReLU :math:`_{>out11}`                                                                                                                                                                                                                                                                                           Convolutional Bottleneck Training Forward Pattern.    
ReLUBackward :math:`_{>t1}` + BatchNormTrainingBackward :math:`_{>t2,>out1}` + ConvolutionBackwardData + ReLUBackward + BatchNormTrainingBackward :math:`_{>t3,>out2}` + ConvolutionBackwardData + ReLUBackward + BatchNormTrainingBackward :math:`_{>t4,>out3}` + ConvolutionBackwardData + Add :math:`_{<t1,>out4}` , ConvolutionBackwardWeights :math:`_{<t2,>out5}` , ConvolutionBackwardWeights :math:`_{<t3,>out6}` , ConvolutionBackwardWeights :math:`_{<t4,>out7}`                                                                                                                                                                                                                            Identical Bottleneck Training Backward Pattern.       
ReLUBackward :math:`_{>t1}` + BatchNormTrainingBackward :math:`_{>t2,>out1}` + ConvolutionBackwardData + ReLUBackward + BatchNormTrainingBackward :math:`_{>t3,>out2}` + ConvolutionBackwardData + ReLUBackward + BatchNormTrainingBackward :math:`_{>t4,>out3}` + ConvolutionBackwardData + Add :math:`_{<t6,>out4}` , BatchNormTrainingBackward :math:`_{<t1,>t5,>out5}` + ConvolutionBackwardData :math:`_{>t6}` , ConvolutionBackwardWeights :math:`_{<t2,>out6}` , ConvolutionBackwardWeights :math:`_{<t3,>out7}` , ConvolutionBackwardWeights :math:`_{<t4,>out8}` , ConvolutionBackwardWeights :math:`_{<t5,>out9}`                                                                            Convolutional Bottleneck Training Backward Pattern.   
=====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ====================================================

