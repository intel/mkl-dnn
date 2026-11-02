Fusion Patterns
###############

.. default-role:: math
.. toctree::
   :maxdepth: 1
   :hidden:

   dev_guide_graph_gated_mlp
   dev_guide_graph_gqa
   dev_guide_graph_sdpa_compressed_kv
   dev_guide_graph_sdpa


The following fusion patterns are subgraphs that the oneDNN Graph API
recognizes as candidates for fusion. The patterns are described using
oneDNN Graph operation (op) names with the following convention.

.. note::
   oneDNN Graph performs limited input validation to minimize 
   the performance overheads. The application is responsible for 
   sanitizing inputs passed to the library. Because large ``u8`` or 
   ``s8`` inputs may lead to accumulator overflow, you can use 
   floating-point patterns instead of quantized patterns.

``"+"`` describes a chain of two ops. The preceding op produces an
output tensor, which is consumed by the following op as its first
operand.

``"[]"`` describes a component of the overall pattern description. For
example, it could include a subgraph or all the op choices within the
bracket.

``"|"`` describes choices of multiple operations, say A+[B|C] means the
graph partition contains A followed by B or C.

``","`` describes a graph composed of multiple subgraphs, each subgraph
marks its output tensor explicitly, which is consumed by other
subgraphs.

``Superscript`` denotes the numbers of repetition pattern. For example,
A+[B|C] `^{3}` means the graph partition
contains A followed by three ops, each of them is either B or C. The
superscript could be a range of number meaning allowing a range of
repetition. If the range is between 0 and 1, we use superscript ``"?"``.

``Subscript`` denotes the input and output tensors which need to
explicitly mark the producer and consumer relation within one graph
partition. For example,
A `_{>t1}` +B+C `_{<t1}`
refers to the pattern started with A followed by B and C, and C takes an
implicit input tensor from B and an extra tensor t1 output from A.
``">"`` refers to the output tensor, and ``"<"`` for input tensor. Input
and output tensors between neighbor ops are not explicitly marked, for
example, B consumes t1 implicitly in the example above.

Subscript ``"out"`` marks the output tensor of a certain op to be the
output of a graph partition. For example, in
A `_{>t1}` +B `_{>out}`\ +C `_{<t1,>out}`,
B's output and C's output are marked as output tensors.

Subscript ``"in"`` marks the input tensor of a certain op to be the
input of a graph partition. For example, in
A `_{<in1}`\ +B `_{<in1}`
A's input and B's second input are graph partition input, and they share
the same input tensor in1. Most input tensors of a graph partition are
not explicitly marked. For example, the input tensors of the first op
are implicitly regarded as graph partition inputs. Besides, for input
tensors of other ops, if they are not produced by any proceeding ops,
they are regarded as implicit graph partition inputs. In the example
A `_{>t1}`\ +B+C `_{<t1}`,
A's inputs are regarded as implicit graph partition inputs, and if B is
a binary operation, the second input tensor is an implicit graph
partition input.

The following categories will be used in describing a fusion pattern.

Unary = [Abs \| Clamp \| Elu \| Exp \| GELU \| HardSwish \| LeakyReLU \|
Log \| Sigmoid \| SoftPlus \| Pow \| ReLU \| Round \| Sqrt \| Square \|
Tanh]

Binary = [Add \| Divide \| Maximum \| Minimum \| Multiply \| Subtract]

Reduction = [ReduceL1 \| ReduceL2 \| ReduceMax \| ReduceMean \|
ReduceMin \| ReduceProd \| ReduceSum]

Inference
~~~~~~~~~

Floating Point Patterns
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 75 25
   :header-rows: 1

   * - Pattern
     - Description
   * - Scaled Dot-Product Attention
     - Refer to `Scaled Dot-Product Attention (SDPA) <dev_guide_graph_sdpa.html>`_ for more details.
   * - Grouped Query Attention
     - Refer to `Grouped Query Attention (GQA) <dev_guide_graph_gqa.html>`_ for more details.
   * - Scaled Dot-Product Attention with Compressed Key/Value
     - Refer to `Scaled Dot-Product Attention with Compressed Key/Value <dev_guide_graph_sdpa_compressed_kv.html>`_ for more details.
   * - Gated Multi-Layer Perceptron (Gated-MLP)
     - Refer to `Gated Multi-Layer Perceptron (Gated-MLP) <dev_guide_graph_gated_mlp.html>`_ for more details.
   * - Convolution + BiasAdd `^?` + BatchNormInference `^?` + [Unary \| Binary] `^{0-3}` `_{>out}`
     - This pattern is widely used in Convolution Neural Networks, for example ResNet, ResNext, SSD, etc.
   * - ConvTranspose + BiasAdd `^?` + [Unary \| Binary] `^{0-3}` `_{>out}`
     - This pattern is widely used in Generative Adversarial Networks.
   * - Interpolate + [Unary \| Binary] `^{0-3}` `_{>out}`
     - This pattern is widely used for image processing.
   * - MatMul + BiasAdd `^?` + [Unary \| Binary] `^{0-3}` + Select `^?` `_{>out}`
     - This pattern is widely used in language models and recommendation models, for example BERT, DLRM, etc.
   * - Reduction + [Unary \| Binary] `^{0-3}` `_{>out}`
     - This pattern is widely used for data processing, for example loss reduction.
   * - Unary + Binary `^{0-3}` `_{>out}`
     - This pattern is widely used in Convolution Neural Networks. 
   * - Binary + [Unary \| Binary] `^{0-3}` `_{>out}`
     - This pattern is widely used in language models and recommendation models, for example BERT, DLRM, etc.
   * - [AvgPool \| MaxPool] + Binary `^{0-3}` `_{>out}`
     - This pattern is widely used in Convolution Neural Networks.
   * - BatchNormInference + ReLU `_{>out}`
     - This pattern is widely used in Convolution Neural Networks, for example DenseNet.
   * - Reciprocal + Multiply `_{>out}`
     - N/A
   * - Reorder + Add `_{>out}`
     - N/A
   



Quantized Patterns
^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 75 25
   :header-rows: 1

   * - Pattern
     - Description
   * - Quantize `^?` + Dequantize `_{>t1}`, Dequantize `_{>t2}` `^{0-3}`, Dequantize + Convolution `_{<t1}` + BiasAdd `^?` + [Unary \| Binary `_{<t2}`] `^{0-3}` + Quantize `^? _{>out}`
     - N/A
   * - Quantize `^?` + Dequantize `_{>t1}`, Dequantize `_{>t2}` `^{0-3}`, Dequantize + ConvTranspose `_{<t1}` + BiasAdd `^?` + [Unary \| Binary `_{<t2}`] `^{0-3}` + Quantize `^?` `_{>out}`
     - N/A
   * - Quantize `^?` + Dequantize `_{>t1}`, Dequantize `_{>t2}` `^{0-3}`, Dequantize + MatMul `_{<t1}` + BiasAdd `^?` + [Unary \| Binary `_{<t2}`] `^{0-3}` + Select `^?` + Quantize `^?` `_{>out}`
     - N/A
   * - Dequantize + [AvgPool \| MaxPool] + Quantize `_{>out}``
     - N/A
   * - Dequantize `_{>t1}`, Dequantize + [AvgPool \| MaxPool] + Add `_{<t1}` + Quantize `_{>out}`
     - N/A
   * - Dequantize + Reorder + Quantize `_{>out}`
     - This pattern is widely used in Generative Adversarial Networks.
   * - Dequantize `_{>t1}`, Dequantize + Reorder + Add `_{<t1}` + Quantize `_{>out}`
     - This pattern is widely used for image processing.
   * - [SoftMax \| LayerNorm \| GroupNorm] + [Unary \| Binary `_{<t2}`] `^{0-3}` + Quantize `^? _{>out}`
     - This pattern is used in SmoothQuant to fuse scales and quantization into previous layers.

Training
~~~~~~~~

.. list-table:: 
   :widths: 75 25
   :header-rows: 1

   * - Pattern
     - Description
   * - ConvolutionBackwardWeights + BiasAddBackward `_{>out}`
     - N/A
   * - ReLUBackward + BatchNormTrainingBackward `_{>out}`
     - N/A
