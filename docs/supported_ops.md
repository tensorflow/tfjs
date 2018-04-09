# Supported Tensorflow Ops

## Arithmetic Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Add|add|
|BiasAdd|add|
|Sub|sub|
|RealDiv|div|
|Div|div|
|Mul|mul|
|Maximum|maximum|
|Minimum|minimum|
|Pow|pow|


## Basic Math Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Abs|abs|
|Acos|acos|
|Asin|asin|
|atan|atan|
|Ceil|ceil|
|ClipByValue|clipByValue|
|Cos|cos|
|Cosh|cosh|
|Elu|elu|
|Exp|exp|
|Floor|floor|
|Log|log|
|Neg|neg|
|Relu|relu|
|Relu6|clipByValue|
|Selu|selu|
|Sigmoid|sigmoid|
|Sin|sin|
|Sinh|sinh|
|Sqrt|sqrt|
|Rsqrt|rsqrt|
|Square|square|
|Tan|tan|
|Tanh|tanh|


## Convolution Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|AvgPool|avgPool|
|MaxPool|maxPool|
|Conv1D|conv1d|
|Conv2D|conv2d|
|Conv2DTranspose|conv2dTranspose|
|DepthwiseConv2d|depthwiseConv2d|
|DepthwiseConv2dNative|depthwiseConv2d|


## Tensor Creation Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Fill|fill|
|LinSpace|linspace|
|OneHot|oneHot|
|Ones|ones|
|OnesLike|onesLike|
|RandomUniform|randomUniform|
|Range|range|
|truncatedNormal|truncatedNormal|
|Zeros|zeros|
|ZerosLike|zerosLike|


## Tensorflow Graph Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|PlaceholderWithDefault|placeholder|
|Placeholder|placeholder|
|Const|const|
|Identity|identity|
|Shape|shape|
|Print|print|
|NoOp|noop|


## Logical Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Equal|equal|
|Greater|greater|
|GreaterEqual|greaterEqual|
|Less|less|
|LessEqual|lessEqual|
|LogicalAnd|logicalAnd|
|LogicalNot|logicalNot|
|LogicalOr|logicalOr|
|Select|where|


## Matrices Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|MatMul|matMul|
|Transpose|transpose|


## Normalization Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|FusedBatchNorm|batchNormalization|
|FusedBatchNormV2|batchNormalization|
|LRN|localResponseNormalization|
|Softmax|softmax|


## Image Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|ResizeBilinear|resizeBilinear|


## Reduction Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Max|max|
|Mean|mean|
|Min|min|
|Sum|sum|
|ArgMax|argMax|
|ArgMin|argMin|


## Slice and Join Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|ConcatV2|concat|
|Concat|concat|
|GatherV2|gather|
|Gather|gather|
|Reverse|reverse|
|ReverseV2|reverse|
|Slice|slice|
|Pack|stack|
|Tile|tile|


## Transformation Ops

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Cast|cast|
|ExpandDims|expandDims|
|Pad|pad|
|PadV2|pad|
|Reshape|reshape|
|Squeeze|squeeze|


