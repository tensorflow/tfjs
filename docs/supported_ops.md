# Supported Tensorflow Ops

## Operations - Arithmetic

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Add|add|
|BiasAdd|add|
|Div|div|
|Maximum|maximum|
|Minimum|minimum|
|Mod|mod|
|Mul|mul|
|Pow|pow|
|RealDiv|div|
|SquaredDifference|squaredDifference|
|Sub|sub|
|Not mapped|floorDiv|


## Operations - Basic math

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Abs|abs|
|Acos|acos|
|Acosh|acosh|
|Asin|asin|
|Asinh|asinh|
|atan|atan|
|Atanh|atanh|
|Ceil|ceil|
|ClipByValue|clipByValue|
|Cos|cos|
|Cosh|cosh|
|Elu|elu|
|Erf|erf|
|Exp|exp|
|Expm1|expm1|
|Floor|floor|
|Log|log|
|Log1p|log1p|
|Neg|neg|
|Reciprocal|reciprocal|
|Reciprocal|reciprocal|
|Relu|relu|
|Relu6|clipByValue|
|Round|round|
|Rsqrt|rsqrt|
|Selu|selu|
|Sigmoid|sigmoid|
|Sign|sign|
|Sin|sin|
|Sinh|sinh|
|Softplus|softplus|
|Sqrt|sqrt|
|Square|square|
|Tan|tan|
|Tanh|tanh|
|Not mapped|leakyRelu|
|Not mapped|logSigmoid|
|Not mapped|prelu|
|Not mapped|step|


## Operations - Control Flow

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Enter|enter|
|Exit|exit|
|LoopCond|loopCond|
|Merge|merge|
|NextIteration|nextIteration|
|Switch|switch|


## Operations - Convolution

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|AvgPool|avgPool|
|Conv1D|conv1d|
|Conv2D|conv2d|
|Conv2DBackpropInput|conv2dTranspose|
|DepthwiseConv2d|depthwiseConv2d|
|DepthwiseConv2dNative|depthwiseConv2d|
|MaxPool|maxPool|
|Not mapped|separableConv2d|


## Tensors - Creation

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
|Not mapped|buffer|
|Not mapped|clone|
|Not mapped|eye|
|Not mapped|fromPixels|
|Not mapped|print|
|Not mapped|randomNormal|
|Not mapped|scalar|
|Not mapped|tensor|
|Not mapped|tensor1d|
|Not mapped|tensor2d|
|Not mapped|tensor3d|
|Not mapped|tensor4d|
|Not mapped|tensor5d|
|Not mapped|variable|


## Tensorflow - Graph

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Const|const|
|FakeQuantWithMinMaxVars|fakeQuantWithMinMaxVars|
|Identity|identity|
|NoOp|noop|
|Placeholder|placeholder|
|PlaceholderWithDefault|placeholder|
|Print|print|
|Rank|rank|
|Shape|shape|
|Size|size|
|Snapshot|snapshot|
|StopGradient|stopGradient|


## Operations - Logical

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
|NotEqual|notEqual|
|Select|where|
|Not mapped|logicalXor|


## Operations - Matrices

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|MatMul|matMul|
|Transpose|transpose|
|Not mapped|dot|
|Not mapped|norm|
|Not mapped|outerProduct|


## Operations - Normalization

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|FusedBatchNorm|batchNormalization|
|FusedBatchNormV2|batchNormalization|
|LRN|localResponseNormalization|
|Softmax|softmax|
|Not mapped|moments|


## Operations - Images

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|ResizeBilinear|resizeBilinear|
|ResizeNearestNeighbor|resizeNearestNeighbor|


## Operations - Reduction

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|ArgMax|argMax|
|ArgMin|argMin|
|Max|max|
|Mean|mean|
|Min|min|
|Sum|sum|
|Not mapped|logSumExp|
|Not mapped|unsortedSegmentSum|


## Tensors - Slicing and Joining

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Concat|concat|
|ConcatV2|concat|
|Gather|gather|
|GatherV2|gather|
|Pack|stack|
|Reverse|reverse|
|ReverseV2|reverse|
|Slice|slice|
|Split|split|
|StridedSlice|stridedSlice|
|Tile|tile|
|Unpack|unstack|


## Tensors - Transformations

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Cast|cast|
|ExpandDims|expandDims|
|Pad|pad|
|PadV2|pad|
|Reshape|reshape|
|Squeeze|squeeze|


