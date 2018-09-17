# Supported Tensorflow Ops

## Operations - Arithmetic

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Add|add|
|AddN|addN|
|BiasAdd|add|
|Div|div|
|FloorDiv|floorDiv|
|Maximum|maximum|
|Minimum|minimum|
|Mod|mod|
|Mul|mul|
|Pow|pow|
|RealDiv|div|
|SquaredDifference|squaredDifference|
|Sub|sub|


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
|Not mapped|atan2|
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
|TensorArrayCloseV3|tensorArrayClose|
|TensorArrayConcatV3|tensorArrayConcat|
|TensorArrayGatherV3|tensorArrayGather|
|TensorArrayReadV3|tensorArrayRead|
|TensorArrayScatterV3|tensorArrayScatter|
|TensorArraySizeV3|tensorArraySize|
|TensorArraySplitV3|tensorArraySplit|
|TensorArrayV3|tensorArray|
|TensorArrayWriteV3|tensorArrayWrite|


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
|Not mapped|scalar|
|Not mapped|tensor|
|Not mapped|tensor1d|
|Not mapped|tensor2d|
|Not mapped|tensor3d|
|Not mapped|tensor4d|
|Not mapped|tensor5d|
|Not mapped|tensor6d|
|Not mapped|variable|


## Operations - Dynamic

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|NonMaxSuppressionV2|nonMaxSuppression|
|NonMaxSuppressionV3|nonMaxSuppression|
|Where|whereAsync|


## Operations - Evaluation

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|TopKV2|topK|


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
|ShapeN|shapeN|
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
|BatchMatMul|matMul|
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
|CropAndResize|cropAndResize|
|ResizeBilinear|resizeBilinear|
|ResizeNearestNeighbor|resizeNearestNeighbor|


## Operations - Reduction

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|All|all|
|Any|any|
|ArgMax|argMax|
|ArgMin|argMin|
|Max|max|
|Mean|mean|
|Min|min|
|Sum|sum|
|Not mapped|logSumExp|


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
|BatchToSpaceND|batchToSpaceND|
|Cast|cast|
|ExpandDims|expandDims|
|Pad|pad|
|PadV2|pad|
|Reshape|reshape|
|SpaceToBatchND|spaceToBatchND|
|Squeeze|squeeze|


