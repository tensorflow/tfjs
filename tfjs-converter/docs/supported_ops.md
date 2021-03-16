# Supported Tensorflow Ops

## Operations - Arithmetic

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Add|add|
|AddN|addN|
|AddV2|AddV2|
|BiasAdd|BiasAdd|
|Div|div|
|DivNoNan|divNoNan|
|FloorDiv|floorDiv|
|FloorMod|FloorMod|
|Maximum|maximum|
|Minimum|minimum|
|Mod|mod|
|Mul|mul|
|Pow|pow|
|RealDiv|RealDiv|
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
|Atan|atan|
|Atan2|atan2|
|Atanh|atanh|
|Ceil|ceil|
|ClipByValue|clipByValue|
|Complex|Complex|
|ComplexAbs|ComplexAbs|
|Cos|cos|
|Cosh|cosh|
|Elu|elu|
|Erf|erf|
|Exp|exp|
|Expm1|expm1|
|Floor|floor|
|Imag|Imag|
|LeakyRelu|leakyRelu|
|Log|log|
|Log1p|log1p|
|Neg|neg|
|Prelu|prelu|
|Prod|Prod|
|Real|Real|
|Reciprocal|reciprocal|
|Relu|relu|
|Relu6|relu6|
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
|Not mapped|isFinite|
|Not mapped|isInf|
|Not mapped|isNaN|
|Not mapped|logSigmoid|
|Not mapped|step|

## Operations - Control Flow

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|EmptyTensorList|EmptyTensorList|
|Enter|Enter|
|Exit|Exit|
|If|If|
|LoopCond|LoopCond|
|Merge|Merge|
|NextIteration|NextIteration|
|StatelessIf|StatelessIf|
|StatelessWhile|StatelessWhile|
|Switch|Switch|
|TensorArrayCloseV3|TensorArrayCloseV3|
|TensorArrayConcatV3|TensorArrayConcatV3|
|TensorArrayGatherV3|TensorArrayGatherV3|
|TensorArrayReadV3|TensorArrayReadV3|
|TensorArrayScatterV3|TensorArrayScatterV3|
|TensorArraySizeV3|TensorArraySizeV3|
|TensorArraySplitV3|TensorArraySplitV3|
|TensorArrayV3|TensorArrayV3|
|TensorArrayWriteV3|TensorArrayWriteV3|
|TensorListConcat|TensorListConcat|
|TensorListFromTensor|TensorListFromTensor|
|TensorListGather|TensorListGather|
|TensorListGetItem|TensorListGetItem|
|TensorListPopBack|TensorListPopBack|
|TensorListPushBack|TensorListPushBack|
|TensorListReserve|TensorListReserve|
|TensorListScatter|TensorListScatter|
|TensorListScatterV2|TensorListScatterV2|
|TensorListSetItem|TensorListSetItem|
|TensorListSplit|TensorListSplit|
|TensorListStack|TensorListStack|
|While|While|

## Operations - Convolution

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|_FusedConv2D|_FusedConv2D|
|AvgPool|AvgPool|
|AvgPool3D|avgPool3d|
|Conv1D|conv1d|
|Conv2D|conv2d|
|Conv2DBackpropInput|Conv2DBackpropInput|
|Conv3D|conv3d|
|DepthwiseConv2d|depthwiseConv2d|
|DepthwiseConv2dNative|DepthwiseConv2dNative|
|Dilation2D|Dilation2D|
|FusedDepthwiseConv2dNative|FusedDepthwiseConv2dNative|
|MaxPool|MaxPool|
|MaxPool3D|maxPool3d|
|MaxPoolWithArgmax|maxPoolWithArgmax|
|Not mapped|conv2dTranspose|
|Not mapped|conv3dTranspose|
|Not mapped|pool|
|Not mapped|separableConv2d|

## Tensors - Creation

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Fill|fill|
|LinSpace|linspace|
|Multinomial|Multinomial|
|OneHot|oneHot|
|Ones|ones|
|OnesLike|onesLike|
|RandomUniform|RandomUniform|
|Range|range|
|TruncatedNormal|truncatedNormal|
|Zeros|zeros|
|ZerosLike|zerosLike|
|Not mapped|eye|

## Operations - Dynamic

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|ListDiff|ListDiff|
|NonMaxSuppressionV2|NonMaxSuppressionV2|
|NonMaxSuppressionV3|NonMaxSuppressionV3|
|NonMaxSuppressionV4|NonMaxSuppressionV4|
|NonMaxSuppressionV5|NonMaxSuppressionV5|
|Where|Where|

## Operations - Evaluation

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|TopKV2|TopKV2|
|Unique|Unique|
|UniqueV2|UniqueV2|
|Not mapped|confusionMatrix|
|Not mapped|inTopKAsync|
|Not mapped|topk|

## Tensorflow - Graph

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Const|Const|
|FakeQuantWithMinMaxVars|FakeQuantWithMinMaxVars|
|Identity|Identity|
|IdentityN|IdentityN|
|NoOp|NoOp|
|Placeholder|Placeholder|
|PlaceholderWithDefault|PlaceholderWithDefault|
|Print|Print|
|Rank|Rank|
|Shape|Shape|
|ShapeN|ShapeN|
|Size|Size|
|Snapshot|Snapshot|
|StopGradient|StopGradient|

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
|Select|Select|
|SelectV2|SelectV2|
|Not mapped|logicalXor|

## Operations - Hashtable

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|HashTable|HashTable|
|HashTableV2|HashTableV2|
|LookupTableFind|LookupTableFind|
|LookupTableFindV2|LookupTableFindV2|
|LookupTableImport|LookupTableImport|
|LookupTableImportV2|LookupTableImportV2|
|LookupTableSize|LookupTableSize|
|LookupTableSizeV2|LookupTableSizeV2|

## Operations - Images

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|CropAndResize|cropAndResize|
|ResizeBilinear|resizeBilinear|
|ResizeNearestNeighbor|resizeNearestNeighbor|
|Not mapped|flipLeftRight|
|Not mapped|rotateWithOffset|

## Operations - Matrices

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|_FusedMatMul|_FusedMatMul|
|BatchMatMul|BatchMatMul|
|BatchMatMulV2|BatchMatMulV2|
|MatMul|matMul|
|Transpose|transpose|
|Not mapped|dot|
|Not mapped|norm|
|Not mapped|outerProduct|

## Operations - Moving Average

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Not mapped|movingAverage|

## Operations - Normalization

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|FusedBatchNorm|FusedBatchNorm|
|FusedBatchNormV2|FusedBatchNormV2|
|FusedBatchNormV3|FusedBatchNormV3|
|LogSoftmax|logSoftmax|
|LRN|LRN|
|Softmax|softmax|
|SparseToDense|sparseToDense|
|Not mapped|batchNorm|
|Not mapped|moments|

## Operations - Reduction

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|All|all|
|Any|any|
|ArgMax|argMax|
|ArgMin|argMin|
|Bincount|bincount|
|DenseBincount|denseBincount|
|Max|max|
|Mean|mean|
|Min|min|
|Prod|prod|
|Sum|sum|
|Not mapped|logSumExp|

## Tensors - RNN

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|

## Operations - Scan

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Cumsum|cumsum|

## Operations - Segment

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Not mapped|unsortedSegmentSum|

## Tensors - Slicing and Joining

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Concat|concat|
|ConcatV2|ConcatV2|
|Gather|gather|
|GatherNd|GatherNd|
|GatherV2|GatherV2|
|Pack|Pack|
|Reverse|reverse|
|ReverseV2|ReverseV2|
|ScatterNd|ScatterNd|
|Slice|slice|
|SparseToDense|SparseToDense|
|Split|split|
|SplitV|SplitV|
|StridedSlice|StridedSlice|
|Tile|tile|
|Unpack|Unpack|
|Not mapped|booleanMaskAsync|
|Not mapped|stack|
|Not mapped|unstack|

## Operations - Spectral

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|FFT|fft|
|IFFT|ifft|
|IRFFT|irfft|
|RFFT|rfft|

## Operations - Signal

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Not mapped|frame|
|Not mapped|hammingWindow|
|Not mapped|hannWindow|
|Not mapped|stft|

## Operations - Linear Algebra

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|Not mapped|bandPart|
|Not mapped|gramSchmidt|
|Not mapped|qr|

## Tensors - Transformations

|Tensorflow Op Name|Tensorflow.js Op Name|
|---|---|
|BatchToSpaceND|batchToSpaceND|
|BroadcastTo|broadcastTo|
|Cast|cast|
|DepthToSpace|depthToSpace|
|ExpandDims|expandDims|
|MirrorPad|MirrorPad|
|Pad|pad|
|PadV2|PadV2|
|Reshape|reshape|
|SpaceToBatchND|spaceToBatchND|
|Squeeze|squeeze|
|Not mapped|setdiff1dAsync|

