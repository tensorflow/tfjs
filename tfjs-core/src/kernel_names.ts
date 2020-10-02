/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// Allow UpperCamelCase variable names
// tslint:disable: variable-name
// Unfortunately just enabling PascalCase per file (tslint:enable:
// allow-pascal-case) doesn't work.
import {NamedTensorInfoMap, TensorInfo} from './kernel_registry';
import {ExplicitPadding} from './ops/conv_util';
import {Activation} from './ops/fused_types';
import {DataType, PixelData} from './types';

export const Abs = 'Abs';
export type AbsInputs = UnaryInputs;

export const Acos = 'Acos';
export type AcosInputs = UnaryInputs;

export const Acosh = 'Acosh';
export type AcoshInputs = UnaryInputs;

export const Add = 'Add';
export type AddInputs = BinaryInputs;

export const AddN = 'AddN';
export type AddNInputs = TensorInfo[];

export const All = 'All';
export type AllInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface AllAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const Any = 'Any';
export type AnyInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface AnyAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const ArgMax = 'ArgMax';
export type ArgMaxInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface ArgMaxAttrs {
  axis: number;
}

export const ArgMin = 'ArgMin';
export type ArgMinInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface ArgMinAttrs {
  axis: number;
}

export const Asin = 'Asin';
export type AsinInputs = UnaryInputs;

export const Asinh = 'Asinh';
export type AsinhInputs = UnaryInputs;

export const Atan = 'Atan';
export type AtanInputs = UnaryInputs;

export const Atanh = 'Atanh';
export type AtanhInputs = UnaryInputs;

export const Atan2 = 'Atan2';
export type Atan2Inputs = BinaryInputs;

export const AvgPool = 'AvgPool';
export type AvgPoolInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface AvgPoolAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const AvgPoolBackprop = 'AvgPoolBackprop';
export type AvgPoolBackpropInputs = Pick<NamedTensorInfoMap, 'dy'|'input'>;
export interface AvgPoolBackpropAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
}

export const AvgPool3D = 'AvgPool3D';
export type AvgPool3DInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface AvgPool3DAttrs {
  filterSize: [number, number, number]|number;
  strides: [number, number, number]|number;
  pad: 'valid'|'same'|number;
  dimRoundingMode?: 'floor'|'round'|'ceil';
  dataFormat: 'NDHWC'|'NCDHW';
  dilations?: [number, number, number]|number;
}

export const AvgPool3DBackprop = 'AvgPool3DBackprop';
export type AvgPool3DBackpropInputs = Pick<NamedTensorInfoMap, 'dy'|'input'>;
export interface AvgPool3DBackpropAttrs {
  filterSize: [number, number, number]|number;
  strides: [number, number, number]|number;
  pad: 'valid'|'same'|number;
  dilations: [number, number, number]|number;
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const BatchMatMul = 'BatchMatMul';
export type BatchMatMulInputs = Pick<NamedTensorInfoMap, 'a'|'b'>;
export interface BatchMatMulAttrs {
  transposeA: boolean;
  transposeB: boolean;
}

export const BatchToSpaceND = 'BatchToSpaceND';
export type BatchToSpaceNDInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface BatchToSpaceNDAttrs {
  blockShape: number[];
  crops: number[][];
}

export type BinaryInputs = Pick<NamedTensorInfoMap, 'a'|'b'>;

export const BroadcastTo = 'BroadcastTo';
export type BroadcastToInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface BroadCastToAttrs {
  shape: number[];
  inputShape: number[];  // for gradient
}

export const Cast = 'Cast';
export type CastInputs = UnaryInputs;
export interface CastAttrs {
  dtype: DataType;
}

export const Ceil = 'Ceil';
export type CeilInputs = UnaryInputs;

export const ClipByValue = 'ClipByValue';
export type ClipByValueInputs = UnaryInputs;
export interface ClipByValueAttrs {
  clipValueMin: number;
  clipValueMax: number;
}

export const Complex = 'Complex';
export type ComplexInputs = Pick<NamedTensorInfoMap, 'real'|'imag'>;

export const Concat = 'Concat';
export type ConcatInputs = TensorInfo[];
export interface ConcatAttrs {
  axis: number;
}

export const Conv2D = 'Conv2D';
export type Conv2DInputs = Pick<NamedTensorInfoMap, 'x'|'filter'>;
export interface Conv2DAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  dataFormat: 'NHWC'|'NCHW';
  dilations: [number, number]|number;
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const Conv2DBackpropFilter = 'Conv2DBackpropFilter';
export type Conv2DBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x'|'dy'>;
export interface Conv2DBackpropFilterAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  dataFormat: 'NHWC'|'NCHW';
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const Conv2DBackpropInput = 'Conv2DBackpropInput';
export type Conv2DBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy'|'filter'>;
export interface Conv2DBackpropInputAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  dataFormat: 'NHWC'|'NCHW';
  dimRoundingMode?: 'floor'|'round'|'ceil';
  inputShape: [number, number, number, number];
}

export const Conv3D = 'Conv3D';
export type Conv3DInputs = Pick<NamedTensorInfoMap, 'x'|'filter'>;
export interface Conv3DAttrs {
  strides: [number, number, number]|number;
  pad: 'valid'|'same';
  dataFormat: 'NDHWC'|'NCDHW';
  dilations: [number, number, number]|number;
}

export const Conv3DBackpropFilterV2 = 'Conv3DBackpropFilterV2';
export type Conv3DBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x'|'y'>;

export interface Conv3DBackpropFilterAttrs {
  strides: [number, number, number]|number;
  pad: 'valid'|'same';
}

export const Conv3DBackpropInputV2 = 'Conv3DBackpropInputV2';
export type Conv3DBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy'>;
export interface Conv3DBackpropInputAttrs {
  pad: 'valid'|'same';
}

export const Cos = 'Cos';
export type CosInputs = UnaryInputs;

export const Cosh = 'Cosh';
export type CoshInputs = UnaryInputs;

export const Cumsum = 'Cumsum';
export type CumsumInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface CumsumAttrs {
  axis: number;
  exclusive: boolean;
  reverse: boolean;
}

export const CropAndResize = 'CropAndResize';
export type CropAndResizeInputs =
    Pick<NamedTensorInfoMap, 'image'|'boxes'|'boxInd'>;
export interface CropAndResizeAttrs {
  cropSize: [number, number];
  method: 'bilinear'|'nearest';
  extrapolationValue: number;
}

export const DepthToSpace = 'DepthToSpace';
export type DepthToSpaceInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface DepthToSpaceAttrs {
  blockSize: number;
  dataFormat: 'NHWC'|'NCHW';
}

export const DepthwiseConv2dNative = 'DepthwiseConv2dNative';
export type DepthwiseConv2dNativeInputs =
    Pick<NamedTensorInfoMap, 'x'|'filter'>;
export interface DepthwiseConv2dNativeAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  dataFormat: 'NHWC'|'NCHW';
  dilations: [number, number]|number;
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const DepthwiseConv2dNativeBackpropFilter =
    'DepthwiseConv2dNativeBackpropFilter';
export type DepthwiseConv2dNativeBackpropFilterInputs =
    Pick<NamedTensorInfoMap, 'x'|'dy'>;

export const DepthwiseConv2dNativeBackpropInput =
    'DepthwiseConv2dNativeBackpropInput';
export type DepthwiseConv2dNativeBackpropInputInputs =
    Pick<NamedTensorInfoMap, 'dy'>;

export const Diag = 'Diag';
export type DiagInputs = Pick<NamedTensorInfoMap, 'x'>;

export const Dilation2D = 'Dilation2D';
export type Dilation2DInputs = Pick<NamedTensorInfoMap, 'x'|'filter'>;
export interface Dilation2DAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  dilations: [number, number]|number;
}

export const Dilation2DBackpropInput = 'Dilation2DBackpropInput';
export type Dilation2DBackpropInputInputs =
    Pick<NamedTensorInfoMap, 'x'|'filter'|'dy'>;

export const Dilation2DBackpropFilter = 'Dilation2DBackpropFilter';
export type Dilation2DBackpropFilterInputs =
    Pick<NamedTensorInfoMap, 'x'|'filter'|'dy'>;

export const Div = 'Div';
export type DivInputs = BinaryInputs;

export const Elu = 'Elu';
export type EluInputs = Pick<NamedTensorInfoMap, 'x'>;

export const EluGrad = 'EluGrad';
export type EluGradInputs = Pick<NamedTensorInfoMap, 'dy'|'y'>;

export const Erf = 'Erf';
export type ErfInputs = UnaryInputs;

export const Equal = 'Equal';
export type EqualInputs = BinaryInputs;

export const Exp = 'Exp';
export type ExpInputs = UnaryInputs;

export const Expm1 = 'Expm1';
export type Expm1Inputs = UnaryInputs;

export const FFT = 'FFT';
export type FFTInputs = Pick<NamedTensorInfoMap, 'input'>;

export const Fill = 'Fill';
export interface FillAttrs {
  shape: number[];
  value: number|string;
  dtype: DataType;
}

export const FlipLeftRight = 'FlipLeftRight';
export type FlipLeftRightInputs = Pick<NamedTensorInfoMap, 'image'>;

export const Floor = 'Floor';
export type FloorInputs = UnaryInputs;

export const FloorDiv = 'FloorDiv';
export type FloorDivInputs = BinaryInputs;

export const FusedBatchNorm = 'FusedBatchNorm';
export type FusedBatchNormInputs =
    Pick<NamedTensorInfoMap, 'x'|'scale'|'offset'|'mean'|'variance'>;
export interface FusedBatchNormAttrs {
  varianceEpsilon: number;
}

export const GatherV2 = 'GatherV2';
export type GatherV2Inputs = Pick<NamedTensorInfoMap, 'x'|'indices'>;
export interface GatherV2Attrs {
  axis: number;
}

export const GatherNd = 'GatherNd';
export type GatherNdInputs = Pick<NamedTensorInfoMap, 'params'|'indices'>;

export const Greater = 'Greater';
export type GreaterInputs = BinaryInputs;

export const GreaterEqual = 'GreaterEqual';
export type GreaterEqualInputs = BinaryInputs;

export const Identity = 'Identity';
export type IdentityInputs = Pick<NamedTensorInfoMap, 'x'>;

export const IFFT = 'IFFT';
export type IFFTInputs = Pick<NamedTensorInfoMap, 'input'>;

export const Imag = 'Imag';
export type ImagInputs = Pick<NamedTensorInfoMap, 'input'>;

export const IsFinite = 'IsFinite';
export type IsFiniteInputs = UnaryInputs;

export const IsInf = 'IsInf';
export type IsInfInputs = UnaryInputs;

export const IsNan = 'IsNan';
export type IsNanInputs = UnaryInputs;

export const Less = 'Less';
export type LessInputs = BinaryInputs;

export const LessEqual = 'LessEqual';
export type LessEqualInputs = BinaryInputs;

export const LinSpace = 'LinSpace';
export interface LinSpaceAttrs {
  start: number;
  stop: number;
  num: number;
}
export const Log = 'Log';
export type LogInputs = UnaryInputs;

export const Log1p = 'Log1p';
export type Log1pInputs = UnaryInputs;

export const LogicalAnd = 'LogicalAnd';
export type LogicalAndInputs = BinaryInputs;

export const LogicalNot = 'LogicalNot';
export type LogicalNotInputs = Pick<NamedTensorInfoMap, 'x'>;

export const LogicalOr = 'LogicalOr';
export type LogicalOrInputs = BinaryInputs;

export const LogSoftmax = 'LogSoftmax';
export type LogSoftmaxInputs = Pick<NamedTensorInfoMap, 'logits'>;
export interface LogSoftmaxAttrs {
  axis: number;
}

export const LRN = 'LRN';
export type LRNInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface LRNAttrs {
  depthRadius: number;
  bias: number;
  alpha: number;
  beta: number;
}

export const LRNBackprop = 'LRNBackprop';
export type LRNBackpropInputs = Pick<NamedTensorInfoMap, 'x'|'y'|'dy'>;
export interface LRNBackpropAttrs {
  depthRadius: number;
  bias: number;
  alpha: number;
  beta: number;
}

export const Max = 'Max';
export type MaxInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxAttrs {
  reductionIndices: number|number[];
  keepDims: boolean;
}

export const Maximum = 'Maximum';
export type MaximumInputs = BinaryInputs;

export const MaxPool = 'MaxPool';
export type MaxPoolInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxPoolAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const MaxPoolBackprop = 'MaxPoolBackprop';
export type MaxPoolBackpropInputs =
    Pick<NamedTensorInfoMap, 'dy'|'input'|'output'>;
export interface MaxPoolBackpropAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const MaxPool3D = 'MaxPool3D';
export type MaxPool3DInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxPool3DAttrs {
  filterSize: [number, number, number]|number;
  strides: [number, number, number]|number;
  pad: 'valid'|'same'|number;
  dataFormat: 'NDHWC'|'NCDHW';
  dilations?: [number, number, number]|number;
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const MaxPool3DBackprop = 'MaxPool3DBackprop';
export type MaxPool3DBackpropInputs =
    Pick<NamedTensorInfoMap, 'dy'|'input'|'output'>;
export interface MaxPool3DBackpropAttrs {
  filterSize: [number, number, number]|number;
  strides: [number, number, number]|number;
  pad: 'valid'|'same'|number;
  dilations?: [number, number, number]|number;
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const MaxPoolWithArgmax = 'MaxPoolWithArgmax';
export type MaxPoolWithArgmaxInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxPoolWithArgmaxAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  includeBatchInIndex: boolean;
}

export const Mean = 'Mean';
export type MeanInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MeanAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const Min = 'Min';
export type MinInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MinAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const Minimum = 'Minimum';
export type MinimumInputs = BinaryInputs;

export const Mod = 'Mod';
export type ModInputs = BinaryInputs;

export const Multiply = 'Multiply';
export type MultiplyInputs = BinaryInputs;

export const Negate = 'Negate';
export type NegateInputs = UnaryInputs;

export const NotEqual = 'NotEqual';
export type NotEqualInputs = BinaryInputs;

export const NonMaxSuppressionV3 = 'NonMaxSuppressionV3';
export type NonMaxSuppressionV3Inputs =
    Pick<NamedTensorInfoMap, 'boxes'|'scores'>;
export interface NonMaxSuppressionV3Attrs {
  maxOutputSize: number;
  iouThreshold: number;
  scoreThreshold: number;
}

export const NonMaxSuppressionV4 = 'NonMaxSuppressionV4';
export type NonMaxSuppressionV4Inputs =
    Pick<NamedTensorInfoMap, 'boxes'|'scores'>;
export interface NonMaxSuppressionV4Attrs {
  maxOutputSize: number;
  iouThreshold: number;
  scoreThreshold: number;
  padToMaxOutputSize: boolean;
}

export const NonMaxSuppressionV5 = 'NonMaxSuppressionV5';
export type NonMaxSuppressionV5Inputs =
    Pick<NamedTensorInfoMap, 'boxes'|'scores'>;
export interface NonMaxSuppressionV5Attrs {
  maxOutputSize: number;
  iouThreshold: number;
  scoreThreshold: number;
  softNmsSigma: number;
}

export const OnesLike = 'OnesLike';
export type OnesLikeInputs = UnaryInputs;

export const OneHot = 'OneHot';
export type OneHotInputs = Pick<NamedTensorInfoMap, 'indices'>;
export interface OneHotAttrs {
  depth: number;
  onValue: number;
  offValue: number;
}

export const PadV2 = 'PadV2';
export type PadV2Inputs = Pick<NamedTensorInfoMap, 'x'>;
export interface PadV2Attrs {
  paddings: Array<[number, number]>;
  constantValue: number;
}

export const Pool = 'Pool';
export type PoolInputs = Pick<NamedTensorInfoMap, 'input'>;

export const Pow = 'Pow';
export type PowInputs = BinaryInputs;

export const Prelu = 'Prelu';
export type PreluInputs = Pick<NamedTensorInfoMap, 'x'|'alpha'>;

export const Prod = 'Prod';
export type ProdInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface ProdAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const Range = 'Range';
export interface RangeAttrs {
  start: number;
  stop: number;
  step: number;
  dtype: 'float32'|'int32';
}

export const Real = 'Real';
export type RealInputs = Pick<NamedTensorInfoMap, 'input'>;

export const Reciprocal = 'Reciprocal';
export type ReciprocalInputs = UnaryInputs;

export const Relu = 'Relu';
export type ReluInputs = Pick<NamedTensorInfoMap, 'x'>;

export const Reshape = 'Reshape';
export type ReshapeInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface ReshapeAttrs {
  shape: number[];
}

export const ResizeNearestNeighbor = 'ResizeNearestNeighbor';
export type ResizeNearestNeighborInputs = Pick<NamedTensorInfoMap, 'images'>;
export interface ResizeNearestNeighborAttrs {
  alignCorners: boolean;
  size: [number, number];
}

export const ResizeNearestNeighborGrad = 'ResizeNearestNeighborGrad';
export type ResizeNearestNeighborGradInputs =
    Pick<NamedTensorInfoMap, 'images'>;

export const ResizeBilinear = 'ResizeBilinear';
export type ResizeBilinearInputs = Pick<NamedTensorInfoMap, 'images'>;
export interface ResizeBilinearAttrs {
  alignCorners: boolean;
  size: [number, number];
}

export const ResizeBilinearGrad = 'ResizeBilinearGrad';
export type ResizeBilinearGradInputs = Pick<NamedTensorInfoMap, 'images'>;

export const Relu6 = 'Relu6';
export type Relu6Inputs = Pick<NamedTensorInfoMap, 'x'>;

export const Reverse = 'Reverse';
export type ReverseInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface ReverseAttrs {
  dims: number|number[];
}

export const Round = 'Round';
export type RoundInputs = UnaryInputs;

export const Rsqrt = 'Rsqrt';
export type RsqrtInputs = UnaryInputs;

export const ScatterNd = 'ScatterNd';
export type ScatterNdInputs = Pick<NamedTensorInfoMap, 'indices'|'updates'>;
export interface ScatterNdAttrs {
  shape: number[];
}

export const SelectV2 = 'SelectV2';
export type SelectV2Inputs = Pick<NamedTensorInfoMap, 'condition'|'t'|'e'>;

export const Selu = 'Selu';
export type SeluInputs = Pick<NamedTensorInfoMap, 'x'>;

export const Slice = 'Slice';
export type SliceInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface SliceAttrs {
  begin: number|number[];
  size: number|number[];
}
export const Sin = 'Sin';
export type SinInputs = UnaryInputs;

export const Sinh = 'Sinh';
export type SinhInputs = UnaryInputs;

export const Sign = 'Sign';
export type SignInputs = UnaryInputs;

export const Sigmoid = 'Sigmoid';
export type SigmoidInputs = UnaryInputs;

export const Softplus = 'Softplus';
export type SoftplusInputs = UnaryInputs;

export const Sqrt = 'Sqrt';
export type SqrtInputs = UnaryInputs;

export const Sum = 'Sum';
export type SumInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface SumAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const SpaceToBatchND = 'SpaceToBatchND';
export type SpaceToBatchNDInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface SpaceToBatchNDAttrs {
  blockShape: number[];
  paddings: number[][];
}

export const SplitV = 'SplitV';
export type SplitVInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface SplitVAttrs {
  numOrSizeSplits: number[]|number;
  axis: number;
}

export const Softmax = 'Softmax';
export type SoftmaxInputs = Pick<NamedTensorInfoMap, 'logits'>;
export interface SoftmaxAttrs {
  dim: number;
}

export const SquaredDifference = 'SquaredDifference';
export type SquaredDifferenceInputs = BinaryInputs;

export const Square = 'Square';
export type SquareInputs = Pick<NamedTensorInfoMap, 'x'>;

export const Sub = 'Sub';
export type SubInputs = BinaryInputs;

export const SparseToDense = 'SparseToDense';
export type SparseToDenseInputs =
    Pick<NamedTensorInfoMap, 'sparseIndices'|'sparseValues'|'defaultValue'>;
export interface SparseToDenseAttrs {
  outputShape: number[];
}

export const StridedSlice = 'StridedSlice';
export type StridedSliceInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface StridedSliceAttrs {
  begin: number[];
  end: number[];
  strides: number[];
  beginMask: number;
  endMask: number;
  ellipsisMask: number;
  newAxisMask: number;
  shrinkAxisMask: number;
}

export const Tan = 'Tan';
export type TanInputs = UnaryInputs;

export const Tanh = 'Tanh';
export type TanhInputs = UnaryInputs;

export const Tile = 'Tile';
export type TileInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface TileAttrs {
  reps: number[];
}

export const TopK = 'TopK';
export type TopKInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface TopKAttrs {
  k: number;
  sorted: boolean;
}

export const Transpose = 'Transpose';
export type TransposeInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface TransposeAttrs {
  perm: number[];
}

export const Unique = 'Unique';
export type UniqueInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface UniqueAttrs {
  axis: number;
}

export type UnaryInputs = Pick<NamedTensorInfoMap, 'x'>;

export const Unpack = 'Unpack';
export type UnpackInputs = Pick<NamedTensorInfoMap, 'value'>;
export interface UnpackAttrs {
  axis: number;
}

export const UnsortedSegmentSum = 'UnsortedSegmentSum';
export type UnsortedSegmentSumInputs =
    Pick<NamedTensorInfoMap, 'x'|'segmentIds'>;
export interface UnsortedSegmentSumAttrs {
  numSegments: number;
}

export const ZerosLike = 'ZerosLike';
export type ZerosLikeInputs = UnaryInputs;

/**
 * TensorFlow.js-only kernels
 */
export const Step = 'Step';
export type StepInputs = UnaryInputs;
export interface StepAttrs {
  alpha: number;
}

export const FromPixels = 'FromPixels';
export interface FromPixelsInputs {
  pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement;
}
export interface FromPixelsAttrs {
  numChannels: number;
}

export const RotateWithOffset = 'RotateWithOffset';
export type RotateWithOffsetInputs = Pick<NamedTensorInfoMap, 'image'>;
export interface RotateWithOffsetAttrs {
  radians: number;
  fillValue: number|[number, number, number];
  center: number|[number, number];
}

export const _FusedMatMul = '_FusedMatMul';
// tslint:disable-next-line: class-name
export interface _FusedMatMulInputs extends NamedTensorInfoMap {
  a: TensorInfo;
  b: TensorInfo;
  bias?: TensorInfo;
  preluActivationWeights?: TensorInfo;
}
// tslint:disable-next-line: class-name
export interface _FusedMatMulAttrs {
  transposeA: boolean;
  transposeB: boolean;
  activation: Activation;
}

export const FusedConv2D = 'FusedConv2D';
export interface FusedConv2DInputs extends NamedTensorInfoMap {
  x: TensorInfo;
  filter: TensorInfo;
  bias?: TensorInfo;
  preluActivationWeights?: TensorInfo;
}
export interface FusedConv2DAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  dataFormat: 'NHWC'|'NCHW';
  dilations: [number, number]|number;
  dimRoundingMode: 'floor'|'round'|'ceil';
  activation: Activation;
}

export const FusedDepthwiseConv2D = 'FusedDepthwiseConv2D';
export interface FusedDepthwiseConv2DInputs extends NamedTensorInfoMap {
  x: TensorInfo;
  filter: TensorInfo;
  bias?: TensorInfo;
  preluActivationWeights?: TensorInfo;
}
export interface FusedDepthwiseConv2DAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  dataFormat: 'NHWC'|'NCHW';
  dilations: [number, number]|number;
  dimRoundingMode: 'floor'|'round'|'ceil';
  activation: Activation;
}
