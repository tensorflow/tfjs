/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import {ExplicitPadding} from '../src/ops/conv_util';

import {NamedTensorInfoMap, TensorInfo} from './kernel_registry';
import {DataType, PixelData} from './types';

export const Add = 'Add';
export type AddInputs = BinaryInputs;

export const AddN = 'AddN';
export type AddNInputs = TensorInfo[];

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

export const Cumsum = 'Cumsum';
export type CumsumInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface CumsumAttrs {
  axis: number;
  exclusive: boolean;
  reverse: boolean;
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

export const Div = 'Div';
export type DivInputs = BinaryInputs;

export const Elu = 'Elu';
export type EluInputs = Pick<NamedTensorInfoMap, 'x'>;

export const EluGrad = 'EluGrad';
export type EluGradInputs = Pick<NamedTensorInfoMap, 'dy'|'y'>;

export const Equal = 'Equal';
export type EqualInputs = BinaryInputs;

export const FloorDiv = 'FloorDiv';
export type FloorDivInputs = BinaryInputs;

export const Fill = 'Fill';
export interface FillAttrs {
  shape: number[];
  value: number|string;
  dtype: DataType;
}

export const FusedBatchNorm = 'FusedBatchNorm';
export type FusedBatchNormInputs =
    Pick<NamedTensorInfoMap, 'x'|'scale'|'offset'|'mean'|'variance'>;
export interface FusedBatchNormAttrs {
  varianceEpsilon: number;
}

export const GatherNd = 'GatherNd';
export type GatherNdInputs = Pick<NamedTensorInfoMap, 'params'|'indices'>;

export const Greater = 'Greater';
export type GreaterInputs = BinaryInputs;

export const GreaterEqual = 'GreaterEqual';
export type GreaterEqualInputs = BinaryInputs;

export const Identity = 'Identity';
export type IdentityInputs = Pick<NamedTensorInfoMap, 'x'>;

export const Imag = 'Imag';
export type ImagInputs = Pick<NamedTensorInfoMap, 'input'>;

export const Less = 'Less';
export type LessInputs = BinaryInputs;

export const LessEqual = 'LessEqual';
export type LessEqualInputs = BinaryInputs;

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

export const Minimum = 'Minimum';
export type MinimumInputs = BinaryInputs;

export const Mod = 'Mod';
export type ModInputs = BinaryInputs;

export const Multiply = 'Multiply';
export type MultiplyInputs = BinaryInputs;

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

export const NonMaxSuppressionV5 = 'NonMaxSuppressionV5';
export type NonMaxSuppressionV5Inputs =
    Pick<NamedTensorInfoMap, 'boxes'|'scores'>;
export interface NonMaxSuppressionV5Attrs {
  maxOutputSize: number;
  iouThreshold: number;
  scoreThreshold: number;
  softNmsSigma: number;
}

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

export const Real = 'Real';
export type RealInputs = Pick<NamedTensorInfoMap, 'input'>;

export const Relu = 'Relu';
export type ReluInputs = Pick<NamedTensorInfoMap, 'x'>;

export const Relu6 = 'Relu6';
export type Relu6Inputs = Pick<NamedTensorInfoMap, 'x'>;

export const SelectV2 = 'SelectV2';
export type SelectV2Inputs = Pick<NamedTensorInfoMap, 'condition'|'t'|'e'>;

export const Selu = 'Selu';
export type SeluInputs = Pick<NamedTensorInfoMap, 'x'>;

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

export const SquaredDifference = 'SquaredDifference';
export type SquaredDifferenceInputs = BinaryInputs;

export const Square = 'Square';
export type SquareInputs = Pick<NamedTensorInfoMap, 'x'>;

export const Sub = 'Sub';
export type SubInputs = BinaryInputs;

export const Tile = 'Tile';
export type TileInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface TileAttrs {
  reps: number[];
}

export const Transpose = 'Transpose';
export type TransposeInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface TransposeAttrs {
  perm: number[];
}

/**
 * TensorFlow.js-only kernels
 */
export const FromPixels = 'FromPixels';
export interface FromPixelsInputs {
  pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement;
}
export interface FromPixelsAttrs {
  numChannels: number;
}
