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
import {NamedTensorInfoMap, TensorInfo} from './kernel_registry';
import {PixelData} from './types';

export const Add = 'Add';
export type AddInputs = BinaryInputs;

export const AddN = 'AddN';
export type AddNInputs = TensorInfo[];

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

export const Concat = 'Concat';
export type ConcatInputs = TensorInfo[];
export interface ConcatAttrs {
  axis: number;
}

export const Conv2D = 'Conv2D';
export type Conv2DInputs = Pick<NamedTensorInfoMap, 'x'|'filter'>;
export interface Conv2DAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  dataFormat: 'NHWC'|'NCHW';
  dilations: [number, number]|number;
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const Conv2DBackpropFilter = 'Conv2DBackpropFilter';
export type Conv2DBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x'|'dy'>;
export interface Conv2DBackpropFilterAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  dataFormat: 'NHWC'|'NCHW';
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const Conv2DBackpropInput = 'Conv2DBackpropInput';
export type Conv2DBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy'|'filter'>;
export interface Conv2DBackpropInputAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
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

export const Equal = 'Equal';
export type EqualInputs = BinaryInputs;

export const FusedBatchNorm = 'FusedBatchNorm';
export type FusedBatchNormInputs =
    Pick<NamedTensorInfoMap, 'x'|'scale'|'offset'|'mean'|'variance'>;
export interface FusedBatchNormAttrs {
  varianceEpsilon: number;
}

export const Greater = 'Greater';
export type GreaterInputs = BinaryInputs;

export const GreaterEqual = 'GreaterEqual';
export type GreaterEqualInputs = BinaryInputs;

export const Identity = 'Identity';
export type IdentityInputs = Pick<NamedTensorInfoMap, 'x'>;

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

export const MaxPoolWithArgmax = 'MaxPoolWithArgmax';
export type MaxPoolWithArgmaxInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxPoolWithArgmaxAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  includeBatchInIndex: boolean;
}

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

export const Max = 'Max';
export type MaxInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxAttrs {
  reductionIndices: number|number[];
  keepDims: boolean;
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
