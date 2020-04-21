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

export type BinaryInputs = Pick<NamedTensorInfoMap, 'a'|'b'>;

export const Div = 'Div';
export type DivInputs = BinaryInputs;

export const FusedBatchNorm = 'FusedBatchNorm';
export type FusedBatchNormInputs =
    Pick<NamedTensorInfoMap, 'x'|'scale'|'offset'|'mean'|'variance'>;
export interface FusedBatchNormAttrs {
  varianceEpsilon: number;
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

export const Conv2DBackpropInput = 'Conv2DBackpropInput';
export type Conv2DBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy'|'filter'>;
export interface Conv2DBackpropInputAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  dataFormat: 'NHWC'|'NCHW';
  dimRoundingMode?: 'floor'|'round'|'ceil';
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

export const Conv3D = 'Conv3D';
export type Conv3DInputs = Pick<NamedTensorInfoMap, 'x'|'filter'>;
export interface Conv3DAttrs {
  strides: [number, number, number]|number;
  pad: 'valid'|'same';
  dataFormat: 'NDHWC'|'NCDHW';
  dilations: [number, number, number]|number;
}

export const Conv3DBackpropFilter = 'Conv3DBackpropFilter';
export type Conv3DBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x'|'y'>;

export const Conv3DBackpropInput = 'Conv3DBackpropInput';
export type Conv3DBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy'>;

export const NotEqual = 'NotEqual';
export type NotEqualInputs = BinaryInputs;

export const SquaredDifference = 'SquaredDifference';
export type SquaredDifferenceInputs = BinaryInputs;

export const Square = 'Square';
export type SquareInputs = Pick<NamedTensorInfoMap, 'x'>;

export const Sub = 'Sub';
export type SubInputs = BinaryInputs;

export const Transpose = 'Transpose';
export type TransposeInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface TransposeAttrs {
  perm: number[];
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

export const BroadcastTo = 'BroadcastTo';
export type BroadcastToInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface BroadCastToAttrs {
  shape: number[];
  inputShape: number[];  // for gradient
}

export const OneHot = 'OneHot';
export type OneHotInputs = Pick<NamedTensorInfoMap, 'indices'>;
export interface OneHotAttrs {
  depth: number;
  onValue: number;
  offValue: number;
}

export const Identity = 'Identity';
export type IdentityInputs = Pick<NamedTensorInfoMap, 'x'>;

export const Tile = 'Tile';
export type TileInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface TileAttrs {
  reps: number[];
}

export const PadV2 = 'PadV2';
export type PadV2Inputs = Pick<NamedTensorInfoMap, 'x'>;
export interface PadV2Attrs {
  paddings: Array<[number, number]>;
  constantValue: number;
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

export const MaxPoolWithArgmax = 'MaxPoolWithArgmax';
export type MaxPoolWithArgmaxInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxPoolWithArgmaxAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  includeBatchInIndex: boolean;
}
