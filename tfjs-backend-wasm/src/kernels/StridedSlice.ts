/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {backend_util, KernelConfig, KernelFunc, StridedSlice, StridedSliceAttrs, StridedSliceInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {reshape} from './Reshape';
import {slice} from './Slice';

let wasmStridedSlice: (
    xId: number, blockSize: number, channelsLast: number, xStrides: Uint8Array,
    xStridesLength: number, outputShape: Uint8Array, outputStrides: Uint8Array,
    outSize: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmStridedSlice = backend.wasm.cwrap(StridedSlice, null /*void*/, [
    'number',  // xId
    'number',  // blockSize
    'number',  // channelsLast
    'array',   // xStrides
    'number',  // xStridesLength
    'array',   // outputShape
    'array',   // outputStrides
    'number',  // outSize
    'number',  // outId
  ]);
}

export function stridedSlice(args: {
  backend: BackendWasm,
  inputs: StridedSliceInputs,
  attrs: StridedSliceAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {x} = inputs;

  let {begin, end, strides} = attrs;
  const {beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask} = attrs;

  const ellipsisAxes = backend_util.slice_util.maskToAxes(ellipsisMask);
  if (ellipsisAxes.length > 1) {
    throw new Error('Multiple ellipses in slice is not allowed.');
  }

  if (ellipsisMask !== 0 && newAxisMask !== 0) {
    throw new Error(
        'Using both ellipsisMask and newAxisMask is not yet supported.');
  }

  if (ellipsisMask !== 0 && shrinkAxisMask !== 0) {
    throw new Error(
        'Using both ellipsisMask and shrinkAxisMask is not yet supported.');
  }

  const numInterpolatedAxes = x.shape.length - begin.length;

  // Expand the dims of x based on the newAxisMask.
  const expandAxes = backend_util.slice_util.maskToAxes(newAxisMask);
  const newShape = x.shape.slice();
  expandAxes.forEach(axis => {
    begin[axis] = 0;
    end[axis] = 1;
    newShape.splice(axis, 0, 1);
  });

  const xReshaped = reshape({inputs: {x}, attrs: {shape: newShape}, backend});

  // Normalize the start, end and strides.
  if (ellipsisAxes.length && numInterpolatedAxes > 0) {
    const fullIndex = ellipsisAxes[0];

    // The ellipsis applies to the masked index as well as any dimensions
    // that are interpolated.
    const numElidedAxes = numInterpolatedAxes + 1;
    begin = backend_util.slice_util.startIndicesWithElidedDims(
        beginMask, fullIndex, numElidedAxes, begin, xReshaped.shape);
    end = backend_util.slice_util.stopIndicesWithElidedDims(
        endMask, fullIndex, numElidedAxes, end, xReshaped.shape);
    strides = backend_util.slice_util.stridesWithElidedDims(
        strides, fullIndex, numElidedAxes, xReshaped.shape);
  } else {
    for (let axis = 0; axis < xReshaped.shape.length; axis++) {
      begin[axis] = backend_util.slice_util.startForAxis(
          beginMask, begin, strides, xReshaped.shape, axis, ellipsisMask);
      end[axis] = backend_util.slice_util.stopForAxis(
          endMask, end, strides, xReshaped.shape, axis, ellipsisMask);
      strides[axis] =
          backend_util.slice_util.stridesForAxis(strides, axis, ellipsisMask);
    }
  }

  const shrinkAxes = backend_util.slice_util.maskToAxes(shrinkAxisMask);
  // Adjust the ends based on the shrink mask.
  shrinkAxes.forEach(axis => {
    end[axis] = begin[axis] + 1;
    strides[axis] = 1;
  });

  // Figure out the output shape.
  const size = backend_util.slice_util.computeOutShape(begin, end, strides);
  // Remove the axes based on shrinkMask.
  const outShape = size.filter((_, axis) => shrinkAxes.indexOf(axis) === -1);

  const nonStrided = strides.every(v => v === 1);
  if (nonStrided) {
    const xSliced = slice({inputs: {x}, attrs: {begin, size}, backend});
    return reshape({inputs: {x: xSliced}, attrs: {shape: outShape}, backend});
  }

  const out = backend.makeOutput(outShape, 'float32');

  wasmStridedSlice(xReshaped, begin, end, strides);

  return reshape({inputs: {x: out}, attrs: {shape: outShape}, backend});

  // const batchSize = x.shape[0];
  // const inputHeight = (dataFormat === 'NHWC') ? x.shape[1] : x.shape[2];
  // const inputWidth = (dataFormat === 'NHWC') ? x.shape[2] : x.shape[3];
  // const inputDepth = (dataFormat === 'NHWC') ? x.shape[3] : x.shape[1];

  // const outputHeight = inputHeight * blockSize;
  // const outputWidth = inputWidth * blockSize;
  // const outputDepth = inputDepth / (blockSize * blockSize);

  // const outputShape = (dataFormat === 'NHWC') ?
  //     [batchSize, outputHeight, outputWidth, outputDepth] :
  //     [batchSize, outputDepth, outputHeight, outputWidth];

  // const out = backend.makeOutput(outputShape, 'float32');

  // const xData = backend.dataIdMap.get(x.dataId);
  // const xId = xData.id;
  // const xStridesBytes =
  //     new Uint8Array(new Int32Array(util.computeStrides(x.shape)).buffer);

  // const outputShapeBytes = new Uint8Array(new
  // Int32Array(outputShape).buffer); const outStridesBytes =
  //     new Uint8Array(new
  //     Int32Array(util.computeStrides(outputShape)).buffer);

  // const outId = backend.dataIdMap.get(out.dataId).id;
  // const channelsLast = dataFormat === 'NHWC' ? 1 : 0;
  // wasmStridedSlice(
  //     xId, blockSize, channelsLast, xStridesBytes, x.shape.length - 1,
  //     outputShapeBytes, outStridesBytes, outputShape.length, outId);

  // return out;
}

export const stridedSliceConfig: KernelConfig = {
  kernelName: StridedSlice,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: stridedSlice as {} as KernelFunc
};
