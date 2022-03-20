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

import {backend_util, FusedDepthwiseConv2D, FusedDepthwiseConv2DAttrs, FusedDepthwiseConv2DInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {DepthwiseConv2D3x3PHWC4Program} from '../depthwise_conv2d_3x3_phwc4_webgpu';
import {DepthwiseConv2D3x3Phwc4Stride2Program} from '../depthwise_conv2d_3x3_stride2_phwc4_webgpu';
import {DepthwiseConv2D3x3Program} from '../depthwise_conv2d_3x3_webgpu';
import {DepthwiseConv2DProgram} from '../depthwise_conv2d_webgpu';
import {DataLayout, DivideRoundUp} from '../webgpu_util'

import {toPHWC4} from './ToPHWC4'

export function fusedDepthwiseConv2D(args: {
  inputs: FusedDepthwiseConv2DInputs,
  attrs: FusedDepthwiseConv2DAttrs,
  backend: WebGPUBackend
}) {
  const {inputs, backend, attrs} = args;
  const {x, filter, bias, preluActivationWeights} = inputs;
  const {strides, pad, dilations, dimRoundingMode, activation} = attrs;

  let $dilations = dilations;
  if ($dilations == null) {
    $dilations = [1, 1];
  }

  util.assert(
      backend_util.eitherStridesOrDilationsAreOne(strides, $dilations),
      () => 'Error in depthwiseConv2d: Either strides or dilations must be ' +
          `1. Got strides ${strides} and dilations '${$dilations}'`);

  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, $dilations,
      pad, dimRoundingMode, true /* depthwise */);

  const programInputs: TensorInfo[] = [];

  const hasBias = bias != null;
  const hasPreluActivationWeights = preluActivationWeights != null;

  let program: DepthwiseConv2DProgram|DepthwiseConv2D3x3Program|
      DepthwiseConv2D3x3PHWC4Program|DepthwiseConv2D3x3Phwc4Stride2Program;
  // TODO: To see if we need to relax the limitation. Currently, it's only for
  // filter size 3x3.
  if (convInfo.batchSize === 1 && convInfo.inHeight === convInfo.outHeight &&
      convInfo.inWidth === convInfo.outWidth && convInfo.strideHeight === 1 &&
      convInfo.strideWidth === 1 &&
      convInfo.filterHeight === convInfo.filterWidth &&
      convInfo.inChannels === convInfo.outChannels &&
      convInfo.filterHeight === 3 && convInfo.inChannels % 4 === 0) {
    //   program = new DepthwiseConv2D3x3Program(
    //       convInfo, hasBias, activation, hasPreluActivationWeights);
    const xInfo = backend.tensorMap.get(x.dataId);
    const programInputs: TensorInfo[] = [];
    let tX: TensorInfo;
    if (xInfo.layout == DataLayout.NHWC) {
      tX = toPHWC4(x, backend);
      programInputs.push(tX);
    } else {
      programInputs.push(x);
    }
    programInputs.push(filter);

    const weightsInfo = backend.tensorMap.get(filter.dataId);
    weightsInfo.layout = DataLayout.OHW10;
    if (hasBias) {
      programInputs.push(bias);
    }
    if (hasPreluActivationWeights) {
      programInputs.push(preluActivationWeights);
    }
    program = new DepthwiseConv2D3x3PHWC4Program(
        convInfo, hasBias, activation, hasPreluActivationWeights);
    const dst_slices = DivideRoundUp(x.shape[3], 4);
    const src_sliceStride = x.shape[1] * x.shape[2];
    const dimensions = [
      {type: 'int32', data: [src_sliceStride]},
      {type: 'int32', data: [dst_slices]}
    ];
    const res =
        backend.runWebGPUProgram(program, programInputs, 'float32', dimensions);
    const resInfo = backend.tensorMap.get(res.dataId);
    resInfo.layout = DataLayout.PHWC4;
    if (tX != null) {
      backend.disposeData(tX.dataId);
    }
    return res;
  } else if (
      convInfo.batchSize === 1 && convInfo.strideHeight === 2 &&
      convInfo.strideWidth === 2 &&
      convInfo.filterHeight === convInfo.filterWidth &&
      convInfo.filterHeight === 3 && convInfo.inChannels % 4 === 0) {
    const xInfo = backend.tensorMap.get(x.dataId);
    const programInputs: TensorInfo[] = [];
    let tX: TensorInfo;
    if (xInfo.layout == DataLayout.NHWC) {
      tX = toPHWC4(x, backend);
      programInputs.push(tX);
    } else {
      programInputs.push(x);
    }
    programInputs.push(filter);

    const weightsInfo = backend.tensorMap.get(filter.dataId);
    weightsInfo.layout = DataLayout.OHW10;
    if (hasBias) {
      programInputs.push(bias);
    }
    if (hasPreluActivationWeights) {
      programInputs.push(preluActivationWeights);
    }
    program = new DepthwiseConv2D3x3Phwc4Stride2Program(
        convInfo, hasBias, activation, hasPreluActivationWeights);
    const dst_slices = DivideRoundUp(x.shape[3], 4);
    const src_sliceStride = x.shape[1] * x.shape[2];
    const dimensions = [
      {type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left]},
      {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
      {type: 'int32', data: [src_sliceStride]},
      {type: 'int32', data: [dst_slices]}
    ];
    const res =
        backend.runWebGPUProgram(program, programInputs, 'float32', dimensions);
    const resInfo = backend.tensorMap.get(res.dataId);
    resInfo.layout = DataLayout.PHWC4;
    if (tX != null) {
      backend.disposeData(tX.dataId);
    }
    return res;
  } else {
    program = new DepthwiseConv2DProgram(
        convInfo, hasBias, activation, hasPreluActivationWeights);
    const dimensions = [
      {type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left]},
      {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
      {type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth]},
      {type: 'int32', data: [convInfo.inHeight, convInfo.inWidth]},
      {type: 'int32', data: [convInfo.filterHeight]},
      {type: 'int32', data: [convInfo.filterWidth]},
      {type: 'int32', data: [convInfo.outChannels / convInfo.inChannels]}
    ];
    programInputs.push(x);
    programInputs.push(filter);
    if (hasBias) {
      programInputs.push(bias);
    }
    if (hasPreluActivationWeights) {
      programInputs.push(preluActivationWeights);
    }
    const result =
        backend.runWebGPUProgram(program, programInputs, 'float32', dimensions);

    return result;
  }
}

export const fusedDepthwiseConv2DConfig: KernelConfig = {
  kernelName: FusedDepthwiseConv2D,
  backendName: 'webgpu',
  kernelFunc: fusedDepthwiseConv2D as {} as KernelFunc,
};
