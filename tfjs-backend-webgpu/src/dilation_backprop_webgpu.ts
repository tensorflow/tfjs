/**
 * @license
 * Copyright 2023 Google LLC.
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

import {backend_util, DataType} from '@tensorflow/tfjs-core';

import {atomicAddSnippet} from './shader_util';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class Dilation2DBackpropInputProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'w', 'dy'];
  uniforms =
      'filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>, dySize: i32,';
  workgroupSize: [number, number, number] = [64, 1, 1];
  atomic = true;
  type: DataType;

  constructor(convInfo: backend_util.Conv2DInfo, outputDtype: DataType) {
    this.outputShape = convInfo.inShape;
    this.dispatchLayout = flatDispatchLayout(convInfo.outShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, convInfo.outShape, this.workgroupSize);

    if (outputDtype !== 'float32' && outputDtype !== 'int32') {
      throw new Error(`Dilation2DBackpropInput only supports float32 and int32
          types, does not support ${outputDtype} type.`);
    }
    this.type = outputDtype;
    this.shaderKey = 'dilation2DBackpropInput';
  }

  getUserCode(): string {
    // This implementation follows the TF c++ cuda implementation:
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/dilation_ops_gpu.cu.cc
    const userCode = `
       ${main('index')} {
         if (index < uniforms.dySize) {
           let coords = getDyCoordsFromIndex(index);
           let b = coords[0];
           let r = coords[1];
           let c = coords[2];
           let d = coords[3];

           let dyCorner = vec2<i32>(r, c) * uniforms.strides - uniforms.pads;
           var curVal = -3.4e38;  // neg_infinity
           var xRMax = 0;
           var xCMax = 0;

           // In the case of multiple argmax branches, we only back-propagate
           // along the last branch, i.e., the one with largest value of
           // 'wR * uniforms.filterDims[1] + wC', similarly to the max-pooling
           // backward routines.
           for (var wR = 0; wR < uniforms.filterDims[0]; wR++) {
             let xR = dyCorner.x + wR * uniforms.dilations[0];

             if (xR >= 0 && xR < uniforms.xShape[1]) {
               for (var wC = 0; wC < uniforms.filterDims[1]; wC++) {
                 let xC = dyCorner.y + wC * uniforms.dilations[1];

                 if (xC >= 0 && xC < uniforms.xShape[2]) {
                   let val = getX(b, xR, xC, d) + getW(wR, wC, d);
                   if (val > curVal) {
                     curVal = val;
                     xRMax = xR;
                     xCMax = xC;
                   }
                 }
               }
             }
           }

           let flatIndexIn = d + uniforms.xShape[3] *
               (xCMax + uniforms.xShape[2] * (xRMax + uniforms.xShape[1] * b));
           let value = getDy(b, r, c, d);
           ${
        atomicAddSnippet(
            '&result[flatIndexIn]', 'value', this.type as 'float32' | 'int32')}
         }
       }
     `;
    return userCode;
  }
}

export class Dilation2DBackpropFilterProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'w', 'dy'];
  uniforms =
      'filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>, dySize: i32,';
  workgroupSize: [number, number, number] = [64, 1, 1];
  atomic = true;
  type: DataType;

  constructor(
      convInfo: backend_util.Conv2DInfo, shape: number[],
      outputDtype: DataType) {
    this.outputShape = convInfo.filterShape;
    this.dispatchLayout = flatDispatchLayout(convInfo.outShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, convInfo.outShape, this.workgroupSize);

    if (outputDtype !== 'float32' && outputDtype !== 'int32') {
      throw new Error(`Dilation2DBackpropFilter only supports float32 and int32
          types, does not support ${outputDtype} type.`);
    }
    this.type = outputDtype;
    this.shaderKey = 'dilation2DBackpropFilter';
  }

  getUserCode(): string {
    // This implementation follows the TF c++ cuda implementation:
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/dilation_ops_gpu.cu.cc
    const userCode = `
       ${main('index')} {
         if (index < uniforms.dySize) {
           let coords = getDyCoordsFromIndex(index);
           let b = coords[0];
           let r = coords[1];
           let c = coords[2];
           let d = coords[3];

           let dyCorner = vec2<i32>(r, c) * uniforms.strides - uniforms.pads;
           var curVal = -3.4e38;  // neg_infinity
           var wRMax = 0;
           var wCMax = 0;

           // In the case of multiple argmax branches, we only back-propagate
           // along the last branch, i.e., the one with largest value of
           // 'wR * uniforms.filterDims[1] + wC', similarly to the max-pooling
           // backward routines.
           for (var wR = 0; wR < uniforms.filterDims[0]; wR++) {
             let xR = dyCorner.x + wR * uniforms.dilations[0];

             if (xR >= 0 && xR < uniforms.xShape[1]) {
               for (var wC = 0; wC < uniforms.filterDims[1]; wC++) {
                 let xC = dyCorner.y + wC * uniforms.dilations[1];

                 if (xC >= 0 && xC < uniforms.xShape[2]) {
                   let val = getX(b, xR, xC, d) + getW(wR, wC, d);
                   if (val > curVal) {
                     curVal = val;
                     wRMax = wR;
                     wCMax = wC;
                   }
                 }
               }
             }
           }

           let flatIndexIn = d + uniforms.wShape[2] * (wCMax + wRMax * uniforms.wShape[1]);
           let value = getDy(b, r, c, d);
           ${
        atomicAddSnippet(
            '&result[flatIndexIn]', 'value', this.type as 'float32' | 'int32')}
         }
       }
     `;
    return userCode;
  }
}
