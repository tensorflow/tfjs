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
      'filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>, outSize: i32,';
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
    this.shaderKey = `dilation2DBackpropInput_${this.type}`;
  }

  getUserCode(): string {
    // This implementation follows the TF c++ cuda implementation:
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/dilation_ops_gpu.cu.cc
    const userCode = `
       ${main('index')} {
         if (index < uniforms.outSize) {
           let d = index % uniforms.dyShape[3];
           let out_idx2 = index / uniforms.dyShape[3];
           let w_out = out_idx2 % uniforms.dyShape[2];
           let out_idx3 = out_idx2 / uniforms.dyShape[2];
           let h_out = out_idx3 % uniforms.dyShape[1];
           let batch = out_idx3 / uniforms.dyShape[1];

           let h_beg = h_out * uniforms.strides[0] - uniforms.pads[0];
           let w_beg = w_out * uniforms.strides[1] - uniforms.pads[1];

           var curVal = -3.4e38;  // neg_infinity
           var h_in_max = select(h_beg, 0, h_beg < 0);
           var w_in_max = select(w_beg, 0, w_beg < 0);

           // In the case of multiple argmax branches, we only back-propagate
           // along the last branch, i.e., the one with largest value of
           // 'h * filter_cols + w', similarly to the max-pooling backward
           // routines.
           for (var h = 0; h < uniforms.filterDims[0]; h++) {
             let h_in = h_beg + h * uniforms.dilations[0];

             if (h_in >= 0 && h_in < uniforms.xShape[1]) {
               for (var w = 0; w < uniforms.filterDims[1]; w++) {
                 let w_in = w_beg + w * uniforms.dilations[1];

                 if (w_in >= 0 && w_in < uniforms.xShape[2]) {
                   let val = getX(batch, h_in, w_in, d) + getW(h, w, d);
                   if (val > curVal) {
                     curVal = val;
                     h_in_max = h_in;
                     w_in_max = w_in;
                   }
                 }
               }
             }
           }

           let flatIndexIn = d + uniforms.xShape[3] *
               (w_in_max + uniforms.xShape[2] * (h_in_max + uniforms.xShape[1] * batch));
           let value = getDy(batch, h_out, w_out, d);
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
      'filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>, outSize: i32,';
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
    this.shaderKey = `dilation2DBackpropFilter_${this.type}`;
  }

  getUserCode(): string {
    // This implementation follows the TF c++ cuda implementation:
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/dilation_ops_gpu.cu.cc
    const userCode = `
       ${main('index')} {
         if (index < uniforms.outSize) {
           let d = index % uniforms.dyShape[3];
           let out_idx2 = index / uniforms.dyShape[3];
           let w_out = out_idx2 % uniforms.dyShape[2];
           let out_idx3 = out_idx2 / uniforms.dyShape[2];
           let h_out = out_idx3 % uniforms.dyShape[1];
           let batch = out_idx3 / uniforms.dyShape[1];

           let h_beg = h_out * uniforms.strides[0] - uniforms.pads[0];
           let w_beg = w_out * uniforms.strides[1] - uniforms.pads[1];

           var curVal = -3.4e38;  // neg_infinity
           var h_w_max = 0;
           var w_w_max = 0;

           // In the case of multiple argmax branches, we only back-propagate
           // along the last branch, i.e., the one with largest value of
           // 'h * filter_cols + w', similarly to the max-pooling backward
           // routines.
           for (var h = 0; h < uniforms.filterDims[0]; h++) {
             let h_in = h_beg + h * uniforms.dilations[0];

             if (h_in >= 0 && h_in < uniforms.xShape[1]) {
               for (var w = 0; w < uniforms.filterDims[1]; w++) {
                 let w_in = w_beg + w * uniforms.dilations[1];

                 if (w_in >= 0 && w_in < uniforms.xShape[2]) {
                   let val = getX(batch, h_in, w_in, d) + getW(h, w, d);
                   if (val > curVal) {
                     curVal = val;
                     h_w_max = h;
                     w_w_max = w;
                   }
                 }
               }
             }
           }

           let flatIndexIn = d + uniforms.wShape[2] * (w_w_max + h_w_max * uniforms.wShape[1]);
           let value = getDy(batch, h_out, w_out, d);
           ${
        atomicAddSnippet(
            '&result[flatIndexIn]', 'value', this.type as 'float32' | 'int32')}
         }
       }
     `;
    return userCode;
  }
}
