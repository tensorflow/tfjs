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

import {util} from '@tensorflow/tfjs-core';

import {getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class Im2ColProgram implements WebGPUProgram {
  variableNames = ['A'];
  uniforms = `ivec2 pad, stride, dilation; int outWidth, itemsPerBlockRow,
      inChannels;`;
  uniformsWgsl =
      `pad : vec2<i32>; stride : vec2<i32>; dilation : vec2<i32>; outWidth : i32; itemsPerBlockRow : i32;
      inChannels : i32;`;
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workPerThread = 4;
  workGroupSize: [number, number, number] = [64, 1, 1];
  isChannelsLast: boolean;
  size: number;
  useWgsl: boolean;

  constructor(outputShape: number[], isChannelsLast: boolean) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.isChannelsLast = isChannelsLast;
    this.shaderKey = `im2col_${this.isChannelsLast}`;
    this.size = util.sizeFromShape(this.outputShape);
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const rowDim = this.isChannelsLast ? 0 : 1;
    const colDim = this.isChannelsLast ? 1 : 2;

    const userCode = `
      void main() {
        int index = getGlobalIndex();

        for(int i=0; i<${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          ivec2 rc = getCoordsFromFlatIndex(flatIndex);

          if(flatIndex < size) {
            int blockIndex = rc[0];
            int pos = rc[1];

            int offsetY = int(blockIndex / outWidth) * stride[1] - pad[1];
            int d0 = offsetY + dilation[1] * (pos / itemsPerBlockRow);
            float value = 0.0;
            if(d0 < aShape[${rowDim}] && d0 >= 0) {
              int offsetX = int(mod(blockIndex, outWidth) * stride[0] -
                pad[0]);
              int d1 = offsetX + dilation[0] * (int(mod(pos,
                itemsPerBlockRow) / inChannels));
              int ch = int(mod(pos, inChannels));
              if(d1 < aShape[${colDim}] && d1 >= 0) {
                value = getA(d0, d1, ch);
              }
            }
            setOutput(flatIndex, value);
          }
        }
      }
    `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const rowDim = this.isChannelsLast ? 0 : 1;
    const colDim = this.isChannelsLast ? 1 : 2;

    const userCode = `
    ${getMainHeaderStringWgsl()} {
      ${getGlobalIndexStringWgsl()}

      for(var i = 0; i<${this.workPerThread}; i = i + 1) {
        let flatIndex = index * ${this.workPerThread} + i;

        let rc = getCoordsFromFlatIndex(flatIndex);

        if(flatIndex < uniforms.size) {
          let blockIndex = rc[0];
          let pos = rc[1];

          let offsetY = blockIndex / uniforms.outWidth * uniforms.stride[1] - uniforms.pad[1];
          let d0 = offsetY + uniforms.dilation[1] * pos / uniforms.itemsPerBlockRow;
          var value = 0.0;
          if(d0 < uniforms.aShape[${rowDim}] && d0 >= 0) {
            let offsetX = (blockIndex % uniforms.outWidth) * uniforms.stride[0] -
              uniforms.pad[0];
            let d1 = offsetX + uniforms.dilation[0] * ((pos %
              uniforms.itemsPerBlockRow) / uniforms.inChannels);
            let ch = pos % uniforms.inChannels;
            if(d1 < uniforms.aShape[${colDim}] && d1 >= 0) {
              value = getA(d0, d1, ch);
            }
          }
          setOutputFlat(flatIndex, value);
        }
      }
    }
  `;
    return userCode;
  }
}
