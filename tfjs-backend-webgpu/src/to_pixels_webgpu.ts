/**
 * @license
 * Copyright 2022 Google LLC.
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

import {DataType} from '@tensorflow/tfjs-core';

import {getMainHeaderString as main, PixelsOpType, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ToPixelsProgram implements WebGPUProgram {
  variableNames = ['Image'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [64, 1, 1];
  workPerThread = 4;
  type: DataType;
  textureFormat: GPUTextureFormat;
  isPixelsOp = PixelsOpType.TO_PIXELS;

  constructor(
      outShape: number[], type: DataType, textureFormat: GPUTextureFormat) {
    this.outputShape = outShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize,
        [this.workPerThread, 1, 1]);
    this.shaderKey = `toPixels_${type}`;
    this.type = type;
    this.textureFormat = textureFormat;
  }

  getUserCode(): string {
    const value = this.type === 'float32' ? 'value' : 'value / 255.0';
    const calculateResult = `
      if (uniforms.numChannels == 1) {
        rgba[0] = ${value};
        rgba[1] = ${value};
        rgba[2] = ${value};
      } else {
        rgba[d] = ${value};
      }`;

    const userCode = `
       @group(0) @binding(0) var outImage : texture_storage_2d<${
        this.textureFormat}, write>;
       ${main('index')} {
         let flatIndex = index * 4;
         if (flatIndex < uniforms.size) {
           var rgba = vec4<f32>(0.0, 0.0, 0.0, 1.0);
           for (var d = 0; d < uniforms.numChannels; d = d + 1) {
             let value = f32(inBuf[index * uniforms.numChannels + d]);
             ${calculateResult}
           }
           let coords = getCoordsFromIndex(flatIndex);
           textureStore(outImage, vec2<i32>(coords.yx), rgba);
         }
       }
      `;
    return userCode;
  }
}
