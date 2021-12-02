/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {getMainHeaderAndGlobalIndexString} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class ToCanvasProgram implements WebGPUProgram {
  variableNames = ['Image'];
  outputShape: number[];
  uniforms = 'depth: i32;';
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  workPerThread = 4;
  size = true;
  type: DataType;

  constructor(outShape: number[], type: DataType) {
    this.outputShape = outShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.shaderKey = `toCanvas_${type}`;
    this.type = type;
  }

  getUserCode(): string {
    let calculateResult;
    if (this.type === 'float32') {
      calculateResult = `
      if (uniforms.numChannels == 1) {
        rgba[0] = value;
        rgba[1] = value;
        rgba[2] = value;
      } else {
        rgba[d] = value;
      }`;
    } else {
      calculateResult = `
      if (uniforms.numChannels == 1) {
        rgba[0] = value / 255.0;
        rgba[1] = value / 255.0;
        rgba[2] = value / 255.0;
      } else {
        rgba[d] = value / 255.0;
      }`;
    }

    const userCode = `
      [[group(0), binding(0)]] var outImage : texture_storage_2d<rgba8unorm, write>;
      [[group(0), binding(1)]] var<storage, read> inBuf : Matrix0;

      ${getMainHeaderAndGlobalIndexString()}
            let flatIndex = index * 4;
            if (flatIndex < uniforms.size) {
              var rgba = vec4<f32>(0.0, 0.0, 0.0, 1.0);

              for (var d = 0; d < uniforms.numChannels; d = d + 1) {
                let value = f32(inBuf.numbers[index * uniforms.numChannels + d]);
                ${calculateResult}
              }

              let coords = getCoordsFromFlatIndex(flatIndex);
              textureStore(outImage, vec2<i32>(coords.yx), rgba);
            }
          }
        `;
    return userCode;
  }
}
