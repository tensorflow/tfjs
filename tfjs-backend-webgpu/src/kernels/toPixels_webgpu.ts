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

import {getMainHeaderAndGlobalIndexString} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class ToPixelsProgram implements WebGPUProgram {
  variableNames = ['Image'];
  outputShape: number[];
  uniforms = 'multiplier : f32; depth: i32;';
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  workPerThread = 4;
  size = true;

  constructor(outShape: number[]) {
    this.outputShape = outShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.shaderKey = 'toPixels';
  }

  getUserCode(): string {
    const userCode = `
          ${getMainHeaderAndGlobalIndexString()}
            if (index < uniforms.size) {
              let rgba = vec4<f32>(0, 0, 0, 255);

              for (let d = 0; d < uniforms.depth; d++) {
                let value = f32(Image[index * uniforms.depth + d]);

                if (uniforms.depth === 1) {
                  rgba[0] = value * uniforms.multiplier;
                  rgba[1] = value * uniforms.multiplier;
                  rgba[2] = value * uniforms.multiplier;
                } else {
                  rgba[d] = value * uniforms.multiplier;
                }
              }

              let j = index * 4;
              setOutputFlat(j, round(rgba[0]));
              setOutputFlat(j + 1, round(rgba[1]));
              setOutputFlat(j + 2, round(rgba[2]));
              setOutputFlat(j + 3, round(rgba[3]));
            }
          }
        `;
    return userCode;
  }
}
