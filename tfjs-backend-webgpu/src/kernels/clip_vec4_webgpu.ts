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

import {util} from '@tensorflow/tfjs-core';
import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class ClipVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  variableNames = ['A'];
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workPerThread = 4;
  workGroupSize: [number, number, number] = [64, 1, 1];
  isVec4 = true;
  minVal: number;
  maxVal: number;

  constructor(outputShape: number[], minVal: number, maxVal: number) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.minVal = minVal;
    this.maxVal = maxVal;
    this.shaderKey = `clipVec4_${minVal}_${maxVal}`;
  }

  getUserCode(): string {
    const type = getCoordsDataType(this.outputShape.length);
    const size = util.sizeFromShape(this.outputShape);
    const userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);
          if(index < ${size / 4}) {
            ${type} coords = getCoordsFromFlatIndex(index * 4);

            vec4 value = getAAtOutCoords(coords);
            if (any(isnan(value))) {
              setOutput(index, value);
              return;
            }

            setOutput(index, clamp(value, ${this.minVal}, ${this.maxVal}));
          }
      }
    `;
    return userCode;
  }
}
