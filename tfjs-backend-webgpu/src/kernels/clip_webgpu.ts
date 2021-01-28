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
import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class ClipProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  variableNames = ['A'];
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workPerThread = 1;
  workGroupSize: [number, number, number] = [64, 1, 1];
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
    this.shaderKey = `clip_${minVal}_${maxVal}`;
  }

  getUserCode(): string {
    const type = getCoordsDataType(this.outputShape.length);
    const size = util.sizeFromShape(this.outputShape);
    const userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);
        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < ${size}) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);

            float value = getAAtOutCoords(coords);
            if (isnan(value)) {
              setOutput(flatIndex, value);
              return;
            }

            setOutput(flatIndex, clamp(value, ${this.minVal}, ${this.maxVal}));
          }
        }
      }
    `;
    return userCode;
  }
}
