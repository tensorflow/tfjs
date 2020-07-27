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

import {backend_util, util} from '@tensorflow/tfjs-core';
import {getCoordsDataType} from '../shader_preprocessor';

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class BinaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workPerThread: number;
  workGroupSize: [number, number, number];

  constructor(op: string, aShape: number[], bShape: number[]) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    const size = util.sizeFromShape(this.outputShape);
    const sizeFit = size % workGroupSizeX === 0;
    const shapesFit = util.arraysEqual(aShape, bShape) && sizeFit;
    this.workPerThread = shapesFit || sizeFit ? 1 : 2;

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    if (shapesFit) {
      this.userCode = `
          float binaryOperation(float a, float b) {
            ${op}
          }

          void main() {
            int index = int(gl_GlobalInvocationID.x);

            float a = A[index];
            float b = B[index];
            setOutput(index, binaryOperation(a, b));
          }
        `;
      this.shaderKey = `binary2${op}`;
    } else if (sizeFit) {
      const type = getCoordsDataType(this.outputShape.length);
      this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        ${type} coords = getCoordsFromFlatIndex(index);

        float a = getAAtOutCoords(coords);
        float b = getBAtOutCoords(coords);
        setOutput(index, binaryOperation(a, b));
      }
      `;
    } else {
      const type = getCoordsDataType(this.outputShape.length);

      this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          if(flatIndex < ${size}) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);

            float a = getAAtOutCoords(coords);
            float b = getBAtOutCoords(coords);
            setOutput(flatIndex, binaryOperation(a, b));
          }
        }
      }
      `;
      this.shaderKey = `binary${op}${type}${size}`;
    }
  }
}
