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

import {backend_util, util} from '@tensorflow/tfjs-core';
import {getCoordsDataType} from '../shader_preprocessor';

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class BinaryOpUniformProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workPerThread: number;
  workGroupSize: [number, number, number];
  needsShapesUniforms = true;
  useVariableUniforms: boolean[];
  variableUniforms: string[];

  constructor(
      op: string, aShape: number[], bShape: number[],
      useUniformWithB: boolean) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 256;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    const size = util.sizeFromShape(this.outputShape);
    const lastDimensionSize = useUniformWithB ? bShape[0] : aShape[0];
    this.workPerThread = 1;
    if (lastDimensionSize === 1) {
      this.variableUniforms =
          useUniformWithB ? ['', 'float B'] : ['float A', ''];
    } else if (lastDimensionSize === 2) {
      this.variableUniforms = useUniformWithB ? ['', 'vec2 B'] : ['vec2 A', ''];
    } else if (lastDimensionSize === 3) {
      this.variableUniforms = useUniformWithB ? ['', 'vec3 B'] : ['vec3 A', ''];
    } else if (lastDimensionSize === 4) {
      this.variableUniforms = useUniformWithB ? ['', 'vec4 B'] : ['vec4 A', ''];
    } else {
      this.workGroupSize = [16, 1, 1];
      this.workPerThread = 4;
      this.variableUniforms = useUniformWithB ?
          ['', `vec4 B[${lastDimensionSize / 4}]`] :
          [`vec4 A[${lastDimensionSize / 4}]`, ''];
    }

    this.useVariableUniforms = useUniformWithB ? [false, true] : [true, false];
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    const type = getCoordsDataType(this.outputShape.length);
    const sharedIndexSnippet = `coords[${this.outputShape.length - 1}] / 4`;
    if (this.workPerThread === 4) {
      const accessDataSnippet = useUniformWithB ?
          `vec4 a = vec4(A[flatIndex], A[flatIndex+1], A[flatIndex + 2], A[flatIndex +3]);
           vec4 b = B[${sharedIndexSnippet}];` :
          `vec4 a = A[${sharedIndexSnippet}];
          vec4 b = vec4(B[flatIndex], B[flatIndex+1], B[flatIndex + 2], B[flatIndex + 3]);`;
      this.userCode = `
          vec4 binaryOperation(vec4 a, vec4 b) {
            ${op}
          }

          void main() {
            int index = int(gl_GlobalInvocationID.x);
            int flatIndex = index * 4;
            ${type} coords = getCoordsFromFlatIndex(flatIndex);
            ${accessDataSnippet}
            vec4 resData = binaryOperation(a, b);
            result[flatIndex] = resData.x;
            result[flatIndex + 1] = resData.y;
            result[flatIndex + 2] = resData.z;
            result[flatIndex + 3] = resData.w;
          }
          `;
    } else {
      const sharedIndexSnippet = lastDimensionSize > 1 ?
          `[coords[${this.outputShape.length - 1}]]` :
          '';
      const accessDataSnippet = useUniformWithB ?
          `float a = getAAtOutCoords(coords);
           float b = B${sharedIndexSnippet};` :
          `float a = A${sharedIndexSnippet};
           float b = getBAtOutCoords(coords);`;
      const sizeFit = size % (workGroupSizeX * this.workPerThread) === 0;
      const writeDataSnippet = sizeFit ?
          `${type} coords = getCoordsFromFlatIndex(index);

           ${accessDataSnippet}
           setOutput(index, binaryOperation(a, b));` :
          `if(index < ${size}) {
              ${type} coords = getCoordsFromFlatIndex(index);

              ${accessDataSnippet}
              setOutput(index, binaryOperation(a, b));
            }`;
      this.userCode = `
          float binaryOperation(float a, float b) {
            ${op}
          }

          void main() {
            int index = int(gl_GlobalInvocationID.x);
            ${writeDataSnippet}
          }
          `;
    }
  }
}
