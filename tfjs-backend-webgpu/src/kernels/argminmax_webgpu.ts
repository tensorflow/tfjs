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

import {getCoordsDataType, getShapeCoords} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class ArgMinMaxProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number];
  variableNames = ['x'];
  uniforms = 'int axis;';
  inputShape: number[];
  reductionFactor: number;
  op: string;

  constructor(inputShape: number[], axis: number, reduceType: 'min'|'max') {
    const axes = [axis];
    backend_util.assertAxesAreInnerMostDims(
        'arg' + reduceType.charAt(0).toUpperCase() + reduceType.slice(1), axes,
        inputShape.length);

    this.op = reduceType === 'min' ? '<' : '>';

    // |outShape| is the shape with the removed axis
    // |reduceShape| is the shape we are reducing. i.e. [ inputShape[axis] ]
    const [outputShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(inputShape, axes);

    this.outputShape = outputShape.length === 0 ? [1] : outputShape;

    // Length of the axis we're reducing on.
    const reduceSize = util.sizeFromShape(reduceShape);

    // The number of comparisons each thread will do
    this.reductionFactor = 2;
    const xMaxThreads = 1024;  // gl_MaxComputeWorkGroupSize
    const xThreads =
        Math.min(Math.ceil(reduceSize / this.reductionFactor), xMaxThreads);

    this.workGroupSize = [xThreads, 1, 1];

    this.dispatchLayout = {x: [], y: this.outputShape.map((d, i) => i)};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.inputShape = inputShape;
    this.shaderKey = `argMinMax_${this.op}_${reduceSize}`;
  }

  getUserCode(): string {
    // When this.workGroupSize[0] > 1, each thread reduces Length /
    // this.workGroupSize[0] values. Thes results are stored in shared memory
    // and iteratively reduced.
    const reduceInSharedMemory = this.workGroupSize[0] > 1;
    const sharedMemorySnippet = `
      shared int xBestIndices[WorkGroupSize];
      shared float xBestValues[WorkGroupSize];
    `;

    const sharedMemoryReduceSnippet = `
      xBestIndices[gl_LocalInvocationID.x] = bestIndex;
      xBestValues[gl_LocalInvocationID.x] = bestValue;

      int currentSize = WorkGroupSize;
      while (currentSize > 1) {
        barrier();

        for (int w = 0; w < ${this.reductionFactor}; ++w) {
          int i = int(gl_LocalInvocationID.x) * ${this.reductionFactor} + w;
          if (i < currentSize) {
            int candidateIndex = xBestIndices[i];
            float candidate = xBestValues[i];
            if (candidate ${this.op} bestValue && !isnan(candidate)) {
              bestValue = candidate;
              bestIndex = candidateIndex;
            }
          }
        }

        xBestIndices[gl_LocalInvocationID.x] = bestIndex;
        xBestValues[gl_LocalInvocationID.x] = bestValue;

        currentSize = DIV_CEIL(currentSize, ${this.reductionFactor});
      }

      if (gl_LocalInvocationID.x == 0) {
        setOutput(flatOutputIndex, int(bestIndex));
      }
    `;

    const outputCoordsType = getCoordsDataType(this.outputShape.length);

    const indexOutputCoords = (outputCoords: string, index: string) => {
      if (this.outputShape.length === 1) {
        return outputCoords;
      } else {
        return `${outputCoords}[${index}]`;
      }
    };

    const indexInputShape = (index: string) => {
      if (this.inputShape.length === 1) {
        return `${getShapeCoords(this.inputShape)}`;
      } else {
        return `${getShapeCoords(this.inputShape)}[${index}]`;
      }
    };

    const userCode = `
      #define DIV_CEIL(x, y) (((x) - 1) / (y) + 1)

      const int WorkGroupSize = int(gl_WorkGroupSize.x);

      ${reduceInSharedMemory ? sharedMemorySnippet : ''}

      // In order to get a flattened index into the input tensor, we need to
      // add back the index along the reduced dimension to |outputCoords|.
      // This function outputs the offset to the first value along
      // |axis| and the stride to get the next value of the input along |axis|.
      ivec2 getInputCoordInfo() {
        const ${outputCoordsType} outputCoords = getOutputCoords();
        int i = ${this.outputShape.length - 1};

        int stride = 1;
        int inputStride = 1;
        int offset = 0;

        for (int r = 1; r <= ${this.inputShape.length}; ++r) {
          int length = ${indexInputShape(`${this.inputShape.length} - r`)};
          if (${this.inputShape.length} - r == axis) {
            inputStride = stride;
          } else {
            offset += ${indexOutputCoords('outputCoords', 'i--')} * stride;
          }
          stride *= length;
        }

        return ivec2(offset, inputStride);
      }

      int getInputIndex(ivec2 coordInfo, int index) {
        return coordInfo[0] + coordInfo[1] * index;
      }

      void main() {
        const ivec2 coordInfo = getInputCoordInfo();

        int bestIndex = 0;
        float bestValue = x[getInputIndex(coordInfo, bestIndex)];

        const int Length = ${indexInputShape('axis')};
        const int WorkPerThread = DIV_CEIL(Length, WorkGroupSize);

        for (int w = 0; w < WorkPerThread; ++w) {
          int i = int(gl_GlobalInvocationID.x) * WorkPerThread + w;
          if (i < Length) {
            float candidate = x[getInputIndex(coordInfo, i)];
            if (candidate ${this.op} bestValue && !isnan(candidate)) {
              bestValue = candidate;
              bestIndex = i;
            }
          }
        }

        const int flatOutputIndex = int(gl_GlobalInvocationID.y);
        ${
        reduceInSharedMemory ? sharedMemoryReduceSnippet :
                               'setOutput(flatOutputIndex, int(bestIndex));'}
      }
    `;
    return userCode;
  }
}
