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

export class ReduceProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number];
  variableNames = ['x'];
  reduceType: 'max'|'min'|'sum';
  inputShape: number[];

  constructor(
      reduceInfo: backend_util.ReduceInfo, reduceType: 'max'|'min'|'sum') {
    this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
    const [outputShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(this.inputShape, [1]);
    this.outputShape = outputShape.length === 0 ? [1] : outputShape;
    const reduceSize = util.sizeFromShape(reduceShape);

    const reductionFactor = 2;
    const xMaxThreads = 1024;
    const xThreads =
        Math.min(Math.ceil(reduceSize / reductionFactor), xMaxThreads);

    this.workGroupSize = [xThreads, 1, 1];
    this.dispatchLayout = {x: [], y: this.outputShape.map((d, i) => i)};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.reduceType = reduceType;
    this.shaderKey = `reduce_${reduceType}`;
  }

  getUserCode(): string {
    const reduceInSharedMemory = this.workGroupSize[0] > 1;
    const reductionFactor = 2;
    const minmaxOp = `
          if (candidate ${this.reduceType === 'min' ? '<' : '>'} bestValue
          && !isnan(candidate))
          {  bestValue = candidate; }
      `;
    const sumOp = ' bestValue += candidate; ';
    const op = (this.reduceType === 'min' || this.reduceType === 'max') ?
        minmaxOp :
        sumOp;

    const sharedMemorySnippet = `
        shared float xBestValues[WorkGroupSize];
      `;
    const sharedMemoryReduceSnippet = `
      xBestValues[gl_LocalInvocationID.x] = bestValue;
      ${this.reduceType === 'sum' ? 'bestValue=0;' : ' '}
      int currentSize = WorkGroupSize;
      while (currentSize > 1) {
        barrier();
        for (int w = 0; w < ${reductionFactor}; ++w) {
          int i = int(gl_LocalInvocationID.x) * ${reductionFactor} + w;
          if (i < currentSize) {
            float candidate = xBestValues[i];
            ${op}
          }
        }
        barrier();
        xBestValues[gl_LocalInvocationID.x] = bestValue;
        currentSize = DIV_CEIL(currentSize, ${reductionFactor});
        ${this.reduceType === 'sum' ? 'if(currentSize > 1) bestValue=0;' : ''}
      }
      if (gl_LocalInvocationID.x == 0) {
        setOutput(flatOutputIndex, bestValue);
      }
    `;

    const outputCoordsType = getCoordsDataType(this.outputShape.length);

    const userCode = `
      #define DIV_CEIL(x, y) (((x) - 1) / (y) + 1)
      const int WorkGroupSize = int(gl_WorkGroupSize.x);
      ${reduceInSharedMemory ? sharedMemorySnippet : ''}
      int getOffset() {
        const ${outputCoordsType} outputCoords = getOutputCoords();
        int offset = ${
        this.outputShape.length === 1 ?
            'outputCoords' :
            'outputCoords[0]'} * ${getShapeCoords(this.inputShape)}[1];
        return offset;
      }
      void main() {
        const int offset= getOffset();
        ${
        this.reduceType === 'sum' ? 'float bestValue = 0;' :
                                    'float bestValue = x[offset];'}
        const int Length = ${
        this.inputShape.length === 1 ? `${getShapeCoords(this.inputShape)}` :
                                       `${getShapeCoords(this.inputShape)}[1]`};
        const int WorkPerThread = DIV_CEIL(Length, WorkGroupSize);
        for (int w = 0; w < WorkPerThread; ++w) {
          int i = int(gl_GlobalInvocationID.x) * WorkPerThread + w;
          if (i < Length) {
            float candidate = x[offset + i];
            ${
        (this.reduceType === 'max' || this.reduceType === 'min') ? minmaxOp :
                                                                   sumOp}
          }
        }
        const int flatOutputIndex = int(gl_GlobalInvocationID.y);
        ${
        reduceInSharedMemory ? sharedMemoryReduceSnippet :
                               'setOutput(flatOutputIndex, bestValue);'}
      }
    `;
    return userCode;
  }
}
