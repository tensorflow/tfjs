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

import {backend_util, DataType} from '@tensorflow/tfjs-core';
import {getMainHeaderString} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class ReduceProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number];
  variableNames = ['x'];
  uniforms = 'reduceSize : i32;';
  reduceType: 'max'|'mean'|'min'|'prod'|'sum';
  inputShape: number[];
  reductionFactor: number;

  constructor(
      reduceInfo: backend_util.ReduceInfo,
      reduceType: 'max'|'mean'|'min'|'prod'|'sum', outputDtype: DataType) {
    this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
    const [outputShape, ] =
        backend_util.computeOutAndReduceShapes(this.inputShape, [1]);
    this.outputShape = outputShape.length === 0 ? [1] : outputShape;

    this.reductionFactor = 2;
    // Note that the maximum of workgroup X dimension is 256.
    const xMaxThreads = 256;
    const xThreads = Math.min(
        Math.ceil(reduceInfo.inSize / this.reductionFactor), xMaxThreads);

    this.workGroupSize = [xThreads, 1, 1];
    this.dispatchLayout = {x: [], y: this.outputShape.map((d, i) => i)};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.reduceType = reduceType;
    this.shaderKey = `reduce_${reduceType}_${outputDtype}`;
  }

  getUserCode(): string {
    const reduceInSharedMemory = this.workGroupSize[0] > 1;

    let reduceOp = ``;
    let initValue = '0.0';
    if (this.reduceType === 'min' || this.reduceType === 'max') {
      reduceOp = `
         if (isNanCustom(candidate)) {
          bestValue = uniforms.NAN;
         } elseif (candidate ${this.reduceType === 'min' ? '<' : '>'}
           bestValue)
           {  bestValue = candidate; }`;
      initValue = 'f32(x.numbers[offset])';
    } else if (this.reduceType === 'sum' || this.reduceType === 'mean') {
      reduceOp = ' bestValue = bestValue + candidate; ';
    } else if (this.reduceType === 'prod') {
      reduceOp = ' bestValue = bestValue * candidate; ';
      initValue = '1.0';
    }

    const outputSnippet = this.reduceType === 'mean' ?
        // tslint:disable-next-line:max-line-length
        `setOutputFlat(flatOutputIndex, bestValue / f32(uniforms.reduceSize));` :
        `setOutputFlat(flatOutputIndex, bestValue);`;

    const sharedMemorySnippet = `
         var<workgroup> xBestValues : array<f32, ${this.workGroupSize[0]}>;
       `;
    const sharedMemoryReduceSnippet = `
       xBestValues[localId.x] = bestValue;
       ${
        this.reduceType === 'sum' || this.reduceType === 'mean' ||
                this.reduceType === 'prod' ?
            `bestValue = ${initValue};` :
            ' '}
       var currentSize = WorkGroupSize;
       for(; currentSize > 1;) {
         workgroupBarrier();
         for (var w = 0; w < ${this.reductionFactor}; w = w + 1) {
           let i = i32(localId.x) * ${this.reductionFactor} + w;
           if (i < currentSize) {
             let candidate = xBestValues[i];
             ${reduceOp}
           }
         }
         workgroupBarrier();
         xBestValues[localId.x] = bestValue;
         currentSize = DIV_CEIL(currentSize, ${this.reductionFactor});
         ${
        this.reduceType === 'sum' || this.reduceType === 'mean' ||
                this.reduceType === 'prod' ?
            `if(currentSize > 1) { bestValue = ${initValue}; }` :
            ''}
       }
       if (localId.x == 0u) {
         ${outputSnippet}
       }
     `;

    const userCode = `
       fn DIV_CEIL(a : i32, b : i32) -> i32 {
        return ((a - 1) / b + 1);
       }
       let WorkGroupSize = ${this.workGroupSize[0]};
       ${reduceInSharedMemory ? sharedMemorySnippet : ''}
       fn getOffset(globalId : vec3<u32>) -> i32 {
         let outputCoords = getOutputCoords(globalId, i32(globalId.x));
         let offset = ${
        this.outputShape.length === 1 ?
            'outputCoords' :
            'outputCoords[0]'} * uniforms.reduceSize;
         return offset;
       }
       ${getMainHeaderString()} {
         let offset = getOffset(globalId);
         var bestValue = ${initValue};
         let Length = uniforms.reduceSize;
         let WorkPerThread = DIV_CEIL(Length, WorkGroupSize);
         for (var w = 0; w < WorkPerThread; w = w + 1) {
           let i = i32(globalId.x) * WorkPerThread + w;
           if (i < Length) {
             let candidate = f32(x.numbers[offset + i]);
             ${reduceOp}
           }
         }
         let flatOutputIndex = i32(globalId.y);
         ${reduceInSharedMemory ? sharedMemoryReduceSnippet : outputSnippet}
       }
     `;
    return userCode;
  }
}
