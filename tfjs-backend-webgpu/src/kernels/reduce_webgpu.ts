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
import {getCoordsDataType} from '../shader_preprocessor';
import {getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class ReduceProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number];
  variableNames = ['x'];
  uniforms = 'int reduceSize;';
  uniformsWgsl = 'reduceSize : i32;';
  reduceType: 'max'|'mean'|'min'|'prod'|'sum';
  inputShape: number[];
  reductionFactor: number;
  useWgsl: boolean;

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
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const reduceInSharedMemory = this.workGroupSize[0] > 1;

    let reduceOp = ``;
    let initValue = '0.0';
    if (this.reduceType === 'min' || this.reduceType === 'max') {
      reduceOp = `
         if (isnan(candidate)) {
          bestValue = float(NAN);
         } else if (candidate ${this.reduceType === 'min' ? '<' : '>'}
           bestValue)
           {  bestValue = candidate; }`;
      initValue = 'float(x[offset])';
    } else if (this.reduceType === 'sum' || this.reduceType === 'mean') {
      reduceOp = ' bestValue += candidate; ';
    } else if (this.reduceType === 'prod') {
      reduceOp = ' bestValue *= candidate; ';
      initValue = '1.0';
    }

    const outputSnippet = this.reduceType === 'mean' ?
        `setOutput(flatOutputIndex, bestValue / float(reduceSize));` :
        `setOutput(flatOutputIndex, bestValue);`;

    const sharedMemorySnippet = `
         shared float xBestValues[WorkGroupSize];
       `;
    const sharedMemoryReduceSnippet = `
       xBestValues[gl_LocalInvocationID.x] = bestValue;
       ${
        this.reduceType === 'sum' || this.reduceType === 'mean' ||
                this.reduceType === 'prod' ?
            `bestValue=${initValue};` :
            ' '}
       int currentSize = WorkGroupSize;
       while (currentSize > 1) {
         barrier();
         for (int w = 0; w < ${this.reductionFactor}; ++w) {
           int i = int(gl_LocalInvocationID.x) * ${this.reductionFactor} + w;
           if (i < currentSize) {
             float candidate = xBestValues[i];
             ${reduceOp}
           }
         }
         barrier();
         xBestValues[gl_LocalInvocationID.x] = bestValue;
         currentSize = DIV_CEIL(currentSize, ${this.reductionFactor});
         ${
        this.reduceType === 'sum' || this.reduceType === 'mean' ||
                this.reduceType === 'prod' ?
            `if(currentSize > 1) bestValue=${initValue};` :
            ''}
       }
       if (gl_LocalInvocationID.x == 0) {
         ${outputSnippet}
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
        this.outputShape.length === 1 ? 'outputCoords' :
                                        'outputCoords[0]'} * reduceSize;
         return offset;
       }
       void main() {
         const int offset= getOffset();
         float bestValue = ${initValue};
         const int Length = reduceSize;
         const int WorkPerThread = DIV_CEIL(Length, WorkGroupSize);
         for (int w = 0; w < WorkPerThread; ++w) {
           int i = int(gl_GlobalInvocationID.x) * WorkPerThread + w;
           if (i < Length) {
             float candidate = float(x[offset + i]);
             ${reduceOp}
           }
         }
         const int flatOutputIndex = int(gl_GlobalInvocationID.y);
         ${reduceInSharedMemory ? sharedMemoryReduceSnippet : outputSnippet}
       }
     `;
    return userCode;
  }

  getUserCodeWgsl(): string {
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
       fn getOffset(globalId : vec3<i32>, index : i32) -> i32 {
         let outputCoords = getOutputCoords(vec3<i32>(globalId), index);
         let offset = ${
        this.outputShape.length === 1 ?
            'outputCoords' :
            'outputCoords[0]'} * uniforms.reduceSize;
         return offset;
       }
       ${getMainHeaderStringWgsl()} {
         ${getGlobalIndexStringWgsl()}
         let offset= getOffset(vec3<i32>(globalId), index);
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
