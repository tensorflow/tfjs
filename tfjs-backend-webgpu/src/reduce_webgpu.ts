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

import {backend_util} from '@tensorflow/tfjs-core';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ReduceProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number];
  variableNames = ['x'];
  uniforms = 'reduceSize : i32,';
  reduceType: 'all'|'any'|'max'|'mean'|'min'|'prod'|'sum';
  inputShape: number[];
  size = true;

  constructor(
      reduceInfo: backend_util.ReduceInfo,
      reduceType: 'all'|'any'|'max'|'mean'|'min'|'prod'|'sum',
      maxComputeWorkgroupSizeX: number) {
    this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
    const [outputShape, ] =
        backend_util.computeOutAndReduceShapes(this.inputShape, [1]);
    this.outputShape = outputShape.length === 0 ? [1] : outputShape;
    // If reduceSize |reduceInfo.inSize| is very large, the I/O accessing will
    // become the bottleneck. Increasing workgroupSize can reduce the times of
    // accessing global memory. The threshold value is just to make sure the
    // reduceSize is large enough for a bigger workgroupSize.
    if (reduceInfo.inSize >= 32768 && maxComputeWorkgroupSizeX >= 512) {
      this.workgroupSize = [512, 1, 1];
    } else if (reduceInfo.inSize >= 4096) {
      this.workgroupSize = [256, 1, 1];
    } else {
      this.workgroupSize = [64, 1, 1];
    }
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    // A work group only outputs a data, so we transfer [1, 1, 1] to compute
    // dispatch size.
    this.dispatch =
        computeDispatch(this.dispatchLayout, this.outputShape, [1, 1, 1]);

    this.reduceType = reduceType;
    this.shaderKey = `reduce_${reduceType}`;
  }

  getUserCode(): string {
    let reduceOp = ``;
    let initValue = '0.0';
    const workgroupSizeX = this.workgroupSize[0];
    if (this.reduceType === 'min' || this.reduceType === 'max') {
      reduceOp = `
         if (isnan(candidate)) {
          bestValue = uniforms.NAN;
         } else if (!isnan(bestValue) && candidate ${
          this.reduceType === 'min' ? '<' : '>'} bestValue)
           {  bestValue = candidate; }`;
      initValue = 'f32(x[offset])';
    } else if (this.reduceType === 'sum' || this.reduceType === 'mean') {
      reduceOp = ' bestValue = bestValue + candidate; ';
    } else if (this.reduceType === 'prod') {
      reduceOp = ' bestValue = bestValue * candidate; ';
      initValue = '1.0';
    } else if (this.reduceType === 'all') {
      reduceOp = ' bestValue = f32(bestValue >= 1.0 && candidate >= 1.0); ';
      initValue = '1.0';
    } else if (this.reduceType === 'any') {
      reduceOp = ' bestValue = f32(bestValue >= 1.0 || candidate >= 1.0); ';
      initValue = '0.0';
    }

    const outputSnippet = this.reduceType === 'mean' ?
        // tslint:disable-next-line:max-line-length
        `setOutputAtIndex(outputIndex, bestValue / f32(uniforms.reduceSize));` :
        `setOutputAtIndex(outputIndex, bestValue);`;

    const sharedMemorySnippet = `
         var<workgroup> xBestValues : array<f32, ${workgroupSizeX}>;
       `;

    const userCode = `
       fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
       }

       ${sharedMemorySnippet}
       fn getOffset(outputIndex : i32) -> i32 {
         let outputCoords = getCoordsFromIndex(outputIndex);
         let offset = ${
        this.outputShape.length === 1 ?
            'outputCoords' :
            'outputCoords[0]'} * uniforms.reduceSize;
          return offset;
       }
       ${main('index')} {
         let outputIndex = index / ${workgroupSizeX};
         let offset = getOffset(outputIndex);
         var bestValue = ${initValue};
         let Length = uniforms.reduceSize;
         let WorkPerThread = DIV_CEIL(u32(Length), ${workgroupSizeX}u);
         for (var k = i32(localId.x); k < Length && outputIndex < uniforms.size;
             k = k + ${workgroupSizeX}) {
           let candidate = f32(x[offset + k]);
           ${reduceOp}
         }
         xBestValues[localId.x] = bestValue;
         workgroupBarrier();

         var reduceSize = min(u32(Length), ${workgroupSizeX}u);
         for (var currentSize = reduceSize / 2u; reduceSize > 1u;
             currentSize = reduceSize / 2u) {
           let interval = DIV_CEIL(reduceSize, 2u);
           if (localId.x < currentSize) {
            let candidate = xBestValues[localId.x + interval];
            ${reduceOp}
            xBestValues[localId.x] = bestValue;
           }
           reduceSize = interval;
           workgroupBarrier();
         }

         if (localId.x == 0u && outputIndex < uniforms.size) {
          ${outputSnippet}
        }
       }
     `;
    return userCode;
  }
}
