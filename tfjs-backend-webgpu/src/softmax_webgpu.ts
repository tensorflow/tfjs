/**
 * @license
 * Copyright 2023 Google LLC.
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

import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {flatDispatchLayout} from './webgpu_util';

export class SoftmaxProgram implements WebGPUProgram {
  variableNames = ['logits'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number];

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;  // [rows, cols]
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = [this.outputShape[0], 1, 1];
    if (this.outputShape[1] >= 4096) {
      this.workgroupSize = [256, 1, 1];
    } else {
      this.workgroupSize = [64, 1, 1]
    }
    this.shaderKey = 'softmax';
  }

  getUserCode(): string {
    const userCode = `
    var<workgroup> buf : array<f32, ${this.workgroupSize[0]}>;
    var<workgroup> rowMaxShared : f32;
    var<workgroup> rowSumShared : f32;
    const block_size = ${this.workgroupSize[0]};
    ${main('index')} {
      let row = index / block_size;
      let tid = i32(localId.x);
      let cols = uniforms.outShape[1];

      var thread_max = -1.0 / 1e-20;
      for (var col = tid; col < cols; col += block_size) {
        let pack = getLogits(row, col);
        thread_max = max(thread_max, pack);
      }
      if (tid < cols)
      {
        buf[tid] = thread_max;
      }
      workgroupBarrier();

      for (var currSize = block_size >> 1;  currSize > 0; currSize = currSize >> 1)
      {
        if (tid < currSize) {
          buf[tid] = max(buf[tid], buf[tid + currSize]);
        }
        workgroupBarrier();
      }

      if (tid == 0) {
        rowMaxShared = buf[0];
      }
      workgroupBarrier();

      var thread_sum = 0.0;
      for (var col = tid; col < cols; col += block_size) {
        let subExp = exp(getLogits(row, col) - rowMaxShared);
        thread_sum += subExp;
      }
      if (tid < cols)
      {
        buf[tid] = thread_sum;
      }
      workgroupBarrier();

      for (var currSize = block_size >> 1;  currSize > 0; currSize = currSize >> 1)
      {
        if (tid < currSize) {
          buf[tid] = buf[tid] + buf[tid + currSize];
        }
        workgroupBarrier();
      }

      if (tid == 0) {
        rowSumShared = buf[0];
      }
      workgroupBarrier();

      for (var col = tid; col < cols; col += block_size) {
        let pack = exp(getLogits(row, col) - rowMaxShared) / rowSumShared;
        setOutputAtCoords(row, col, pack);
      }
  }
    `;
    return userCode;
  }
}
