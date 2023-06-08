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
import {getCoordsXYZ, getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ArgMinMaxProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [64, 1, 1];
  variableNames = ['x'];
  uniforms = 'infinityValue : f32,';
  inputShape: number[];
  reductionFactor: number;
  op: string;
  size = true;
  private type: string;

  constructor(inputShape: number[], axis: number, reduceType: 'min'|'max') {
    const axes = [axis];

    this.op = reduceType === 'min' ? '<' : '>';

    // |outShape| is the shape with the removed axis
    const [outputShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(inputShape, axes);

    this.outputShape = outputShape.length === 0 ? [1] : outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    // The shared algorithm is mainly used for large reduce size. It fully
    // utilizes the threads in one workgroup to do the reduction. However,
    // when the reduce size is very small, it's better to use the plain
    // algorithm to reduce the number of workgroups to speedup. The threthold
    // can be further tuned.
    if (util.sizeFromShape(reduceShape) < 32) {
      this.type = 'plain';
      this.dispatch = computeDispatch(
          this.dispatchLayout, this.outputShape, this.workgroupSize);
    } else {
      this.type = 'shared';
      // A work group only outputs a data, so we transfer [1, 1, 1] to compute
      // dispatch size.
      this.dispatch =
          computeDispatch(this.dispatchLayout, this.outputShape, [1, 1, 1]);
    }

    this.inputShape = inputShape;
    this.shaderKey = `argMinMax_${this.op}_${this.type}`;
  }

  getUserCode(): string {
    const workgroupSizeX = this.workgroupSize[0];
    const getInputShapeLastDim = () => {
      if (this.inputShape.length === 1) {
        return 'uniforms.xShape';
      } else {
        return `uniforms.xShape.${getCoordsXYZ(this.inputShape.length - 1)}`;
      }
    };

    const splitOutputCoords = () => {
      let snippet = '';
      if (this.outputShape.length === 1) {
        if (this.inputShape.length !== 1) {
          snippet += 'outputCoords,';
        }
      } else {
        for (let i = 0; i < this.outputShape.length; i++) {
          snippet += `outputCoords.${getCoordsXYZ(i)},`;
        }
      }
      return snippet;
    };

    if (this.type === 'shared') {
      const sharedMemorySnippet = `
      var<workgroup> xBestIndices : array<i32, ${workgroupSizeX}>;
      var<workgroup> xBestValues : array<f32, ${workgroupSizeX}>;
    `;
      const userCode = `
      fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
      }

      ${sharedMemorySnippet}

      ${main('index')} {
        let outputIndex = index / ${workgroupSizeX};
        let reduceLength = ${getInputShapeLastDim()};

        var bestIndex = i32(localId.x);
        var bestValue = uniforms.infinityValue;
        let outputCoords = getCoordsFromIndex(outputIndex);
        for (var k = i32(localId.x); k < reduceLength && outputIndex < uniforms.size;
            k = k + ${workgroupSizeX}) {
          let candidate = getX(${splitOutputCoords()} k);
          if (!isnan(candidate) && candidate ${this.op} bestValue) {
            bestValue = candidate;
            bestIndex = k;
          }
        }
        xBestValues[localId.x] = bestValue;
        xBestIndices[localId.x] = bestIndex;
        workgroupBarrier();

        var reduceSize = min(u32(reduceLength), ${workgroupSizeX}u);
        for (var currentSize = reduceSize / 2u; reduceSize > 1u;
            currentSize = reduceSize / 2u) {
          let interval = DIV_CEIL(reduceSize, 2u);
          if (localId.x < currentSize) {
            let candidate = xBestValues[localId.x + interval];
            if (candidate ${this.op} bestValue) {
              bestValue = candidate;
              xBestValues[localId.x] = bestValue;
              xBestIndices[localId.x] = xBestIndices[localId.x + interval];
            }
          }
          reduceSize = interval;
          workgroupBarrier();
        }

        if (localId.x == 0u && outputIndex < uniforms.size) {
          setOutputAtIndexI32(outputIndex, xBestIndices[localId.x]);
        }
      }
    `;
      return userCode;
    } else {
      const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let outputCoords = getCoordsFromIndex(index);
          var bestIndex = 0;
          var bestValue = getX(${splitOutputCoords()} 0);
          let reduceLength = ${getInputShapeLastDim()};
          for (var i = 1; i < reduceLength; i++) {
            let candidate = getX(${splitOutputCoords()} i);
            if (candidate ${this.op} bestValue) {
              bestValue = candidate;
              bestIndex = i;
            }
          }
          setOutputAtIndexI32(index, bestIndex);
        }
      }
      `;
      return userCode;
    }
  }
}
