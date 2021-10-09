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

import {getCoordsDataType, getMainHeaderString} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class ArgMinMaxProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number];
  variableNames = ['x'];
  uniforms = 'axis : i32;';
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
    // Note that the maximum of workgroup X dimension is 256.
    const xMaxThreads = 256;  // gl_MaxComputeWorkGroupSize.
    const xThreads =
        Math.min(Math.ceil(reduceSize / this.reductionFactor), xMaxThreads);

    this.workGroupSize = [xThreads, 1, 1];

    this.dispatchLayout = {x: [], y: this.outputShape.map((d, i) => i)};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.inputShape = inputShape;
    this.shaderKey = `argMinMax${this.op}`;
  }

  getUserCode(): string {
    // When this.workGroupSize[0] > 1, each thread reduces Length /
    // this.workGroupSize[0] values. Thes results are stored in shared memory
    // and iteratively reduced.
    const reduceInSharedMemory = this.workGroupSize[0] > 1;
    const sharedMemorySnippet = `
      var<workgroup> xBestIndices : array<i32, ${this.workGroupSize[0]}>;
      var<workgroup> xBestValues : array<f32, ${this.workGroupSize[0]}>;
    `;

    const sharedMemoryReduceSnippet = `
      xBestIndices[localId.x] = bestIndex;
      xBestValues[localId.x] = bestValue;

      for(var currentSize = WorkGroupSize; currentSize > 1; currentSize = DIV_CEIL(currentSize, ${
        this.reductionFactor})) {
        workgroupBarrier();

        for (var w = 0; w < ${this.reductionFactor}; w = w + 1) {
          let i = i32(localId.x) * ${this.reductionFactor} + w;
          if (i < currentSize) {
            let candidateIndex = xBestIndices[i];
            let candidate = xBestValues[i];
            if(candidate ${this.op} bestValue && !isNanCustom(candidate)) {
              bestValue = candidate;
              bestIndex = candidateIndex;
            }
          }
        }

        xBestIndices[localId.x] = bestIndex;
        xBestValues[localId.x] = bestValue;
      }

      if (localId.x == 0u) {
        setOutputFlatI32(flatOutputIndex, i32(bestIndex));
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
        return 'uniforms.xShape';
      } else {
        return `uniforms.xShape[${index}]`;
      }
    };

    const userCode = `
      fn DIV_CEIL(a : i32, b : i32) -> i32 {
        return ((a - 1) / b + 1);
      }

      let WorkGroupSize = ${this.workGroupSize[0]};

      ${reduceInSharedMemory ? sharedMemorySnippet : ''}

      // In order to get a flattened index into the input tensor, we need to
      // add back the index along the reduced dimension to |outputCoords|.
      // This function outputs the offset to the first value along
      // |axis| and the stride to get the next value of the input along |axis|.
      fn getInputCoordInfo(globalId : vec3<u32>) -> vec2<i32>{
        let outputCoords : ${
        outputCoordsType} = getOutputCoords(globalId, i32(globalId.x));
        var i = ${this.outputShape.length - 1};

        var stride = 1;
        var inputStride = 1;
        var offset = 0;

        for (var r = 1; r <= ${this.inputShape.length}; r = r + 1) {
          let length = ${indexInputShape(`${this.inputShape.length} - r`)};
          if (${this.inputShape.length} - r == uniforms.axis) {
            inputStride = stride;
          } else {
            offset = offset + ${
        indexOutputCoords('outputCoords', 'i')} * stride;
            i = i - 1;
          }
          stride = stride * length;
        }

        return vec2<i32>(offset, inputStride);
      }

      fn getInputIndex(coordInfo : vec2<i32>, index : i32) -> i32{
        return coordInfo[0] + coordInfo[1] * index;
      }

      ${getMainHeaderString()} {
        let coordInfo = getInputCoordInfo(globalId);

        var bestIndex = 0;
        var bestValue = x.numbers[getInputIndex(coordInfo, bestIndex)];

        let Length = ${indexInputShape('uniforms.axis')};
        let WorkPerThread = DIV_CEIL(Length, WorkGroupSize);

        for (var w = 0; w < WorkPerThread; w = w + 1) {
          let i = i32(globalId.x) * WorkPerThread + w;
          if (i < Length) {
            let candidate = x.numbers[getInputIndex(coordInfo, i)];
            if (candidate ${
        this.op} bestValue && !isNanCustom(f32(candidate))) {
              bestValue = candidate;
              bestIndex = i;
            }
          }
        }

        let flatOutputIndex = i32(globalId.y);
        ${
        reduceInSharedMemory ? sharedMemoryReduceSnippet :
                               'setOutputFlatI32(flatOutputIndex, bestIndex);'}
      }
    `;
    return userCode;
  }
}
