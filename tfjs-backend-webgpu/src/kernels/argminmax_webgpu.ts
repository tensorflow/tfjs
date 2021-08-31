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
import {getCoordsDataTypeWgsl, getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class ArgMinMaxProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number];
  variableNames = ['x'];
  uniforms = 'int axis;';
  uniformsWgsl = 'axis : u32;';
  inputShape: number[];
  reductionFactor: number;
  op: string;
  useWgsl: boolean;

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
    this.useWgsl = getUseWgsl();
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

        currentSize = DIV_CEIL(currentSize, ${this.reductionFactor}u);
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
        return 'xShape';
      } else {
        return `xShape[${index}]`;
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
        float bestValue = float(x[getInputIndex(coordInfo, bestIndex)]);

        const int Length = ${indexInputShape('axis')};
        const int WorkPerThread = DIV_CEIL(Length, WorkGroupSize);

        for (int w = 0; w < WorkPerThread; ++w) {
          int i = int(gl_GlobalInvocationID.x) * WorkPerThread + w;
          if (i < Length) {
            float candidate = float(x[getInputIndex(coordInfo, i)]);
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

  getUserCodeWgsl(): string {
    // When this.workGroupSize[0] > 1, each thread reduces Length /
    // this.workGroupSize[0] values. Thes results are stored in shared memory
    // and iteratively reduced.
    const reduceInSharedMemory = this.workGroupSize[0] > 1;
    const sharedMemorySnippet = `
      var<workgroup> xBestIndices : array<u32, ${this.workGroupSize[0]}>;
      var<workgroup> xBestValues : array<f32, ${this.workGroupSize[0]}>;
    `;

    const sharedMemoryReduceSnippet = `
      xBestIndices[localId.x] = bestIndex;
      xBestValues[localId.x] = bestValue;

      for(var currentSize = WorkGroupSize; currentSize > 1u; currentSize = DIV_CEIL(currentSize, ${
        this.reductionFactor}u)) {
        workgroupBarrier();

        for (var w = 0u; w < ${this.reductionFactor}u; w = w + 1u) {
          let i = localId.x * ${this.reductionFactor}u + w;
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

    const outputCoordsType = getCoordsDataTypeWgsl(this.outputShape.length);

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
      fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
      }

      let WorkGroupSize = ${this.workGroupSize[0]}u;

      ${reduceInSharedMemory ? sharedMemorySnippet : ''}

      // In order to get a flattened index into the input tensor, we need to
      // add back the index along the reduced dimension to |outputCoords|.
      // This function outputs the offset to the first value along
      // |axis| and the stride to get the next value of the input along |axis|.
      fn getInputCoordInfo(globalId : vec3<u32>, globalIndex : u32) -> vec2<u32>{
        let outputCoords : ${
        outputCoordsType} = getOutputCoords(globalId, globalIndex);
        var i = ${this.outputShape.length - 1}u;

        var stride = 1u;
        var inputStride = 1u;
        var offset = 0u;

        for (var r = 1u; r <= ${this.inputShape.length}u; r = r + 1u) {
          let length = ${indexInputShape(`${this.inputShape.length}u - r`)};
          if (${this.inputShape.length}u - r == uniforms.axis) {
            inputStride = stride;
          } else {
            offset = offset + ${
        indexOutputCoords('outputCoords', 'i')} * stride;
            i = i - 1u;
          }
          stride = stride * length;
        }

        return vec2<u32>(offset, inputStride);
      }

      fn getInputIndex(coordInfo : vec2<u32>, index : u32) -> u32{
        return coordInfo[0] + coordInfo[1] * index;
      }

      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let coordInfo = getInputCoordInfo(globalId, index);

        var bestIndex = 0u;
        var bestValue = x.numbers[getInputIndex(coordInfo, bestIndex)];

        let Length = ${indexInputShape('uniforms.axis')};
        let WorkPerThread = DIV_CEIL(Length, WorkGroupSize);

        for (var w = 0u; w < WorkPerThread; w = w + 1u) {
          let i = globalId.x * WorkPerThread + w;
          if (i < Length) {
            let candidate = x.numbers[getInputIndex(coordInfo, i)];
            if (candidate ${
        this.op} bestValue && !isNanCustom(f32(candidate))) {
              bestValue = candidate;
              bestIndex = i;
            }
          }
        }

        let flatOutputIndex = globalId.y;
        ${
        reduceInSharedMemory ?
            sharedMemoryReduceSnippet :
            'setOutputFlatI32(flatOutputIndex, i32(bestIndex));'}
      }
    `;
    return userCode;
  }
}
