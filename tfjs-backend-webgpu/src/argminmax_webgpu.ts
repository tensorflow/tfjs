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

import {getCoordsXYZ, getMainHeaderAndGlobalIndexString} from './shader_preprocessor';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ArgMinMaxProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  variableNames = ['x'];
  uniforms = 'infinityValue : f32,';
  inputShape: number[];
  reductionFactor: number;
  op: string;
  size = true;
  axis: number;

  constructor(inputShape: number[], axis: number, reduceType: 'min'|'max') {
    this.axis = axis;
    const axes = [axis];
    backend_util.assertAxesAreInnerMostDims(
        'arg' + reduceType.charAt(0).toUpperCase() + reduceType.slice(1), axes,
        inputShape.length);

    this.op = reduceType === 'min' ? '<' : '>';

    // |outShape| is the shape with the removed axis
    const [outputShape] =
        backend_util.computeOutAndReduceShapes(inputShape, axes);

    this.outputShape = outputShape.length === 0 ? [1] : outputShape;

    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    // A work group only outputs a data, so we transfer [1, 1, 1] to compute
    // dispatch size.
    this.dispatch =
        computeDispatch(this.dispatchLayout, this.outputShape, [1, 1, 1]);

    this.inputShape = inputShape;
    this.shaderKey = `argMinMax${this.op}_${this.axis}`;
  }

  getUserCode(): string {
    const sharedMemorySnippet = `
      var<workgroup> xBestIndices : array<i32, ${this.workGroupSize[0]}>;
      var<workgroup> xBestValues : array<f32, ${this.workGroupSize[0]}>;
    `;

    const getCoordFromIndex =
        (coords: string, index: number, length: number) => {
          if (length === 1) {
            return coords;
          } else {
            return `${coords}.${getCoordsXYZ(index)}`;
          }
        };

    const getInputCoordInfoSnippet = () => {
      const inputSize = this.inputShape.length;
      let i = this.outputShape.length - 1;
      let stride = 'stride';
      let offset = 'offset';
      let snippet = '';
      for (let r = 1; r <= inputSize; r = r + 1) {
        const length = `uniforms.xShape.${getCoordsXYZ(inputSize - r)}`;
        if (inputSize - r === this.axis) {
          snippet += `
          inputStride = ${stride};`;
        } else {
          offset += ` + ${
              getCoordFromIndex(
                  'outputCoords', i, this.outputShape.length)} * ${stride}`;
          i = i - 1;
        }
        stride = `${stride} * ${length}`;
      }
      snippet += `offset = ${offset};`;
      return snippet;
    };

    const userCode = `
      fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
      }

      ${sharedMemorySnippet}

      // In order to get a flattened index into the input tensor, we need to
      // add back the index along the reduced dimension to |outputCoords|.
      // This function outputs the offset to the first value along
      // |axis| and the stride to get the next value of the input along |axis|.
      fn getInputCoordInfo(outputIndex : i32) -> vec2<i32>{
        let outputCoords = getCoordsFromIndex(outputIndex);

        var stride = 1;
        var inputStride = 1;
        var offset = 0;
      ${getInputCoordInfoSnippet()}

        return vec2<i32>(offset, inputStride);
      }

      fn getInputIndex(coordInfo : vec2<i32>, index : i32) -> i32{
        return coordInfo[0] + coordInfo[1] * index;
      }

      ${getMainHeaderAndGlobalIndexString()}
        let outputIndex = index / i32(workGroupSizeX);
        let coordInfo = getInputCoordInfo(outputIndex);
        let Length = ${
        getCoordFromIndex(
            'uniforms.xShape', this.axis, this.inputShape.length)};

        var bestIndex = i32(localId.x);
        var bestValue = uniforms.infinityValue;

        for (var k = i32(localId.x); k < Length && outputIndex < uniforms.size;
            k = k + i32(workGroupSizeX)) {
          let candidate = f32(x[getInputIndex(coordInfo, k)]);
          if (!isnan(candidate) && candidate ${this.op} bestValue) {
            bestValue = candidate;
            bestIndex = k;
          }
        }
        xBestValues[localId.x] = bestValue;
        xBestIndices[localId.x] = bestIndex;
        workgroupBarrier();

        var reduceSize = min(u32(Length), workGroupSizeX);
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
  }
}
