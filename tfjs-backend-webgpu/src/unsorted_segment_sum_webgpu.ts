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

import {DataType} from '@tensorflow/tfjs-core';

import {atomicAddSnippet} from './shader_util';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class UnsortedSegmentSumProgram implements WebGPUProgram {
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'segmentIds'];
  uniforms = 'numSegments : i32, xSize: i32,';
  workgroupSize: [number, number, number] = [64, 1, 1];
  atomic = true;
  type: DataType;

  constructor(inShape: number[], outShape: number[], outputDtype: DataType) {
    this.outputShape = outShape;
    this.dispatchLayout = flatDispatchLayout(inShape);
    this.dispatch =
        computeDispatch(this.dispatchLayout, inShape, this.workgroupSize);
    if (outputDtype !== 'float32' && outputDtype !== 'int32') {
      throw new Error(`UnsortedSegmentSum only supports float32 and int32
              types, does not support ${outputDtype} type.`);
    }
    this.type = outputDtype;
    this.shaderKey = 'unsortedSegmentSum';
  }

  getUserCode(): string {
    const userCode = `
    ${main('index')} {
      if (index < uniforms.xSize) {
        let coords = getXCoordsFromIndex(index);
        let b = coords[0];
        let inCol = coords[1];

        let segmentId = i32(getSegmentIds(inCol));
        if (segmentId >= 0) {
          let flatIndex = b * uniforms.numSegments + segmentId % uniforms.numSegments;
          let value = getX(b, inCol);

          ${
        atomicAddSnippet(
            '&result[flatIndex]', 'value', this.type as 'float32' | 'int32')}
        }
      }
    }
  `;
    return userCode;
  }
}
