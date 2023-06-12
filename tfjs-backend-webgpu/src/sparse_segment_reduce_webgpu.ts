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

export class SparseSegmentSumProgram implements WebGPUProgram {
  variableNames = ['input', 'indices', 'segmentIds'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  uniforms = 'segmentSize : i32, sparseSize : i32,';
  workgroupSize: [number, number, number] = [64, 1, 1];
  atomic = true;
  type: DataType;

  constructor(outShape: number[], sparseSize: number, outputDtype: DataType) {
    this.outputShape = outShape;
    this.type = outputDtype;
    this.dispatchLayout = flatDispatchLayout([sparseSize]);
    this.dispatch =
        computeDispatch(this.dispatchLayout, [sparseSize], this.workgroupSize);

    this.shaderKey = 'sparseSegmentSum';
  }

  getUserCode(): string {
    const userCode = `
    ${main('index')} {
      if (index < uniforms.sparseSize) {
        let indexInSegmentIds = index / uniforms.segmentSize;
        let indexInSegment = index % uniforms.segmentSize;
        let indexInInput = indices[indexInSegmentIds];
        let segmentId = segmentIds[indexInSegmentIds];

        let value = input[indexInInput * uniforms.segmentSize + indexInSegment];
        let outIndex = segmentId * uniforms.segmentSize + indexInSegment;
        ${
        atomicAddSnippet(
            '&result[outIndex]', 'value', this.type as 'float32' | 'int32')}
      }
    }
  `;
    return userCode;
  }
}

export class SparseSegmentIdCountProgram implements WebGPUProgram {
  variableNames = ['segmentIds'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [64, 1, 1];
  atomic = true;

  constructor(outShape: number, segmentIdsShape: number[]) {
    this.outputShape = [outShape];
    this.dispatchLayout = flatDispatchLayout(segmentIdsShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, segmentIdsShape, this.workgroupSize);

    this.shaderKey = 'sparseSegmentIdCountProgram';
  }

  getUserCode(): string {
    const userCode = `
    ${main('index')} {
      if (index < uniforms.segmentIdsShape) {
        let segmentId = segmentIds[index];
        ${atomicAddSnippet('&result[segmentId]', '1', 'int32')}
      }
    }
  `;
    return userCode;
  }
}

export class SparseSegmentMeanProgram implements WebGPUProgram {
  variableNames = ['segmentSum', 'sameSegmentIdCount'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  uniforms = 'segmentSize : i32';
  workgroupSize: [number, number, number] = [64, 1, 1];
  size = true;
  type: DataType;

  constructor(outShape: number[], outputDtype: DataType) {
    this.outputShape = outShape;
    this.type = outputDtype;
    this.dispatchLayout = flatDispatchLayout(outShape);
    this.dispatch =
        computeDispatch(this.dispatchLayout, outShape, this.workgroupSize);

    this.shaderKey = 'sparseSegmentMean';
  }

  getUserCode(): string {
    const userCode = `
    ${main('index')} {
      if (index < uniforms.size) {
        let segmentId = index / uniforms.segmentSize;
        let count = sameSegmentIdCount[segmentId];
        if (count != 0) {
          ${
        this.type === 'float32' ?
            'setOutputAtIndex(index, segmentSum[index] / f32(count));' :
            'setOutputAtIndexI32(index, segmentSum[index] / count);'}
        }
      }
    }
  `;
    return userCode;
  }
}
