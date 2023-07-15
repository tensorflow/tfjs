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
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class RangeSizeProgram implements WebGPUProgram {
  variableNames = ['starts', 'limits', 'deltas'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [64, 1, 1];
  size = true;

  constructor(outLength: number) {
    this.outputShape = [outLength];
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);

    this.shaderKey = `rangeSize`;
  }

  getUserCode(): string {
    const userCode = `
        ${main('index')} {
          if (index < uniforms.size) {
            let start = f32(getStartsByOutputIndex(index));
            let limit = f32(getLimitsByOutputIndex(index));
            let delta = f32(getDeltasByOutputIndex(index));
            var size = 0.0;
            if (((delta > 0) && (limit >= start)) || ((delta < 0) && (limit <= start))) {
              size = ceil(abs((limit - start) / delta));
            }

            setOutputAtIndex(index, size);
          }
        }
      `;
    return userCode;
  }
}

export class RangeDenseValuesProgram implements WebGPUProgram {
  variableNames = ['starts', 'deltas', 'rtNestedSplits'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [64, 1, 1];
  uniforms = 'rtNestedSplitsNum : i32,';
  size = true;

  constructor(outLength: number) {
    this.outputShape = [outLength];
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);

    this.shaderKey = `rangeDenseValues`;
  }

  getUserCode(): string {
    const userCode = `
        ${main('index')} {
          if (index < uniforms.size) {
            var left = 1;
            var right = uniforms.rtNestedSplitsNum;
            var mid = 1;
            while (left < right) {
              mid = (left + right) / 2;
              let midVal = i32(getRtNestedSplitsByOutputIndex(mid));
              let midPreVal = i32(getRtNestedSplitsByOutputIndex(mid - 1));
              if (index >= midPreVal && index < midVal) {
                break;
              }
              if (index < midVal) {
                right = mid;
              } else {
                left = mid;
              }
            }

            let curRangeIndex = f32(index) - getRtNestedSplitsByOutputIndex(mid - 1);
            let curStart = f32(getStartsByOutputIndex(mid - 1));
            let curDelta = f32(getDeltasByOutputIndex(mid - 1));

            setOutputAtIndex(index, curStart + curDelta * curRangeIndex);
          }
        }
      `;
    return userCode;
  }
}
