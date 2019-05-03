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

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class PadProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];

  constructor(
      xShape: number[], paddings: Array<[number, number]>,
      constantValue: number) {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    const rank = xShape.length;
    const type = getCoordsDataType(rank);
    this.dispatchLayout = {x: this.outputShape.map((d, i) => i)};
    this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape);

    const start = paddings.map(p => p[0]).join(',');
    const end = paddings.map((p, i) => p[0] + xShape[i]).join(',');
    const startValue = rank > 1 ? `${type}(${start})` : `${start}`;
    const endValue = rank > 1 ? `${type}(${end})` : `${end}`;

    const xShapeValue =
        rank > 1 ? `${type}(${xShape.join(',')})` : `${xShape[0]}`;

    const leftPadCondition =
        rank > 1 ? `any(lessThan(outC, start))` : `outC < start`;
    const rightPadCondition =
        rank > 1 ? `any(greaterThanEqual(outC, end))` : `outC >= end`;

    this.userCode = `
      ${type} start = ${startValue};
      ${type} end = ${endValue};

      void main() {
        uint index = gl_GlobalInvocationID.x;
        ${type} outC = getOutputCoords();

        if(${leftPadCondition} || ${rightPadCondition}) {
          setOutput(index, ${constantValue});
        } else {
          ${type} coords = outC - start;
          ${type} xShape = ${xShapeValue};
          setOutput(index, x[getFlatIndex(coords, xShape)]);
        }
      }
    `;
  }
}
