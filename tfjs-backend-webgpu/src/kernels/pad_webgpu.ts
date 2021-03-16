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

import {util} from '@tensorflow/tfjs-core';

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class PadProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'float constantValue;';
  workPerThread = 8;
  workGroupSize: [number, number, number] = [16, 1, 1];
  xShape: number[];
  paddings: Array<[number, number]>;

  constructor(xShape: number[], paddings: Array<[number, number]>) {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.xShape = xShape;
    this.paddings = paddings;
    this.shaderKey = `pad_${paddings}`;
  }

  getUserCode(): string {
    const rank = this.xShape.length;
    const size = util.sizeFromShape(this.outputShape);
    const type = getCoordsDataType(rank);
    const start = this.paddings.map(p => p[0]).join(',');
    const end = this.paddings.map((p, i) => p[0] + this.xShape[i]).join(',');
    const startValue = rank > 1 ? `${type}(${start})` : `${start}`;
    const endValue = rank > 1 ? `${type}(${end})` : `${end}`;

    const leftPadCondition =
        rank > 1 ? `any(lessThan(outC, start))` : `outC < start`;
    const rightPadCondition =
        rank > 1 ? `any(greaterThanEqual(outC, end))` : `outC >= end`;

    const unpackedCoords = rank > 1 ?
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
        'coords';

    const userCode = `
      ${type} start = ${startValue};
      ${type} end = ${endValue};

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for (int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          if (flatIndex < ${size}) {
            ${type} outC = getCoordsFromFlatIndex(flatIndex);

            if (${leftPadCondition} || ${rightPadCondition}) {
              setOutput(flatIndex, constantValue);
            } else {
              ${type} coords = outC - start;
              setOutput(flatIndex, getX(${unpackedCoords}));
            }
          }
        }
      }
    `;
    return userCode;
  }
}
