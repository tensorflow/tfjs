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

import {getCoordsDataType, getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export function padCommon(shape: number[], fillZero = false): string {
  const rank = shape.length;
  const type = getCoordsDataType(rank);
  const start = shape.map((_, i) => `uniforms.pad${i}[0]`).join(',');
  const end = shape
                  .map(
                      (_, i) => `uniforms.pad${i}[0] + uniforms.xShape${
                          rank > 1 ? `[${i}]` : ''}`)
                  .join(',');
  const startValue = rank > 1 ? `${type}(${start})` : `${start}`;
  const endValue = rank > 1 ? `${type}(${end})` : `${end}`;

  const leftPadCondition =
      rank > 1 ? `any(paddedCoords < start)` : `paddedCoords < start`;
  const rightPadCondition =
      rank > 1 ? `any(paddedCoords >= end)` : `paddedCoords >= end`;

  const unpackedCoords = rank > 1 ?
      ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
      'coords';
  return `
        let start = ${startValue};
        let end = ${endValue};
        if (${leftPadCondition} || ${rightPadCondition}) {
          setOutputAtIndex(index, ${fillZero ? 0.0 : 'uniforms.constantValue'});
        } else {
          let coords = paddedCoords - start;
          setOutputAtIndex(index, getX(${unpackedCoords}));
        }
  `;
}

export class PadProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'constantValue : f32,';
  workgroupSize: [number, number, number] = [64, 1, 1];
  xShape: number[];
  size = true;

  constructor(xShape: number[], paddings: Array<[number, number]>) {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    paddings.map((_, i) => {
      this.uniforms += ` pad${i} : vec2<i32>,`;
    });
    this.xShape = xShape;
    this.shaderKey = 'pad';
  }

  getUserCode(): string {
    const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let paddedCoords = getCoordsFromIndex(index);
          ${padCommon(this.xShape)}
        }
      }
    `;
    return userCode;
  }
}
